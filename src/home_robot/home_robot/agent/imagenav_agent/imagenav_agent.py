# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.cluster import DBSCAN

from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.navigation_planner.discrete_planner import DiscretePlanner

from .frontier_exploration import FrontierExplorationPolicy
from .obs_preprocessor import ObsPreprocessor
from .visualizer import NavVisualizer


class IINAgentModule(nn.Module):
    """
    An agent module that maintains a 2D map, explores with FBE, and detects and
    localizes object goals from keypoint correspondences.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.semantic_map_module = Categorical2DSemanticMapModule(
            frame_height=config.frame_height,
            frame_width=config.frame_width,
            camera_height=config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[
                1
            ],
            hfov=config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov,
            num_sem_categories=config.semantic_map.num_sem_categories,
            map_size_cm=config.semantic_map.map_size_cm,
            map_resolution=config.semantic_map.map_resolution,
            vision_range=config.semantic_map.vision_range,
            explored_radius=config.semantic_map.explored_radius,
            been_close_to_radius=config.semantic_map.been_close_to_radius,
            global_downscaling=config.semantic_map.global_downscaling,
            du_scale=config.semantic_map.du_scale,
            cat_pred_threshold=config.semantic_map.cat_pred_threshold,
            exp_pred_threshold=config.semantic_map.exp_pred_threshold,
            map_pred_threshold=config.semantic_map.map_pred_threshold,
            must_explore_close=config.semantic_map.must_explore_close,
            min_obs_height_cm=config.semantic_map.min_obs_height_cm,
            dilate_obstacles=config.semantic_map.dilate_obstacles,
            dilate_size=config.semantic_map.dilate_size,
            dilate_iter=config.semantic_map.dilate_iter,
        )
        self.goal_policy_config = config.superglue
        self.exploration_policy = FrontierExplorationPolicy()

    @property
    def goal_update_steps(self) -> int:
        return self.exploration_policy.goal_update_steps

    def superglue(
        self,
        goal_map: torch.Tensor,
        found_goal: torch.Tensor,
        local_map: torch.Tensor,
        matches: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Goal detection and localization via SuperGlue"""

        score_func = self.goal_policy_config.score_function
        assert score_func in ["confidence_sum", "match_count"]

        for e in range(confidence.shape[0]):
            # if the goal category is empty, the goal can't be found
            if not local_map[e, -1].any().item():
                continue

            if score_func == "confidence_sum":
                score = confidence[e][matches[e] != -1].sum()
            else:  # match_count
                score = (matches[e] != -1).sum()

            if score < self.goal_policy_config.score_thresh:
                continue

            found_goal[e] = True
            # Set goal_map to the last channel of the local semantic map
            goal_map[e, 0] = local_map[e, -1]

        return goal_map, found_goal

    def forward(
        self,
        seq_obs: torch.Tensor,
        seq_pose_delta: torch.Tensor,
        seq_dones: torch.Tensor,
        seq_update_global: torch.Tensor,
        seq_camera_poses: Optional[torch.Tensor],
        seq_found_goal: torch.Tensor,
        seq_goal_map: torch.Tensor,
        init_local_map: torch.Tensor,
        init_global_map: torch.Tensor,
        init_local_pose: torch.Tensor,
        init_global_pose: torch.Tensor,
        init_lmb: torch.Tensor,
        init_origins: torch.Tensor,
        matches: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Update maps and poses with a sequence of observations, and predict
        high-level goals from map features.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation)
             of shape (batch_size, sequence_length, 3 + 1 + num_sem_categories,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_dones: sequence of (batch_size, sequence_length) done flags that
             indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            seq_camera_poses: sequence of (batch_size, sequence_length, 4, 4) camera poses
            init_local_map: initial local map before any updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)

        Returns:
            seq_goal_map: sequence of binary maps encoding goal(s) of shape
             (batch_size, sequence_length, M, M)
            seq_found_goal: binary variables to denote whether we found the object
             goal category of shape (batch_size, sequence_length)
            final_local_map: final local map after all updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (batch_size, sequence_length, 3)
            seq_global_pose: sequence of global poses of shape
             (batch_size, sequence_length, 3)
            seq_lmb: sequence of local map boundaries of shape
             (batch_size, sequence_length, 4)
            seq_origins: sequence of local map origins of shape
             (batch_size, sequence_length, 3)
        """
        # Reset the last channel of the local map each step when found_goal=False
        # init_local_map: [8, 21, 480, 480]
        init_local_map[:, -1][seq_found_goal[:, 0] == 0] *= 0.0

        # Update map with observations and generate map features
        batch_size, sequence_length = seq_obs.shape[:2]
        (
            seq_map_features,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.semantic_map_module(
            seq_obs,
            seq_pose_delta,
            seq_dones,
            seq_update_global,
            seq_camera_poses,
            init_local_map,
            init_global_map,
            init_local_pose,
            init_global_pose,
            init_lmb,
            init_origins,
        )

        # Predict high-level goals from map features.
        map_features = seq_map_features.flatten(0, 1)

        # the last channel of map_features is cut off -- used for goal det/loc.
        frontier_map = self.exploration_policy(map_features[:, :-1])
        seq_goal_map[seq_found_goal[:, 0] == 0] = frontier_map[
            seq_found_goal[:, 0] == 0
        ]

        # predict if the goal is found and where it is.
        seq_goal_map, seq_found_goal = self.superglue(
            seq_goal_map, seq_found_goal, final_local_map, matches, confidence
        )
        seq_goal_map = seq_goal_map.view(
            batch_size, sequence_length, *seq_goal_map.shape[-2:]
        )

        seq_frontier_map = frontier_map.view(
            batch_size, sequence_length, *frontier_map.shape[-2:]
        )

        return (
            seq_goal_map,
            seq_found_goal,
            seq_frontier_map,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        )


class ImageNavAgent(Agent):
    """Class for a modular agent that navigates to objects specified by images."""

    def __init__(self, config: DictConfig, device_id: int = 0) -> None:
        self.device = torch.device(f"cuda:{device_id}")
        self.obs_preprocessor = ObsPreprocessor(config, self.device)

        self.max_steps = config.habitat.environment.max_episode_steps
        self.num_environments = 1

        self._module = IINAgentModule(config).to(self.device)

        self.use_dilation_for_stg = config.planner.use_dilation_for_stg
        self.verbose = config.planner.verbose

        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=config.semantic_map.num_sem_categories,
            map_resolution=config.semantic_map.map_resolution,
            map_size_cm=config.semantic_map.map_size_cm,
            global_downscaling=config.semantic_map.global_downscaling,
        )
        agent_radius_cm = config.habitat.simulator.agents.main_agent.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.semantic_map.map_resolution)
        )
        self.planner = DiscretePlanner(
            turn_angle=config.habitat.simulator.turn_angle,
            collision_threshold=config.planner.collision_threshold,
            step_size=config.planner.step_size,
            obs_dilation_selem_radius=config.planner.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.planner.goal_dilation_selem_radius,
            map_size_cm=config.semantic_map.map_size_cm,
            map_resolution=config.semantic_map.map_resolution,
            visualize=False,
            print_images=False,
            dump_location=config.dump_location,
            exp_name=config.exp_name,
            agent_cell_radius=agent_cell_radius,
            min_obs_dilation_selem_radius=config.planner.min_obs_dilation_selem_radius,
            map_downsample_factor=config.planner.map_downsample_factor,
            map_update_frequency=config.planner.map_update_frequency,
            discrete_actions=config.planner.discrete_actions,
        )

        self.goal_filtering = config.semantic_prediction.goal_filtering
        self.goal_update_steps = self._module.goal_update_steps
        self.timesteps = None
        self.timesteps_before_goal_update = None
        self.found_goal = torch.zeros(
            self.num_environments, 1, dtype=bool, device=self.device
        )
        self.goal_map = torch.zeros(
            self.num_environments,
            1,
            *self.semantic_map.local_map.shape[2:],
            dtype=self.semantic_map.local_map.dtype,
            device=self.device,
        )

        self.visualizer = None
        if config.generate_videos:
            self.visualizer = NavVisualizer(
                num_sem_categories=config.semantic_map.num_sem_categories,
                map_size_cm=config.semantic_map.map_size_cm,
                map_resolution=config.semantic_map.map_resolution,
                print_images=config.generate_videos,
                dump_location=config.dump_location,
                exp_name=config.exp_name,
            )

    def reset(self) -> None:
        """Initialize agent state."""
        self.obs_preprocessor.reset()
        if self.visualizer is not None:
            self.visualizer.reset()

        self.timesteps = [0]
        self.timesteps_before_goal_update = [0]
        self.semantic_map.init_map_and_pose()
        self.found_goal[:] = False
        self.goal_map[:] *= 0
        self.planner.reset()

    def act(self, obs: Observations) -> DiscreteNavigationAction:
        """Act end-to-end."""
        (
            obs_preprocessed,
            pose_delta,
            camera_pose,
            matches,
            confidence,
        ) = self.obs_preprocessor.preprocess(obs)

        planner_inputs, vis_inputs = self._prepare_planner_inputs(
            obs_preprocessed, pose_delta, matches, confidence, camera_pose
        )

        closest_goal_map = None
        if self.timesteps[0] >= (self.max_steps - 1):
            action = DiscreteNavigationAction.STOP
        else:
            action, closest_goal_map, _, _ = self.planner.plan(
                **planner_inputs[0],
                use_dilation_for_stg=self.use_dilation_for_stg,
                debug=self.verbose,
            )

        if self.visualizer is not None:
            collision = obs.task_observations.get("collisions")
            if collision is None:
                collision = {"is_collision": False}
            info = {
                **planner_inputs[0],
                **vis_inputs[0],
                "semantic_frame": obs.rgb,
                "closest_goal_map": closest_goal_map,
                "last_goal_image": obs.task_observations["instance_imagegoal"],
                "last_collisions": collision,
                "last_td_map": obs.task_observations.get("top_down_map"),
            }
            self.visualizer.visualize(**info)

        return action

    @torch.no_grad()
    def _prepare_planner_inputs(
        self,
        obs: torch.Tensor,
        pose_delta: torch.Tensor,
        matches: torch.Tensor,
        confidence: torch.Tensor,
        camera_pose: Optional[torch.Tensor] = None,
    ) -> Tuple[List[dict], List[dict]]:
        """
        Determine a long-term navigation goal in 2D map space for a local policy to
        execute.
        """
        dones = torch.zeros(self.num_environments, dtype=torch.bool)
        update_global = torch.tensor(
            [
                self.timesteps_before_goal_update[e] == 0
                for e in range(self.num_environments)
            ]
        )

        (
            self.goal_map,
            self.found_goal,
            frontier_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self._module(
            obs.unsqueeze(1),
            pose_delta.unsqueeze(1),
            dones.unsqueeze(1),
            update_global.unsqueeze(1),
            camera_pose.unsqueeze(1),
            self.found_goal,
            self.goal_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
            matches,
            confidence,
        )

        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]

        goal_map = self._prep_goal_map_input()
        for e in range(self.num_environments):
            self.semantic_map.update_frontier_map(e, frontier_map[e][0].cpu().numpy())
            if self.found_goal[e].item():
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
            elif self.timesteps_before_goal_update[e] == 0:
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
                self.timesteps_before_goal_update[e] = self.goal_update_steps

        self.timesteps = [self.timesteps[e] + 1 for e in range(self.num_environments)]
        self.timesteps_before_goal_update = [
            self.timesteps_before_goal_update[e] - 1
            for e in range(self.num_environments)
        ]

        planner_inputs = [
            {
                "obstacle_map": self.semantic_map.get_obstacle_map(e),
                "goal_map": self.semantic_map.get_goal_map(e),
                "frontier_map": self.semantic_map.get_frontier_map(e),
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(e),
                "found_goal": self.found_goal[e].item(),
            }
            for e in range(self.num_environments)
        ]
        vis_inputs = [
            {
                "explored_map": self.semantic_map.get_explored_map(e),
                "timestep": self.timesteps[e],
            }
            for e in range(self.num_environments)
        ]
        if self.semantic_map.num_sem_categories > 1:
            for e in range(self.num_environments):
                vis_inputs[e]["semantic_map"] = self.semantic_map.get_semantic_map(e)

        return planner_inputs, vis_inputs

    def _prep_goal_map_input(self) -> None:
        """
        Perform optional clustering of the goal channel to mitigate noisy projection
        splatter.
        """
        goal_map = self.goal_map.squeeze(1).cpu().numpy()

        if not self.goal_filtering:
            return goal_map

        for e in range(goal_map.shape[0]):
            if not self.found_goal[e]:
                continue

            # cluster goal points
            c = DBSCAN(eps=4, min_samples=1)
            data = np.array(goal_map[e].nonzero()).T
            c.fit(data)

            # mask all points not in the largest cluster
            mode = scipy.stats.mode(c.labels_, keepdims=False).mode.item()
            mode_mask = (c.labels_ != mode).nonzero()
            x = data[mode_mask]
            goal_map_ = np.copy(goal_map[e])
            goal_map_[x] = 0.0

            # adopt masked map if non-empty
            if goal_map_.sum() > 0:
                goal_map[e] = goal_map_

        return np.ceil(goal_map)
