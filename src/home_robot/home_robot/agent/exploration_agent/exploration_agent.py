# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.nn import DataParallel

import home_robot.utils.pose as pu
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.geometric.geometric_map_state import GeometricMapState
from home_robot.navigation_planner.discrete_planner import DiscretePlanner

from .exploration_agent_module import ExplorationAgentModule

# For visualizing exploration issues
debug_frontier_map = False


class ExplorationAgent(Agent):
    """Simple exploration agent based on a 2D geometric map"""

    # Flag for debugging data flow and task configuraiton
    verbose = False

    def __init__(self, config, device_id: int = 0):
        self.max_steps = config.AGENT.max_steps
        self.num_environments = config.NUM_ENVIRONMENTS
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0

        self._module = ExplorationAgentModule(config)

        if config.NO_GPU:
            self.device = torch.device("cpu")
            self.module = self._module
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")
            self._module = self._module.to(self.device)
            # Use DataParallel only as a wrapper to move model inputs to GPU
            self.module = DataParallel(self._module, device_ids=[self.device_id])

        self.use_dilation_for_stg = config.AGENT.PLANNER.use_dilation_for_stg
        self.geometric_map = GeometricMapState(
            device=self.device,
            num_environments=self.num_environments,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
        )
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.AGENT.PLANNER.goal_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=False,
            print_images=False,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            agent_cell_radius=agent_cell_radius,
        )

        self.goal_update_steps = self._module.goal_update_steps
        self.timesteps = None
        self.timesteps_before_goal_update = None
        self.episode_panorama_start_steps = None
        self.last_poses = None

    # ------------------------------------------------------------------
    # Inference methods to interact with vectorized simulation
    # environments
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_planner_inputs(
        self,
        obs: torch.Tensor,
        pose_delta: torch.Tensor,
        camera_pose: torch.Tensor = None,
    ) -> Tuple[List[dict], List[dict]]:
        """Prepare low-level planner inputs from an observation - this is
        the main inference function of the agent that lets it interact with
        vectorized environments.

        This function assumes that the agent has been initialized.

        Args:
            obs: current frame containing (RGB, depth, segmentation) of shape
             (num_environments, 3 + 1 + num_sem_categories, frame_height, frame_width)
            pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
             of shape (num_environments, 3)
            camera_pose: camera extrinsic pose of shape (num_environments, 4, 4)

        Returns:
            planner_inputs: list of num_environments planner inputs dicts containing
                obstacle_map: (M, M) binary np.ndarray local obstacle map
                 prediction
                sensor_pose: (7,) np.ndarray denoting global pose (x, y, o)
                 and local map boundaries planning window (gx1, gx2, gy1, gy2)
                goal_map: (M, M) binary np.ndarray denoting goal location
            vis_inputs: list of num_environments visualization info dicts containing
                explored_map: (M, M) binary np.ndarray local explored map
                 prediction
        """
        dones = torch.tensor([False] * self.num_environments)
        update_global = torch.tensor(
            [
                self.timesteps_before_goal_update[e] == 0
                for e in range(self.num_environments)
            ]
        )

        (
            goal_map,
            frontier_map,
            self.geometric_map.local_map,
            self.geometric_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.module(
            obs.unsqueeze(1),
            pose_delta.unsqueeze(1),
            dones.unsqueeze(1),
            update_global.unsqueeze(1),
            camera_pose.unsqueeze(1),
            self.geometric_map.local_map,
            self.geometric_map.global_map,
            self.geometric_map.local_pose,
            self.geometric_map.global_pose,
            self.geometric_map.lmb,
            self.geometric_map.origins,
        )

        self.geometric_map.local_pose = seq_local_pose[:, -1]
        self.geometric_map.global_pose = seq_global_pose[:, -1]
        self.geometric_map.lmb = seq_lmb[:, -1]
        self.geometric_map.origins = seq_origins[:, -1]

        goal_map = goal_map.squeeze(1).cpu().numpy()

        for e in range(self.num_environments):
            self.geometric_map.update_frontier_map(e, frontier_map[e][0])
            if self.timesteps_before_goal_update[e] == 0:
                self.geometric_map.update_global_goal_for_env(e, goal_map[e])
                self.timesteps_before_goal_update[e] = self.goal_update_steps

        self.timesteps = [self.timesteps[e] + 1 for e in range(self.num_environments)]
        self.timesteps_before_goal_update = [
            self.timesteps_before_goal_update[e] - 1
            for e in range(self.num_environments)
        ]

        if debug_frontier_map:
            import matplotlib.pyplot as plt

            plt.subplot(131)
            plt.imshow(self.geometric_map.get_frontier_map(e))
            plt.subplot(132)
            plt.imshow(frontier_map[e][0])
            plt.subplot(133)
            plt.imshow(self.geometric_map.get_goal_map(e))
            plt.show()
            breakpoint()

        planner_inputs = [
            {
                "obstacle_map": self.geometric_map.get_obstacle_map(e),
                "goal_map": self.geometric_map.get_goal_map(e),
                "frontier_map": self.geometric_map.get_frontier_map(e),
                "sensor_pose": self.geometric_map.get_planner_pose_inputs(e),
                "found_goal": False,
            }
            for e in range(self.num_environments)
        ]
        vis_inputs = [
            {
                "explored_map": self.geometric_map.get_explored_map(e),
                "been_close_map": self.geometric_map.get_been_close_map(e),
                "timestep": self.timesteps[e],
            }
            for e in range(self.num_environments)
        ]

        return planner_inputs, vis_inputs

    def reset_vectorized(self):
        """Initialize agent state."""
        self.timesteps = [0] * self.num_environments
        self.timesteps_before_goal_update = [0] * self.num_environments
        self.last_poses = [np.zeros(3)] * self.num_environments
        self.geometric_map.init_map_and_pose()

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.timesteps[e] = 0
        self.timesteps_before_goal_update[e] = 0
        self.last_poses[e] = np.zeros(3)
        self.geometric_map.init_map_and_pose_for_env(e)

    # ---------------------------------------------------------------------
    # Inference methods to interact with the robot or a single un-vectorized
    # simulation environment
    # ---------------------------------------------------------------------

    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()
        self.planner.reset()
        self.episode_panorama_start_steps = self.panorama_start_steps

    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """Act end-to-end."""
        # t0 = time.time()

        # 1 - Obs preprocessing
        (
            obs_preprocessed,
            pose_delta,
            camera_pose,
        ) = self._preprocess_obs(obs)

        # t1 = time.time()
        # print(f"[Agent] Obs preprocessing time: {t1 - t0:.2f}")

        # 2 - Semantic mapping + policy
        planner_inputs, vis_inputs = self.prepare_planner_inputs(
            obs_preprocessed,
            pose_delta,
            camera_pose=camera_pose,
        )

        # t2 = time.time()
        # print(f"[Agent] Semantic mapping and policy time: {t2 - t1:.2f}")

        # 3 - Planning
        closest_goal_map = None
        if self.timesteps[0] < self.episode_panorama_start_steps:
            action = DiscreteNavigationAction.TURN_RIGHT
        elif self.timesteps[0] > self.max_steps:
            action = DiscreteNavigationAction.STOP
        else:
            action, closest_goal_map, _, _ = self.planner.plan(
                **planner_inputs[0], use_dilation_for_stg=self.use_dilation_for_stg
            )

        # t3 = time.time()
        # print(f"[Agent] Planning time: {t3 - t2:.2f}")
        # print(f"[Agent] Total time: {t3 - t0:.2f}")
        # print()

        vis_inputs[0]["image_frame"] = obs.task_observations["image_frame"]
        vis_inputs[0]["closest_goal_map"] = closest_goal_map
        vis_inputs[0]["third_person_image"] = obs.third_person_image
        info = {**planner_inputs[0], **vis_inputs[0]}

        return action, info

    def _preprocess_obs(self, obs: Observations):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map.

        Output conventions:
            * obs_preprocessed = (1, 4, H, W) torch.Tensor
                - channels 1-3 are RGB (0 - 255 range)
                - channel 4 is depth (in cm)
            * pose_delta = (1, 3) torch.Tensor:
                - +X is forward
                - +Y is leftward
                - +theta is measured from +X to +Y
            * camera_pose = (1, 4, 4) camera extrinsics
                - +X is forward
                - +Y is leftward
                - +Z is upward
                - rotations are in 3D
        """
        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm
        obs_preprocessed = torch.cat([rgb, depth], dim=-1).unsqueeze(0)
        obs_preprocessed = obs_preprocessed.permute(0, 3, 1, 2)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_poses[0])
        ).unsqueeze(0)
        self.last_poses[0] = curr_pose

        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)
        return (
            obs_preprocessed,
            pose_delta,
            camera_pose,
        )
