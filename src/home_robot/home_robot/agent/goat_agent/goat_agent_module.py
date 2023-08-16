import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)
from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory
from home_robot.navigation_policy.language_navigation.languagenav_frontier_exploration_policy import (
    LanguageNavFrontierExplorationPolicy,
)

from .goat_matching import GoatMatching

# Do we need to visualize the frontier as we explore?
debug_frontier_map = False


class GoatAgentModule(nn.Module):
    def __init__(
        self,
        config,
        matching: GoatMatching,
        instance_memory: Optional[InstanceMemory] = None,
    ):
        super().__init__()
        self.matching = matching
        self.instance_memory = instance_memory
        self.goal_inst = None
        self.instance_goal_found = False
        self.num_sem_categories = config.AGENT.SEMANTIC_MAP.num_sem_categories
        self.semantic_map_module = Categorical2DSemanticMapModule(
            frame_height=config.ENVIRONMENT.frame_height,
            frame_width=config.ENVIRONMENT.frame_width,
            camera_height=config.ENVIRONMENT.camera_height,
            hfov=config.ENVIRONMENT.hfov,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            max_depth=config.AGENT.SEMANTIC_MAP.max_depth,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            vision_range=config.AGENT.SEMANTIC_MAP.vision_range,
            explored_radius=config.AGENT.SEMANTIC_MAP.explored_radius,
            been_close_to_radius=config.AGENT.SEMANTIC_MAP.been_close_to_radius,
            target_blacklisting_radius=config.AGENT.SEMANTIC_MAP.target_blacklisting_radius,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            du_scale=config.AGENT.SEMANTIC_MAP.du_scale,
            cat_pred_threshold=config.AGENT.SEMANTIC_MAP.cat_pred_threshold,
            exp_pred_threshold=config.AGENT.SEMANTIC_MAP.exp_pred_threshold,
            map_pred_threshold=config.AGENT.SEMANTIC_MAP.map_pred_threshold,
            must_explore_close=config.AGENT.SEMANTIC_MAP.must_explore_close,
            min_obs_height_cm=config.AGENT.SEMANTIC_MAP.min_obs_height_cm,
            dilate_obstacles=config.AGENT.SEMANTIC_MAP.dilate_obstacles,
            dilate_size=config.AGENT.SEMANTIC_MAP.dilate_size,
            dilate_iter=config.AGENT.SEMANTIC_MAP.dilate_iter,
            record_instance_ids=getattr(
                config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
            ),
            instance_memory=instance_memory,
            max_instances=getattr(config.AGENT.SEMANTIC_MAP, "max_instances", 0),
            evaluate_instance_tracking=getattr(
                config.ENVIRONMENT, "evaluate_instance_tracking", False
            ),
            exploration_type=config.AGENT.SEMANTIC_MAP.exploration_type,
        )
        self.policy = LanguageNavFrontierExplorationPolicy(
            exploration_strategy=config.AGENT.exploration_strategy
        )
        self.goal_policy_config = config.AGENT.SUPERGLUE
        self.instance_goal_found = False
        self.goal_inst = None

    def reset_sub_episode(self):
        self.instance_goal_found = False
        self.goal_inst = None

    def reset(self):
        self.reset_sub_episode()

    @property
    def goal_update_steps(self):
        return self.policy.goal_update_steps

    def forward(
        self,
        seq_obs,
        seq_pose_delta,
        seq_dones,
        seq_update_global,
        seq_camera_poses,
        seq_found_goal: torch.Tensor,
        seq_goal_map: torch.Tensor,
        init_local_map,
        init_global_map,
        init_local_pose,
        init_global_pose,
        init_lmb,
        init_origins,
        seq_object_goal_category=None,
        reject_visited_targets=False,
        blacklist_target=False,
        matches=None,
        confidence=None,
        local_instance_ids=None,
        all_matches=None,
        all_confidences=None,
        instance_ids=None,
        score_thresh=0.0,
        seq_obstacle_locations=None,
        seq_free_locations=None,
    ):
        """Update maps and poses with a sequence of observations, and predict
        high-level goals from map features.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation, instance_segmentation)
             of shape (batch_size, sequence_length, 3 + 1 + num_sem_categories,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_dones: sequence of (batch_size, sequence_length) done flags that
             indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            seq_camera_poses: sequence of (batch_size, 4, 4) camera poses
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
            seq_object_goal_category: sequence of object goal categories of shape
             (batch_size, sequence_length, 1)
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
        # t0 = time.time()

        # Reset the last channel of the local map each step when found_goal=False
        if matches is not None or confidence is not None:
            # TODO:
            init_local_map[:, 21][seq_found_goal[:, 0] == 0] *= 0.0

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
            seq_obstacle_locations=seq_obstacle_locations,
            seq_free_locations=seq_free_locations,
            blacklist_target=blacklist_target,
        )

        # t1 = time.time()
        # print(f"[Semantic mapping] Total time: {t1 - t0:.2f}")

        map_features = seq_map_features.flatten(0, 1)
        # Compute the frontier map here
        frontier_map = self.policy.get_frontier_map(map_features)

        seq_goal_map[seq_found_goal[:, 0] == 0] = frontier_map[
            seq_found_goal[:, 0] == 0
        ]

        seq_goal_pose = None
        if len(all_matches) > 0 or matches is not None or self.instance_goal_found:
            (
                seq_goal_map,
                seq_found_goal,
                seq_goal_pose,
                self.instance_goal_found,
                self.goal_inst,
            ) = self.matching.select_and_localize_instance(
                seq_goal_map,
                seq_found_goal,
                final_local_map,
                seq_lmb[0],
                matches,
                confidence,
                local_instance_ids,
                self.instance_memory.local_id_to_global_id_map,
                self.instance_goal_found,
                self.goal_inst,
                all_matches=all_matches,
                all_confidences=all_confidences,
                instance_ids=instance_ids,
                score_thresh=score_thresh,
            )

            seq_goal_map = seq_goal_map.view(
                batch_size, sequence_length, *seq_goal_map.shape[-2:]
            )

        else:
            # Predict high-level goals from map features
            # batched across sequence length x num environments
            if seq_object_goal_category is not None:
                seq_object_goal_category = seq_object_goal_category.flatten(0, 1)

            # Compute the goal map
            goal_map, found_goal = self.policy(
                map_features,
                seq_object_goal_category,
                reject_visited_targets=reject_visited_targets,
            )

            seq_goal_map = goal_map.view(
                batch_size, sequence_length, *goal_map.shape[-2:]
            )
            seq_found_goal = found_goal.view(batch_size, sequence_length)

        seq_frontier_map = frontier_map.view(
            batch_size, sequence_length, *frontier_map.shape[-2:]
        )

        return (
            seq_goal_map,
            seq_found_goal,
            seq_goal_pose,
            seq_frontier_map,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        )
