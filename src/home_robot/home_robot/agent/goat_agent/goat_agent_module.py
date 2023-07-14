import time

import torch.nn as nn
import torch
from typing import Tuple

from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)
from home_robot.navigation_policy.language_navigation.languagenav_frontier_exploration_policy import (
    LanguageNavFrontierExplorationPolicy,
)
from home_robot.navigation_policy.exploration.frontier_exploration_policy import (
    FrontierExplorationPolicy,
)

# Do we need to visualize the frontier as we explore?
debug_frontier_map = False


class GoatAgentModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.semantic_map_module = Categorical2DSemanticMapModule(
            frame_height=config.ENVIRONMENT.frame_height,
            frame_width=config.ENVIRONMENT.frame_width,
            camera_height=config.ENVIRONMENT.camera_height,
            hfov=config.ENVIRONMENT.hfov,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
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
        )
        self.policy = LanguageNavFrontierExplorationPolicy(
            exploration_strategy=config.AGENT.exploration_strategy
        )
        self.frontier_only_policy = FrontierExplorationPolicy(exploration_strategy=config.AGENT.exploration_strategy)
        self.goal_policy_config = config.AGENT.SUPERGLUE

    @property
    def goal_update_steps(self):
        return self.policy.goal_update_steps

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
    ):
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
            blacklist_target,
        )

        # t1 = time.time()
        # print(f"[Semantic mapping] Total time: {t1 - t0:.2f}")

        if matches is not None or confidence is not None:
            explore_map = self.frontier_only_policy.get_frontier_map(seq_map_features.flatten(0, 1)[:, :-1])
            seq_goal_map[seq_found_goal[:, 0] == 0] = explore_map[seq_found_goal[:, 0] == 0]
            # predict if the goal is found and where it is.
            seq_goal_map, seq_found_goal = self.superglue(
                seq_goal_map, seq_found_goal, final_local_map, matches, confidence
            )
            seq_goal_map = seq_goal_map.view(
                batch_size, sequence_length, *seq_goal_map.shape[-2:]
            )
            frontier_map = explore_map
        
        else:
            # Predict high-level goals from map features
            # batched across sequence length x num environments
            map_features = seq_map_features.flatten(0, 1)
            if seq_object_goal_category is not None:
                seq_object_goal_category = seq_object_goal_category.flatten(0, 1)

            # Compute the goal map
            goal_map, found_goal = self.policy(
                map_features,
                seq_object_goal_category,
                reject_visited_targets=reject_visited_targets,
            )

            seq_goal_map = goal_map.view(batch_size, sequence_length, *goal_map.shape[-2:])
            seq_found_goal = found_goal.view(batch_size, sequence_length)

            # Compute the frontier map here
            frontier_map = self.policy.get_frontier_map(map_features)
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
