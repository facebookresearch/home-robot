import time

import torch.nn as nn

from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)
from home_robot.navigation_policy.language_navigation.languagenav_frontier_exploration_policy import (
    LanguageNavFrontierExplorationPolicy,
)

# Do we need to visualize the frontier as we explore?
debug_frontier_map = False


class LanguageNavAgentModule(nn.Module):
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
        init_local_map,
        init_global_map,
        init_local_pose,
        init_global_pose,
        init_lmb,
        init_origins,
        seq_object_goal_category=None,
        seq_start_recep_goal_category=None,
        seq_end_recep_goal_category=None,
        seq_nav_to_recep=None,
        reject_visited_targets=False,
        blacklist_target=False,
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
            seq_start_recep_goal_category: sequence of start recep goal categories of shape
             (batch_size, sequence_length, 1)
            seq_end_recep_goal_category: sequence of end recep goal categories of shape
             (batch_size, sequence_length, 1)
            seq_nav_to_recep: sequence of binary digits indicating if navigation is to object or end receptacle of shape
             (batch_size, 1)
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
        if debug_frontier_map:
            import matplotlib.pyplot as plt

            plt.subplot(121)
            plt.imshow(seq_frontier_map[0, 0].numpy())
            plt.subplot(122)
            plt.imshow(goal_map[0].numpy())
            plt.show()
            breakpoint()
        # t2 = time.time()
        # print(f"[Policy] Total time: {t2 - t1:.2f}")

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
