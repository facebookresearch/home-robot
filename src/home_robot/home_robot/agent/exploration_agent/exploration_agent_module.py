# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time

import torch.nn as nn

from home_robot.mapping.geometric.geometric_map_module import GeometricMapModule
from home_robot.navigation_policy.exploration.frontier_exploration_policy import (
    FrontierExplorationPolicy,
)

# Do we need to visualize the frontier as we explore?
debug_frontier_map = False


class ExplorationAgentModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.geometric_map_module = GeometricMapModule(
            frame_height=config.ENVIRONMENT.frame_height,
            frame_width=config.ENVIRONMENT.frame_width,
            camera_height=config.ENVIRONMENT.camera_height,
            hfov=config.ENVIRONMENT.hfov,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            vision_range=config.AGENT.SEMANTIC_MAP.vision_range,
            explored_radius=config.AGENT.SEMANTIC_MAP.explored_radius,
            been_close_to_radius=config.AGENT.SEMANTIC_MAP.been_close_to_radius,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            du_scale=config.AGENT.SEMANTIC_MAP.du_scale,
            exp_pred_threshold=config.AGENT.SEMANTIC_MAP.exp_pred_threshold,
            map_pred_threshold=config.AGENT.SEMANTIC_MAP.map_pred_threshold,
        )
        self.policy = FrontierExplorationPolicy(
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
    ):
        """Update maps and poses with a sequence of observations, and predict
        high-level goals from map features.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation)
             of shape (batch_size, sequence_length, 3 + 1,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_dones: sequence of (batch_size, sequence_length) done flags that
             indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            seq_camera_poses: sequence of (batch_size, sequence_length, 4, 4) camera poses
            init_local_map: initial local map before any updates of shape
             (batch_size, 4, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, 4, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)

        Returns:
            seq_goal_map: sequence of binary maps encoding goal(s) of shape
             (batch_size, sequence_length, M, M)
            final_local_map: final local map after all updates of shape
             (batch_size, 4, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, 4, M * ds, M * ds)
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
        ) = self.geometric_map_module(
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

        # t1 = time.time()
        # print(f"[Semantic mapping] Total time: {t1 - t0:.2f}")

        # Predict high-level goals from map features
        # batched across sequence length x num environments
        map_features = seq_map_features.flatten(0, 1)
        goal_map = self.policy(map_features)
        seq_goal_map = goal_map.view(batch_size, sequence_length, *goal_map.shape[-2:])

        # Compute the frontier map here
        frontier_map = goal_map
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
            seq_frontier_map,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        )
