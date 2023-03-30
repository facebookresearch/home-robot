# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import scipy
import skimage.morphology
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN

from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.utils.morphology import binary_dilation


class FrontierExplorationPolicy(nn.Module):
    """
    Policy to select high-level goals for exploration
    """

    def __init__(self, exploration_strategy: str):
        super().__init__()
        assert exploration_strategy in ["seen_frontier", "been_close_to_frontier"]
        self.exploration_strategy = exploration_strategy

        self.dilate_explored_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(10))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.select_border_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(1))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )

    @property
    def goal_update_steps(self):
        return 1

    def forward(self, map_features):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 9, M, M)

        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
        """
        # otherwise, do frontier exploration
        goal_map = self.explore(map_features)
        return goal_map

    def get_frontier_map(self, map_features):
        # Select unexplored area
        if self.exploration_strategy == "seen_frontier":
            frontier_map = (map_features[:, [MC.EXPLORED_MAP], :, :] == 0).float()
        elif self.exploration_strategy == "been_close_to_frontier":
            frontier_map = (map_features[:, [MC.BEEN_CLOSE_MAP], :, :] == 0).float()

        # Dilate explored area
        frontier_map = 1 - binary_dilation(
            1 - frontier_map, self.dilate_explored_kernel
        )

        # Select the frontier
        frontier_map = (
            binary_dilation(frontier_map, self.select_border_kernel) - frontier_map
        )
        return frontier_map

    def explore(self, map_features):
        """Explore closest unexplored region."""
        goal_map = self.get_frontier_map(map_features)
        return goal_map
