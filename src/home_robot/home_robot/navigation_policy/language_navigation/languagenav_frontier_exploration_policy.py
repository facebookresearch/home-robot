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


class LanguageNavFrontierExplorationPolicy(nn.Module):
    """
    Policy to select high-level goals for Object Goal Navigation:
    go to object goal if it is mapped and explore frontier (closest
    unexplored region) otherwise.
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

    def reach_single_category(self, map_features, category, reject_visited_targets):
        # if the goal is found, reach it
        # goal_map, found_goal = self.reach_goal_if_in_map(
        #     map_features, category, reject_visited_targets=reject_visited_targets
        # )
        # otherwise, do frontier exploration
        batch_size, _, height, width = map_features.shape
        device = map_features.device
        # Otherwise, set closest frontier as the goal
        goal_map = torch.zeros((batch_size, height, width), device=device)
        found_goal = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )
        goal_map = self.explore_otherwise(map_features, goal_map, found_goal)
        return goal_map, found_goal

    def forward(
        self,
        map_features,
        object_category=None,
        reject_visited_targets=False,
    ):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 9 + num_sem_categories, M, M)
            object_category: object goal category
        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
            goal category of shape (batch_size,)
        """
        assert object_category is not None

        # Here, the goal is specified by a single object
        return self.reach_single_category(
            map_features, object_category, reject_visited_targets
        )

    def cluster_filtering(self, m):
        # m is a 480x480 goal map
        if not m.any():
            return m
        device = m.device

        # cluster goal points
        k = DBSCAN(eps=4, min_samples=1)
        m = m.cpu().numpy()
        data = np.array(m.nonzero()).T
        k.fit(data)

        # mask all points not in the largest cluster
        mode = scipy.stats.mode(k.labels_, keepdims=True).mode.item()
        mode_mask = (k.labels_ != mode).nonzero()
        x = data[mode_mask]

        m_filtered = np.copy(m)
        m_filtered[x] = 0.0
        m_filtered = torch.tensor(m_filtered, device=device)

        return m_filtered

    def reach_goal_if_in_map(
        self,
        map_features,
        goal_category,
        reject_visited_targets=False,
    ):
        """If the desired goal is in the semantic map, reach it."""
        batch_size, _, height, width = map_features.shape
        device = map_features.device

        goal_map = torch.zeros((batch_size, height, width), device=device)
        found_goal_current = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for e in range(batch_size):
            # if the category goal was not found previously
            if not found_goal_current[e]:
                # the category to navigate to
                category_map = map_features[
                    e, goal_category[e] + 2 * MC.NON_SEM_CHANNELS, :, :
                ]

                if reject_visited_targets:
                    # remove the target objects that the agent has already been close to
                    category_map = category_map * (
                        1 - map_features[e, MC.BLACKLISTED_TARGETS_MAP, :, :]
                    )
                # if the desired category is found with required constraints, set goal for navigation
                if (category_map == 1).sum() > 0:
                    goal_map[e] = category_map == 1
                    found_goal_current[e] = True
        return goal_map, found_goal_current

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

    def explore_otherwise(self, map_features, goal_map, found_goal):
        """Explore closest unexplored region otherwise."""
        frontier_map = self.get_frontier_map(map_features)
        batch_size = map_features.shape[0]
        for e in range(batch_size):
            if not found_goal[e]:
                goal_map[e] = frontier_map[e]

        return goal_map
