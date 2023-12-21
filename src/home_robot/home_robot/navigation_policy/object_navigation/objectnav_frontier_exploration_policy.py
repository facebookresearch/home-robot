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
from home_robot.utils.morphology import binary_dilation, binary_erosion


class ObjectNavFrontierExplorationPolicy(nn.Module):
    """
    Policy to select high-level goals for Object Goal Navigation:
    go to object goal if it is mapped and explore frontier (closest
    unexplored region) otherwise.
    """

    def __init__(
        self,
        exploration_strategy: str,
        num_sem_categories: int,
        explored_area_dilation_radius=10,
    ):
        super().__init__()
        assert exploration_strategy in ["seen_frontier", "been_close_to_frontier"]
        self.exploration_strategy = exploration_strategy

        self.dilate_explored_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(explored_area_dilation_radius))
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
        self.num_sem_categories = num_sem_categories

    @property
    def goal_update_steps(self):
        return 1

    def reach_single_category(self, map_features, category):
        # if the goal is found, reach it
        goal_map, found_goal = self.reach_goal_if_in_map(map_features, category)
        # otherwise, do frontier exploration
        goal_map = self.explore_otherwise(map_features, goal_map, found_goal)
        return goal_map, found_goal

    def reach_object_recep_combination(
        self, map_features, object_category, recep_category
    ):
        # First check if object (small goal) and recep category are in the same cell of the map. if found, set it as a goal
        goal_map, found_goal = self.reach_goal_if_in_map(
            map_features,
            recep_category,
            small_goal_category=object_category,
        )
        # Then check if the recep category exists in the map. if found, set it as a goal
        goal_map, found_rec_goal = self.reach_goal_if_in_map(
            map_features,
            recep_category,
            reject_visited_regions=True,
            goal_map=goal_map,
            found_goal=found_goal,
        )
        # Otherwise, set closest frontier as the goal
        goal_map = self.explore_otherwise(map_features, goal_map, found_rec_goal)
        return goal_map, found_goal

    def forward(
        self,
        map_features,
        object_category=None,
        start_recep_category=None,
        end_recep_category=None,
        instance_id=None,
        nav_to_recep=None,
    ):
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 9 + num_sem_categories, M, M)
            object_category: object goal category
            start_recep_category: start receptacle category
            end_recep_category: end receptacle category
            nav_to_recep: If both object_category and recep_category are specified, whether to navigate to receptacle
        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
            found_goal: binary variables to denote whether we found the object
            goal category of shape (batch_size,)
        """
        assert (
            object_category is not None
            or end_recep_category is not None
            or instance_id is not None
        )
        if instance_id is not None:
            instance_map = map_features[0][
                2 * MC.NON_SEM_CHANNELS
                + self.num_sem_categories : 2 * MC.NON_SEM_CHANNELS
                + 2 * self.num_sem_categories,
                :,
                :,
            ]
            if len(instance_map) != 0:
                inst_map_idx = instance_map == instance_id
                inst_map_idx = torch.argmax(torch.sum(inst_map_idx, axis=(1, 2)))
                goal_map = (
                    (instance_map[inst_map_idx] == instance_id)
                    .to(torch.float)
                    .unsqueeze(0)
                )
                if torch.sum(goal_map) == 0:
                    found_goal = torch.tensor([0])
                else:
                    found_goal = torch.tensor([1])
            else:
                # try to navigate to instance without an instance map -- explore
                # create an empty goal map
                batch_size, _, height, width = map_features.shape
                device = map_features.device
                goal_map = torch.zeros((batch_size, height, width), device=device)
                found_goal = torch.tensor([0])

            goal_map = self.explore_otherwise(map_features, goal_map, found_goal)
            return goal_map, found_goal

        elif object_category is not None and start_recep_category is not None:
            if nav_to_recep is None or end_recep_category is None:
                nav_to_recep = torch.tensor([0] * map_features.shape[0])

            # there is at least one instance in the batch where the goal is object
            if nav_to_recep.sum() < map_features.shape[0]:
                goal_map_o, found_goal_o = self.reach_object_recep_combination(
                    map_features, object_category, start_recep_category
                )
            # there is at least one instance in the batch where the goal is receptacle
            elif nav_to_recep.sum() > 0:
                goal_map_r, found_goal_r = self.reach_single_category(
                    map_features, end_recep_category
                )
            # some instances in batch may be navigating to objects (before pick skill) and some may be navigating to recep (before place skill)
            if nav_to_recep.sum() == 0:
                return goal_map_o, found_goal_o
            elif nav_to_recep.sum() == map_features.shape[0]:
                return goal_map_r, found_goal_r
            else:
                goal_map = (
                    goal_map_o * nav_to_recep.view(-1, 1, 1)
                    + (1 - nav_to_recep).view(-1, 1, 1) * goal_map_o
                )
                found_goal = (
                    found_goal_r * nav_to_recep + (1 - nav_to_recep) * found_goal_r
                )
                return goal_map, found_goal
        else:
            # Here, the goal is specified by a single object or receptacle to navigate to with no additional constraints (eg. the given object can be on any receptacle)
            goal_category = (
                object_category if object_category is not None else end_recep_category
            )
            return self.reach_single_category(map_features, goal_category)

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
        small_goal_category=None,
        reject_visited_regions=False,
        goal_map=None,
        found_goal=None,
    ):
        """If the desired goal is in the semantic map, reach it."""
        batch_size, _, height, width = map_features.shape
        device = map_features.device
        if goal_map is None and found_goal is None:
            goal_map = torch.zeros((batch_size, height, width), device=device)
            found_goal_current = torch.zeros(
                batch_size, dtype=torch.bool, device=device
            )
        else:
            # crate a fresh map
            found_goal_current = torch.clone(found_goal)
        for e in range(batch_size):
            # if the category goal was not found previously
            if not found_goal_current[e]:
                # the category to navigate to
                category_map = map_features[
                    e, goal_category[e] + 2 * MC.NON_SEM_CHANNELS, :, :
                ]
                if small_goal_category is not None:
                    # additionally check if the category has the required small object on it
                    category_map = (
                        category_map
                        * map_features[
                            e, small_goal_category[e] + 2 * MC.NON_SEM_CHANNELS, :, :
                        ]
                    )
                if reject_visited_regions:
                    # remove the receptacles that the already been close to
                    category_map = category_map * (
                        1 - map_features[e, MC.BEEN_CLOSE_MAP, :, :]
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
        else:
            raise Exception("not implemented")

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
