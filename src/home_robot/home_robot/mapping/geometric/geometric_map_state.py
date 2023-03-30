# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch

from home_robot.mapping.map_utils import MapSizeParameters, init_map_and_pose_for_env
from home_robot.mapping.semantic.constants import MapConstants as MC


class GeometricMapState:
    """
    This class holds a dense 2D geometric map with the global and local map and sensor pose,
    as well as the agent's current goal in the local map.

    Map proposed in:
    Object Goal Navigation using Goal-Oriented Semantic Exploration
    https://arxiv.org/pdf/2007.00643.pdf
    https://github.com/devendrachaplot/Object-Goal-Navigation
    """

    def __init__(
        self,
        device: torch.device,
        num_environments: int,
        map_resolution: int,
        map_size_cm: int,
        global_downscaling: int,
    ):
        """
        Arguments:
            device: torch device on which to store map state
            num_environments: number of parallel maps (always 1 in real-world but
             multiple in simulation)
            map_resolution: size of map bins (in centimeters)
            map_size_cm: global map size (in centimetres)
            global_downscaling: ratio of global over local map
        """
        self.device = device
        self.num_environments = num_environments

        self.map_size_parameters = MapSizeParameters(
            map_resolution, map_size_cm, global_downscaling
        )
        self.resolution = map_resolution
        self.global_map_size_cm = map_size_cm
        self.global_downscaling = global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        self.global_map_size = self.global_map_size_cm // self.resolution
        self.local_map_size = self.local_map_size_cm // self.resolution

        # Map consists of multiple channels (5 NON_SEM_CHANNELS followed by semantic channels) containing the following:
        # 0: Obstacle Map
        # 1: Explored Area
        # 2: Current Agent Location
        # 3: Past Agent Locations
        # 4: Regions agent has been close to
        num_channels = MC.NON_SEM_CHANNELS

        self.global_map = torch.zeros(
            self.num_environments,
            num_channels,
            self.global_map_size,
            self.global_map_size,
            device=self.device,
        )
        self.local_map = torch.zeros(
            self.num_environments,
            num_channels,
            self.local_map_size,
            self.local_map_size,
            device=self.device,
        )

        # Global and local (x, y, o) sensor pose
        self.global_pose = torch.zeros(self.num_environments, 3, device=self.device)
        self.local_pose = torch.zeros(self.num_environments, 3, device=self.device)

        # Origin of local map (3rd dimension stays 0)
        self.origins = torch.zeros(self.num_environments, 3, device=self.device)

        # Local map boundaries
        self.lmb = torch.zeros(
            self.num_environments, 4, dtype=torch.int32, device=self.device
        )

        # Binary map encoding agent high-level goal
        self.goal_map = np.zeros(
            (self.num_environments, self.local_map_size, self.local_map_size)
        )
        self.frontier_map = np.zeros(
            (self.num_environments, self.local_map_size, self.local_map_size)
        )

    def init_map_and_pose(self):
        """Initialize global and local map and sensor pose variables."""
        for e in range(self.num_environments):
            self.init_map_and_pose_for_env(e)

    def init_map_and_pose_for_env(self, e: int):
        """Initialize global and local map and sensor pose variables for
        a specific environment.
        """
        init_map_and_pose_for_env(
            e,
            self.local_map,
            self.global_map,
            self.local_pose,
            self.global_pose,
            self.lmb,
            self.origins,
            self.map_size_parameters,
        )
        self.goal_map[e] *= 0.0

    def update_frontier_map(self, e: int, frontier_map: np.ndarray):
        """Update the current exploration frontier."""
        self.frontier_map[e] = frontier_map

    def get_frontier_map(self, e: int):
        return self.frontier_map[e]

    def update_global_goal_for_env(self, e: int, goal_map: np.ndarray):
        """Update global goal for a specific environment with the goal action chosen
        by the policy.

        Arguments:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
        """
        self.goal_map[e] = goal_map

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_obstacle_map(self, e) -> np.ndarray:
        """Get local obstacle map for an environment."""
        return np.copy(self.local_map[e, MC.OBSTACLE_MAP, :, :].cpu().float().numpy())

    def get_explored_map(self, e) -> np.ndarray:
        """Get local explored map for an environment."""
        return np.copy(self.local_map[e, MC.EXPLORED_MAP, :, :].cpu().float().numpy())

    def get_visited_map(self, e) -> np.ndarray:
        """Get local visited map for an environment."""
        return np.copy(self.local_map[e, MC.VISITED_MAP, :, :].cpu().float().numpy())

    def get_been_close_map(self, e) -> np.ndarray:
        """Get map showing regions the agent has been close to"""
        return np.copy(self.local_map[e, MC.BEEN_CLOSE_MAP, :, :].cpu().float().numpy())

    def get_planner_pose_inputs(self, e) -> np.ndarray:
        """Get local planner pose inputs for an environment.

        Returns:
            planner_pose_inputs with 7 dimensions:
             1-3: Global pose
             4-7: Local map boundaries
        """
        return (
            torch.cat([self.local_pose[e] + self.origins[e], self.lmb[e]])
            .cpu()
            .float()
            .numpy()
        )

    def get_goal_map(self, e) -> np.ndarray:
        """Get binary goal map encoding current global goal for an
        environment."""
        return self.goal_map[e]
