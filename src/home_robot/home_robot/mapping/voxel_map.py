# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.motion.robot import Robot
from home_robot.motion.space import XYT


class SparseVoxelMapNavigationSpace(XYT):
    """subclass for sampling XYT states from explored space"""

    def __init__(
        self,
        voxel_map: SparseVoxelMap,
        robot: Robot,
        step_size: float = 0.1,
        use_orientation: bool = False,
        orientation_resolution: int = 64,
    ):
        self.robot = robot
        self.step_size = step_size
        self.voxel_map = voxel_map
        self.create_collision_masks(orientation_resolution)

        # Always use 3d states
        self.use_orientation = use_orientation
        if self.use_orientation:
            self.dof = 3
        else:
            self.dof = 2

    def create_collision_masks(
        self, orientation_resolution: int, show_all: bool = False
    ):
        """Create a set of orientation masks

        Args:
            orientation_resolution: number of bins to break it into
        """
        self._footprint = self.robot.get_footprint()
        self._orientation_resolution = 64
        self._oriented_masks = []

        # NOTE: this is just debug code - lets you see waht the masks look like
        assert not show_all or orientation_resolution == 64

        for i in range(orientation_resolution):
            theta = i * 2 * np.pi / orientation_resolution
            mask = self._footprint.get_rotated_mask(
                self.voxel_map.grid_resolution, angle_radians=theta
            )
            if show_all:
                plt.subplot(8, 8, i + 1)
                plt.axis("off")
                plt.imshow(mask.cpu().numpy())
            self._oriented_masks.append(mask)
        if show_all:
            plt.show()

    def _get_theta_index(self, theta: float) -> int:
        """gets the index associated with theta here"""
        assert (
            theta >= 0 and theta <= 2 * np.pi
        ), "only angles between 0 and 2*PI allowed"
        theta_idx = np.round((theta / (2 * np.pi) * self._orientation_resolution) - 0.5)
        if theta_idx == self._orientation_resolution:
            theta_idx = 0
        return int(theta_idx)

    def is_valid(self, state: torch.Tensor) -> bool:
        """Check to see if state is valid; i.e. if there's any collisions if mask is at right place"""
        assert len(state) == 3
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        ok = self.voxel_map.xyt_is_safe(state[:2])
        if not ok:
            # This was
            return False

        # Now sample mask at this location
        theta_idx = self._get_theta_index(state[-1])
        mask = self._oriented_masks[theta_idx]
        assert mask.shape[0] == mask.shape[1], "square masks only for now"
        dim = mask.shape[0]
        half_dim = dim // 2
        grid_xy = self.voxel_map.xy_to_grid_coords(state[:2])
        x0 = int(grid_xy[0]) - half_dim
        x1 = int(grid_xy[0]) + half_dim + 1
        y0 = int(grid_xy[1]) - half_dim
        y1 = int(grid_xy[1]) + half_dim + 1

        obstacles, explored = self.voxel_map.get_2d_map()
        crop_obs = obstacles[x0:x1, y0:y1]
        crop_exp = explored[x0:x1, y0:y1]

        collision = torch.any(crop_obs & mask)
        unknown_if_safe = torch.any(~crop_exp & mask)

        return (not collision) and (not unknown_if_safe)

    def sample_valid_location(self, max_tries: int = 100) -> Optional[torch.Tensor]:
        """Return a state that's valid and that we can move to.

        Args:
            max_tries(int): number of times to re-sample if cannot find a viable location.

        Returns:
            xyt(Tensor): a free space location, explored and collision-free
        """

        for i in range(max_tries):
            xyt = torch.rand(3) * np.pi * 2
            point = self.voxel_map.sample_explored()
            xyt[:2] = point
            if self.is_valid(xyt):
                return xyt
        else:
            return None

    def sample(self) -> np.ndarray:
        """Sample any position that corresponds to an "explored" location. Goals are valid if they are within a reasonable distance of explored locations. Paths through free space are ok and don't collide.

        Since our motion planners currently use numpy, we'll stick with that for the return type for now.
        """

        # Sample any point which is explored and not an obstacle
        # Sampled points are convertd to CPU for now
        point = self.voxel_map.sample_explored()

        # Create holder
        state = np.zeros(3)
        state[:2] = point[0].cpu().numpy()

        # Sample a random orientation
        state[-1] = np.random.random() * 2 * np.pi
        return state
