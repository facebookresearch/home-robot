# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.morphology import binary_dilation, disk

from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.motion.robot import Robot
from home_robot.motion.space import XYT
from home_robot.utils.morphology import (
    expand_mask,
    find_closest_point_on_mask,
    get_edges,
)


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

    def draw_state_on_grid(
        self, img: np.ndarray, state: np.ndarray, weight: int = 10
    ) -> np.ndarray:
        """Helper function to draw masks on image"""
        grid_xy = self.voxel_map.xy_to_grid_coords(state[:2])
        mask = self.get_oriented_mask(state[2])
        x0 = int(np.round(float(grid_xy[0] - mask.shape[0] // 2)))
        x1 = int(np.round(float(grid_xy[0] + mask.shape[0] // 2 + 1)))
        y0 = int(np.round(float(grid_xy[1] - mask.shape[1] // 2)))
        y1 = int(np.round(float(grid_xy[1] + mask.shape[1] // 2 + 1)))
        assert x1 - x0 == mask.shape[0], "crop shape incorrect"
        assert y1 - y0 == mask.shape[1], "crop shape incorrect"
        img[x0:x1, y0:y1] += mask * weight
        return img

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

    def distance(self, q0: np.ndarray, q1: np.ndarray) -> float:
        """Return distance between q0 and q1."""
        assert len(q0) == 3, "must use 3 dimensions for current state"
        assert len(q1) == 3 or len(q1) == 2, "2 or 3 dimensions for goal"
        if len(q1) == 3:
            # Measure to the final position exactly
            return np.linalg.norm(q0 - q1)
        else:
            # Measure only to the final goal x/y position
            return np.linalg.norm(q0[:2] - q1[:2])

    def extend(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        """extend towards another configuration in this space.
        TODO: we can set the classes here, right now assuming still np.ndarray"""
        assert len(q0) == 3, "initial configuration must be 3d"
        assert len(q1) == 3 or len(q1) == 2, "final configuration can be 2d or 3d"
        dxy = q1[:2] - q0[:2]
        step = dxy / np.linalg.norm(dxy) * self.step_size
        xy = q0[:2]
        if np.linalg.norm(q1[:2] - q0[:2]) > self.step_size:
            # Compute theta looking at new goal point
            new_theta = math.atan2(-dxy[1], -dxy[0])
            if new_theta < 0:
                new_theta += 2 * np.pi
            xy = q0[:2] + step
            # First, turn in the right direction
            yield np.array([xy[0], xy[1], new_theta])
            # Now take steps towards the right goal
            while np.linalg.norm(xy - q1[:2]) > self.step_size:
                xy = xy + step
                yield np.array([xy[0], xy[1], new_theta])
        # At the end, rotate into the correct orientation
        yield q1

    def _get_theta_index(self, theta: float) -> int:
        """gets the index associated with theta here"""
        if theta < 0:
            theta += 2 * np.pi
        if theta >= 2 * np.pi:
            theta -= 2 * np.pi
        assert (
            theta >= 0 and theta <= 2 * np.pi
        ), "only angles between 0 and 2*PI allowed"
        theta_idx = np.round((theta / (2 * np.pi) * self._orientation_resolution) - 0.5)
        if theta_idx == self._orientation_resolution:
            theta_idx = 0
        return int(theta_idx)

    def get_oriented_mask(self, theta: float) -> torch.Tensor:
        theta_idx = self._get_theta_index(theta)
        return self._oriented_masks[theta_idx]

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
        mask = self.get_oriented_mask(state[-1])
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

    def sample_frontier(
        self,
        max_tries_per_size: int = 100,
        min_size: int = 5,
        max_size: int = 10,
        debug: bool = False,
        verbose: bool = False,
    ) -> Optional[torch.Tensor]:
        """Sample a valid location on the current frontier. Works by finding the edges of "explored" that are not obstacles.

        Args:
            max_tries_per_size(int): number for rejection sampling
            min_size(int): min radius of filter for growing frontier
            max_size(int): max radius of filter for growing frontier
            debug(bool): show visualizations of frontiers
        """

        # Get the masks from our 3d map
        obstacles, explored = self.voxel_map.get_2d_map()

        # Extract edges from our explored mask
        edges = get_edges(explored)

        # Do not explore obstacles any more
        frontier_edges = edges & ~obstacles

        for radius in range(min_size, max_size + 1):
            # Now we apply this filter and try to sample a goal position
            if verbose:
                print("[VOXEL MAP: sampling] sampling margin of size", radius)
            expanded_frontier = expand_mask(frontier_edges, radius)
            # TODO: should we do this or not?
            # Make sure not to sample things that will just be in obstacles
            # expanded_obstacles = expand_mask(obstacles, radius)

            # Mask where we will look at
            outside_frontier = expanded_frontier & ~explored & ~obstacles

            # Mask where we will sample locations to move to
            expanded_frontier = expanded_frontier & explored & ~obstacles

            if debug:
                import matplotlib.pyplot as plt

                plt.subplot(1, 3, 1)
                plt.imshow(frontier_edges.cpu().numpy())
                plt.subplot(1, 3, 2)
                plt.imshow(expanded_frontier.cpu().numpy())
                plt.subplot(1, 3, 3)
                plt.imshow(outside_frontier.cpu().numpy())
                plt.show()

            # TODO: this really should not be random at all
            valid_indices = torch.nonzero(expanded_frontier, as_tuple=False)
            if valid_indices.size(0) == 0:
                continue

            # Rejection sampling:
            # - Find a point that we could potentially move to
            # - Compute a position and orientation
            # - Check to see if we can actually move there
            # - If so, return it
            for i in range(max_tries_per_size):
                random_index = torch.randint(valid_indices.size(0), (1,))
                # self.grid_coords_to_xy(valid_indices[random_index])
                point_grid_coords = valid_indices[random_index]
                outside_point = find_closest_point_on_mask(
                    outside_frontier, point_grid_coords.float()
                )

                # convert back
                point = self.voxel_map.grid_coords_to_xy(point_grid_coords)
                theta = math.atan2(
                    outside_point[1] - point_grid_coords[0, 1],
                    outside_point[0] - point_grid_coords[0, 0],
                )

                # Ensure angle is in 0 to 2 * PI
                if theta < 0:
                    theta += 2 * np.pi

                xyt = torch.zeros(3)
                xyt[:2] = point
                xyt[2] = theta

                # Check to see if this point is valid
                if verbose:
                    print("[VOXEL MAP: sampling]", radius, i, "sampled", xyt)
                if self.is_valid(xyt):
                    return xyt

        # We failed to find anything useful
        return None

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
