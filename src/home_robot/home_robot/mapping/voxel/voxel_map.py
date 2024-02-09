# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import skfmm
import skimage
import skimage.morphology
import torch

from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.motion import XYT, RobotModel
from home_robot.utils.geometry import angle_difference, interpolate_angles
from home_robot.utils.morphology import (
    binary_dilation,
    binary_erosion,
    expand_mask,
    find_closest_point_on_mask,
    get_edges,
)


class SparseVoxelMapNavigationSpace(XYT):
    """subclass for sampling XYT states from explored space"""

    # Used for making sure we do not divide by zero anywhere
    tolerance: float = 1e-8

    def __init__(
        self,
        voxel_map: SparseVoxelMap,
        robot: RobotModel,
        step_size: float = 0.1,
        rotation_step_size: float = 0.5,
        use_orientation: bool = False,
        orientation_resolution: int = 64,
        dilate_frontier_size: int = 12,
        dilate_obstacle_size: int = 2,
        extend_mode: str = "separate",
    ):
        self.robot = robot
        self.step_size = step_size
        self.rotation_step_size = rotation_step_size
        self.voxel_map = voxel_map
        self.create_collision_masks(orientation_resolution)
        self.extend_mode = extend_mode

        # Always use 3d states
        self.use_orientation = use_orientation
        if self.use_orientation:
            self.dof = 3
        else:
            self.dof = 2

        self._kernels = {}

        if dilate_frontier_size > 0:
            self.dilate_explored_kernel = torch.nn.Parameter(
                torch.from_numpy(skimage.morphology.disk(dilate_frontier_size))
                .unsqueeze(0)
                .unsqueeze(0)
                .float(),
                requires_grad=False,
            )
        else:
            self.dilate_explored_kernel = None
        if dilate_obstacle_size > 0:
            self.dilate_obstacles_kernel = torch.nn.Parameter(
                torch.from_numpy(skimage.morphology.disk(dilate_obstacle_size))
                .unsqueeze(0)
                .unsqueeze(0)
                .float(),
                requires_grad=False,
            )
        else:
            self.dilate_obstacles_kernel = None

    def draw_state_on_grid(
        self, img: np.ndarray, state: np.ndarray, weight: int = 10
    ) -> np.ndarray:
        """Helper function to draw masks on image"""
        grid_xy = self.voxel_map.xy_to_grid_coords(state[:2])
        mask = self.get_oriented_mask(state[2])
        x0 = int(np.round(float(grid_xy[0] - mask.shape[0] // 2)))
        x1 = x0 + mask.shape[0]
        y0 = int(np.round(float(grid_xy[1] - mask.shape[1] // 2)))
        y1 = y0 + mask.shape[1]
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
        """extend towards another configuration in this space. Will be either separate or joint depending on if the robot can "strafe":
        separate: move then rotate
        joint: move and rotate all at once."""
        assert len(q0) == 3, f"initial configuration must be 3d, was {q0}"
        assert (
            len(q1) == 3 or len(q1) == 2
        ), f"final configuration can be 2d or 3d, was {q1}"
        if self.extend_mode == "separate":
            return self._extend_separate(q0, q1)
        elif self.extend_mode == "joint":
            # Just default to linear interpolation, does not use rotation_step_size
            return super().extend(q0, q1)
        else:
            raise NotImplementedError(f"not supported: {self.extend_mode=}")

    def _extend_separate(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        """extend towards another configuration in this space.
        TODO: we can set the classes here, right now assuming still np.ndarray"""
        assert len(q0) == 3, f"initial configuration must be 3d, was {q0}"
        assert (
            len(q1) == 3 or len(q1) == 2
        ), f"final configuration can be 2d or 3d, was {q1}"
        dxy = q1[:2] - q0[:2]
        step = dxy / np.linalg.norm(dxy + self.tolerance) * self.step_size
        xy = q0[:2]
        if np.linalg.norm(q1[:2] - q0[:2]) > self.step_size:
            # Compute theta looking at new goal point
            new_theta = math.atan2(dxy[1], dxy[0])
            if new_theta < 0:
                new_theta += 2 * np.pi

            # TODO: oreint towards the new theta
            cur_theta = q0[-1]
            angle_diff = angle_difference(new_theta, cur_theta)
            while angle_diff > self.rotation_step_size:
                # Interpolate
                cur_theta = interpolate_angles(
                    cur_theta, new_theta, self.rotation_step_size
                )
                yield np.array([xy[0], xy[1], cur_theta])
                angle_diff = angle_difference(new_theta, cur_theta)

            xy = q0[:2] + step
            # First, turn in the right direction
            yield np.array([xy[0], xy[1], new_theta])
            # Now take steps towards the right goal
            while np.linalg.norm(xy - q1[:2]) > self.step_size:
                xy = xy + step
                yield np.array([xy[0], xy[1], new_theta])

            # Update current angle
            cur_theta = new_theta
        else:
            cur_theta = q0[-1]

        # now interpolate to goal angle
        angle_diff = angle_difference(q1[-1], cur_theta)
        while angle_diff > self.rotation_step_size:
            # Interpolate
            cur_theta = interpolate_angles(cur_theta, q1[-1], self.rotation_step_size)
            yield np.array([xy[0], xy[1], cur_theta])
            angle_diff = angle_difference(q1[-1], cur_theta)

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

    def is_valid(
        self,
        state: torch.Tensor,
        is_safe_threshold=0.95,
        debug: bool = False,
        verbose: bool = False,
    ) -> bool:
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
        x1 = x0 + dim
        y0 = int(grid_xy[1]) - half_dim
        y1 = y0 + dim

        obstacles, explored = self.voxel_map.get_2d_map()

        crop_obs = obstacles[x0:x1, y0:y1]
        crop_exp = explored[x0:x1, y0:y1]
        assert mask.shape == crop_obs.shape
        assert mask.shape == crop_exp.shape

        collision = torch.any(crop_obs & mask)

        p_is_safe = (
            torch.sum((crop_exp & mask) | ~mask) / (mask.shape[0] * mask.shape[1])
        ).item()
        is_safe = p_is_safe > is_safe_threshold
        if verbose:
            print(f"{collision=}, {is_safe=}, {p_is_safe=}, {is_safe_threshold=}")

        valid = bool((not collision) and is_safe)
        if debug:
            if collision:
                print("- state in collision")
            if not is_safe:
                print("- not safe")

            print(f"{valid=}")
            obs = obstacles.cpu().numpy().copy()
            exp = explored.cpu().numpy().copy()
            obs[x0:x1, y0:y1] = 1
            plt.subplot(321)
            plt.imshow(obs)
            plt.subplot(322)
            plt.imshow(exp)
            plt.subplot(323)
            plt.imshow(crop_obs.cpu().numpy())
            plt.title("obstacles")
            plt.subplot(324)
            plt.imshow(crop_exp.cpu().numpy())
            plt.title("explored")
            plt.subplot(325)
            plt.imshow(mask.cpu().numpy())
            plt.show()

        return valid

    def sample_near_mask(
        self,
        mask: torch.Tensor,
        radius_m: float = 0.7,
        max_tries: int = 1000,
        verbose: bool = True,
        debug: bool = False,
        look_at_any_point: bool = False,
    ) -> Optional[np.ndarray]:
        """Sample a position near the mask and return.

        Args:
            look_at_any_point(bool): robot should look at the closest point on target mask instead of average pt
        """

        obstacles, explored = self.voxel_map.get_2d_map()

        # Extract edges from our explored mask

        # Radius computed from voxel map measurements
        radius = np.ceil(radius_m / self.voxel_map.grid_resolution)
        expanded_mask = expand_mask(mask, radius)

        # TODO: was this:
        # expanded_mask = expanded_mask & less_explored & ~obstacles
        expanded_mask = expanded_mask & explored & ~obstacles

        if debug:
            import matplotlib.pyplot as plt

            plt.imshow(
                mask.int()
                + expanded_mask.int() * 10
                + explored.int()
                + obstacles.int() * 5
            )
            plt.show()

        # Where can the robot go?
        valid_indices = torch.nonzero(expanded_mask, as_tuple=False)
        if valid_indices.size(0) == 0:
            print("[VOXEL MAP: sampling] No valid goals near mask!")
            return None
        if not look_at_any_point:
            mask_indices = torch.nonzero(mask, as_tuple=False)
            outside_point = mask_indices.float().mean(dim=0)

        # maximum number of tries
        for i in range(max_tries):
            random_index = torch.randint(valid_indices.size(0), (1,))
            point_grid_coords = valid_indices[random_index]

            if look_at_any_point:
                outside_point = find_closest_point_on_mask(
                    mask, point_grid_coords.float()
                )

            # convert back
            point = self.voxel_map.grid_coords_to_xy(point_grid_coords)
            if point is None:
                print("[VOXEL MAP: sampling] ERR:", point, point_grid_coords)
                continue
            if outside_point is None:
                print(
                    "[VOXEL MAP: sampling] ERR finding closest pt:",
                    point,
                    point_grid_coords,
                    "closest =",
                    outside_point,
                )
                continue
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
            if self.is_valid(xyt, verbose=verbose):
                yield xyt

        # We failed to find anything useful
        return None

    def has_zero_contour(self, phi):
        """
        Check if a zero contour exists in the given phi array.

        Parameters:
        - phi: 2D NumPy array with boolean values.

        Returns:
        - True if a zero contour exists, False otherwise.
        """
        # Check if there are True and False values in the array
        has_true_values = np.any(phi)
        has_false_values = np.any(~phi)

        # Return True if both True and False values are present
        return has_true_values and has_false_values

    def _get_kernel(self, size: int):
        """Return a kernel for expanding/shrinking areas."""
        if size <= 0:
            return None
        if size not in self._kernels:
            kernel = torch.nn.Parameter(
                torch.from_numpy(skimage.morphology.disk(size))
                .unsqueeze(0)
                .unsqueeze(0)
                .float(),
                requires_grad=False,
            )
            self._kernels[size] = kernel
        return self._kernels[size]

    def get_frontier(
        self, expand_size: int = 5, debug: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute frontier regions of the map"""

        obstacles, explored = self.voxel_map.get_2d_map()
        # Extract edges from our explored mask
        obstacles = binary_dilation(
            obstacles.float().unsqueeze(0).unsqueeze(0), self.dilate_obstacles_kernel
        )[0, 0].bool()
        less_explored = binary_erosion(
            explored.float().unsqueeze(0).unsqueeze(0), self.dilate_explored_kernel
        )[0, 0]

        # Get the masks from our 3d map
        edges = get_edges(less_explored)

        # Do not explore obstacles any more
        traversible = explored & ~obstacles
        frontier_edges = edges & ~obstacles

        kernel = self._get_kernel(expand_size)
        if kernel is not None:
            expanded_frontier = binary_dilation(
                frontier_edges.float().unsqueeze(0).unsqueeze(0),
                kernel,
            )[0, 0].bool()
        else:
            # This is a bad idea, planning will probably fail
            expanded_frontier = frontier_edges

        outside_frontier = expanded_frontier & ~explored
        frontier = expanded_frontier & ~obstacles & explored

        if debug:
            import matplotlib.pyplot as plt

            plt.subplot(221)
            print("obstacles")
            plt.imshow(obstacles.cpu().numpy())
            plt.subplot(222)
            plt.imshow(explored.bool().cpu().numpy())
            plt.title("explored")
            plt.subplot(223)
            plt.imshow((traversible + frontier).cpu().numpy())
            plt.title("traversible + frontier")
            plt.subplot(224)
            plt.imshow((frontier_edges).cpu().numpy())
            plt.title("just frontiers")
            plt.show()

        return frontier, outside_frontier, traversible

    def sample_closest_frontier(
        self,
        xyt: np.ndarray,
        max_tries: int = 1000,
        expand_size: int = 5,
        debug: bool = False,
        verbose: bool = False,
        step_dist: float = 0.5,
        min_dist: float = 0.5,
    ) -> Optional[torch.Tensor]:
        """Sample a valid location on the current frontier using FMM planner to compute geodesic distance. Returns points in order until it finds one that's valid.

        Args:
            xyt(np.ndrray): [x, y, theta] of the agent; must be of size 2 or 3.
            max_tries(int): number of attempts to make for rejection sampling
            debug(bool): show visualizations of frontiers
            step_dist(float): how far apart in geo dist these points should be
        """
        assert (
            len(xyt) == 2 or len(xyt) == 3
        ), f"xyt must be of size 2 or 3 instead of {len(xyt)}"

        frontier, outside_frontier, traversible = self.get_frontier(
            expand_size=expand_size, debug=debug
        )

        # from scipy.ndimage.morphology import distance_transform_edt
        m = np.ones_like(traversible)
        start_x, start_y = self.voxel_map.xy_to_grid_coords(xyt[:2]).int().cpu().numpy()
        if debug:
            print("--- Coordinates ---")
            print(f"{xyt=}")
            print(f"{start_x=}, {start_y=}")

        m[start_x, start_y] = 0
        m = np.ma.masked_array(m, ~traversible)

        # import pickle
        # with open('debug_m.pkl', 'wb') as f:
        #     pickle.dump(m, f)
        # print (~traversible)
        if not self.has_zero_contour(m):
            return None

        distance_map = skfmm.distance(m, dx=1)
        frontier_map = distance_map.copy()
        # Masks are the areas we are ignoring - ignore everything but the frontiers
        frontier_map.mask = np.bitwise_or(frontier_map.mask, ~frontier.cpu().numpy())

        # Get distances of frontier points
        distances = frontier_map.compressed()
        xs, ys = np.where(~frontier_map.mask)

        if debug:
            plt.subplot(121)
            plt.imshow(distance_map, interpolation="nearest")
            plt.title("Distance to start")
            plt.axis("off")

            plt.subplot(122)
            plt.imshow(frontier_map, interpolation="nearest")
            plt.title("Distance to start (edges only)")
            plt.axis("off")
            plt.show()

            print(f"-> found {len(distances)} items")

        assert len(xs) == len(ys) and len(xs) == len(distances)
        tries = 1
        prev_dist = -1 * float("Inf")
        for x, y, dist in sorted(zip(xs, ys, distances), key=lambda x: x[2]):
            if dist < min_dist:
                continue

            # Don't explore too close to where we are
            if dist < prev_dist + step_dist:
                continue
            prev_dist = dist

            point_grid_coords = torch.FloatTensor([[x, y]])
            outside_point = find_closest_point_on_mask(
                outside_frontier, point_grid_coords
            )

            if outside_point is None:
                print(
                    "[VOXEL MAP: sampling] ERR finding closest pt:",
                    point_grid_coords,
                    "closest =",
                    outside_point,
                )
                continue

            # convert back to real-world coordinates
            point = self.voxel_map.grid_coords_to_xy(point_grid_coords)
            if point is None:
                print("[VOXEL MAP: sampling] ERR:", point, point_grid_coords)
                continue

            theta = math.atan2(
                outside_point[1] - point_grid_coords[0, 1],
                outside_point[0] - point_grid_coords[0, 0],
            )
            if debug:
                print(f"{dist=}, {x=}, {y=}, {theta=}")

            # Ensure angle is in 0 to 2 * PI
            if theta < 0:
                theta += 2 * np.pi

            xyt = torch.zeros(3)
            xyt[:2] = point
            xyt[2] = theta

            # Check to see if this point is valid
            if verbose:
                print("[VOXEL MAP: sampling] sampled", xyt)
            if self.is_valid(xyt, debug=debug):
                yield xyt

            tries += 1
            if tries > max_tries:
                break
        yield None

    def sample_random_frontier(
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
        less_explored = binary_erosion(
            explored.float().unsqueeze(0).unsqueeze(0), self.dilate_explored_kernel
        )[0, 0]
        edges = get_edges(less_explored)

        # Do not explore obstacles any more
        frontier_edges = edges & ~obstacles

        # Mask where we will look at
        outside_frontier = ~explored & ~obstacles

        for radius in range(min_size, max_size + 1):
            # Now we apply this filter and try to sample a goal position
            if verbose:
                print("[VOXEL MAP: sampling] sampling margin of size", radius)
            expanded_frontier = expand_mask(frontier_edges, radius)
            # TODO: should we do this or not?
            # Make sure not to sample things that will just be in obstacles
            # expanded_obstacles = expand_mask(obstacles, radius)

            # Mask where we will sample locations to move to
            expanded_frontier = expanded_frontier & explored & ~obstacles

            if debug:
                import matplotlib.pyplot as plt

                plt.subplot(221)
                plt.imshow(frontier_edges.cpu().numpy())
                plt.subplot(222)
                plt.imshow(expanded_frontier.cpu().numpy())
                plt.title("expanded frontier")
                plt.subplot(223)
                plt.imshow(outside_frontier.cpu().numpy())
                plt.title("outside frontier")
                plt.subplot(224)
                plt.imshow((less_explored + explored).cpu().numpy())
                plt.title("explored")
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
                if point is None:
                    print("[VOXEL MAP: sampling] ERR:", point, point_grid_coords)
                    continue
                if outside_point is None:
                    print(
                        "[VOXEL MAP: sampling] ERR finding closest pt:",
                        point,
                        point_grid_coords,
                        "closest =",
                        outside_point,
                    )
                    continue
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
                    yield xyt

        # We failed to find anything useful
        yield None

    def show(
        self,
        instances: bool = False,
        orig: Optional[np.ndarray] = None,
        norm: float = 255.0,
        backend: str = "open3d",
    ):
        """Tool for debugging map representations that we have created"""
        geoms = self.voxel_map._get_open3d_geometries(instances, orig, norm)

        # lazily import open3d - it's a tough dependency
        import open3d

        # Show the geometries of where we have explored
        open3d.visualization.draw_geometries(geoms)

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
                yield xyt
        else:
            yield None

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
