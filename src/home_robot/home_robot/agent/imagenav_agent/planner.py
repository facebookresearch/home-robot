from enum import Enum
from typing import Tuple, List
import os
import shutil
import math

import cv2
import skimage.morphology
import skfmm
import numpy as np
from numpy import ma

import home_robot.utils.pose as pu


# Same enum as HabitatSimActions without Habitat dependency
class DiscreteActions(Enum):
    stop = 0
    move_forward = 1
    turn_left = 2
    turn_right = 3


class FMMPlanner:
    """
    Fast Marching Method Planner.
    """

    def __init__(
        self,
        traversible: np.ndarray,
        scale: int = 1,
        step_size: int = 5,
        vis_dir: str = "data/images/planner",
        print_images=False,
    ):
        """
        Arguments:
            traversible: (M + 1, M + 1) binary map encoding traversible regions
            scale: map scale
            step_size: maximum distance of the short-term goal selected by the
             planner
            vis_dir: folder where to dump visualization
        """
        self.print_images = print_images
        if print_images:
            self.vis_dir = vis_dir
            os.makedirs(self.vis_dir, exist_ok=True)

        self.scale = scale
        self.step_size = step_size
        if scale != 1.0:
            self.traversible = cv2.resize(
                traversible,
                (traversible.shape[1] // scale, traversible.shape[0] // scale),
                interpolation=cv2.INTER_NEAREST,
            )
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.0))
        self.fmm_dist = None
        self.goal_map = None

    def save_planner_viz(self, goal_map: np.ndarray, timestep: int):
        r, c = self.traversible.shape
        dist_vis = np.zeros((r, c * 3))
        dist_vis[:, :c] = np.flipud(self.traversible)
        dist_vis[:, c : 2 * c] = np.flipud(goal_map)
        dist_vis[:, 2 * c :] = np.flipud(self.fmm_dist / self.fmm_dist.max())

        cv2.imwrite(
            os.path.join(self.vis_dir, f"planner_snapshot_{timestep}.png"),
            (dist_vis * 255).astype(int),
        )

    def set_multi_goal(
        self,
        goal_map: np.ndarray,
        goal_dilation: int = 0,
        timestep: int = None,
    ):
        """Set long-term goal(s) used to compute distance from a binary
        goal map.
        """
        # dilate the goal map
        dilated_goal_map = goal_map
        if goal_dilation > 0:
            selem = skimage.morphology.disk(goal_dilation)
            dilated_goal_map = skimage.morphology.binary_dilation(
                dilated_goal_map, selem
            ) is not True
            dilated_goal_map = 1 - dilated_goal_map * 1.0

        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        traversible_ma[dilated_goal_map == 1] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        self.goal_map = dilated_goal_map

        if self.print_images and timestep is not None:
            self.save_planner_viz(dilated_goal_map, timestep)

    def get_short_term_goal(self, state: List[int]):
        """Compute the short-term goal closest to the current state.

        Arguments:
            state: current location
        """
        scale = self.scale * 1.0
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])

        # dx, dy = 0.0, 0.0 (always)

        mask = FMMPlanner.get_mask(dx, dy, scale, self.step_size)
        dist_mask = FMMPlanner.get_dist(dx, dy, scale, self.step_size)

        state = [int(x) for x in state]

        dist = np.pad(
            self.fmm_dist,
            self.du,
            "constant",
            constant_values=self.fmm_dist.shape[0] ** 2,
        )
        subset = dist[
            state[0] : state[0] + 2 * self.du + 1, state[1] : state[1] + 2 * self.du + 1
        ]

        assert (
            subset.shape[0] == 2 * self.du + 1 and subset.shape[1] == 2 * self.du + 1
        ), "Planning error: unexpected subset shape {}".format(subset.shape)

        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2

        stop = subset[self.du, self.du] < self.step_size

        subset -= subset[self.du, self.du]
        ratio1 = subset / dist_mask
        subset[ratio1 < -1.5] = 1

        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        replan = subset[stg_x, stg_y] > -0.0001

        return (
            (stg_x + state[0] - self.du) * scale,
            (stg_y + state[1] - self.du) * scale,
            replan,
            stop,
        )

    @staticmethod
    def get_mask(sx, sy, scale, step_size):
        size = int(step_size // scale) * 2 + 1
        mask = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                cond1 = (
                    ((i + 0.5) - (size // 2 + sx)) ** 2
                    + ((j + 0.5) - (size // 2 + sy)) ** 2
                ) <= step_size**2
                cond2 = (
                    ((i + 0.5) - (size // 2 + sx)) ** 2
                    + ((j + 0.5) - (size // 2 + sy)) ** 2
                ) > (step_size - 1) ** 2
                if cond1 and cond2:
                    mask[i, j] = 1
        mask[size // 2, size // 2] = 1
        return mask

    @staticmethod
    def get_dist(sx, sy, scale, step_size):
        size = int(step_size // scale) * 2 + 1
        mask = np.zeros((size, size)) + 1e-10
        for i in range(size):
            for j in range(size):
                if (
                    ((i + 0.5) - (size // 2 + sx)) ** 2
                    + ((j + 0.5) - (size // 2 + sy)) ** 2
                ) <= step_size**2:
                    mask[i, j] = max(
                        5,
                        (
                            ((i + 0.5) - (size // 2 + sx)) ** 2
                            + ((j + 0.5) - (size // 2 + sy)) ** 2
                        )
                        ** 0.5,
                    )
        return mask


class DiscretePlanner:
    """
    This class translates planner inputs into a discrete low-level action
    using an FMM planner.
    """

    def __init__(
        self,
        turn_angle: float,
        collision_threshold: float,
        obs_dilation_selem_radius: int,
        goal_dilation_selem_radius: int,
        map_size_cm: int,
        map_resolution: int,
        print_images: bool,
        dump_location: str,
        exp_name: str,
    ):
        """
        Arguments:
            turn_angle (float): agent turn angle (in degrees)
            collision_threshold (float): forward move distance under which we
             consider there's a collision (in meters)
            obs_dilation_selem_radius: radius (in cells) of obstacle dilation
             structuring element
            obs_dilation_selem_radius: radius (in cells) of goal dilation
             structuring element
            map_size_cm: global map size (in centimeters)
            map_resolution: size of map bins (in centimeters)
            print_images: if True, save visualization as images
        """
        self.print_images = print_images
        self.default_vis_dir = f"{dump_location}/images/{exp_name}"
        if self.print_images:
            os.makedirs(self.default_vis_dir, exist_ok=True)

        self.map_size_cm = map_size_cm
        self.map_resolution = map_resolution
        self.map_shape = (
            self.map_size_cm // self.map_resolution,
            self.map_size_cm // self.map_resolution,
        )
        self.turn_angle = turn_angle
        self.collision_threshold = collision_threshold
        self.start_obs_dilation_selem_radius = obs_dilation_selem_radius
        self.goal_dilation_selem_radius = goal_dilation_selem_radius

        self.vis_dir = None
        self.collision_map = None
        self.visited_map = None
        self.col_width = None
        self.last_pose = None
        self.curr_pose = None
        self.last_action = None
        self.timestep = None
        self.curr_obs_dilation_selem_radius = None

    def reset(self):
        self.vis_dir = self.default_vis_dir
        self.collision_map = np.zeros(self.map_shape)
        self.visited_map = np.zeros(self.map_shape)
        self.col_width = 1
        self.last_pose = None
        self.curr_pose = [
            self.map_size_cm / 100.0 / 2.0,
            self.map_size_cm / 100.0 / 2.0,
            0.0,
        ]
        self.last_action = None
        self.timestep = 1
        self.curr_obs_dilation_selem_radius = self.start_obs_dilation_selem_radius

    def set_vis_dir(self, episode_id: str):
        self.vis_dir = os.path.join(self.default_vis_dir, str(episode_id))
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def plan(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        sensor_pose: np.ndarray,
        found_goal: bool,
    ) -> Tuple[int, np.ndarray]:
        """Plan a low-level action.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gx1, gx2, gy1, gy2)
            found_goal: whether we found the object goal category

        Returns:
            action: low-level action
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
        """
        print("obstacle_map", obstacle_map.shape, obstacle_map.sum())
        print("goal_map", goal_map.shape, goal_map.sum())
        raise NotImplementedError

        self.last_pose = self.curr_pose
        obstacle_map = np.rint(obstacle_map)

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = sensor_pose
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        start = [
            int(start_y * 100.0 / self.map_resolution - gx1),
            int(start_x * 100.0 / self.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, obstacle_map.shape)

        self.curr_pose = [start_x, start_y, start_o]
        self.visited_map[gx1:gx2, gy1:gy2][
            start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1
        ] = 1

        if self.last_action == DiscreteActions.move_forward:
            self._check_collision()

        (
            short_term_goal, closest_goal_map, did_replan_obstacle, stop
        ) = self._get_short_term_goal(
            obstacle_map, np.copy(goal_map), start, planning_window
        )
        if did_replan_obstacle:
            self.collision_map *= 0

        # Short-term goal -> deterministic local policy
        if stop and found_goal:
            action = DiscreteActions.stop
        else:
            stg_x, stg_y = short_term_goal
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0], stg_y - start[1]))
            angle_agent = start_o % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.turn_angle / 2.0:
                action = DiscreteActions.turn_right
            elif relative_angle < -self.turn_angle / 2.0:
                action = DiscreteActions.turn_left
            else:
                action = DiscreteActions.move_forward

        self.last_action = action
        return action, closest_goal_map

    def _get_short_term_goal(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        start: List[int],
        planning_window: List[int],
    ) -> Tuple[Tuple[int, int], np.ndarray, bool, bool]:
        """Get short-term goal.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            start: start location (x, y)
            planning_window: local map boundaries (gx1, gx2, gy1, gy2)

        Returns:
            short_term_goal: short-term goal position (x, y) in map
            closest_goal_map: (M, M) binary array denoting closest goal
                location in the goal map in geodesic distance
            did_replan_obstacle: binary flag to indicate we replanned the
                obstacle map.
            stop: binary flag to indicate we've reached the goal
        """
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1 : h + 1, 1 : w + 1] = mat
            return new_mat

        def remove_boundary(mat, value=1):
            return mat[value:-value, value:-value]

        gx1, gx2, gy1, gy2 = planning_window
        x2, y2 = obstacle_map.shape
        replan = True
        did_replan_obstacle = False
        goal_map = add_boundary(goal_map, value=0)

        while replan:
            # dilate the obstacle map
            obs_dilation_selem = skimage.morphology.disk(
                self.curr_obs_dilation_selem_radius
            )
            dilated_obstacles = cv2.dilate(
                obstacle_map, obs_dilation_selem, iterations=1
            )

            traversible = 1 - dilated_obstacles
            traversible[self.collision_map[gx1:gx2, gy1:gy2][:x2, :y2] == 1] = 0
            traversible[self.visited_map[gx1:gx2, gy1:gy2][:x2, :y2] == 1] = 1
            traversible[
                int(start[0]) - 1 : int(start[0]) + 2,
                int(start[1]) - 1 : int(start[1]) + 2,
            ] = 1
            traversible = add_boundary(traversible)

            planner = FMMPlanner(
                traversible,
                vis_dir=self.vis_dir,
                print_images=self.print_images,
            )

            state = [start[0] + 1, start[1] + 1]
            planner.set_multi_goal(
                goal_map,
                goal_dilation=self.goal_dilation_selem_radius,
                timestep=self.timestep,
            )
            stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)

            replan = replan and self.curr_obs_dilation_selem_radius > 1
            if replan:
                self.curr_obs_dilation_selem_radius -= 1
                did_replan_obstacle = True

        self.timestep += 1

        stg_x, stg_y = stg_x - 1, stg_y - 1
        short_term_goal = int(stg_x), int(stg_y)

        # Select closest point on goal map for visualization
        # TODO How to do this without the overhead of creating another FMM planner?
        vis_planner = FMMPlanner(traversible)
        curr_loc_map = np.zeros_like(goal_map)
        curr_loc_map[start[0], start[1]] = 1
        vis_planner.set_multi_goal(curr_loc_map, self.goal_dilation_selem_radius)
        fmm_dist_ = vis_planner.fmm_dist.copy()
        goal_map_ = goal_map.copy()
        goal_map_[goal_map_ == 0] = 10000
        fmm_dist_[fmm_dist_ == 0] = 10000
        closest_goal_map = (goal_map_ * fmm_dist_) == (goal_map_ * fmm_dist_).min()
        closest_goal_map = remove_boundary(closest_goal_map)

        return short_term_goal, closest_goal_map, did_replan_obstacle, stop

    def _check_collision(self):
        """Check whether we had a collision and update the collision map."""
        x1, y1, t1 = self.last_pose
        x2, y2, _ = self.curr_pose
        buf = 4
        length = 2

        if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
            self.col_width += 2
            if self.col_width == 7:
                length = 4
                buf = 3
            self.col_width = min(self.col_width, 5)
        else:
            self.col_width = 1

        dist = pu.get_l2_distance(x1, x2, y1, y2)

        if dist < self.collision_threshold:
            # We have a collision
            width = self.col_width

            # Add obstacles to the collision map
            for i in range(length):
                for j in range(width):
                    wx = x1 + 0.05 * (
                        (i + buf) * np.cos(np.deg2rad(t1))
                        + (j - width // 2) * np.sin(np.deg2rad(t1))
                    )
                    wy = y1 + 0.05 * (
                        (i + buf) * np.sin(np.deg2rad(t1))
                        - (j - width // 2) * np.cos(np.deg2rad(t1))
                    )
                    r, c = wy, wx
                    r, c = int(r * 100 / self.map_resolution), int(
                        c * 100 / self.map_resolution
                    )
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r, c] = 1
