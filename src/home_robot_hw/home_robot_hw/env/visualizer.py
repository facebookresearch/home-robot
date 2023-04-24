# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import shutil
from typing import Optional

import cv2
import numpy as np
import skimage.morphology
from PIL import Image

import home_robot.utils.pose as pu
import home_robot.utils.visualization as vu

map_color_palette = [
    int(x * 255.0)
    for x in [
        1.0,
        1.0,
        1.0,  # empty space
        0.6,
        0.6,
        0.6,  # obstacles
        0.95,
        0.95,
        0.95,  # explored area
        0.96,
        0.36,
        0.26,  # visited area
        0.12,
        0.46,
        0.70,  # closest goal
        0.63,
        0.78,
        0.95,  # rest of goal
        1.0,
        0.0,
        0.66,  # goal categories
        0.9400000000000001,
        0.8868,
        0.66,
        0.8882000000000001,
        0.9400000000000001,
        0.66,
        0.7832000000000001,
        0.9400000000000001,
        0.66,
        0.6782000000000001,
        0.9400000000000001,
        0.66,
        0.66,
        0.9400000000000001,
        0.7468000000000001,
        0.66,
        0.9400000000000001,
        0.8518000000000001,
        0.66,
        0.9232,
        0.9400000000000001,
        0.66,
        0.8182,
        0.9400000000000001,
        0.66,
        0.7132,
        0.9400000000000001,
        0.7117999999999999,
        0.66,
        0.9400000000000001,
        0.8168,
        0.66,
        0.9400000000000001,
        0.9218,
        0.66,
        0.9400000000000001,
        0.9400000000000001,
        0.66,
        0.8531999999999998,
        0.9400000000000001,
        0.66,
        0.748199999999999,
    ]
]


class PI:
    EMPTY_SPACE = 0
    OBSTACLES = 1
    EXPLORED = 2
    VISITED = 3
    CLOSEST_GOAL = 4
    REST_OF_GOAL = 5
    BEEN_CLOSE = 6
    SEM_START = 7


class Visualizer:
    """
    This class is intended to visualize a single object goal navigation task.
    """

    def __init__(self, config):
        self.show_images = config.VISUALIZE
        self.print_images = config.PRINT_IMAGES
        self.default_vis_dir = f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}"
        os.makedirs(self.default_vis_dir, exist_ok=True)

        self.num_sem_categories = config.AGENT.SEMANTIC_MAP.num_sem_categories
        self.map_resolution = config.AGENT.SEMANTIC_MAP.map_resolution
        map_size_cm = config.AGENT.SEMANTIC_MAP.map_size_cm
        self.map_shape = (
            map_size_cm // self.map_resolution,
            map_size_cm // self.map_resolution,
        )
        self.agent_cell_radius = int(config.AGENT.radius * 100 / self.map_resolution)

        self.vis_dir = None
        self.image_vis = None
        self.visited_map_vis = None
        self.last_xy = None
        self.legend = cv2.imread("assets/legend.png")

    def reset(self):
        self.vis_dir = self.default_vis_dir
        self.image_vis = None
        self.visited_map_vis = np.zeros(self.map_shape)
        self.last_xy = None

    def set_vis_dir(self, scene_id: str, episode_id: str):
        self.print_images = True
        self.vis_dir = os.path.join(self.default_vis_dir, f"{scene_id}_{episode_id}")
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def disable_print_images(self):
        self.print_images = False

    def visualize(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        frontier_map: np.ndarray,
        closest_goal_map: Optional[np.ndarray],
        sensor_pose: np.ndarray,
        found_goal: bool,
        explored_map: np.ndarray,
        semantic_map: np.ndarray,
        semantic_frame: np.ndarray,
        goal_name: str,
        timestep: int,
        visualize_goal: bool = True,
        been_close_map: Optional[np.ndarray] = None,
        third_person_image: Optional[np.ndarray] = None,
        pfs_vis: Optional[np.ndarray] = None,
    ):
        """Visualize frame input and semantic map.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gy1, gy2, gx1, gy2)
            found_goal: whether we found the object goal category
            explored_map: (M, M) binary local explored map prediction
            semantic_map: (M, M) local semantic map predictions
            semantic_frame: semantic frame visualization
            goal_name: semantic goal category
            timestep: time step within the episode
            visualize_goal: if True, visualize goal
        """
        if self.image_vis is None:
            self.image_vis = self._init_vis_image(goal_name)

        curr_x, curr_y, curr_o, gy1, gy2, gx1, gx2 = sensor_pose
        gy1, gy2, gx1, gx2 = int(gy1), int(gy2), int(gx1), int(gx2)

        # Update visited map with last visited area
        if self.last_xy is not None:
            last_x, last_y = self.last_xy
            last_pose = [
                int(last_y * 100.0 / self.map_resolution - gy1),
                int(last_x * 100.0 / self.map_resolution - gx1),
            ]
            last_pose = pu.threshold_poses(last_pose, obstacle_map.shape)
            curr_pose = [
                int(curr_y * 100.0 / self.map_resolution - gy1),
                int(curr_x * 100.0 / self.map_resolution - gx1),
            ]
            curr_pose = pu.threshold_poses(curr_pose, obstacle_map.shape)
            self.visited_map_vis[gy1:gy2, gx1:gx2] = vu.draw_line(
                last_pose, curr_pose, self.visited_map_vis[gy1:gy2, gx1:gx2]
            )
        self.last_xy = (curr_x, curr_y)

        semantic_map += PI.SEM_START

        # Obstacles, explored, and visited areas
        no_category_mask = (
            semantic_map == PI.SEM_START + self.num_sem_categories - 1
        )  # Assumes the last category is "other"
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        visited_mask = self.visited_map_vis[gy1:gy2, gx1:gx2] == 1
        semantic_map[no_category_mask] = PI.EMPTY_SPACE
        semantic_map[np.logical_and(no_category_mask, explored_mask)] = PI.EXPLORED
        semantic_map[np.logical_and(no_category_mask, obstacle_mask)] = PI.OBSTACLES
        semantic_map[visited_mask] = PI.VISITED

        # Goal
        if visualize_goal:
            selem = skimage.morphology.disk(4)
            goal_mat = (1 - skimage.morphology.binary_dilation(goal_map, selem)) != 1
            goal_mask = goal_mat == 1
            semantic_map[goal_mask] = PI.REST_OF_GOAL
            if closest_goal_map is not None:
                closest_goal_mat = (
                    1 - skimage.morphology.binary_dilation(closest_goal_map, selem)
                ) != 1
                closest_goal_mask = closest_goal_mat == 1
                semantic_map[closest_goal_mask] = PI.CLOSEST_GOAL

        # Semantic categories
        semantic_map_vis = Image.new(
            "P", (semantic_map.shape[1], semantic_map.shape[0])
        )
        semantic_map_vis.putpalette(map_color_palette)
        semantic_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        semantic_map_vis = semantic_map_vis.convert("RGB")
        semantic_map_vis = np.flipud(semantic_map_vis)
        semantic_map_vis = semantic_map_vis[:, :, [2, 1, 0]]
        semantic_map_vis = cv2.resize(
            semantic_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST
        )
        self.image_vis[50:530, 670:1150] = semantic_map_vis

        # First-person semantic frame
        self.image_vis[50:530, 15:655] = cv2.resize(semantic_frame, (640, 480))

        # Agent arrow
        pos = (
            (curr_x * 100.0 / self.map_resolution - gx1) * 480 / obstacle_map.shape[0],
            (obstacle_map.shape[1] - curr_y * 100.0 / self.map_resolution + gy1)
            * 480
            / obstacle_map.shape[1],
            np.deg2rad(-curr_o),
        )
        color = map_color_palette[9:12][::-1]
        cv2.circle(
            self.image_vis,
            (int(pos[0] + 670), int(pos[1] + 50)),
            self.agent_cell_radius,
            color,
            -1,
        )
        forward_arrow = vu.get_forward_contour_points(
            pos, origin=(670, 50), size=self.agent_cell_radius + 10
        )
        cv2.drawContours(self.image_vis, [forward_arrow], 0, color, -1)

        image_vis = self.image_vis
        if pfs_vis is not None:
            reqd_H = int(pfs_vis.shape[0] * image_vis.shape[1] / pfs_vis.shape[1])
            pfs_vis = cv2.resize(
                pfs_vis,
                (image_vis.shape[1], reqd_H),
                interpolation=cv2.INTER_AREA,
            )
            pfs_vis = pfs_vis[..., ::-1]
            image_vis = np.concatenate([image_vis, pfs_vis], axis=0)

        if self.show_images:
            cv2.imshow("Visualization", image_vis)
            cv2.waitKey(1)

        if self.print_images:
            cv2.imwrite(
                os.path.join(self.vis_dir, "snapshot_{:03d}.png".format(timestep)),
                image_vis,
            )

    def _init_vis_image(self, goal_name: str):
        vis_image = np.ones((655, 1165, 3)).astype(np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (20, 20, 20)  # BGR
        thickness = 2

        text = "Observations (Goal: {})".format(goal_name)
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (640 - textsize[0]) // 2 + 15
        textY = (50 + textsize[1]) // 2
        vis_image = cv2.putText(
            vis_image,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        text = "Predicted Semantic Map"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = 640 + (480 - textsize[0]) // 2 + 30
        textY = (50 + textsize[1]) // 2
        vis_image = cv2.putText(
            vis_image,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        # Draw outlines
        color = [100, 100, 100]
        vis_image[49, 15:655] = color
        vis_image[49, 670:1150] = color
        vis_image[50:530, 14] = color
        vis_image[50:530, 655] = color
        vis_image[50:530, 669] = color
        vis_image[50:530, 1150] = color
        vis_image[530, 15:655] = color
        vis_image[530, 670:1150] = color

        # Draw legend
        lx, ly, _ = self.legend.shape
        vis_image[537 : 537 + lx, 155 : 155 + ly, :] = self.legend

        return vis_image


class ExplorationVisualizer:
    """
    This class is intended to visualize an exploration task.
    """

    def __init__(self, config):
        self.show_images = config.VISUALIZE
        self.print_images = config.PRINT_IMAGES
        self.default_vis_dir = f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}"
        os.makedirs(self.default_vis_dir, exist_ok=True)

        self.map_resolution = config.AGENT.SEMANTIC_MAP.map_resolution
        map_size_cm = config.AGENT.SEMANTIC_MAP.map_size_cm
        self.map_shape = (
            map_size_cm // self.map_resolution,
            map_size_cm // self.map_resolution,
        )
        self.agent_cell_radius = int(config.AGENT.radius * 100 / self.map_resolution)
        self.vis_dir = None
        self.image_vis = None
        self.visited_map_vis = None
        self.last_xy = None

    def reset(self):
        self.vis_dir = self.default_vis_dir
        self.image_vis = None
        self.visited_map_vis = np.zeros(self.map_shape)
        self.last_xy = None

    def set_vis_dir(self, scene_id: str, episode_id: str):
        self.print_images = True
        self.vis_dir = os.path.join(self.default_vis_dir, f"{scene_id}_{episode_id}")
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def disable_print_images(self):
        self.print_images = False

    def visualize(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        frontier_map: np.ndarray,
        closest_goal_map: Optional[np.ndarray],
        sensor_pose: np.ndarray,
        found_goal: bool,
        explored_map: np.ndarray,
        image_frame: np.ndarray,
        timestep: int,
        occupancy_vis: np.ndarray = None,
        visualize_goal: bool = True,
        been_close_map: Optional[np.ndarray] = None,
        third_person_image: Optional[np.ndarray] = None,
    ):
        """Visualize frame input and geometric map.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gy1, gy2, gx1, gy2)
            explored_map: (M, M) binary local explored map prediction
            image_frame: image frame visualization
            timestep: time step within the episode
            visualize_goal: if True, visualize goal
        """
        if self.image_vis is None:
            self.image_vis = self._init_vis_image("Exploration")

        curr_x, curr_y, curr_o, gy1, gy2, gx1, gx2 = sensor_pose
        gy1, gy2, gx1, gx2 = int(gy1), int(gy2), int(gx1), int(gx2)

        # Update visited map with last visited area
        if self.last_xy is not None:
            last_x, last_y = self.last_xy
            last_pose = [
                int(last_y * 100.0 / self.map_resolution - gy1),
                int(last_x * 100.0 / self.map_resolution - gx1),
            ]
            last_pose = pu.threshold_poses(last_pose, obstacle_map.shape)
            curr_pose = [
                int(curr_y * 100.0 / self.map_resolution - gy1),
                int(curr_x * 100.0 / self.map_resolution - gx1),
            ]
            curr_pose = pu.threshold_poses(curr_pose, obstacle_map.shape)
            self.visited_map_vis[gy1:gy2, gx1:gx2] = vu.draw_line(
                last_pose, curr_pose, self.visited_map_vis[gy1:gy2, gx1:gx2]
            )
        self.last_xy = (curr_x, curr_y)

        # Obstacles, explored, and visited areas
        geometric_map = np.zeros_like(obstacle_map)
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        visited_mask = self.visited_map_vis[gy1:gy2, gx1:gx2] == 1
        geometric_map[:, :] = PI.EMPTY_SPACE
        geometric_map[explored_mask] = PI.EXPLORED
        geometric_map[obstacle_mask] = PI.OBSTACLES
        geometric_map[visited_mask] = PI.VISITED

        # Goal
        if visualize_goal:
            selem = skimage.morphology.disk(4)
            goal_mat = (1 - skimage.morphology.binary_dilation(goal_map, selem)) != 1
            goal_mask = goal_mat == 1
            geometric_map[goal_mask] = PI.REST_OF_GOAL
            if closest_goal_map is not None:
                closest_goal_mat = (
                    1 - skimage.morphology.binary_dilation(closest_goal_map, selem)
                ) != 1
                closest_goal_mask = closest_goal_mat == 1
                geometric_map[closest_goal_mask] = PI.CLOSEST_GOAL

        geometric_map_vis = Image.new(
            "P", (geometric_map.shape[1], geometric_map.shape[0])
        )
        geometric_map_vis.putpalette(map_color_palette)
        geometric_map_vis.putdata(geometric_map.flatten().astype(np.uint8))
        geometric_map_vis = geometric_map_vis.convert("RGB")
        geometric_map_vis = np.flipud(geometric_map_vis)
        geometric_map_vis = geometric_map_vis[:, :, [2, 1, 0]]
        geometric_map_vis = cv2.resize(
            geometric_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST
        )
        self.image_vis[50:530, 670:1150] = geometric_map_vis

        # First-person frame
        self.image_vis[50:530, 15:655] = cv2.resize(image_frame, (640, 480))

        # Agent arrow
        pos = (
            (curr_x * 100.0 / self.map_resolution - gx1) * 480 / obstacle_map.shape[0],
            (obstacle_map.shape[1] - curr_y * 100.0 / self.map_resolution + gy1)
            * 480
            / obstacle_map.shape[1],
            np.deg2rad(-curr_o),
        )
        color = map_color_palette[9:12][::-1]
        cv2.circle(
            self.image_vis,
            (int(pos[0] + 670), int(pos[1] + 50)),
            self.agent_cell_radius,
            color,
            -1,
        )
        forward_arrow = vu.get_forward_contour_points(
            pos, origin=(670, 50), size=self.agent_cell_radius + 10
        )
        cv2.drawContours(self.image_vis, [forward_arrow], 0, color, -1)

        image_vis = self.image_vis
        if occupancy_vis is not None:
            occupancy_vis = cv2.resize(
                occupancy_vis, (image_vis.shape[0], image_vis.shape[0])
            )
            image_vis = np.concatenate([image_vis, occupancy_vis], axis=1)
        if self.show_images:
            cv2.imshow("Visualization", image_vis)
            cv2.waitKey(1)

        if self.print_images:
            cv2.imwrite(
                os.path.join(self.vis_dir, "snapshot_{:03d}.png".format(timestep)),
                image_vis,
            )

    def _init_vis_image(self, goal_name: str):
        vis_image = np.ones((655, 1165, 3)).astype(np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (20, 20, 20)  # BGR
        thickness = 2

        text = "Observations (Goal: {})".format(goal_name)
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (640 - textsize[0]) // 2 + 15
        textY = (50 + textsize[1]) // 2
        vis_image = cv2.putText(
            vis_image,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        text = "Predicted Geometric Map"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = 640 + (480 - textsize[0]) // 2 + 30
        textY = (50 + textsize[1]) // 2
        vis_image = cv2.putText(
            vis_image,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        # Draw outlines
        color = [100, 100, 100]
        vis_image[49, 15:655] = color
        vis_image[49, 670:1150] = color
        vis_image[50:530, 14] = color
        vis_image[50:530, 655] = color
        vis_image[50:530, 669] = color
        vis_image[50:530, 1150] = color
        vis_image[530, 15:655] = color
        vis_image[530, 670:1150] = color

        # Draw legend
        # lx, ly, _ = self.legend.shape
        # vis_image[537 : 537 + lx, 155 : 155 + ly, :] = self.legend

        return vis_image
