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

from .constants import FloorplannertoMukulIndoor, HM3DtoCOCOIndoor
from .constants import PaletteIndices as PI
from .constants import RearrangeCategories


class VIS_LAYOUT:
    HEIGHT = 480
    FIRST_PERSON_W = 360
    TOP_DOWN_W = HEIGHT
    THIRD_PERSON_W = HEIGHT
    LEFT_PADDING = 15
    TOP_PADDING = 50
    BOTTOM_PADDING = 80
    Y1 = TOP_PADDING
    Y2 = TOP_PADDING + HEIGHT
    FIRST_RGB_X1 = LEFT_PADDING
    FIRST_RGB_X2 = LEFT_PADDING + FIRST_PERSON_W
    FIRST_SEM_X1 = LEFT_PADDING + FIRST_RGB_X2
    FIRST_SEM_X2 = FIRST_SEM_X1 + FIRST_PERSON_W
    TOP_DOWN_X1 = FIRST_SEM_X2 + LEFT_PADDING
    TOP_DOWN_X2 = TOP_DOWN_X1 + TOP_DOWN_W
    THIRD_PERSON_X1 = TOP_DOWN_X2 + LEFT_PADDING
    THIRD_PERSON_X2 = THIRD_PERSON_X1 + THIRD_PERSON_W
    IMAGE_HEIGHT = Y2 + BOTTOM_PADDING
    IMAGE_WIDTH = THIRD_PERSON_X2 + LEFT_PADDING


V = VIS_LAYOUT


class Visualizer:
    """
    This class is intended to visualize a single object goal navigation task.
    """

    def __init__(self, config):
        self.show_images = config.VISUALIZE
        self.print_images = config.PRINT_IMAGES
        self.default_vis_dir = f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}"
        os.makedirs(self.default_vis_dir, exist_ok=True)
        if hasattr(config, "habitat"):  # hydra configs
            self.episodes_data_path = config.habitat.dataset.data_path
        else:
            self.episodes_data_path = config.TASK_CONFIG.DATASET.DATA_PATH
        assert (
            "floorplanner" in self.episodes_data_path
            or "hm3d" in self.episodes_data_path
            or "mp3d" in self.episodes_data_path
        )
        if "hm3d" in self.episodes_data_path:
            if config.AGENT.SEMANTIC_MAP.semantic_categories == "coco_indoor":
                self.semantic_category_mapping = HM3DtoCOCOIndoor()
            else:
                raise NotImplementedError
        elif (
            "floorplanner" in self.episodes_data_path
            and hasattr(config, "habitat")
            and "CatNavToObjTask" in config.habitat.task.type
        ):
            self.semantic_category_mapping = RearrangeCategories()
            self._obj_name_to_id_mapping = {
                "action_figure": 0,
                "cup": 1,
                "dishtowel": 2,
                "hat": 3,
                "sponge": 4,
                "stuffed_toy": 5,
                "tape": 6,
                "vase": 7,
            }
            self._rec_name_to_id_mapping = {
                "armchair": 0,
                "armoire": 1,
                "bar_stool": 2,
                "coffee_table": 3,
                "desk": 4,
                "dining_table": 5,
                "kitchen_island": 6,
                "sofa": 7,
                "stool": 8,
            }
            self._obj_id_to_name_mapping = {
                k: v for v, k in self._obj_name_to_id_mapping.items()
            }
            self._rec_id_to_name_mapping = {
                k: v for v, k in self._rec_name_to_id_mapping.items()
            }
        elif "floorplanner" in self.episodes_data_path:
            if config.AGENT.SEMANTIC_MAP.semantic_categories == "mukul_indoor":
                self.semantic_category_mapping = FloorplannertoMukulIndoor()
            else:
                raise NotImplementedError
        elif "mp3d" in self.episodes_data_path:
            # TODO This is a hack to get unit tests running, we'll need to
            #  adapt this if we want to run ObjectNav on MP3D
            if config.AGENT.SEMANTIC_MAP.semantic_categories == "mukul_indoor":
                self.semantic_category_mapping = FloorplannertoMukulIndoor()
            else:
                raise NotImplementedError

        self.legend = cv2.imread(self.semantic_category_mapping.categories_legend_path)

        self.num_sem_categories = config.AGENT.SEMANTIC_MAP.num_sem_categories
        self.map_resolution = config.AGENT.SEMANTIC_MAP.map_resolution
        map_size_cm = config.AGENT.SEMANTIC_MAP.map_size_cm
        self.map_shape = (
            map_size_cm // self.map_resolution,
            map_size_cm // self.map_resolution,
        )

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

    def get_semantic_vis(self, semantic_map):
        semantic_map_vis = Image.new(
            "P", (semantic_map.shape[1], semantic_map.shape[0])
        )
        semantic_map_vis.putpalette(self.semantic_category_mapping.map_color_palette)
        semantic_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        semantic_map_vis = semantic_map_vis.convert("RGB")
        semantic_map_vis = np.asarray(semantic_map_vis)
        semantic_map_vis = semantic_map_vis[:, :, [2, 1, 0]]
        return semantic_map_vis

    def visualize(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        closest_goal_map: Optional[np.ndarray],
        sensor_pose: np.ndarray,
        found_goal: bool,
        explored_map: np.ndarray,
        semantic_map: np.ndarray,
        been_close_map: np.ndarray,
        semantic_frame: np.ndarray,
        goal_name: str,
        timestep: int,
        visualize_goal: bool = True,
        third_person_image=None,
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
        # Do nothing if visualization is off
        if not self.show_images and not self.print_images:
            return

        # Initialize
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
            goal_mat = 1 - skimage.morphology.binary_dilation(goal_map, selem) != 1
            goal_mask = goal_mat == 1
            semantic_map[goal_mask] = PI.REST_OF_GOAL
            if closest_goal_map is not None:
                closest_goal_mat = (
                    1 - skimage.morphology.binary_dilation(closest_goal_map, selem) != 1
                )
                closest_goal_mask = closest_goal_mat == 1
                semantic_map[closest_goal_mask] = PI.CLOSEST_GOAL

        # Semantic categories
        semantic_map_vis = self.get_semantic_vis(semantic_map)
        semantic_map_vis = np.flipud(semantic_map_vis)

        # overlay the regions the agent has been close to
        been_close_map = np.flipud(np.rint(been_close_map) == 1)
        color_index = PI.BEEN_CLOSE * 3
        color = self.semantic_category_mapping.map_color_palette[
            color_index : color_index + 3
        ][::-1]
        semantic_map_vis[been_close_map] = (
            semantic_map_vis[been_close_map] + color
        ) / 2

        semantic_map_vis = cv2.resize(
            semantic_map_vis, (V.TOP_DOWN_W, V.HEIGHT), interpolation=cv2.INTER_NEAREST
        )
        self.image_vis[V.Y1 : V.Y2, V.TOP_DOWN_X1 : V.TOP_DOWN_X2] = semantic_map_vis

        # First-person semantic frame
        self.image_vis[V.Y1 : V.Y2, V.FIRST_RGB_X1 : V.FIRST_RGB_X2] = cv2.resize(
            semantic_frame[:, :, [2, 1, 0]], (V.FIRST_PERSON_W, V.HEIGHT)
        )
        # Semantic categories
        first_person_semantic_map_vis = self.get_semantic_vis(
            semantic_frame[:, :, 3] + PI.SEM_START
        )
        # First-person semantic frame
        self.image_vis[V.Y1 : V.Y2, V.FIRST_SEM_X1 : V.FIRST_SEM_X2] = cv2.resize(
            first_person_semantic_map_vis,
            (V.FIRST_PERSON_W, V.HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )

        if third_person_image is not None:
            self.image_vis[
                V.Y1 : V.Y2, V.THIRD_PERSON_X1 : V.THIRD_PERSON_X2
            ] = cv2.resize(
                third_person_image[:, :, [2, 1, 0]],
                (V.THIRD_PERSON_W, V.HEIGHT),
            )

        # Agent arrow
        pos = (
            (curr_x * 100.0 / self.map_resolution - gx1) * 480 / obstacle_map.shape[0],
            (obstacle_map.shape[1] - curr_y * 100.0 / self.map_resolution + gy1)
            * 480
            / obstacle_map.shape[1],
            np.deg2rad(-curr_o),
        )
        agent_arrow = vu.get_contour_points(pos, origin=(V.TOP_DOWN_X1, V.Y1))
        color = self.semantic_category_mapping.map_color_palette[9:12][::-1]
        cv2.drawContours(self.image_vis, [agent_arrow], 0, color, -1)

        if self.show_images:
            cv2.imshow("Visualization", self.image_vis)
            cv2.waitKey(1)

        if self.print_images:
            cv2.imwrite(
                os.path.join(self.vis_dir, "snapshot_{:03d}.png".format(timestep)),
                self.image_vis,
            )

    def _init_vis_image(self, goal_name: str):
        vis_image = np.ones((V.IMAGE_HEIGHT, V.IMAGE_WIDTH, 3)).astype(np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (20, 20, 20)  # BGR
        thickness = 2

        text = goal_name
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (
            2 * V.FIRST_PERSON_W + V.LEFT_PADDING - textsize[0]
        ) // 2 + V.LEFT_PADDING
        textY = (V.TOP_PADDING + textsize[1]) // 2
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
        textX = (V.TOP_DOWN_W - textsize[0]) // 2 + V.TOP_DOWN_X1
        textY = (V.TOP_PADDING + textsize[1]) // 2
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

        text = "Third person image"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = V.THIRD_PERSON_X1 + (V.THIRD_PERSON_W - textsize[0]) // 2
        textY = (V.TOP_PADDING + textsize[1]) // 2
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
        color = (100, 100, 100)
        for y in [V.Y1 - 1, V.Y2]:
            for x_start, x_len in [
                (V.FIRST_RGB_X1, V.FIRST_PERSON_W),
                (V.FIRST_SEM_X1, V.FIRST_PERSON_W),
                (V.TOP_DOWN_X1, V.TOP_DOWN_W),
                (V.THIRD_PERSON_X1, V.THIRD_PERSON_W),
            ]:
                vis_image[y, x_start - 1 : x_start + x_len] = color

        for x in [
            V.FIRST_RGB_X1 - 1,
            V.FIRST_RGB_X2,
            V.FIRST_SEM_X1 - 1,
            V.FIRST_SEM_X2,
            V.TOP_DOWN_X1 - 1,
            V.TOP_DOWN_X2,
            V.THIRD_PERSON_X1 - 1,
            V.THIRD_PERSON_X2,
        ]:
            vis_image[V.Y1 - 1 : V.Y2, x] = color

        # Draw legend
        lx, ly, _ = self.legend.shape
        vis_image[
            V.Y2 : V.Y2 + lx, V.FIRST_SEM_X1 : V.FIRST_SEM_X1 + ly, :
        ] = self.legend

        return vis_image
