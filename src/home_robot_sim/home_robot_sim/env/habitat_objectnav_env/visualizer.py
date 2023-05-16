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
from .constants import RearrangeBasicCategories, RearrangeDETICCategories


class VIS_LAYOUT:
    HEIGHT = 480
    FIRST_PERSON_W = 360
    TOP_DOWN_W = HEIGHT
    THIRD_PERSON_W = HEIGHT
    LEFT_PADDING = 15
    TOP_PADDING = 50
    LEGEND_TOP_PADDING = 5
    BOTTOM_PADDING = 240  # 80
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

    def __init__(self, config, dataset=None):
        self.show_images = config.VISUALIZE
        self.print_images = config.PRINT_IMAGES
        self.default_vis_dir = f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}"
        self._dataset = dataset
        os.makedirs(self.default_vis_dir, exist_ok=True)
        if hasattr(config, "habitat"):  # hydra configs
            self.episodes_data_path = config.habitat.dataset.data_path
        else:
            self.episodes_data_path = config.TASK_CONFIG.DATASET.DATA_PATH
        assert (
            "rearrange" in self.episodes_data_path
            or "hm3d" in self.episodes_data_path
            or "mp3d" in self.episodes_data_path
        )
        if "hm3d" in self.episodes_data_path:
            if config.AGENT.SEMANTIC_MAP.semantic_categories == "coco_indoor":
                self.semantic_category_mapping = HM3DtoCOCOIndoor()
            else:
                raise NotImplementedError
        elif (
            "rearrange" in self.episodes_data_path
            and hasattr(config, "habitat")
            and "CatNavToObjTask" in config.habitat.task.type
        ):
            if self._dataset is None:
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
            else:
                self._obj_name_to_id_mapping = (
                    self._dataset.obj_category_to_obj_category_id
                )
                self._rec_name_to_id_mapping = (
                    self._dataset.recep_category_to_recep_category_id
                )
            self._obj_id_to_name_mapping = {
                k: v for v, k in self._obj_name_to_id_mapping.items()
            }
            self._rec_id_to_name_mapping = {
                k: v for v, k in self._rec_name_to_id_mapping.items()
            }

            if config.GROUND_TRUTH_SEMANTICS:
                self.semantic_category_mapping = RearrangeBasicCategories()
            else:
                # combining objs and receps ids into one dictionary
                self.obj_rec_combined_mapping = {}
                for i in range(
                    len(self._obj_id_to_name_mapping)
                    + len(self._rec_id_to_name_mapping)
                ):
                    if i < len(self._obj_id_to_name_mapping):
                        self.obj_rec_combined_mapping[
                            i + 1
                        ] = self._obj_id_to_name_mapping[i]
                    else:
                        self.obj_rec_combined_mapping[
                            i + 1
                        ] = self._rec_id_to_name_mapping[
                            i - len(self._obj_id_to_name_mapping)
                        ]

                self.semantic_category_mapping = RearrangeDETICCategories(
                    self.obj_rec_combined_mapping
                )

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
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.text_color = (20, 20, 20)  # BGR
        self.text_thickness = 2

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

    def get_semantic_vis(self, semantic_map, rgb_frame=None):
        semantic_map_vis = Image.new(
            "P", (semantic_map.shape[1], semantic_map.shape[0])
        )
        semantic_map_vis.putpalette(self.semantic_category_mapping.map_color_palette)
        semantic_map_vis.putdata(semantic_map.flatten().astype(np.uint8))

        if rgb_frame is not None:
            # overlaying semantics on RGB frame
            rgb_frame = rgb_frame[:, :, [2, 1, 0]]
            mask = np.array(semantic_map_vis)
            mask = (
                mask == self.semantic_category_mapping.num_sem_categories - 1 + 7
            ).astype(np.uint8) * 255
            mask = Image.fromarray(mask)
            rgb_pil = Image.fromarray(rgb_frame)
            semantic_map_vis = semantic_map_vis.convert("RGB")
            semantic_map_vis.paste(rgb_pil, mask=mask)
        else:
            semantic_map_vis = semantic_map_vis.convert("RGB")

        semantic_map_vis = np.asarray(semantic_map_vis)[:, :, [2, 1, 0]]

        return semantic_map_vis

    def visualize(
        self,
        timestep: int,
        semantic_frame: np.ndarray,
        obstacle_map: np.ndarray = None,
        goal_map: np.ndarray = None,
        closest_goal_map: Optional[np.ndarray] = None,
        sensor_pose: np.ndarray = None,
        found_goal: bool = None,
        explored_map: np.ndarray = None,
        semantic_map: np.ndarray = None,
        been_close_map: np.ndarray = None,
        frontier_map: np.ndarray = None,
        goal_name: str = None,
        visualize_goal: bool = True,
        third_person_image: np.ndarray = None,
        curr_skill: str = None,
        curr_action: str = None,
        short_term_goal: np.ndarray = None,
        dilated_obstacle_map: np.ndarray = None,
        **kwargs,
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
            curr_skill: the skill currently being executed
        """
        # Do nothing if visualization is off
        if not self.show_images and not self.print_images:
            return

        # Initialize
        if self.image_vis is None:
            self.image_vis = self._init_vis_image(goal_name)

        image_vis = self.image_vis.copy()

        # if curr_skill is not None, place the skill name below the third person image
        if curr_skill is not None and curr_action is not None:
            image_vis = self._put_text_on_image(
                image_vis,
                curr_skill + "\n" + curr_action,
                V.THIRD_PERSON_X1,
                V.Y2,
                V.THIRD_PERSON_W,
                V.BOTTOM_PADDING,
            )

        if dilated_obstacle_map is not None:
            obstacle_map = dilated_obstacle_map

        if obstacle_map is not None:
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
                        1 - skimage.morphology.binary_dilation(closest_goal_map, selem)
                        != 1
                    )
                    closest_goal_mask = closest_goal_mat == 1
                    semantic_map[closest_goal_mask] = PI.CLOSEST_GOAL
                if short_term_goal is not None:
                    short_term_goal_mask = np.zeros(goal_mask.shape)
                    short_term_goal_mask[short_term_goal[0], short_term_goal[1]] = 1
                    short_term_goal_mask = (
                        1
                        - skimage.morphology.binary_dilation(
                            short_term_goal_mask, selem
                        )
                        != 1
                    )
                    short_term_goal_mask = short_term_goal_mask == 1
                    semantic_map[short_term_goal_mask] = PI.SHORT_TERM_GOAL

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
                semantic_map_vis,
                (V.TOP_DOWN_W, V.HEIGHT),
                interpolation=cv2.INTER_NEAREST,
            )
            image_vis[V.Y1 : V.Y2, V.TOP_DOWN_X1 : V.TOP_DOWN_X2] = semantic_map_vis

            # Agent arrow
            pos = (
                (curr_x * 100.0 / self.map_resolution - gx1)
                * 480
                / obstacle_map.shape[0],
                (obstacle_map.shape[1] - curr_y * 100.0 / self.map_resolution + gy1)
                * 480
                / obstacle_map.shape[1],
                np.deg2rad(-curr_o),
            )
            agent_arrow = vu.get_contour_points(pos, origin=(V.TOP_DOWN_X1, V.Y1))
            color = self.semantic_category_mapping.map_color_palette[9:12][::-1]
            cv2.drawContours(image_vis, [agent_arrow], 0, color, -1)

        # First-person RGB frame
        rgb_frame = semantic_frame[:, :, [2, 1, 0]]
        image_vis[V.Y1 : V.Y2, V.FIRST_RGB_X1 : V.FIRST_RGB_X2] = cv2.resize(
            rgb_frame, (V.FIRST_PERSON_W, V.HEIGHT)
        )
        # Semantic categories
        first_person_semantic_map_vis = self.get_semantic_vis(
            semantic_frame[:, :, 3] + PI.SEM_START, rgb_frame
        )
        # First-person semantic frame
        image_vis[V.Y1 : V.Y2, V.FIRST_SEM_X1 : V.FIRST_SEM_X2] = cv2.resize(
            first_person_semantic_map_vis,
            (V.FIRST_PERSON_W, V.HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )

        if third_person_image is not None:
            image_vis[V.Y1 : V.Y2, V.THIRD_PERSON_X1 : V.THIRD_PERSON_X2] = cv2.resize(
                third_person_image[:, :, [2, 1, 0]],
                (V.THIRD_PERSON_W, V.HEIGHT),
            )

        if self.show_images:
            cv2.imshow("Visualization", image_vis)
            cv2.waitKey(1)
        if self.print_images:
            cv2.imwrite(
                os.path.join(self.vis_dir, "snapshot_{:03d}.png".format(timestep)),
                image_vis,
            )

    def _put_text_on_image(
        self,
        vis_image,
        text: str,
        bbox_x_start: int,
        bbox_y_start: int,
        bbox_x_len: int,
        bbox_y_len: int,
    ):
        """
        Place text at the center of the given bounding box.
        """
        textsize = cv2.getTextSize(
            text, self.font, self.font_scale, self.text_thickness
        )[0]
        # The x coordinate at which the left edge of text needs to be placed
        textX = (bbox_x_len - textsize[0]) // 2 + bbox_x_start
        # The height at which base needs to be placed
        textY = (bbox_y_len + textsize[1]) // 2 + bbox_y_start
        return cv2.putText(
            vis_image,
            text,
            (textX, textY),
            self.font,
            self.font_scale,
            self.text_color,
            self.text_thickness,
            cv2.LINE_AA,
        )

    def _init_vis_image(self, goal_name: str):
        vis_image = np.ones((V.IMAGE_HEIGHT, V.IMAGE_WIDTH, 3)).astype(np.uint8) * 255

        vis_image = self._put_text_on_image(
            vis_image, goal_name, V.LEFT_PADDING, 0, 2 * V.FIRST_PERSON_W, V.TOP_PADDING
        )

        text = "Predicted Semantic Map"
        vis_image = self._put_text_on_image(
            vis_image, text, V.TOP_DOWN_X1, 0, V.TOP_DOWN_W, V.TOP_PADDING
        )

        text = "Third person image"
        vis_image = self._put_text_on_image(
            vis_image, text, V.THIRD_PERSON_X1, 0, V.THIRD_PERSON_W, V.TOP_PADDING
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
            V.Y2 + V.LEGEND_TOP_PADDING : V.Y2 + lx + V.LEGEND_TOP_PADDING, 0:ly, :
        ] = self.legend

        return vis_image
