# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import glob
import os
import shutil
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import skimage.morphology
from habitat.utils.render_wrapper import append_text_to_image
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import draw_collision, images_to_video
from natsort import natsorted
from PIL import Image

import home_robot.utils.pose as pu
from home_robot.perception.constants import PaletteIndices as PI
from home_robot.perception.constants import languagenav_2categories_map_color_palette, languagenav_2categories_color_palette
from home_robot.perception.detection.maskrcnn.coco_categories import (
    coco_categories_color_palette,
)
from home_robot.utils.visualization import draw_line, get_contour_points

MAP_COLOR_PALETTE = [
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
        0.6,
        0.87,
        0.54,  # been close map
        0.0,
        1.0,
        0.0,  # short term goal
        0.6,
        0.17,
        0.54,  # blacklisted targets map
        0.0,
        0.0,
        0.0,  # instance border
        *coco_categories_color_palette,
    ]
]

MAP_COLOR_PALETTE = languagenav_2categories_map_color_palette


def append_text_to_image_right_align(
    image: np.ndarray, text: List[str], font_size: float = 0.5
) -> np.ndarray:
    """Write lines of text over the top of an image. Text is aligned top-right."""
    h, w, c = image.shape
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    for line in text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        if y > h:
            y = textsize[1] + 10

        x = w - (textsize[0] + 10)

        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (0, 0, 0),
            font_thickness * 2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return np.clip(image, 0, 255)


def record_video(
    target_dir: str,
    image_dir: str,
    episode_name: str = "0",
) -> None:
    """Converts a directory of image snapshots into a video."""
    print(f"Recording video {episode_name}")

    # Semantic map vis
    fnames = natsorted(glob.glob(f"{image_dir}/snapshot*.png"))
    imgs = [cv2.imread(fname) for fname in fnames]
    images_to_video(
        [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs],
        target_dir,
        f"{episode_name}",
        fps=10,
        quality=5,
        verbose=True,
    )


class NavVisualizer:
    """
    This class is intended to visualize a single image goal navigation task.
    """

    def __init__(
        self,
        num_sem_categories: int,
        map_size_cm: int,
        map_resolution: int,
        print_images: bool,
        dump_location: str,
        exp_name: str,
    ) -> None:
        """
        Arguments:
            num_sem_categories: number of semantic segmentation categories
            map_size_cm: global map size (in centimeters)
            map_resolution: size of map bins (in centimeters)
            print_images: if True, save visualization as images
            coco_categories_legend: path to the legend image of coco categories
        """
        self.print_images = print_images
        self.default_vis_dir = f"{dump_location}/images/{exp_name}"
        if self.print_images:
            os.makedirs(self.default_vis_dir, exist_ok=True)

        self.num_sem_categories = num_sem_categories
        self.map_resolution = map_resolution
        self.map_shape = (
            map_size_cm // self.map_resolution,
            map_size_cm // self.map_resolution,
        )

        self.vis_dir = None
        self.image_vis = None
        self.visited_map_vis = None
        self.last_xy = None
        self.ind_frame_height = 450

    def reset(self) -> None:
        self.vis_dir = self.default_vis_dir
        self.image_vis = None
        self.visited_map_vis = np.zeros(self.map_shape)
        self.last_xy = None

    def set_vis_dir(self, episode_id: str) -> None:
        self.vis_dir = os.path.join(self.default_vis_dir, str(episode_id))
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def visualize(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        closest_goal_map: Optional[np.ndarray],
        sensor_pose: np.ndarray,
        found_goal: bool,
        explored_map: np.ndarray,
        rgb_frame: np.ndarray,
        semantic_frame: np.ndarray,
        timestep: int,
        last_goal_image,
        last_td_map: Dict[str, Any] = None,
        last_collisions: Dict[str, Any] = None,
        semantic_map: Optional[np.ndarray] = None,
        visualize_goal: bool = True,
        metrics: Dict[str, Any] = None,
        been_close_map=None,
        blacklisted_targets_map=None,
        frontier_map: Optional[np.ndarray] = None,
        dilated_obstacle_map: Optional[np.ndarray] = None,
        instance_map: Optional[np.ndarray] = None,
        short_term_goal: Optional[np.ndarray] = None,
        goal_pose = None,
    ) -> None:
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
            rgb_frame: rgb frame visualization
            semantic_frame: semantic frame visualization
            timestep: time step within the episode
            last_td_map: habitat oracle top down map
            last_collisions: collisions dictionary
            visualize_goal: if True, visualize goal
            metrics: can populate for last frame
        """
        if not self.print_images:
            return

        if last_collisions is None:
            last_collisions = {"is_collision": False}

        if dilated_obstacle_map is not None:
            obstacle_map = dilated_obstacle_map

        goal_frame = self.make_goal(last_goal_image)

        obs_frame = self.make_observations(
            rgb_frame,
            last_collisions["is_collision"],
            found_goal,
            metrics,
        )
        print(np.unique(semantic_frame))
        sem_frame = self.make_sem_observations(
            semantic_frame,
            last_collisions["is_collision"],
            found_goal,
            metrics,
        )
        map_pred_frame = self.make_map_preds(
            sensor_pose,
            obstacle_map,
            explored_map,
            semantic_map,
            closest_goal_map,
            goal_map,
            visualize_goal,
        )
        td_map_frame = None if last_td_map is None else self.make_td_map(last_td_map)

        kp_frame = np.ones_like(goal_frame) * 255
        # kp_frame = self.make_keypoint(timestep)

        if td_map_frame is None:
            frame = np.concatenate(
                [goal_frame, obs_frame, map_pred_frame, sem_frame], axis=1
            )
        else:
            # upper_frame = np.concatenate([goal_frame, obs_frame, sem_frame], axis=1)
            # lower_frame = self.pad_frame(
            #     np.concatenate([map_pred_frame, td_map_frame], axis=1),
            #     upper_frame.shape[1],
            # )
            lower_frame = np.concatenate([map_pred_frame, td_map_frame], axis=1)
            upper_frame = self.pad_frame(
                np.concatenate([goal_frame, obs_frame, sem_frame], axis=1),
                lower_frame.shape[1]
            )
            frame = np.concatenate([upper_frame, lower_frame], axis=0)

        nframes = 1 if metrics is None else 5
        for i in range(nframes):
            name = f"snapshot_{timestep}_{i}.png"
            cv2.imwrite(os.path.join(self.vis_dir, name), frame)

    def pad_frame(self, frame: np.ndarray, width: int) -> np.ndarray:
        """Pad the width of a frame to `width` centered white sides."""
        h = frame.shape[0]
        w = frame.shape[1]
        left_bar = np.ones((h, (width - w) // 2, 3), dtype=np.uint8) * 255
        right_bar = (
            np.ones((h, (width - w - left_bar.shape[1]), 3), dtype=np.uint8) * 255
        )
        return np.concatenate([left_bar, frame, right_bar], axis=1)

    def make_keypoint(self, timestep: int) -> np.ndarray:
        """Create the keypoint-matching sub-frame."""
        fname = os.path.join(self.vis_dir, f"superglue_{timestep}.png")
        assert os.path.exists(fname), f"keypoint frame does not exist at `{fname}`."

        border_size = 10
        text_bar_height = 50 - border_size
        kp_img = cv2.imread(fname)
        os.remove(fname)

        new_h = self.ind_frame_height - text_bar_height - 2 * border_size
        new_w = int((new_h / kp_img.shape[0]) * kp_img.shape[1])
        kp_img = cv2.resize(kp_img, (new_w, new_h))

        kp_img = self._add_border(kp_img, border_size)

        w = kp_img.shape[1]
        top_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, kp_img.astype(np.uint8)], axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (20, 20, 20)
        thickness = 2

        text = "Keypoint Matching"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (w - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        return frame

    def make_goal(self, goal_img: np.ndarray) -> np.ndarray:
        """make the goal image sub-frame."""
        border_size = 10
        text_bar_height = 50 - border_size
        new_h = self.ind_frame_height - text_bar_height - 2 * border_size
        goal_img = cv2.resize(goal_img, (new_h, new_h))
        goal_img = cv2.cvtColor(goal_img, cv2.COLOR_RGB2BGR)
        goal_img = self._add_border(goal_img, border_size)
        w = goal_img.shape[1]

        top_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, goal_img.astype(np.uint8)], axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (20, 20, 20)
        thickness = 2

        text = "Goal Image"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (w - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return frame

    def make_observations(
        self,
        sem_img: np.ndarray,
        collision: bool,
        found_goal: bool,
        metrics: Dict[str, float],
    ) -> np.ndarray:
        """
        make the egocentric RGB observation sub-frame. Overlay a goal detected banner
        and a collision border.
        """
        border_size = 10
        text_bar_height = 50 - border_size
        new_h = self.ind_frame_height - text_bar_height - 2 * border_size
        new_w = int(new_h / sem_img.shape[0] * sem_img.shape[1])
        sem_img = cv2.resize(sem_img, (new_w, new_h))

        if found_goal:
            sem_img = self._found_goal_detection(sem_img)

        sem_img = self._write_metrics(sem_img, metrics)

        if collision:
            sem_img = draw_collision(sem_img)

        sem_img = cv2.cvtColor(sem_img, cv2.COLOR_RGB2BGR)
        sem_img = self._add_border(sem_img, border_size)
        w = sem_img.shape[1]

        top_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, sem_img.astype(np.uint8)], axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (20, 20, 20)
        thickness = 2

        text = "Observation"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (w - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return frame


    def make_sem_observations(
        self,
        sem_img: np.ndarray,
        collision: bool,
        found_goal: bool,
        metrics: Dict[str, float],
    ) -> np.ndarray:
        """
        make the egocentric RGB observation sub-frame. Overlay a goal detected banner
        and a collision border.
        """

        semantic_map_vis = Image.new(
            "P", (sem_img.shape[1], sem_img.shape[0])
        )

        sem_img = sem_img % 255

        semantic_map_vis.putpalette(languagenav_2categories_color_palette)
        semantic_map_vis.putdata(sem_img.flatten().astype(np.uint8))
        semantic_map_vis = semantic_map_vis.convert("RGB")

        sem_img = np.asarray(semantic_map_vis)[:, :, [2, 1, 0]]

        border_size = 10
        text_bar_height = 50 - border_size
        new_h = self.ind_frame_height - text_bar_height - 2 * border_size
        new_w = int(new_h / sem_img.shape[0] * sem_img.shape[1])

        sem_img = cv2.resize(sem_img, (new_w, new_h))

        if found_goal:
            sem_img = self._found_goal_detection(sem_img)

        sem_img = self._write_metrics(sem_img, metrics)

        if collision:
            sem_img = draw_collision(sem_img)

        sem_img = cv2.cvtColor(sem_img, cv2.COLOR_RGB2BGR)
        sem_img = self._add_border(sem_img, border_size)
        w = sem_img.shape[1]

        top_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, sem_img.astype(np.uint8)], axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (20, 20, 20)
        thickness = 2

        text = "Observation"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (w - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return frame

    def make_map_preds(
        self,
        sensor_pose: np.ndarray,
        obstacle_map: np.ndarray,
        explored_map: np.ndarray,
        semantic_map: np.ndarray,
        closest_goal_map: np.ndarray,
        goal_map: np.ndarray,
        visualize_goal: bool,
    ) -> np.ndarray:
        """make the predicted map sub-frame."""
        if semantic_map is None:
            fill_val = self.num_sem_categories - 1
            semantic_map = np.zeros_like(obstacle_map) + fill_val

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
            self.visited_map_vis[gy1:gy2, gx1:gx2] = draw_line(
                last_pose, curr_pose, self.visited_map_vis[gy1:gy2, gx1:gx2]
            )
        self.last_xy = (curr_x, curr_y)

        semantic_map += PI.SEM_START

        # Obstacles, explored, and visited areas
        no_category_mask = semantic_map == PI.SEM_START + self.num_sem_categories - 1
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
        semantic_map_vis = Image.new(
            "P", (semantic_map.shape[1], semantic_map.shape[0])
        )

        semantic_map_vis.putpalette(MAP_COLOR_PALETTE)
        semantic_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        semantic_map_vis = semantic_map_vis.convert("RGB")
        semantic_map_vis = np.flipud(semantic_map_vis)
        semantic_map_vis = semantic_map_vis[:, :, [2, 1, 0]]
        semantic_map_vis = cv2.resize(
            semantic_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST
        )

        border_size = 10
        text_bar_height = 50 - border_size
        old_h, old_w = semantic_map_vis.shape[:2]
        new_h = self.ind_frame_height - text_bar_height - 2 * border_size
        new_w = int(new_h / semantic_map_vis.shape[0] * semantic_map_vis.shape[1])
        semantic_map_vis = cv2.resize(semantic_map_vis, (new_w, new_h))

        # Agent arrow
        pos = (
            (curr_x * 100.0 / self.map_resolution - gx1) * 480 / obstacle_map.shape[0],
            (obstacle_map.shape[1] - curr_y * 100.0 / self.map_resolution + gy1)
            * 480
            / obstacle_map.shape[1],
            np.deg2rad(-curr_o),
        )
        pos = (pos[0] * new_w / old_w, pos[1] * new_h / old_h, pos[2])
        agent_arrow = get_contour_points(pos, origin=(0, 0))
        color = MAP_COLOR_PALETTE[9:12][::-1]
        cv2.drawContours(semantic_map_vis, [agent_arrow], 0, color, -1)

        # semantic_map_vis = cv2.cvtColor(semantic_map_vis, cv2.COLOR_RGB2BGR)

        # add map outline
        color = [100, 100, 100]
        h, w = semantic_map_vis.shape[:2]
        semantic_map_vis[0, 0:] = color
        semantic_map_vis[h - 1, 0:] = color
        semantic_map_vis[0:, 0] = color
        semantic_map_vis[0:, w - 1] = color

        semantic_map_vis = self._add_border(semantic_map_vis, border_size)
        w = semantic_map_vis.shape[1]

        top_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, semantic_map_vis.astype(np.uint8)], axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (20, 20, 20)
        thickness = 2

        text = "Predicted Map"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (w - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return frame

    def make_td_map(self, top_down_map: np.ndarray) -> np.ndarray:
        """
        In Habitat Simulation, an oracle top-down map may be provided.
        Visualize that sub-frame.
        """
        border_size = 10
        text_bar_height = 50 - border_size
        new_h = self.ind_frame_height - text_bar_height - 2 * border_size

        td_map = maps.colorize_draw_agent_and_fit_to_height(top_down_map, new_h)
        td_map = cv2.cvtColor(td_map, cv2.COLOR_RGB2BGR)

        # add map outline
        color = [100, 100, 100]
        h, w = td_map.shape[:2]
        td_map[0, 0:] = color
        td_map[h - 1, 0:] = color
        td_map[0:, 0] = color
        td_map[0:, w - 1] = color

        td_map = self._add_border(td_map, border_size)
        w = td_map.shape[1]

        top_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, td_map.astype(np.uint8)], axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (20, 20, 20)
        thickness = 2

        text = "Oracle Top-Down Map"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (w - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return frame

    def _write_metrics(
        self, frame: np.ndarray, metrics: Dict[str, float]
    ) -> np.ndarray:
        """If metrics are provided, write them on the RGB frame."""
        if metrics is None:
            return frame

        lines = []
        for k, v in {"success": "SR", "spl": "SPL"}.items():
            if k in metrics:
                lines.append(f"{v}: {metrics[k]:.3f}")

        return append_text_to_image_right_align(frame, lines, font_size=0.8)

    def _add_border(self, frame: np.ndarray, border_size: int) -> np.ndarray:
        """Add a white border to a frame."""
        h, w = frame.shape[:2]
        side = np.ones((h, border_size, 3), dtype=np.uint8) * 255
        frame = np.concatenate([side, frame, side], axis=1)
        top = np.ones((border_size, w + 2 * border_size, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top, frame, top], axis=0)
        return frame

    def _found_goal_detection(self, view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """overlay a green goal detected banner"""
        strip_width = view.shape[0] // 15
        mask = np.ones(view.shape)
        mask[strip_width:-strip_width] = 0
        mask = mask == 1
        view[mask] = (alpha * np.array([0, 255, 0]) + (1.0 - alpha) * view)[mask]
        return append_text_to_image(view, ["Goal Detected"], font_size=0.5)
