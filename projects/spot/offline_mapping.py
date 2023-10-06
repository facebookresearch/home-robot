#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import click
import cv2
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import skimage.morphology
import torch
from habitat_sim.utils.common import d3_40_colors_rgb
from PIL import Image, ImageDraw, ImageFont

# TODO Install home_robot and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)

from collections import defaultdict

import home_robot.utils.pose as pu
import home_robot.utils.visualization as vu
from home_robot.agent.goat_agent.goat_matching import GoatMatching
from home_robot.core.interfaces import Observations
from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory
from home_robot.perception.detection.maskrcnn.coco_categories import (
    coco_categories,
    coco_categories_color_palette,
    coco_category_id_to_coco_category,
)
from home_robot.perception.detection.maskrcnn.maskrcnn_perception import (
    MaskRCNNPerception,
)
from home_robot.utils.config import get_config


class PI:
    EMPTY_SPACE = 0
    OBSTACLES = 1
    EXPLORED = 2
    VISITED = 3
    GOAL = 4
    SEM_START = 5


def generate_legend(
    vis_image: np.ndarray,
    colors: np.ndarray,
    texts: List[str],
    start_x: int,
    start_y: int,
    total_w: int,
    total_h: int,
):
    font = 0
    font_scale = 0.5
    font_color = (0, 0, 0)
    font_thickness = 1

    # grid size - number of labels in each column/row
    grid_w, grid_h = 7, 6
    int_w = total_w / grid_w
    int_h = total_h / grid_h
    ctr = 0
    for y in range(grid_h):
        for x in range(grid_w):
            if ctr > len(colors) - 1:
                break
            rect_start_x = int(total_w * x / grid_w) + start_x
            rect_start_y = int(total_h * y / grid_h) + start_y
            rect_start = [rect_start_x, rect_start_y]
            rect_end_x = rect_start_x + int(int_h * 0.2) + 20
            rect_end_y = rect_start_y + int(int_h * 0.2) + 10
            rect_end = [rect_end_x, rect_end_y]
            vis_image = cv2.rectangle(
                vis_image, rect_start, rect_end, colors[ctr].tolist(), thickness=-1
            )
            vis_image = cv2.putText(
                vis_image,
                texts[ctr],
                (rect_end_x + 5, rect_end_y - 5),
                font,
                font_scale,
                font_color,
                font_thickness,
                cv2.LINE_AA,
            )
            ctr += 1
    return vis_image


def get_semantic_map_vis(
    semantic_map: Categorical2DSemanticMapState,
    # To visualize a trajectory
    semantic_frame: Optional[np.array] = None,
    depth_frame: Optional[np.array] = None,
    # To visualize matching a goal to memory
    goal_image: Optional[np.array] = None,
    instance_image: Optional[np.array] = None,
    instance_memory: Optional[InstanceMemory] = None,
    visualize_instances: bool = False,
    legend: Optional[np.array] = None,
):
    vis_image = np.ones((655, 1820, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    if semantic_frame is not None:
        text = "Segmentation"
    elif goal_image is not None:
        text = "Goal"
    else:
        raise NotImplementedError

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

    if depth_frame is not None:
        text = "Depth"
    elif instance_image is not None:
        text = "Matching Instance"
    else:
        raise NotImplementedError

    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (640 - textsize[0]) // 2 + 30
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

    text = "Predicted Instance Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 1280 + (480 - textsize[0]) // 2 + 45
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

    map_color_palette = [
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
        0.70,  # goal
    ]
    map_color_palette = [int(x * 255.0) for x in map_color_palette]
    map_color_palette += d3_40_colors_rgb.flatten().tolist()[1:]

    new_colors = d3_40_colors_rgb.copy()
    new_colors[:, 0] = np.minimum(new_colors[:, 0] + 15, 255)
    new_colors[:, 2] = np.maximum(new_colors[:, 2] - 15, 0)
    map_color_palette += new_colors.flatten().tolist()

    semantic_categories_map = semantic_map.get_semantic_map(0)
    obstacle_map = semantic_map.get_obstacle_map(0)
    explored_map = semantic_map.get_explored_map(0)
    visited_map = semantic_map.get_visited_map(0)
    goal_map = semantic_map.get_goal_map(0)
    instance_map = semantic_map.get_instance_map(0)

    no_category_mask = semantic_categories_map == semantic_map.num_sem_categories - 1
    if not visualize_instances:
        semantic_categories_map += PI.SEM_START
    else:
        unique_instances, remapped_instances = np.unique(
            instance_map, return_inverse=True
        )

        # project instance map
        projected_instance_map = instance_map.max(0)

        semantic_categories_map = projected_instance_map
        semantic_categories_map += PI.SEM_START - 1
        semantic_categories_map[
            semantic_categories_map == PI.SEM_START - 1
        ] = PI.EMPTY_SPACE

        num_instances = int(np.max(unique_instances))

        if num_instances > 2 * len(d3_40_colors_rgb) - 1:
            raise NotImplementedError

    obstacle_mask = np.rint(obstacle_map) == 1
    explored_mask = np.rint(explored_map) == 1
    visited_mask = visited_map == 1
    semantic_categories_map[no_category_mask] = PI.EMPTY_SPACE
    semantic_categories_map[
        np.logical_and(no_category_mask, explored_mask)
    ] = PI.EXPLORED
    semantic_categories_map[
        np.logical_and(no_category_mask, obstacle_mask)
    ] = PI.OBSTACLES
    semantic_categories_map[visited_mask] = PI.VISITED

    # Goal
    selem = skimage.morphology.disk(4)
    goal_mat = (1 - skimage.morphology.binary_dilation(goal_map, selem)) != 1
    goal_mask = goal_mat == 1
    semantic_categories_map[goal_mask] = PI.GOAL

    # Draw semantic map
    semantic_map_vis = Image.new("P", semantic_categories_map.shape)
    semantic_map_vis.putpalette(map_color_palette)
    semantic_map_vis.putdata(semantic_categories_map.flatten().astype(np.uint8))
    semantic_map_vis = semantic_map_vis.convert("RGB")
    semantic_map_vis = np.flipud(semantic_map_vis)
    semantic_map_vis = cv2.resize(
        semantic_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST
    )

    vis_image[50:530, 1325:1805] = semantic_map_vis

    if semantic_frame is not None:
        vis_image[50:530, 15:655] = cv2.resize(semantic_frame[:, :, ::-1], (640, 480))
    elif goal_image is not None:
        vis_image[50:530, 15:655] = cv2.resize(goal_image, (640, 480))
    else:
        raise NotImplementedError

    if depth_frame is not None:
        vis_image[50:530, 670:1310] = cv2.resize(depth_frame, (640, 480))
    elif instance_image is not None:
        vis_image[50:530, 670:1310] = cv2.resize(instance_image, (640, 480))
    else:
        raise NotImplementedError

    # Draw legend
    if legend is not None:
        lx, ly, _ = legend.shape
        vis_image[537 : 537 + lx, 155 : 155 + ly, :] = legend[:, :, ::-1]
    elif visualize_instances:
        # Name instances as chair-1, chair-2 and so on
        category_counts = defaultdict(int)
        instance_to_name = {}
        for instance in range(1, num_instances + 1):
            if instance == 0:
                continue
            if instance_memory is not None:
                # retrieve name
                category = instance_memory.instances[0][int(instance)].category_id
                category_counts[category] += 1
                instance_to_name[instance] = (
                    coco_category_id_to_coco_category[category]
                    + f" - {category_counts[category]}"
                )
            else:
                instance_to_name[instance] = f"Instance - {instance}"
        vis_image = generate_legend(
            vis_image,
            np.array(
                map_color_palette[3 * PI.SEM_START : (PI.SEM_START + num_instances) * 3]
            ).reshape(-1, 3),
            [instance_to_name[i] for i in range(1, num_instances + 1)],
            155,
            537,
            1250,
            115,
        )

    # Draw agent arrow
    curr_x, curr_y, curr_o, gy1, _, gx1, _ = semantic_map.get_planner_pose_inputs(0)
    pos = (
        (curr_x * 100.0 / semantic_map.resolution - gx1)
        * 480
        / semantic_map.local_map_size,
        (semantic_map.local_map_size - curr_y * 100.0 / semantic_map.resolution + gy1)
        * 480
        / semantic_map.local_map_size,
        np.deg2rad(-curr_o),
    )
    agent_arrow = vu.get_contour_points(pos, origin=(1325, 50), size=10)
    color = map_color_palette[9:12]
    cv2.drawContours(vis_image, [agent_arrow], 0, color, -1)
    return vis_image


def create_video(images, output_file, fps):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for image in images:
        video_writer.write(image)
    video_writer.release()


def text_to_image(
    text,
    width,
    height,
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
):
    # Create a blank image with the specified dimensions
    image = Image.new(
        "RGB", (width, height), color=(73, 109, 137)
    )  # RGB color can be any combination you like
    # Set up the drawing context
    d = ImageDraw.Draw(image)
    # Set the font and size. Font path might be different in your system. Install a font if necessary.
    font = ImageFont.truetype(font_path, 15)
    # Calculate width and height of the text to center it
    text_width, text_height = d.textsize(text, font=font)
    position = ((width - text_width) / 2, (height - text_height) / 2)
    # Add the text to the image
    d.text(position, text, fill=(255, 255, 255), font=font)
    # Convert the PIL image to a NumPy array
    image_array = np.array(image)
    return image_array


record_instance_ids = True
save_map_and_instances = True
load_map_and_instances = False


def print_metrics(
    metrics_list: List[Dict],
    goal_type: Optional[str] = None,
    goal_category: Optional[str] = None,
):
    metrics_df = pd.DataFrame.from_records(metrics_list)

    if goal_type is None and goal_category is None:
        print("all goals:")
    elif goal_type is not None and goal_category is None:
        print(f"{goal_type} goals: ")
        metrics_df = metrics_df[metrics_df["type"] == goal_type]
    elif goal_type is None and goal_category is not None:
        print(f"{goal_category} goals: ")
        metrics_df = metrics_df[metrics_df["category"] == goal_category]
    else:
        print(f"{goal_type} {goal_category} goals: ")
        metrics_df = metrics_df[metrics_df["type"] == goal_type][
            metrics_df["category"] == goal_category
        ]

    print(
        f"total: {len(metrics_df)}, "
        f"matched correctly: {metrics_df['success'].sum()}, "
        f"false positive: {metrics_df['false_positive'].sum()}, "
        f"false negative: {metrics_df['false_negative'].sum()}, "
        f"at least mapped: {metrics_df['instance_detected'].sum()}"
    )
    print(
        f"matching accuracy: {metrics_df['success'].sum() / metrics_df['instance_detected'].sum() * 100.0:.1f}%, "
        f"at least mapped: {metrics_df['instance_detected'].sum() / len(metrics_df) * 100.0:.1f}%"
    )


@click.command()
@click.option(
    "--base_dir",
    default=f"{str(Path(__file__).resolve().parent)}/trajectories/trajectory1",
)
@click.option(
    "--legend_path",
    default=f"{str(Path(__file__).resolve().parent)}/coco_categories_legend.png",
)
@click.option("--device", default="cuda:0")
def main(base_dir: str, legend_path: str, device: str):
    obs_dir = f"{base_dir}/obs/"
    map_vis_dir = f"{base_dir}/map_vis/"
    goal_grounding_vis_dir = f"{base_dir}/goal_grounding_vis/"
    if legend_path is not None:
        legend = cv2.imread(legend_path)
    else:
        legend = None

    device = torch.device(device)

    categories = list(coco_categories.keys())
    num_sem_categories = len(coco_categories)

    if load_map_and_instances:
        print("Loading prebuilt map and instance memory...")
        semantic_map = pickle.load(open(f"{base_dir}/semantic_map.pkl", "rb"))
        instance_memory = pickle.load(open(f"{base_dir}/instance_memory.pkl", "rb"))

    else:
        # --------------------------------------------------------------------------------------------
        # Load trajectory of home_robot Observations
        # --------------------------------------------------------------------------------------------

        observations = []
        for path in natsort.natsorted(glob.glob(f"{obs_dir}/*.pkl")):
            print("- loading", path)
            with open(path, "rb") as f:
                try:
                    observations.append(pickle.load(f))
                except Exception as e:
                    print(e)

        # Predict semantic segmentation
        segmentation = MaskRCNNPerception(
            sem_pred_prob_thr=0.8,
            sem_gpu_id=0,
        )

        observations = [
            segmentation.predict(obs, depth_threshold=0.5) for obs in observations
        ]
        for obs in observations:
            obs.semantic[obs.semantic == 0] = num_sem_categories
            obs.semantic = obs.semantic - 1
            obs.task_observations["instance_map"] += 1
            obs.task_observations["instance_map"] = obs.task_observations[
                "instance_map"
            ].astype(int)

        print()
        print("home_robot observations:")
        print("------------------------")
        obs = observations[0]
        print("obs.gps", obs.gps)
        print("obs.compass", obs.compass)
        print("obs.rgb", obs.rgb.shape, obs.rgb.dtype, obs.rgb.min(), obs.rgb.max())
        print(
            "obs.depth",
            obs.depth.shape,
            obs.depth.dtype,
            obs.depth.min(),
            obs.depth.max(),
        )
        print(
            "obs.semantic",
            obs.semantic.shape,
            obs.semantic.dtype,
            obs.semantic.min(),
            obs.semantic.max(),
        )
        print("obs.camera_pose", obs.camera_pose)
        print("obs.task_observations", obs.task_observations.keys())

        # --------------------------------------------------------------------------------------------
        # Map initialization
        # --------------------------------------------------------------------------------------------

        # Instance memory is responsible for tracking instances in the map
        instance_memory = None
        if record_instance_ids:
            instance_memory = InstanceMemory(
                1,
                4,
                debug_visualize=True,
                save_dir=f"{base_dir}/instances",
                mask_cropped_instances=False,
                padding_cropped_instances=0,
                category_id_to_category_name=coco_category_id_to_coco_category,
            )

        # State holds global and local map and sensor pose
        # See class definition for argument info
        semantic_map = Categorical2DSemanticMapState(
            device=device,
            num_environments=1,
            num_sem_categories=num_sem_categories,
            map_resolution=5,
            map_size_cm=4800,
            global_downscaling=2,
            record_instance_ids=record_instance_ids,
            instance_memory=instance_memory,
        )
        semantic_map.init_map_and_pose()

        # Module is responsible for updating the local and global maps and poses
        # See class definition for argument info
        semantic_map_module = Categorical2DSemanticMapModule(
            frame_height=obs.rgb.shape[0],
            frame_width=obs.rgb.shape[1],
            camera_height=obs.camera_pose[2, 3],
            hfov=60.2,
            num_sem_categories=num_sem_categories,
            map_size_cm=4800,
            map_resolution=5,
            vision_range=100,
            explored_radius=150,
            been_close_to_radius=200,
            global_downscaling=2,
            du_scale=4,
            cat_pred_threshold=1.0,
            exp_pred_threshold=1.0,
            map_pred_threshold=15.0,
            min_depth=0.5,
            max_depth=5.95,
            must_explore_close=False,
            min_obs_height_cm=10,
            record_instance_ids=record_instance_ids,
            instance_memory=instance_memory,
        ).to(device)

        # --------------------------------------------------------------------------------------------
        # Map building
        # --------------------------------------------------------------------------------------------

        def preprocess_obs(obs: Observations, last_pose: np.array):
            """Take a home-robot observation, preprocess it to put it into the
            correct format for the semantic map.

            Output conventions:
                * obs_preprocessed = (1, 4 + num_sem_categories + num_instances, H, W) torch.Tensor
                    - channels 1-3 are RGB (0 - 255 range)
                    - channel 4 is depth (in cm)
                * pose_delta = (1, 3) torch.Tensor:
                    - +X is forward
                    - +Y is leftward
                    - +theta is measured from +X to +Y
                * camera_pose = (1, 4, 4) camera extrinsics
                    - +X is forward
                    - +Y is leftward
                    - +Z is upward
            """
            rgb = torch.from_numpy(obs.rgb).to(device)
            depth = (
                torch.from_numpy(obs.depth).unsqueeze(-1).to(device) * 100.0
            )  # m to cm
            semantic = one_hot_encoding[torch.from_numpy(obs.semantic).to(device)]
            obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)

            if record_instance_ids:
                instances = obs.task_observations["instance_map"]
                # first create a mapping to 1, 2, ... num_instances
                instance_ids = np.unique(instances)
                # map instance id to index
                instance_id_to_idx = {
                    instance_id: idx for idx, instance_id in enumerate(instance_ids)
                }
                # convert instance ids to indices, use vectorized lookup
                instances = torch.from_numpy(
                    np.vectorize(instance_id_to_idx.get)(instances)
                ).to(device)
                # create a one-hot encoding
                instances = torch.eye(len(instance_ids), device=device)[instances]
                obs_preprocessed = torch.cat([obs_preprocessed, instances], dim=-1)

            obs_preprocessed = obs_preprocessed.unsqueeze(0)
            obs_preprocessed = obs_preprocessed.permute(0, 3, 1, 2)

            curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
            pose_delta = (
                torch.tensor(pu.get_rel_pose_change(curr_pose, last_pose))
                .unsqueeze(0)
                .to(device)
            )

            camera_pose = obs.camera_pose
            if camera_pose is not None:
                camera_pose = (
                    torch.tensor(np.asarray(camera_pose)).unsqueeze(0).to(device)
                )
            return (obs_preprocessed, pose_delta, camera_pose, curr_pose)

        last_pose = np.zeros(3)
        one_hot_encoding = torch.eye(num_sem_categories, device=device)
        Path(map_vis_dir).mkdir(parents=True, exist_ok=True)
        vis_images = []

        for i, obs in enumerate(observations):
            # Preprocess observation
            (obs_preprocessed, pose_delta, camera_pose, last_pose) = preprocess_obs(
                obs, last_pose
            )

            if i == 0:
                print()
                print("preprocessed observations:")
                print("-------------------------")
                print("obs_preprocessed", obs_preprocessed.shape)
                print("pose_delta", pose_delta)
                print("camera_pose", camera_pose)

            # Update map
            dones = torch.tensor([False]).to(device)
            update_global = torch.tensor([True]).to(device)
            (
                seq_map_features,
                semantic_map.local_map,
                semantic_map.global_map,
                seq_local_pose,
                seq_global_pose,
                seq_lmb,
                seq_origins,
            ) = semantic_map_module(
                obs_preprocessed.unsqueeze(1),
                pose_delta.unsqueeze(1),
                dones.unsqueeze(1),
                update_global.unsqueeze(1),
                camera_pose,
                semantic_map.local_map,
                semantic_map.global_map,
                semantic_map.local_pose,
                semantic_map.global_pose,
                semantic_map.lmb,
                semantic_map.origins,
            )

            semantic_map.local_pose = seq_local_pose[:, -1]
            semantic_map.global_pose = seq_global_pose[:, -1]
            semantic_map.lmb = seq_lmb[:, -1]
            semantic_map.origins = seq_origins[:, -1]

            # Visualize map
            depth_frame = obs.depth
            if depth_frame.max() > 0:
                depth_frame = depth_frame / depth_frame.max()
            depth_frame = (depth_frame * 255).astype(np.uint8)
            depth_frame = np.repeat(depth_frame[:, :, np.newaxis], 3, axis=2)
            vis_image = get_semantic_map_vis(
                semantic_map,
                semantic_frame=obs.task_observations["semantic_frame"],
                depth_frame=depth_frame,
                legend=None,
                instance_memory=instance_memory,
                visualize_instances=True,
            )
            vis_images.append(vis_image)
            plt.imsave(Path(map_vis_dir) / f"{i}.png", vis_image)

        create_video(
            [v[:, :, ::-1] for v in vis_images],
            f"{map_vis_dir}/video.mp4",
            fps=20,
        )

    if save_map_and_instances:
        print("Saving map and instance memory...")
        pickle.dump(semantic_map, open(f"{base_dir}/semantic_map.pkl", "wb"))
        pickle.dump(instance_memory, open(f"{base_dir}/instance_memory.pkl", "wb"))

    # --------------------------------------------------------------------------------------------
    # Ground goals in memory
    # --------------------------------------------------------------------------------------------

    config_path = "projects/spot/configs/config.yaml"
    config, _ = get_config(config_path)
    matching = GoatMatching(
        device=device.index,
        score_func="confidence_sum",
        num_sem_categories=num_sem_categories,
        config=config.AGENT.SUPERGLUE,
        default_vis_dir=map_vis_dir,
        print_images=True,
        instance_memory=instance_memory,
    )

    goals_file = Path(f"{base_dir}/goals.json")
    goals = []
    if Path.exists(goals_file):
        with open(goals_file) as f:
            goals = json.load(f)
    else:
        print("No goals specified: goals.json does not exist")
        sys.exit()

    metrics = []
    for i, goal in enumerate(goals):
        categories = goal["categories"]
        goal_type = goal["type"]
        name = goal["name"]
        print()
        print(f"Goal {i}")
        print(f"Type: {goal_type}")
        category_ids = None
        if categories is not None:
            category_ids = [coco_categories.get(c) for c in categories]
        if goal_type == "image":
            image_goal = cv2.imread(str(Path(base_dir) / "image_goals" / name))
            goal_vis_image = image_goal[:, :, ::-1]
            image_goal, goal_image_keypoints = matching.get_goal_image_keypoints(
                image_goal
            )
            language_goal = None
            matching_fn = matching.match_image_to_image
            score_thresh = config.AGENT.SUPERGLUE.score_thresh_image
            agg_fn = config.AGENT.SUPERGLUE.agg_fn_image
            feat_agg_fn = config.AGENT.SUPERGLUE.feat_agg_fn_image
            aggregate_feats = config.AGENT.SUPERGLUE.aggregate_feats_image
        elif goal_type == "language":
            language_goal = name
            goal_vis_image = text_to_image(language_goal, 640, 480)
            image_goal, goal_image_keypoints = None, None
            matching_fn = matching.match_language_to_image
            score_thresh = config.AGENT.SUPERGLUE.score_thresh_lang
            agg_fn = config.AGENT.SUPERGLUE.agg_fn_lang
            feat_agg_fn = config.AGENT.SUPERGLUE.feat_agg_fn_lang
            aggregate_feats = config.AGENT.SUPERGLUE.aggregate_feats_lang
        else:
            raise ValueError(
                "Invalid goal type. Only image and language goals supported currently"
            )

        (
            all_matches,
            all_confidences,
            instance_ids,
        ) = matching.get_matches_against_memory(
            matching_fn,
            0,
            language_goal=language_goal,
            image_goal=image_goal,
            use_full_image=True,
            categories=category_ids,
            goal_image_keypoints=goal_image_keypoints,
            aggregate_feats=aggregate_feats,
            feat_agg_fn=feat_agg_fn,
        )
        stats = {
            i: {
                "mean": float(scores.mean()),
                "median": float(np.median(scores)),
                "max": float(scores.max()),
                "min": float(scores.min()),
                "all": scores.flatten().tolist(),
            }
            for i, scores in zip(instance_ids, all_confidences)
        }
        Path(goal_grounding_vis_dir).mkdir(parents=True, exist_ok=True)
        with open(
            Path(goal_grounding_vis_dir) / f"{goal_type}_goal{i}_stats.json", "w"
        ) as f:
            json.dump(stats, f, indent=4)
        (
            goal_map,
            _,
            _,
            instance_goal_found,
            goal_inst,
        ) = matching.select_and_localize_instance(
            goal_map=None,
            found_goal=torch.Tensor([False]),
            local_map=semantic_map.local_map,
            lmb=semantic_map.lmb,
            matches=None,
            confidence=None,
            local_instance_ids=None,
            local_id_to_global_id_map=None,
            instance_goal_found=False,
            goal_inst=None,
            all_matches=all_matches,
            all_confidences=all_confidences,
            instance_ids=instance_ids,
            score_thresh=score_thresh,
            agg_fn=agg_fn,
        )
        if instance_goal_found:
            semantic_map.update_global_goal_for_env(0, goal_map.cpu().numpy())

            vis_image = get_semantic_map_vis(
                semantic_map,
                goal_image=goal_vis_image,
                # Visualize the first cropped view of the instance
                instance_image=instance_memory.instances[0][goal_inst]
                .instance_views[0]
                .cropped_image[:, :, ::-1],
                legend=None,
                instance_memory=instance_memory,
                visualize_instances=True,
            )
            plt.imsave(
                Path(goal_grounding_vis_dir) / f"{goal_type}_goal{i}.png", vis_image
            )

            print("Found goal:", instance_goal_found)
            print("Goal instance ID:", goal_inst)

        correct_instances = [
            instance["id"] for instance in goal["ground_truth_instances"]
        ]

        print(f"Correct instances were {correct_instances}")
        metrics_per_goal = {
            "type": goal_type,
            "category": categories[0] if categories is not None else "None",
            "name": name,
            "success": int(goal_inst in correct_instances),
            "false_positive": int(
                goal_inst is not None and goal_inst not in correct_instances
            ),
            "false_negative": int(goal_inst is None),
            "instance_detected": np.any(
                [inst in instance_memory.instances[0] for inst in correct_instances]
            ),
        }
        metrics.append(metrics_per_goal)
    print()
    print_metrics(metrics)
    print("-" * 50)
    print_metrics(metrics, goal_type="image")
    print("-" * 50)
    print_metrics(metrics, goal_type="language")
    for category in np.unique([m["category"] for m in metrics]):
        print("-" * 50)
        print_metrics(metrics, goal_category=category)


if __name__ == "__main__":
    main()
