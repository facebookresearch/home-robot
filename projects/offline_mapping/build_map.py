#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import pickle
import sys
from pathlib import Path
from typing import List

import click
import cv2
import matplotlib.pyplot as plt
import natsort
import numpy as np
import torch
from PIL import Image

# TODO Install home_robot and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)

import home_robot.utils.pose as pu
import home_robot.utils.visualization as vu
from home_robot.core.interfaces import Observations
from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.mapping.semantic.instance_tracking_modules import InstanceMemory
from home_robot.perception.detection.detic.detic_perception import DeticPerception

# Semantic segmentation categories predicted from frames and projected in the map
coco_categories = [
    "chair",
    "couch",
    "potted_plant",
    "bed",
    "toilet",
    "tv",
    "dining_table",
    "oven",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "cup",
    "bottle",
]

# Color palette for semantic categories
coco_categories_color_palette = [
    0.9400000000000001,
    0.7818,
    0.66,  # chair
    0.9400000000000001,
    0.8868,
    0.66,  # couch
    0.8882000000000001,
    0.9400000000000001,
    0.66,  # potted plant
    0.7832000000000001,
    0.9400000000000001,
    0.66,  # bed
    0.6782000000000001,
    0.9400000000000001,
    0.66,  # toilet
    0.66,
    0.9400000000000001,
    0.7468000000000001,  # tv
    0.66,
    0.9400000000000001,
    0.8518000000000001,  # dining-table
    0.66,
    0.9232,
    0.9400000000000001,  # oven
    0.66,
    0.8182,
    0.9400000000000001,  # sink
    0.66,
    0.7132,
    0.9400000000000001,  # refrigerator
    0.7117999999999999,
    0.66,
    0.9400000000000001,  # book
    0.8168,
    0.66,
    0.9400000000000001,  # clock
    0.9218,
    0.66,
    0.9400000000000001,  # vase
    0.9400000000000001,
    0.66,
    0.8531999999999998,  # cup
    0.9400000000000001,
    0.66,
    0.748199999999999,  # bottle
]


def get_semantic_map_vis(
    semantic_map: Categorical2DSemanticMapState,
    semantic_frame: np.array,
    depth_frame: np.array,
    color_palette: List[float],
    legend=None,
):
    vis_image = np.ones((655, 1820, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Segmentation"
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

    text = "Depth"
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

    text = "Predicted Semantic Map"
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
        *color_palette,
    ]
    map_color_palette = [int(x * 255.0) for x in map_color_palette]

    semantic_categories_map = semantic_map.get_semantic_map(0)
    obstacle_map = semantic_map.get_obstacle_map(0)
    explored_map = semantic_map.get_explored_map(0)
    visited_map = semantic_map.get_visited_map(0)

    semantic_categories_map += 4
    no_category_mask = (
        semantic_categories_map == 4 + semantic_map.num_sem_categories - 1
    )
    obstacle_mask = np.rint(obstacle_map) == 1
    explored_mask = np.rint(explored_map) == 1
    visited_mask = visited_map == 1
    semantic_categories_map[no_category_mask] = 0
    semantic_categories_map[np.logical_and(no_category_mask, explored_mask)] = 2
    semantic_categories_map[np.logical_and(no_category_mask, obstacle_mask)] = 1
    semantic_categories_map[visited_mask] = 3

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

    # Draw semantic frame
    # vis_image[50:530, 15:655] = cv2.resize(semantic_frame[:, :, ::-1], (640, 480))
    vis_image[50:530, 15:655] = cv2.resize(semantic_frame, (640, 480))

    # Draw depth frame
    vis_image[50:530, 670:1310] = cv2.resize(depth_frame, (640, 480))

    # Draw legend
    if legend is not None:
        lx, ly, _ = legend.shape
        vis_image[537 : 537 + lx, 155 : 155 + ly, :] = legend[:, :, ::-1]

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


record_instance_ids = True


@click.command()
@click.option(
    "--input_trajectory_dir",
    default=f"{str(Path(__file__).resolve().parent)}/trajectory/",
)
@click.option(
    "--output_visualization_dir",
    default=f"{str(Path(__file__).resolve().parent)}/map_visualization/",
)
@click.option(
    "--legend_path",
    default=f"{str(Path(__file__).resolve().parent)}/coco_categories_legend.png",
)
def main(input_trajectory_dir: str, output_visualization_dir: str, legend_path: str):
    # --------------------------------------------------------------------------------------------
    # Load trajectory of home_robot Observations
    # --------------------------------------------------------------------------------------------
    print("Read trajectories from:", input_trajectory_dir)
    print("        Output data to:", output_visualization_dir)
    print("          Legend is at:", legend_path)
    observations = []
    for path in natsort.natsorted(glob.glob(f"{input_trajectory_dir}/*.pkl")):
        with open(path, "rb") as f:
            observations.append(pickle.load(f))

    # Predict semantic segmentation
    categories = [
        "other",
        *coco_categories,
        "other",
    ]
    num_sem_categories = len(categories) - 1
    segmentation = DeticPerception(
        vocabulary="custom",
        custom_vocabulary=",".join(categories),
        sem_gpu_id=0,
    )
    observations = [
        segmentation.predict(obs, depth_threshold=0.5) for obs in observations
    ]
    for obs in observations:
        obs.semantic[obs.semantic == 0] = len(categories) - 1
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
    print("obs.rgb", obs.rgb.shape, obs.rgb.min(), obs.rgb.max())
    print("obs.depth", obs.depth.shape, obs.depth.min(), obs.depth.max())
    print("obs.semantic", obs.semantic.shape, obs.semantic.min(), obs.semantic.max())
    print("obs.camera_pose", obs.camera_pose)
    print("obs.task_observations", obs.task_observations.keys())

    # --------------------------------------------------------------------------------------------
    # Map initialization
    # --------------------------------------------------------------------------------------------

    device = torch.device("cuda:0")

    # Instance memory is responsible for tracking instances in the map
    instance_memory = None
    if record_instance_ids:
        instance_memory = InstanceMemory(
            1,
            4,
            instance_association="map_overlap",
            debug_visualize=True,
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
        hfov=42.0,  #  60.0,  # 47.0
        num_sem_categories=num_sem_categories,
        map_size_cm=4800,
        map_resolution=5,
        vision_range=100,
        explored_radius=150,
        been_close_to_radius=200,
        global_downscaling=2,
        du_scale=4,
        cat_pred_threshold=5.0,
        exp_pred_threshold=1.0,
        map_pred_threshold=1.0,
        min_depth=0.5,
        max_depth=3.5,  # 1.5,
        must_explore_close=False,
        min_obs_height_cm=10,
        record_instance_ids=record_instance_ids,
        instance_memory=instance_memory,
        instance_association="map_overlap",
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
        depth = torch.from_numpy(obs.depth).unsqueeze(-1).to(device) * 100.0  # m to cm
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
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0).to(device)
        return (obs_preprocessed, pose_delta, camera_pose, curr_pose)

    last_pose = np.zeros(3)
    one_hot_encoding = torch.eye(num_sem_categories, device=device)
    if legend_path is not None:
        legend = cv2.imread(legend_path)
    else:
        legend = None
    Path(output_visualization_dir).mkdir(parents=True, exist_ok=True)
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
            obs.task_observations["semantic_frame"],
            depth_frame,
            coco_categories_color_palette,
            legend,
        )
        vis_images.append(vis_image)
        plt.imsave(Path(output_visualization_dir) / f"{i}.png", vis_image)

    create_video(
        [v[:, :, ::-1] for v in vis_images],
        f"{output_visualization_dir}/video.mp4",
        fps=20,
    )


if __name__ == "__main__":
    main()
