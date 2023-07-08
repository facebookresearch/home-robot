#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import click
import glob
import pickle
import sys
import torch
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# TODO Install home_robot and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)

import home_robot.utils.pose as pu
from home_robot.core.interfaces import Observations
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import Categorical2DSemanticMapState
from home_robot.mapping.semantic.categorical_2d_semantic_map_module import Categorical2DSemanticMapModule


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


@click.command()
@click.option(
    "--trajectory_path",
    default="trajectories/fremont1",
)
def main(trajectory_path):
    # --------------------------------------------------------------------------------------------
    # Load trajectory of home_robot Observations
    # --------------------------------------------------------------------------------------------

    observations = []
    for path in sorted(glob.glob(str(Path(__file__).resolve().parent) + f"/{trajectory_path}/*.pkl")):
        with open(path, "rb") as f:
            observations.append(pickle.load(f))
    observations = observations[:5]  # TODO Remove

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
        segmentation.predict(obs, depth_threshold=0.5)
        for obs in observations
    ]
    for obs in observations:
        obs.semantic[obs.semantic == 0] = len(categories) - 1
        obs.semantic = obs.semantic - 1

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

    # State holds global and local map and sensor pose
    # See class definition for argument info
    semantic_map = Categorical2DSemanticMapState(
        device=device,
        num_environments=1,
        num_sem_categories=16,
        map_resolution=5,
        map_size_cm=4800,
        global_downscaling=2,
    )
    semantic_map.init_map_and_pose()

    # Module is responsible for updating the local and global maps and poses
    # See class definition for argument info
    semantic_map_module = Categorical2DSemanticMapModule(
        frame_height=640,
        frame_width=480,
        camera_height=1.31,
        hfov=42.0,
        num_sem_categories=16,
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
        must_explore_close=False,
        min_obs_height_cm=10,
    ).to(device)

    # --------------------------------------------------------------------------------------------
    # Map building
    # --------------------------------------------------------------------------------------------

    def preprocess_obs(obs: Observations, last_pose: np.array):
        """Take a home-robot observation, preprocess it to put it into the
        correct format for the semantic map.

        Output conventions:
            * obs_preprocessed = (1, 4 + num_sem_categories, H, W) torch.Tensor
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
                - rotations are in 3D
        """
        rgb = torch.from_numpy(obs.rgb).to(device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(device) * 100.0
        )  # m to cm
        semantic = one_hot_encoding[torch.from_numpy(obs.semantic).to(device)]
        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1).unsqueeze(0)
        obs_preprocessed = obs_preprocessed.permute(0, 3, 1, 2)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, last_pose)
        ).unsqueeze(0)

        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)
        return (
            obs_preprocessed,
            pose_delta,
            camera_pose,
            curr_pose
        )

    last_pose = np.zeros(3)
    one_hot_encoding = torch.eye(num_sem_categories, device=device)

    for i, obs in enumerate(observations):
        # Preprocess observation
        (
            obs_preprocessed,
            pose_delta,
            camera_pose,
            last_pose
        ) = preprocess_obs(obs, last_pose)

        if i == 0:
            print()
            print("preprocessed observations:")
            print("-------------------------")
            print("obs_preprocessed", obs_preprocessed.shape)
            print("pose_delta", pose_delta)
            print("camera_pose", camera_pose)


        for x in [obs_preprocessed.unsqueeze(1),
                pose_delta.unsqueeze(1),
                dones.unsqueeze(1),
                update_global.unsqueeze(1),
                camera_pose,
                semantic_map.local_map,
                semantic_map.global_map,
                semantic_map.local_pose,
                semantic_map.global_pose,
                semantic_map.lmb,
                semantic_map.origins]:
            print(x.device)

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

    # --------------------------------------------------------------------------------------------
    # Map visualization
    # --------------------------------------------------------------------------------------------

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
        *coco_categories_color_palette,
    ]
    map_color_palette = [int(x * 255.0) for x in map_color_palette]

    semantic_categories_map = semantic_map.get_semantic_map(0)
    obstacle_map = semantic_map.get_obstacle_map(0)
    explored_map = semantic_map.get_explored_map(0)
    visited_map = semantic_map.get_visited_map(0)

    semantic_categories_map += 4
    no_category_mask = semantic_categories_map == 4 + num_sem_categories - 1
    obstacle_mask = np.rint(obstacle_map) == 1
    explored_mask = np.rint(explored_map) == 1
    visited_mask = visited_map == 1
    semantic_categories_map[no_category_mask] = 0
    semantic_categories_map[np.logical_and(no_category_mask, explored_mask)] = 2
    semantic_categories_map[np.logical_and(no_category_mask, obstacle_mask)] = 1
    semantic_categories_map[visited_mask] = 3

    semantic_map_vis = Image.new("P", semantic_categories_map.shape)
    semantic_map_vis.putpalette(map_color_palette)
    semantic_map_vis.putdata(semantic_categories_map.flatten().astype(np.uint8))
    semantic_map_vis = semantic_map_vis.convert("RGB")
    semantic_map_vis = np.flipud(semantic_map_vis)
    plt.imsave("semantic_map.png", semantic_map_vis)


if __name__ == "__main__":
    main()
