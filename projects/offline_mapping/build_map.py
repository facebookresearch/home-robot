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

# TODO Install home_robot and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)

from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import Categorical2DSemanticMapState
from home_robot.mapping.semantic.categorical_2d_semantic_map_module import Categorical2DSemanticMapModule
from home_robot.utils.config import get_config


@click.command()
@click.option(
    "--trajectory_path",
    default="trajectories/fremont1",
)
def main(trajectory_path):
    config_path = "projects/offline_mapping/configs/agent/eval.yaml"
    config, config_str = get_config(config_path)

    # --------------------------------------------------------------------------------------------
    # Load trajectory of home_robot Observations
    # --------------------------------------------------------------------------------------------

    observations = []
    for path in sorted(glob.glob(str(Path(__file__).resolve().parent) + f"/{trajectory_path}/*.pkl")):
        with open(path, "rb") as f:
            observations.append(pickle.load(f))

    # --------------------------------------------------------------------------------------------
    # Preprocess trajectory
    # --------------------------------------------------------------------------------------------

    # Predict semantic segmentation
    categories = [
        "other",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "toilet",
        "tv",
        "dining table",
        "oven",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "cup",
        "bottle",
        "other",
    ]
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

    obs = observations[0]
    print()
    print("obs.gps", obs.gps)
    print("obs.compass", obs.compass)
    print("obs.rgb", obs.rgb.shape, obs.rgb.min(), obs.rgb.max())
    print("obs.depth", obs.depth.shape, obs.depth.min(), obs.depth.max())
    print("obs.semantic", obs.semantic.shape, obs.semantic.min(), obs.semantic.max())
    print("obs.camera_pose", obs.camera_pose)
    print("obs.task_observations", obs.task_observations.keys())
    print()

    # --------------------------------------------------------------------------------------------
    # Build semantic map
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
        frame_height=480,
        frame_width=640,
        camera_height=0.88,
        hfov=79.0,
        num_sem_categories=16,
        map_size_cm=4800,
        map_resolution=5,
        vision_range=100,
        global_downscaling=2,
        du_scale=4,
        cat_pred_threshold=5.0,
        exp_pred_threshold=1.0,
        map_pred_threshold=1.0,
    ).to(device)


if __name__ == "__main__":
    main()
