#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import click
import glob
import pickle
import sys
from pathlib import Path

# TODO Install home_robot and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)

from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.utils.config import get_config


@click.command()
@click.option(
    "--trajectory_path",
    default="trajectories/fremont1",
)
def main(trajectory_path):
    config_path = "projects/offline_mapping/configs/agent/eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.EXP_NAME = "debug"
    config.freeze()

    agent = ObjectNavAgent(config=config)
    agent.reset()

    # Load trajectory
    observations = []
    for path in sorted(glob.glob(str(Path(__file__).resolve().parent) + f"/{trajectory_path}/*.pkl")):
        with open(path, "rb") as f:
            observations.append(pickle.load(f))

    # If the trajectory doesn't contain semantics, predict them here
    obs = observations[0]
    if obs.semantic is None:
        categories = [
            "other",
            "chair",
            "sofa",
            "plant",
            "bed",
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
            obs.task_observations = {
                "goal_id": 1,
                "goal_name": 1,
                "object_goal": 1,
                "recep_goal": 1,
            }

    print()
    print("obs.gps", obs.gps)
    print("obs.compass", obs.compass)
    print("obs.rgb", obs.rgb.shape, obs.rgb.min(), obs.rgb.max())
    print("obs.depth", obs.depth.shape, obs.depth.min(), obs.depth.max())
    print("obs.semantic", obs.semantic.shape, obs.semantic.min(), obs.semantic.max())
    print("obs.camera_pose", obs.camera_pose)
    print("obs.task_observations", obs.task_observations.keys())
    print()

    print(f"Iterating over {len(observations)} observations")
    for obs in observations:
        agent.act(obs)


if __name__ == "__main__":
    main()
