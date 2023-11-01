# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import natsort

# TODO Install home_robot, home_robot_sim and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot_sim"),
)

from config_utils import get_config
from habitat.core.env import Env

from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot_sim.env.habitat_objectnav_env.habitat_objectnav_env import (
    HabitatObjectNavEnv,
)


def create_video(images, output_file, fps):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for image in images:
        video_writer.write(image)
    video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="objectnav/modular_objectnav_hm3d.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/habitat_objectnav/configs/agent/hm3d_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    print("Arguments:")
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    print("-" * 100)

    config = get_config(args.habitat_config_path, args.baseline_config_path)

    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.habitat.dataset.split = "val"
    config.EXP_NAME = "debug"

    agent = ObjectNavAgent(config=config)
    env = HabitatObjectNavEnv(Env(config=config), config=config)

    env.reset()
    agent.reset()

    scene_id = env.habitat_env.current_episode.scene_id.split("/")[-1].split(".")[0]
    agent.planner.set_vis_dir(scene_id, env.habitat_env.current_episode.episode_id)

    t = 0

    while not env.episode_over:
        t += 1
        print(t)
        obs = env.get_observation()
        action, info = agent.act(obs)
        env.apply_action(action, info=info)

    print(env.get_episode_metrics())

    # Record video
    images = []
    for path in natsort.natsorted(glob.glob(f"{env.visualizer.vis_dir}/*.png")):
        images.append(cv2.imread(path))
    create_video(images, f"{env.visualizer.vis_dir}/video.mp4", fps=20)

    if config.AGENT.SEMANTIC_MAP.record_instance_ids:
        # TODO Can we create a visualization of the instance memory here?
        print("Let's generate visualization")
        pass
