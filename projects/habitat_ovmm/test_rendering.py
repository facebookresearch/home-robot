#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
import time

import cv2
import habitat
from gym import spaces
from habitat_baselines.config.default import get_config
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cfg-path",
    type=str,
    default="src/third_party/habitat-lab/habitat-lab/habitat/config/benchmark/ovmm/ovmm.yaml",
)
parser.add_argument("--num-eps", type=int, default=50)
parser.add_argument(
    "opts",
    default=None,
    nargs=argparse.REMAINDER,
    help="Modify config options from command line",
)
args = parser.parse_args()


def get_ac_cont(env):
    ac_names = ["base_velocity"]
    ac_args = {}
    for ac_name in ac_names:
        ac_args.update(env.action_space.spaces[ac_name].sample())
    return {"action": "base_velocity", "action_args": ac_args}


def get_ac_disc(env):
    ac_names = list(env.action_space.keys())
    ac_names.remove("stop")
    return {"action": random.choice(ac_names), "action_args": {}}


def get_ac(env):
    if "stop" in env.action_space.spaces:
        return get_ac_disc(env)
    else:
        return get_ac_cont(env)


def set_episode(env, episode_id):
    episode = [ep for ep in env.episodes if ep.episode_id == episode_id][0]
    env.current_episode = episode


def save_image(image, step):
    folder_name = "scene_visuals_with_hbao"
    os.makedirs(folder_name, exist_ok=True)
    cv2.imwrite(
        f"{folder_name}/{env.current_episode.scene_id.split('/')[-1].split('.')[0]}_{i}_{step}.png",
        image,
    )


config = get_config(args.cfg_path, args.opts)

with habitat.Env(config=config) as env:
    env.reset()
    for i in range(args.num_eps):
        observations = env.reset()

        save_image(observations["robot_third_rgb"][:, :, [2, 1, 0]], 0)
        step = 0
        while not env.episode_over:
            step += 1
            ac = get_ac(env)
            print(ac)

            observations = env.step(ac)
            save_image(observations["robot_third_rgb"][:, :, [2, 1, 0]], step)
