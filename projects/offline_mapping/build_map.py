#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import click
import glob
import pickle

from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.utils.config import get_config


@click.command()
@click.option(
    "--trajectory_path",
    default="trajectories/fremont1",
)
def main(trajectory_path):
    config_path = "projects/offline_mapping/configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.EXP_NAME = "debug"
    config.freeze()

    agent = ObjectNavAgent(config=config)
    agent.reset()

    observations = []
    for path in sorted(glob.glob(f"{trajectory_path}/*.pkl")):
        with open(path, "rb") as f:
            observations.append(pickle.load(f))

    for obs in observations:
        agent.act(obs)


if __name__ == "__main__":
    main()
