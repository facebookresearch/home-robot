#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time

import click
import rospy

from home_robot.agent.exploration_agent.exploration_agent import ExplorationAgent
from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot.utils.config import get_config
from home_robot_hw.env.stretch_exploration_env import StretchExplorationEnv


@click.command()
@click.option(
    "--agent",
    default="discrete",
    type=click.Choice(["discrete", "sampling"], case_sensitive=False),
)
@click.option(
    "--dry-run",
    default=False,
    is_flag=True,
    help="do not execute any actions, just print them",
)
def main(agent, dry_run):
    config_path = "projects/stretch_exploration/configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.EXP_NAME = "debug"
    config.freeze()

    rospy.init_node("eval_episode_stretch_exploration")
    if agent == "discrete":
        agent = ExplorationAgent(config=config)
    else:
        raise NotImplementedError(f"agent {agent} not recognized")
    env = StretchExplorationEnv(config=config, visualize=True, dry_run=dry_run)

    agent.reset()
    env.reset()

    t = 0
    while not env.episode_over:
        t += 1
        obs = env.get_observation()
        action, info = agent.act(obs)
        print("STEP =", t)
        print("=======> Submitted action")
        start_ = time.time()
        env.apply_action(action, info=info)
        print("=======> Action completed in {} secs".format(time.time() - start_))

    print(env.get_episode_metrics())


if __name__ == "__main__":
    main()
