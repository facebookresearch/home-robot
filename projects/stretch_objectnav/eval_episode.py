#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import click
import rospy

from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.agent.objectnav_agent.sampling_agent import SamplingBasedObjectNavAgent
from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot.utils.config import get_config
from home_robot_hw.env.stretch_object_nav_env import StretchObjectNavEnv


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
    config_path = "projects/stretch_objectnav/configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.EXP_NAME = "debug"
    config.freeze()

    rospy.init_node("eval_episode_stretch_objectnav")
    if agent == "discrete":
        agent = ObjectNavAgent(config=config)
    elif agent == "sampling":
        agent = SamplingBasedObjectNavAgent(config=config)
    else:
        raise NotImplementedError(f"agent {agent} not recognized")
    env = StretchObjectNavEnv(config=config, dry_run=dry_run)

    agent.reset()
    env.reset()

    t = 0
    while not env.episode_over:
        t += 1
        obs = env.get_observation()
        action, info = agent.act(obs)
        print("STEP =", t)
        env.apply_action(action, info=info)

    print(env.get_episode_metrics())


if __name__ == "__main__":
    main()
