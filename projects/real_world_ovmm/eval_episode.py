#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from datetime import datetime
from typing import Optional, Tuple

import click
import rospy

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot.utils.config import load_config
from home_robot_hw.env.stretch_pick_and_place_env import StretchPickandPlaceEnv


@click.command()
@click.option("--test-pick", default=False, is_flag=True)
@click.option("--test-gaze", default=False, is_flag=True)
@click.option("--test-place", default=False, is_flag=True)
@click.option("--skip-gaze", default=True, is_flag=True)
@click.option("--reset-nav", default=False, is_flag=True)
@click.option("--dry-run", default=False, is_flag=True)
@click.option("--pick-object", default="cup")
@click.option("--start-recep", default="chair")
@click.option("--goal-recep", default="chair")
@click.option(
    "--cat-map-file", default="projects/real_world_ovmm/configs/example_cat_map.json"
)
@click.option("--max-num-steps", default=200)
@click.option("--visualize-maps", default=True, is_flag=True)
@click.option("--visualize-grasping", default=True, is_flag=True)
@click.option(
    "--debug",
    default=False,
    is_flag=True,
    help="Add pauses for debugging manipulation behavior.",
)
def main(
    test_pick=False,
    reset_nav=False,
    pick_object="cup",
    start_recep="chair",
    goal_recep="chair",
    dry_run=False,
    visualize_maps=True,
    visualize_grasping=True,
    test_place=False,
    cat_map_file=None,
    max_num_steps=200,
    config_path="projects/real_world_ovmm/configs/agent/eval.yaml",
    **kwargs,
):
    print("- Starting ROS node")
    rospy.init_node("eval_episode_stretch_objectnav")

    print("- Loading configuration")
    config = load_config(config_path=config_path, visualize=visualize_maps, **kwargs)

    print("- Creating environment")
    env = StretchPickandPlaceEnv(
        config=config,
        test_grasping=test_pick,
        dry_run=dry_run,
        cat_map_file=cat_map_file,
        visualize_grasping=visualize_grasping,
    )

    print("- Creating agent")
    agent = OpenVocabManipAgent(config=config)

    robot = env.get_robot()
    if reset_nav:
        print("- Sending the robot to [0, 0, 0]")
        # Send it back to origin position to make testing a bit easier
        robot.nav.navigate_to([0, 0, 0])

    agent.reset()
    if hasattr(agent, "planner"):
        now = datetime.now()
        agent.planner.set_vis_dir("real_world", now.strftime("%Y_%m_%d_%H_%M_%S"))
    env.reset(start_recep, pick_object, goal_recep)

    t = 0
    while not env.episode_over and not rospy.is_shutdown():
        t += 1
        print("STEP =", t)
        obs = env.get_observation()
        action, info, obs = agent.act(obs)
        done = env.apply_action(action, info=info, prev_obs=obs)
        if done:
            print("Done.")
            break
        elif t >= max_num_steps:
            print("Reached maximum step limit.")
            break

    print("Metrics:", env.get_episode_metrics())


if __name__ == "__main__":
    print("---- Starting real-world evaluation ----")
    main()
    print("==================================")
    print("Done real world evaluation.")
