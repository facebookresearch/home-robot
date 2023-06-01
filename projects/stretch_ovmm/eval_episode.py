#!/usr/bin/env python
from typing import Optional, Tuple

import click
import rospy

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot_hw.env.stretch_pick_and_place_env import (
    StretchPickandPlaceEnv,
    load_config,
)


@click.command()
@click.option("--test-pick", default=False, is_flag=True)
@click.option("--test-gaze", default=False, is_flag=True)
@click.option("--test-place", default=False, is_flag=True)
@click.option("--skip-gaze", default=True, is_flag=True)
@click.option("--reset-nav", default=False, is_flag=True)
@click.option("--dry-run", default=False, is_flag=True)
@click.option("--pick-object", default="cup")
@click.option("--start-recep", default="table")
@click.option("--goal-recep", default="chair")
@click.option(
    "--cat-map-file", default="projects/stretch_ovmm/configs/example_cat_map.json"
)
@click.option("--visualize-maps", default=False, is_flag=True)
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
    start_recep="table",
    goal_recep="chair",
    dry_run=False,
    visualize_maps=False,
    cat_map_file=None,
    **kwargs,
):
    print("- Starting ROS node")
    rospy.init_node("eval_episode_stretch_objectnav")

    print("- Loading configuration")
    config = load_config(visualize=visualize_maps, **kwargs)

    print("- Creating environment")
    env = StretchPickandPlaceEnv(
        config=config,
        test_grasping=test_pick,
        dry_run=dry_run,
        cat_map_file=cat_map_file,
    )

    print("- Creating agent")
    agent = OpenVocabManipAgent(config=config)

    robot = env.get_robot()
    if reset_nav:
        print("- Sending the robot to [0, 0, 0]")
        # Send it back to origin position to make testing a bit easier
        robot.nav.navigate_to([0, 0, 0])

    agent.reset()
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

    print("Metrics:", env.get_episode_metrics())


if __name__ == "__main__":
    print("---- Starting real-world evaluation ----")
    main()
    print("==================================")
    print("Done real world evaluation.")
