#!/usr/bin/env python
from typing import Optional, Tuple

import click
import rospy

from home_robot.agent.ovmm_agent.pick_and_place_agent import PickAndPlaceAgent
from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot_hw.env.stretch_pick_and_place_env import (
    REAL_WORLD_CATEGORIES,
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
@click.option("--object", default="cup")
@click.option("--start-recep", default="table")
@click.option("--goal-recep", default="chair")
@click.option("--visualize-maps", default=False, is_flag=True)
@click.option(
    "--debug",
    default=False,
    is_flag=True,
    help="Add pauses for debugging manipulation behavior.",
)
def main(
    test_pick=False,
    test_gaze=False,
    skip_gaze=True,
    reset_nav=False,
    object="cup",
    start_recep="table",
    goal_recep="chair",
    dry_run=False,
    visualize_maps=False,
    test_place=False,
    **kwargs,
):
    print("- Starting ROS node")
    rospy.init_node("eval_episode_stretch_objectnav")

    REAL_WORLD_CATEGORIES[1] = start_recep
    REAL_WORLD_CATEGORIES[2] = object
    REAL_WORLD_CATEGORIES[3] = goal_recep
    print("- Loading configuration")
    config = load_config(visualize=visualize_maps, **kwargs)

    print("- Creating environment")
    env = StretchPickandPlaceEnv(
        goal_options=REAL_WORLD_CATEGORIES,
        config=config,
        test_grasping=test_pick,
        dry_run=dry_run,
    )

    # TODO: May be a bit easier if we just read skip_{skill} from command line - similar to habitat_ovmm
    print("- Creating agent")
    agent = PickAndPlaceAgent(
        config=config,
        skip_find_object=test_pick or test_gaze,
        skip_orient=False,
        skip_gaze=test_pick or skip_gaze,
        skip_pick=test_gaze,
        skip_place=test_pick or test_gaze,
        test_place=test_place,
    )

    robot = env.get_robot()
    if reset_nav:
        print("- Sending the robot to [0, 0, 0]")
        # Send it back to origin position to make testing a bit easier
        robot.nav.navigate_to([0, 0, 0])

    agent.reset()
    env.reset(start_recep, object, goal_recep)

    t = 0
    while not env.episode_over and not rospy.is_shutdown():
        t += 1
        print("STEP =", t)
        obs = env.get_observation()
        action, info = agent.act(obs)
        done = env.apply_action(action, info=info)
        if done:
            break

    print(env.get_episode_metrics())


if __name__ == "__main__":
    print("---- Starting real-world evaluation ----")
    main()
