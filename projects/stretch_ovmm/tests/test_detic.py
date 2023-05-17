#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
This file is a test for inverse kinematics on the stretch robot. It makes sure we can reach and execute different positions, which have been generated by the grasp planner. It's a useful utility for now.
"""

#!/usr/bin/env python
from typing import Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import rospy

from home_robot.agent.ovmm_agent.pick_and_place_agent import PickAndPlaceAgent
from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot.utils.config import get_config
from home_robot.utils.geometry import posquat2sophus, sophus2posquat, xyt2sophus
from home_robot.utils.pose import to_pos_quat
from home_robot_hw.env.stretch_pick_and_place_env import (
    REAL_WORLD_CATEGORIES,
    StretchPickandPlaceEnv,
    load_config,
)
from home_robot_hw.ros.utils import matrix_to_pose_msg, ros_pose_to_transform


@click.command()
@click.option("--reset-nav", default=False, is_flag=True)
@click.option("--object", default="cup")
@click.option("--start-recep", default="table")
@click.option("--goal-recep", default="chair")
@click.option(
    "--debug",
    default=False,
    is_flag=True,
    help="Add pauses for debugging manipulation behavior.",
)
def main(
    reset_nav=False,
    object="cup",
    start_recep="table",
    goal_recep="chair",
    **kwargs,
):
    REAL_WORLD_CATEGORIES[2] = object
    config = load_config(**kwargs)

    rospy.init_node("eval_episode_stretch_objectnav")
    env = StretchPickandPlaceEnv(
        goal_options=REAL_WORLD_CATEGORIES,
        config=config,
        dry_run=True,
        ros_grasping=False,
    )

    robot = env.get_robot()
    if reset_nav:
        # Send it back to origin position to make testing a bit easier
        robot.nav.navigate_to([0, 0, 0])

    env.reset(start_recep, object, goal_recep)

    while not rospy.is_shutdown():
        obs = env.get_observation()
        # obs = env.segmentation.predict(obs, draw_instance_predictions=True)
        vis = obs.task_observations["semantic_frame"]

        # Visualize what we can see
        plt.figure(1)
        plt.imshow(vis)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
