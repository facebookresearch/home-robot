#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
This test is for robot base motion back and forth
"""

import click
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped

from home_robot.agent.ovmm_agent.pick_and_place_agent import PickAndPlaceAgent
from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot.utils.config import get_config
from home_robot.utils.geometry import posquat2sophus, sophus2posquat, xyt2sophus
from home_robot.utils.pose import to_pos_quat
from home_robot_hw.env.stretch_pick_and_place_env import (
    StretchPickandPlaceEnv,
    load_config,
)


@click.command()
@click.option("--test-id", default=0, type=int)
def run_experiment(visualize_maps=False, test_id=0, reset_nav=False, **kwargs):
    config = load_config(visualize=visualize_maps, **kwargs)
    rospy.init_node("eval_episode_stretch_objectnav")
    env = StretchPickandPlaceEnv(config=config, ros_grasping=False)
    env.reset("table", "cup", "chair")
    robot = env.get_robot()

    # Put it into initial posture
    env.robot.move_to_nav_posture()

    i = 0
    while not rospy.is_shutdown():
        # Send it back to origin position to make testing a bit easier
        print(i)
        if i % 2 == 0:
            robot.nav.navigate_to([0, 0, 0])
        else:
            robot.nav.navigate_to([0.2, 0, 0])
        i += 1

        # Done
        input("--- Press enter to continue ---")


if __name__ == "__main__":
    run_experiment()
