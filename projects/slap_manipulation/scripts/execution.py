# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

import click
import numpy as np
import yaml

from home_robot.motion.stretch import STRETCH_HOME_Q, HelloStretchKinematics
from home_robot_hw.remote import StretchClient


@click.command()
@click.option(
    "--path",
    default="projects/slap_manipulation/actions.yaml",
    help="Path to action file",
)
def main(path):
    # load action dict from "~/actions.yaml"
    actions = yaml.unsafe_load(open(path, "r"))
    robot = StretchClient()
    debug = False

    # Switch to manipulation mode
    robot.switch_to_manipulation_mode()
    assert robot.in_manipulation_mode()

    # Get gripper pose
    pos, quat = robot.manip.get_ee_pose()
    print(f"EE pose: pos={pos}, quat={quat}")
    base_pose_start = robot.nav.get_base_pose()

    # Command the robot arm 2: Absolute EE control
    move_base_flag = True
    for i, action in enumerate(actions):
        base_pose = robot.nav.get_base_pose()
        if debug:
            print(base_pose)
            input("Press Enter")
        pos_desired = action["pos"]
        quat_desired = action["ori"]
        q0 = robot.manip.get_joint_positions()
        if i > 0:
            pos_desired[0] = 0.0
            # move_base_flag = False
            action["gripper"] = 1
        print(f"EE pose: pos={pos_desired}, quat={quat_desired}")
        robot.manip.goto_ee_pose(
            pos_desired,
            quat_desired,
            relative=False,
            world_frame=True,
            initial_cfg=q0,
            # move_base=move_base_flag,
        )
        if action["gripper"] == 1:
            robot.manip.close_gripper(blocking=True)
        else:
            robot.manip.open_gripper(blocking=True)
        if debug:
            input("Press enter to continue")

    # Stop all robot motion
    robot.stop()


if __name__ == "__main__":
    main()
