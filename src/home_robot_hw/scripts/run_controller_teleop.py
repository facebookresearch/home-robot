#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import rospy

from home_robot.motion.stretch import HelloStretchKinematics
from home_robot_hw.ros.path import get_urdf_dir
from home_robot_hw.teleop.stretch_xbox_controller import StretchXboxController

if __name__ == "__main__":
    stretch_planner_urdf_path = get_urdf_dir()
    model = HelloStretchKinematics(
        visualize=False,
        root="",
        urdf_path=stretch_planner_urdf_path,
    )
    controller = StretchXboxController(model)
    rospy.spin()
