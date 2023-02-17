#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import rospy
import os
from home_robot_hw.teleop.stretch_xbox_controller import StretchXboxController
from home_robot.agent.motion.stretch import HelloStretch
from home_robot_hw.ros.path import get_urdf_dir


if __name__ == "__main__":
    rospy.init_node("xbox_controller")

    stretch_planner_urdf_path = get_urdf_dir()
    print(stretch_planner_urdf_path)
    model = HelloStretch(
        visualize=False,
        root="",
        urdf_path=stretch_planner_urdf_path,
    )
    controller = StretchXboxController(model)
    rospy.spin()
