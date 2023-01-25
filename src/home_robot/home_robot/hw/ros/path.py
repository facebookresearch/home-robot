# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import rospy
import rospkg

from home_robot.agent.motion.robot import PLANNER_STRETCH_URDF


def get_package_path():
    r = rospkg.RosPack()
    return r.get_path("home_robot")


def get_urdf_path():
    return os.path.join(get_package_path(), "..", "..", PLANNER_STRETCH_URDF)
