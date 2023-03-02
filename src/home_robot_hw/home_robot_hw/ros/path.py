# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

import rospkg

from home_robot.motion.stretch import PLANNER_STRETCH_URDF


def get_package_path():
    r = rospkg.RosPack()
    return r.get_path("home_robot_hw")


def get_urdf_path():
    return os.path.join(get_package_path(), PLANNER_STRETCH_URDF)


def get_urdf_dir():
    """location of the calibrated urdfs for use in planning"""
    return os.path.join(get_package_path(), "assets/hab_stretch/urdf")
