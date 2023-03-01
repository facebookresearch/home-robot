# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pybullet as pb

from home_robot.motion.robot import Robot

DEFAULT_FRANKA_URDF = "assets/franka_panda/panda.urdf"
DEFAULT_CALVIN_FRANKA_URDF = "assets/franka_panda/panda_calvin.urdf"


class Franka(Robot):
    def __init__(self, name="robot", urdf_path=None, visualize=False):
        """Create the robot in bullet for things like kinematics; extract information"""
        # urdf path
        if urdf_path is None:
            urdf_path = DEFAULT_FRANKA_URDF
        super(Franka, self).__init__(name, urdf_path, visualize)


class CalvinFranka(Franka):
    def __init__(self, name="robot", urdf_path=None, visualize=False):
        """Create the robot in bullet for things like kinematics; extract information"""
        # urdf path
        if urdf_path is None:
            urdf_path = DEFAULT_CALVIN_FRANKA_URDF
        super(CalvinFranka, self).__init__(name, urdf_path, visualize)
