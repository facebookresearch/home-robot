# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional

import ros_numpy
import rospy
import tf2_ros

from home_robot.motion.robot import Robot
from home_robot.motion.stretch import HelloStretchKinematics
from home_robot_hw.constants import ControlMode

from .modules.head import StretchHeadClient
from .modules.manip import StretchManipulationClient
from .modules.nav import StretchNavigationClient
from .ros import StretchRosInterface


class StretchClient:
    """Defines a ROS-based interface to the real Stretch robot. Collect observations and command the robot."""

    def __init__(
        self,
        init_node: bool = True,
        camera_overrides: Optional[Dict] = None,
        visualize_ik: bool = False,
    ):
        # Ros
        if init_node:
            rospy.init_node("stretch_user_client")

        if camera_overrides is None:
            camera_overrides = {}
        self._ros_client = StretchRosInterface(**camera_overrides)

        # Robot model
        self._robot_model = HelloStretchKinematics(visualize=visualize_ik)

        # Interface modules
        self.nav = StretchNavigationClient(self._ros_client, self._robot_model)
        self.manip = StretchManipulationClient(self._ros_client, self._robot_model)
        self.head = StretchHeadClient(self._ros_client, self._robot_model)

        # Init control mode
        self._base_control_mode = ControlMode.IDLE

    # Mode interfaces

    def switch_to_navigation_mode(self):
        """Switch stretch to navigation control
        Robot base is now controlled via continuous velocity feedback.
        """
        result_pre = True
        if self.manip.is_enabled:
            result_pre = self.manip.disable()

        result_post = self.nav.enable()

        self._base_control_mode = ControlMode.NAVIGATION

        return result_pre and result_post

    def switch_to_manipulation_mode(self):
        """Switch stretch to manipulation control
        Robot base is now controlled via position control.
        Base rotation is locked.
        """
        result_pre = True
        if self.nav.is_enabled:
            result_pre = self.nav.disable()

        result_post = self.manip.enable()

        self._base_control_mode = ControlMode.MANIPULATION

        return result_pre and result_post

    def in_manipulation_mode(self):
        return self._base_control_mode == ControlMode.MANIPULATION

    def in_navigation_mode(self):
        return self._base_control_mode == ControlMode.NAVIGATION

    # General control methods

    def wait(self):
        self.nav.wait()
        self.manip.wait()
        self.head.wait()

    def reset(self):
        self.stop()
        self.switch_to_manipulation_mode()
        self.manip.home()
        self.switch_to_navigation_mode()
        self.nav.home()
        self.stop()

    def stop(self):
        self.nav.disable()
        self.manip.disable()
        self._base_control_mode = ControlMode.IDLE

    # Other interfaces

    @property
    def robot_model(self) -> Robot:
        return self._robot_model
