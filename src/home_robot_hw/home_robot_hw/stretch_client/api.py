# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import rospy

from home_robot_hw.constants import ControlMode

from .ros import StretchRosInterface
from .modules.nav import StretchNavigationInterface
from .modules.manip import StretchManipulationInterface
from .modules.camera import StretchCameraInterface

class StretchClient:
    """Defines a ROS-based interface to the real Stretch robot. Collect observations and command the robot."""

    def __init__(self, init_cameras: bool = True, init_node: bool = False):
        # Ros
        if init_node:
            rospy.init_node("stretch_user_client")
        self._ros_client = StretchRosInterface()

        # Interface modules
        self.nav = StretchNavigationInterface(self._ros_client)
        self.manip = StretchManipulationInterface(self._ros_client)
        self.camera = StretchCameraInterface(self._ros_client)

        # Init control mode
        self._base_control_mode = ControlMode.IDLE

    # Mode interfaces

    def switch_to_navigation_mode(self):
        """Switch stretch to navigation control
        Robot base is now controlled via continuous velocity feedback.
        """
        self._base_control_mode = ControlMode.NAVIGATION

        result_pre = True
        if self.in_manipulation_mode():
            result_pre = self.manip.disable()

        result_post = self.nav.enable()

        return result_pre and result_post

    def switch_to_manipulation_mode(self):
        """Switch stretch to manipulation control
        Robot base is now controlled via position control.
        Base rotation is locked.
        """
        self._base_control_mode = ControlMode.MANIPULATION
        
        result_pre = True
        if self.in_navigation_mode():
            result_pre = self.nav.disable()

        result_post = self.manip.enable()

        return result_pre and result_post

    def in_manipulation_mode(self):
        return self._base_control_mode == ControlMode.MANIPULATION

    def in_navigation_mode(self):
        return self._base_control_mode == ControlMode.NAVIGATION

    # General state getters

    def get_frame_pose(self, frame, base_frame=None, lookup_time=None, timeout_s=None):
        """look up a particular frame in base coords"""
        if lookup_time is None:
            lookup_time = rospy.Time(0)  # return most recent transform
        if timeout_s is None:
            timeout_ros = rospy.Duration(0.1)
        else:
            timeout_ros = rospy.Duration(timeout_s)
        if base_frame is None:
            base_frame = self.odom_link
        try:
            stamped_transform = self.tf2_buffer.lookup_transform(
                base_frame, frame, lookup_time, timeout_ros
            )
            pose_mat = ros_numpy.numpify(stamped_transform.transform)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            print("!!! Lookup failed from", self.base_link, "to", self.odom_link, "!!!")
            return None
        return pose_mat

