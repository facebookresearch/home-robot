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
        urdf_path: str = "",
        ik_type: str = "pybullet",
    ):
        # Ros
        if init_node:
            rospy.init_node("stretch_user_client")

        if camera_overrides is None:
            camera_overrides = {}
        self._ros_client = StretchRosInterface(**camera_overrides)

        # Robot model
        self._robot_model = HelloStretchKinematics(urdf_path=urdf_path, ik_type=ik_type)

        # Interface modules
        self.nav = StretchNavigationClient(
            self._ros_client, self._robot_model
        )  # -- TODO: whatever is supposed to be populating se3_base_filtered never is...
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

    @property
    def robot_joint_pos(self):
        return self._ros_client.pos

    @property
    def camera_pose(self):
        return self._ros_client.se3_camera_pose

    @property
    def rgb_cam(self):
        return self._ros_client.rgb_cam

    @property
    def dpt_cam(self):
        return self._ros_client.dpt_cam

    def get_frame_pose(
        self, frame, base_frame=None, lookup_time=None, timeout_s=None
    ):  # TODO: this won't work at all? no odom_link, no tf2_buffer...
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
        ):
            print("!!! Lookup failed from", self.base_link, "to", self.odom_link, "!!!")
            return None
        return pose_mat
