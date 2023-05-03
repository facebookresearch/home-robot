# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Optional

import rospy

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
        ik_type: str = "pinocchio",
        visualize_ik: bool = False,
        grasp_frame: Optional[str] = None,
        ee_link_name: Optional[str] = None,
        manip_mode_controlled_joints: Optional[List[str]] = None,
    ):
        """Create an interface into ROS execution here. This one needs to connect to:
            - joint_states to read current position
            - tf for SLAM
            - FollowJointTrajectory for arm motions

        Based on this code:
        https://github.com/hello-robot/stretch_ros/blob/master/hello_helpers/src/hello_helpers/hello_misc.py
        """
        # Ros
        if init_node:
            rospy.init_node("stretch_user_client")

        if camera_overrides is None:
            camera_overrides = {}
        self._ros_client = StretchRosInterface(**camera_overrides)

        # Robot model
        self._robot_model = HelloStretchKinematics(
            urdf_path=urdf_path,
            ik_type=ik_type,
            visualize=visualize_ik,
            grasp_frame=grasp_frame,
            ee_link_name=ee_link_name,
            manip_mode_controlled_joints=manip_mode_controlled_joints,
        )

        # Interface modules
        self.nav = StretchNavigationClient(self._ros_client, self._robot_model)
        self.manip = StretchManipulationClient(self._ros_client, self._robot_model)
        self.head = StretchHeadClient(self._ros_client, self._robot_model)

        # Init control mode
        self._base_control_mode = ControlMode.IDLE

        # Initially start in navigation mode all the time - in order to make sure we are initialized into a decent state. Otherwise we need to check the different components and safely figure out control mode, which can be inaccurate.
        self.switch_to_navigation_mode()

    @property
    def model(self):
        return self._robot_model

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
        return self.head.get_pose_in_base_coords(rotated=False)

    @property
    def rgb_cam(self):
        return self._ros_client.rgb_cam

    @property
    def dpt_cam(self):
        return self._ros_client.dpt_cam

    def get_frame_pose(self, frame, base_frame=None, lookup_time=None):
        """look up a particular frame in base coords"""
        return self._ros_client.get_frame_pose(frame, base_frame, lookup_time)
