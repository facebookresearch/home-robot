# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Optional

import numpy as np
import rospy

from home_robot.core.interfaces import Observations
from home_robot.motion.robot import Robot
from home_robot.motion.stretch import (
    STRETCH_DEMO_PREGRASP_Q,
    STRETCH_NAVIGATION_Q,
    STRETCH_POSTNAV_Q,
    STRETCH_PREDEMO_Q,
    STRETCH_PREGRASP_Q,
    HelloStretchKinematics,
)
from home_robot.utils.geometry import xyt2sophus
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

    def get_joint_state(self):
        return self._ros_client.get_joint_state()

    def get_frame_pose(self, frame, base_frame=None, lookup_time=None):
        """look up a particular frame in base coords"""
        return self._ros_client.get_frame_pose(frame, base_frame, lookup_time)

    def move_to_manip_posture(self):
        """Move the arm and head into manip mode posture: gripper down, head facing the gripper."""
        self.switch_to_manipulation_mode()
        self.head.look_at_ee(blocking=False)
        self.manip.goto_joint_positions(
            self.manip._extract_joint_pos(STRETCH_PREGRASP_Q)
        )
        print("- Robot switched to manipulation mode.")

    def move_to_demo_pregrasp_posture(self):
        """Move the arm and head into pre-demo posture: gripper straight, arm way down, head facing the gripper."""
        self.switch_to_manipulation_mode()
        self.head.look_at_ee(blocking=False)
        self.manip.goto_joint_positions(
            self.manip._extract_joint_pos(STRETCH_DEMO_PREGRASP_Q)
        )

    def move_to_pre_demo_posture(self):
        """Move the arm and head into pre-demo posture: gripper straight, arm way down, head facing the gripper."""
        self.switch_to_manipulation_mode()
        self.head.look_at_ee(blocking=False)
        self.manip.goto_joint_positions(
            self.manip._extract_joint_pos(STRETCH_PREDEMO_Q)
        )

    def move_to_nav_posture(self):
        """Move the arm and head into nav mode. The head will be looking front."""

        # First retract the robot's joints
        self.switch_to_manipulation_mode()
        self.head.look_front(blocking=False)
        self.manip.goto_joint_positions(
            self.manip._extract_joint_pos(STRETCH_NAVIGATION_Q)
        )
        self.switch_to_navigation_mode()
        print("- Robot switched to navigation mode.")

    def move_to_post_nav_posture(self):
        """Move the arm to nav mode, head to nav mode with PREGRASP's tilt. The head will be looking front."""
        self.switch_to_manipulation_mode()
        self.head.look_front(blocking=False)
        self.manip.goto_joint_positions(
            self.manip._extract_joint_pos(STRETCH_POSTNAV_Q)
        )
        self.switch_to_navigation_mode()

    def get_base_pose(self) -> np.ndarray:
        """Get the robot's base pose as XYT."""
        return self.nav.get_base_pose()

    def get_observation(
        self, rotate_head_pts=False, start_pose: Optional[np.ndarray] = None
    ) -> Observations:
        """Get an observation from the current robot.

        Parameters:
            rotate_head_pts: this is true to put things into the same format as Habitat; generally we do not want to do this"""
        rgb, depth, xyz = self.head.get_images(
            compute_xyz=True,
        )
        current_pose = xyt2sophus(self.nav.get_base_pose())

        if start_pose is not None:
            # use sophus to get the relative translation
            relative_pose = start_pose.inverse() * current_pose
        else:
            relative_pose = current_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]

        # GPS in robot coordinates
        gps = relative_pose.translation()[:2]

        # Get joint state information
        joint_positions, _, _ = self.get_joint_state()

        # Create the observation
        obs = Observations(
            rgb=rgb.copy(),
            depth=depth.copy(),
            xyz=xyz.copy(),
            gps=gps,
            compass=np.array([theta]),
            camera_pose=self.head.get_pose(rotated=rotate_head_pts),
            joint=self.model.config_to_hab(joint_positions),
        )
        return obs
