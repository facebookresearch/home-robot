# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import TriggerRequest

from home_robot.core.state import ManipulatorBaseParams
from home_robot.motion.robot import Robot
from home_robot.motion.stretch import STRETCH_HOME_Q, HelloStretchIdx
from home_robot.utils.geometry import posquat2sophus, sophus2posquat, xyt2sophus

from .abstract import AbstractControlModule, enforce_enabled


class StretchManipulationClient(AbstractControlModule):
    def __init__(self, ros_client, robot_model: Robot):
        super().__init__()

        self._ros_client = ros_client
        self._robot_model = robot_model

        # Tmp: keep track of base_x movement
        self.base_x = 0.0

    # Enable / disable

    def _enable_hook(self) -> bool:
        """Called when interface is enabled."""
        # Switch interface mode & print messages
        result = self._ros_client.pos_mode_service(TriggerRequest())
        rospy.loginfo(result.message)
        self.base_x = 0.0

        return result.success

    def _disable_hook(self) -> bool:
        """Called when interface is disabled."""
        return True

    # Interface methods

    def get_ee_pose(self, world_frame=False):
        q, _, _ = self._ros_client.get_joint_state()
        pos_base, quat_base = self._robot_model.manip_fk(q)
        pos_base[0] = self.base_x

        if world_frame:
            pose_base2ee = posquat2sophus(pos_base, quat_base)
            pose_world2base = self._ros_client.se3_base_filtered
            pose_world2ee = pose_world2base * pose_base2ee

            pos, quat = sophus2posquat(pose_world2ee)

        else:
            pos, quat = pos_base, quat_base

        return pos, quat

    def get_joint_positions(self):
        q, _, _ = self._ros_client.get_joint_state()
        return [
            self.base_x,
            q[HelloStretchIdx.LIFT],
            q[HelloStretchIdx.ARM],
            q[HelloStretchIdx.WRIST_YAW],
            q[HelloStretchIdx.WRIST_PITCH],
            q[HelloStretchIdx.WRIST_ROLL],
        ]

    @enforce_enabled
    def goto(self, q, move_base=False, wait=True, max_wait_t=10.0, verbose=False):
        """Directly command the robot using generalized coordinates
        some of these params are unsupported
        """
        goal = self._ros_client.config_to_ros_trajectory_goal(q)
        self._ros_client.trajectory_client.send_goal(goal)

        self._register_wait(self._ros_client.wait_for_trajectory_action)
        if wait:
            self.wait()

        return True

    @enforce_enabled
    def home(self):
        self.goto(STRETCH_HOME_Q, wait=True)

    @enforce_enabled
    def goto_joint_positions(
        self,
        joint_positions: List[float],
        relative: bool = False,
        blocking: bool = True,
    ):
        """
        list of robot arm joint positions:
            BASE_TRANSLATION = 0
            LIFT = 1
            ARM = 2
            WRIST_YAW = 3
            WRIST_PITCH = 4
            WRIST_ROLL = 5

        Args:
            joint_positions: List of length 6 containing desired joint positions
            relative_base: Whether the base joint moves relative to current base position
            blocking: Whether command blocks until completetion
        """
        assert len(joint_positions) == 6, "Joint position vector must be of length 6."

        # Compute joint states
        joint_pos_goal = np.array(joint_positions)
        if relative:
            joint_pos_goal += self.get_joint_positions()

        # Base x translation command is relative
        base_x_new = joint_pos_goal[0]
        joint_pos_goal[0] = base_x_new - self.base_x
        self.base_x = base_x_new

        # Construct and send command
        joint_goals = {
            self._ros_client.BASE_TRANSLATION_JOINT: joint_pos_goal[0],
            self._ros_client.LIFT_JOINT: joint_pos_goal[1],
            self._ros_client.ARM_JOINT: joint_pos_goal[2],
            self._ros_client.WRIST_YAW: joint_pos_goal[3],
            self._ros_client.WRIST_PITCH: joint_pos_goal[4],
            self._ros_client.WRIST_ROLL: joint_pos_goal[5],
        }

        self._ros_client.send_trajectory_goals(joint_goals)

        self._register_wait(self._ros_client.wait_for_trajectory_action)
        if blocking:
            self.wait()

    @enforce_enabled
    def goto_ee_pose(
        self,
        pos: List[float],
        quat: Optional[List[float]] = None,
        relative: bool = False,
        world_frame: bool = False,
        blocking: bool = True,
    ):
        """Command gripper to pose
        Does not rotate base.
        Cannot be used in navigation mode.

        Args:
            pos: Desired position
            quat: Desired orientation in quaternion (xyzw)
            relative: Whether specified pose is relative to current pose
            world_frame: Infer poses in world frame instead of base frame
            blocking: Whether command blocks until completetion
        """
        pos_ee_curr, quat_ee_curr = self.get_ee_pose(world_frame=world_frame)
        if quat is None:
            quat = [0, 0, 0, 1] if relative else quat_ee_curr

        # Compute IK goal: pose relative to base
        pose_input = posquat2sophus(np.array(pos), np.array(quat))

        if world_frame:
            pose_world2ee = pose_input
            pose_world2base = self._ros_client.se3_base_filtered
            pose_desired = pose_world2base.inverse() * pose_world2ee
        else:
            pose_desired = pose_input

        if relative:
            pose_base2ee_curr = posquat2sophus(pos_ee_curr, quat_ee_curr)
            pose_base2ee_desired = pose_desired * pose_base2ee_curr
        else:
            pose_base2ee_desired = pose_desired

        pos_ik_goal, quat_ik_goal = sophus2posquat(pose_base2ee_desired)

        # Perform IK
        q = self._robot_model.manip_ik((pos_ik_goal, quat_ik_goal))
        if q is None:
            return False
        joint_pos = self._extract_joint_pos(q)

        # Execute joint command
        breakpoint()
        self.goto_joint_positions(joint_pos, blocking=blocking)

        return True

    @enforce_enabled
    def open_gripper(self, blocking: bool = True):
        gripper_target = self._robot_model.range[HelloStretchIdx.GRIPPER][1]
        self.move_gripper(gripper_target, blocking=blocking)

    @enforce_enabled
    def close_gripper(self, blocking: bool = True):
        gripper_target = self._robot_model.range[HelloStretchIdx.GRIPPER][0]
        self.move_gripper(gripper_target, blocking=blocking)

    @enforce_enabled
    def move_gripper(self, target, blocking: bool = True):
        joint_goals = {
            self._ros_client.GRIPPER_FINGER: target,
        }
        self._ros_client.send_trajectory_goals(joint_goals)

        self._register_wait(self._ros_client.wait_for_trajectory_action)
        if blocking:
            self.wait()

    # Helper methods

    def _compute_base_translation_pos(self):
        l0_pose = self._manipulator_params.se3_base
        l1_pose = self._ros_client.se3_base_odom
        return (l0_pose.inverse() * l1_pose).translation()[0]

    def _extract_joint_pos(self, q):
        return [
            q[HelloStretchIdx.BASE_X],
            q[HelloStretchIdx.LIFT],
            q[HelloStretchIdx.ARM],
            q[HelloStretchIdx.WRIST_YAW],
            q[HelloStretchIdx.WRIST_PITCH],
            q[HelloStretchIdx.WRIST_ROLL],
        ]
