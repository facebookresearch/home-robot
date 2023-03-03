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
from home_robot.motion.stretch import HelloStretchIdx
from home_robot.utils.geometry import posquat2sophus, xyt2sophus

from .abstract import AbstractControlModule, enforce_enabled


class StretchManipulationInterface(AbstractControlModule):
    def __init__(self, ros_client, robot_model):
        self._ros_client = ros_client
        self._robot_model = robot_model

    def _enable_hook(self) -> bool:
        """Called when interface is enabled."""
        # Switch interface mode & print messages
        result = self._ros_client.pos_mode_service(TriggerRequest())
        rospy.loginfo(result.message)

        # Set manipulator params
        self._manipulator_params = ManipulatorBaseParams(
            se3_base=self._ros_client.se3_base_odom,
        )

        return result.success

    def _disable_hook(self) -> bool:
        """Called when interface is disabled."""
        return True

    # Interface methods

    def get_joint_state(self):
        with self._js_lock:
            return self.ros_client.pos, self.ros_client.vel, self.ros_client.frc

    @enforce_enabled
    def wait(self):
        self._ros_client.wait_for_trajectory_action()

    @enforce_enabled
    def goto(self, q, move_base=False, wait=True, max_wait_t=10.0, verbose=False):
        """Directly command the robot using generalized coordinates
        some of these params are unsupported
        """
        goal = self._ros_client.config_to_ros_trajectory_goal(q)
        self._ros_client.trajectory_client.send_goal(goal)
        if wait:
            self.wait()
        return True

    @enforce_enabled
    def set_joint_positions(
        self,
        joint_positions: List[float],
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

        # Construct and send command
        joint_goals = {
            self._ros_client.BASE_TRANSLATION_JOINT: joint_positions[0],
            self._ros_client.LIFT_JOINT: joint_positions[1],
            self._ros_client.ARM_JOINT: joint_positions[2],
            self._ros_client.WRIST_YAW: joint_positions[3],
            self._ros_client.WRIST_PITCH: joint_positions[4],
            self._ros_client.WRIST_ROLL: joint_positions[5],
        }

        self._ros_client.send_trajectory_goals(joint_goals)

        if blocking:
            self.wait()

    @enforce_enabled
    def set_ee_pose(
        self,
        pos: List[float],
        quat: Optional[List[float]] = None,
        relative: bool = True,
        blocking: bool = True,
    ):
        """Command gripper to pose
        Does not rotate base.
        Cannot be used in navigation mode.

        Args:
            pos: Desired position
            quat: Desired orientation in quaternion (xyzw)
            relative: Whether specified pose is relative to base (relative to world if set to False)
        """
        # Compute pose relative to base
        if relative:
            pos_rel = np.array(pos)
            quat_rel = np.array(quat)
        else:
            pose_base_abs = xyt2sophus(self.get_base_state()["pose_se2"])
            pose_ee_abs = posquat2sophus(pos, quat)
            pose_ee_rel = pose_base_abs.inverse() * pose_ee_abs

            pos_rel = pose_ee_rel.translation()
            quat_rel = R.from_matrix(pose_ee_rel.so3().matrix()).as_quat()

        # Perform IK
        q = self._robot_model.manip_ik((pos_rel, quat_rel))
        joint_pos = self._extract_joint_pos(q)

        # Execute joint command
        self.set_joint_positions(joint_pos, blocking=blocking)

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
            q[HelloStretchIdx.WRIST_PITCH],
            q[HelloStretchIdx.WRIST_ROLL],
            q[HelloStretchIdx.WRIST_YAW],
        ]
