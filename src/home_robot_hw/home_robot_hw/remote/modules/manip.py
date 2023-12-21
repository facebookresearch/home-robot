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
from home_robot.motion.robot import RobotModel
from home_robot.motion.stretch import STRETCH_HOME_Q, HelloStretchIdx
from home_robot.utils.geometry import posquat2sophus, sophus2posquat, xyt2sophus

from .abstract import AbstractControlModule, enforce_enabled

GRIPPER_MOTION_SECS = 2.2
JOINT_POS_TOL = 0.015
JOINT_ANG_TOL = 0.05


class StretchManipulationClient(AbstractControlModule):
    def __init__(self, ros_client, robot_model: RobotModel):
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

    def get_ee_pose(self, world_frame=False, matrix=False):
        q, _, _ = self._ros_client.get_joint_state()
        pos_base, quat_base = self._robot_model.manip_fk(q)
        pos_base[0] += self.base_x

        if world_frame:
            pose_base2ee = posquat2sophus(pos_base, quat_base)
            pose_world2base = self._ros_client.se3_base_filtered
            pose_world2ee = pose_world2base * pose_base2ee

            pos, quat = sophus2posquat(pose_world2ee)

        else:
            pos, quat = pos_base, quat_base
        if matrix:
            pose = posquat2sophus(pos, quat)
            return pose.matrix()
        else:
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

    def get_gripper_position(self) -> float:
        """get current gripper position as a float"""
        q, _, _ = self._ros_client.get_joint_state()
        return q[HelloStretchIdx.GRIPPER]

    @enforce_enabled
    def goto(
        self,
        q,
        dq: List = None,
        ddq: List = None,
        move_base=False,
        wait=True,
        max_wait_t=10.0,
        verbose=False,
    ):
        """Directly command the robot using generalized coordinates
        some of these params are unsupported
        """
        goal = self._ros_client.config_to_ros_trajectory_goal(q, dq, ddq)
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
        debug: bool = False,
        move_base: bool = True,
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
            blocking: Whether command blocks until completion
        """
        assert len(joint_positions) == 6, "Joint position vector must be of length 6."

        # Compute joint states
        joint_pos_init = self.get_joint_positions()
        joint_pos_goal = np.array(joint_positions)
        if relative:
            joint_pos_goal += np.array(joint_pos_init)

        # Construct command
        #   (note: base translation joint command is relative)
        joint_goals = {
            self._ros_client.LIFT_JOINT: joint_pos_goal[1],
            self._ros_client.ARM_JOINT: joint_pos_goal[2],
            self._ros_client.WRIST_YAW: joint_pos_goal[3],
            self._ros_client.WRIST_PITCH: joint_pos_goal[4],
            self._ros_client.WRIST_ROLL: joint_pos_goal[5],
        }
        if move_base:
            joint_goals[self._ros_client.BASE_TRANSLATION_JOINT] = (
                joint_pos_goal[0] - self.base_x
            )
        self.base_x = joint_pos_goal[0]

        # Send command to trajectory server
        self._ros_client.send_trajectory_goals(joint_goals)

        # Wait logic
        def joint_move_wait():
            # Wait for action to complete
            self._ros_client.wait_for_trajectory_action()

            # Check final joint states
            joint_pos_final = self.get_joint_positions()
            joint_err = np.array(joint_pos_final) - np.array(joint_pos_goal)
            arm_success = np.allclose(joint_err[:3], 0.0, atol=JOINT_POS_TOL)
            wrist_success = np.allclose(joint_err[3:], 0.0, atol=JOINT_ANG_TOL)
            if not (arm_success and wrist_success):
                print("Warning: Joint goal not achieved.")

            # Debug print
            if debug:
                print("-- joint goto cmd --")
                print(
                    "Initial joint pos: [", *(f"{x:.3f}" for x in joint_pos_init), "]"
                )
                print(
                    "Desired joint pos: [", *(f"{x:.3f}" for x in joint_pos_goal), "]"
                )
                print(
                    "Achieved joint pos: [",
                    *(f"{x:.3f}" for x in joint_pos_final),
                    "]",
                )
                print("--------------------")

        self._register_wait(joint_move_wait)

        if blocking:
            self.wait()

    def solve_fk(self, full_body_cfg):
        pos, quat = self._robot_model.manip_fk(full_body_cfg)
        return pos, quat

    def solve_ik(
        self,
        pos: List[float],
        quat: Optional[List[float]] = None,
        relative: bool = False,
        world_frame: bool = False,
        initial_cfg: np.ndarray = None,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Solve inverse kinematics appropriately (or at least try to) and get the joint position
        that we will be moving to.

        Note: When relative==True, the delta orientation is still defined in the world frame

        Returns None if no solution is found, else returns an executable solution
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

            pos_desired = pos_ee_curr + pose_input.translation()
            so3_desired = pose_input.so3() * pose_base2ee_curr.so3()
            quat_desired = R.from_matrix(so3_desired.matrix()).as_quat()

            pose_base2ee_desired = posquat2sophus(pos_desired, quat_desired)

        else:
            pose_base2ee_desired = pose_desired

        pos_ik_goal, quat_ik_goal = sophus2posquat(pose_base2ee_desired)

        # Execute joint command
        if debug:
            print("=== EE goto command ===")
            print(f"Initial EE pose: pos={pos_ee_curr}; quat={quat_ee_curr}")
            print(f"Input EE pose: pos={np.array(pos)}; quat={np.array(quat)}")
            print(f"Desired EE pose: pos={pos_ik_goal}; quat={quat_ik_goal}")

        # Perform IK
        full_body_cfg, ik_success, ik_debug_info = self._robot_model.manip_ik(
            (pos_ik_goal, quat_ik_goal), q0=initial_cfg
        )

        # Expected to return None if we did not get a solution
        if not ik_success or full_body_cfg is None:
            return None
        # Return a valid solution to the IK problem here
        return full_body_cfg

    @enforce_enabled
    def goto_ee_pose(
        self,
        pos: List[float],
        quat: Optional[List[float]] = None,
        relative: bool = False,
        world_frame: bool = False,
        blocking: bool = True,
        debug: bool = False,
        initial_cfg: np.ndarray = None,
    ) -> bool:
        """Command gripper to pose
        Does not rotate base.
        Cannot be used in navigation mode.

        Args:
            pos: Desired position
            quat: Desired orientation in quaternion (xyzw)
            relative: Whether specified pose is relative to current pose
            world_frame: Infer poses in world frame instead of base frame
            blocking: Whether command blocks until completetion
            initial_cfg: Preferred (initial) joint state configuration
        """
        full_body_cfg = self.solve_ik(
            pos, quat, relative, world_frame, initial_cfg, debug
        )
        if full_body_cfg is None:
            print("Warning: Cannot find an IK solution for desired EE pose!")
            return False

        joint_pos = self._extract_joint_pos(full_body_cfg)
        self.goto_joint_positions(joint_pos, blocking=blocking, debug=debug)

        # Debug print
        if debug and blocking:
            pos, quat = self.get_ee_pose()
            print(f"Achieved EE pose: pos={pos}; quat={quat}")
            print("=======================")

        return True

    @enforce_enabled
    def rotate_ee(self, axis: int, angle: float, **kwargs) -> bool:
        """Rotates the gripper by one of 3 principal axes (X, Y, Z)"""
        assert axis in [0, 1, 2], "'axis' must be 0, 1, or 2! (x, y, z)"

        r = np.zeros(3)
        r[axis] = angle
        quat_desired = R.from_rotvec(r).as_quat().tolist()

        return self.goto_ee_pose([0, 0, 0], quat_desired, relative=True, **kwargs)

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

        def wait_for_gripper():
            rospy.sleep(GRIPPER_MOTION_SECS)

        self._register_wait(wait_for_gripper)
        if blocking:
            self.wait()

    # Helper methods

    def _compute_base_translation_pos(self):
        l0_pose = self._manipulator_params.se3_base
        l1_pose = self._ros_client.se3_base_odom
        return (l0_pose.inverse() * l1_pose).translation()[0]

    def _extract_joint_pos(self, q):
        """Helper to convert from the general-purpose config including full robot state, into the command space used in just the manip controller. Extracts just lift/arm/wrist information."""
        return self._robot_model.config_to_manip_command(q)
