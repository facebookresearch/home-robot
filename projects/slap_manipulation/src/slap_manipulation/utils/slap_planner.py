# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

from home_robot.motion.stretch import (
    STRETCH_GRASP_OFFSET,
    HelloStretchIdx,
    HelloStretchKinematics,
)
from home_robot.utils.pose import to_matrix, to_pos_quat
from home_robot_hw.remote import StretchClient
from home_robot_hw.ros.utils import matrix_to_pose_msg, ros_pose_to_transform


class CombinedSLAPPlanner(object):
    """Simple skill motion planner to connect three-six waypoints into a continuous motion"""

    def __init__(self, robot: StretchClient):
        """
        Solve IK
        """
        if not isinstance(robot, StretchClient):
            raise RuntimeError(
                "The SimpleSkillMotionPlanner was designed only for Stretch."
            )
        self.robot = robot
        self.broadcaster = tf2_ros.TransformBroadcaster()
        # TODO: find an elegant way of including this through Client
        # or something rather than hardcoding here
        # Offset from STRETCH_GRASP_FRAME to predicted grasp point
        self._robot_ee_to_grasp_offset = STRETCH_GRASP_OFFSET.copy()
        self._robot_ee_to_grasp_offset[2, 3] += 0.11
        self._robot_max_grasp = 0  # 0.13, empirically found
        self.mode = "open"

    def linear_interpolation(self, waypoints, grippers, num_points_per_segment=3):
        num_waypoints = len(waypoints)
        num_segments = num_waypoints - 1

        interpolated_trajectory = []
        interpolated_gripper = []

        for i in range(num_segments):
            start_pose = waypoints[i]
            end_pose = waypoints[i + 1]

            spaced_idx = np.linspace(0, 1, num_points_per_segment)
            for t in spaced_idx:
                interpolated_pose = start_pose
                interpolated_pose = (1 - t) * start_pose + t * end_pose
                interpolated_trajectory.append(interpolated_pose)
                if t == 1:
                    interpolated_gripper.append(grippers[i + 1])
                else:
                    interpolated_gripper.append(grippers[i])
        interpolated_trajectory = np.array(interpolated_trajectory)
        interpolated_gripper = np.array(interpolated_gripper)

        return interpolated_trajectory, interpolated_gripper

    def plan_for_skill(
        self, actions_pose_mat: np.ndarray, action_gripper: np.ndarray, p2p_motion=False
    ) -> Optional[List[Tuple]]:
        """Simple trajectory generator which moves to an offset from 0th action,
        and then executes the given trajectory."""
        # grasp_pos, grasp_quat = to_pos_quat(grasp)
        self.robot.switch_to_manipulation_mode()
        trajectory = []
        num_pts_per_segment = 4

        joint_pos_pre = self.robot.manip.get_joint_positions()

        if p2p_motion:
            current_pose = self.robot.manip.get_ee_pose(matrix=True)
            current_gripper = np.array([-1])  # do not change state
            actions_pose_mat = np.concatenate(
                (np.expand_dims(current_pose, 0), actions_pose_mat), axis=0
            )
            actions_gripper = np.concatenate((current_gripper, action_gripper), axis=0)
            actions_pose_mat, action_gripper = self.linear_interpolation(
                actions_pose_mat, actions_gripper
            )
            self._send_action_to_tf(actions_pose_mat)
            for i in range(actions_pose_mat.shape[0]):
                desired_pos, desired_quat = to_pos_quat(actions_pose_mat[i])
                desired_cfg, success, _ = self.robot.model.manip_ik(
                    (desired_pos, desired_quat), q0=None
                )
                if success and desired_cfg is not None:
                    desired_pt = (
                        f"action_{i}",
                        self.robot.model.config_to_manip_command(desired_cfg),
                        int(action_gripper[i]),
                    )
                    trajectory.append(desired_pt)
                else:
                    print(f"-> could not solve for skill; action_{i} unreachable")
                    return None
        else:
            # smoothen trajectory via linear interpolation
            actions_pose_mat, action_gripper = self.linear_interpolation(
                actions_pose_mat,
                action_gripper,
                num_points_per_segment=num_pts_per_segment,
            )
            assert len(actions_pose_mat) == len(action_gripper)
            # 1st bring the ee up to the height of 1st action
            begin_pose = self.robot.manip.get_ee_pose()
            begin_pose = to_matrix(*begin_pose)
            begin_pose[2, 3] = actions_pose_mat[0, 2, 3]
            # TODO: also rotate the gripper as per 1st action
            begin_pose[:3, :3] = actions_pose_mat[0, :3, :3]
            # get current gripper reading, keep gripper as is during this motion
            # gripper = np.array([int(self.robot.manip.get_gripper_position() < -0.01)])
            gripper = np.array([-1])  # do not change state
            actions_pose_mat = np.concatenate(
                (
                    np.expand_dims(begin_pose, 0),
                    actions_pose_mat,
                    np.expand_dims(begin_pose, 0),
                ),
                axis=0,
            )
            self._send_action_to_tf(actions_pose_mat)
            action_gripper = np.concatenate(
                (
                    gripper,
                    action_gripper.reshape(-1),
                    np.expand_dims(action_gripper.reshape(-1)[-1], axis=0),
                ),
                axis=-1,
            )
            initial_pt = ("initial", joint_pos_pre, gripper)
            trajectory.append(initial_pt)

            num_actions = actions_pose_mat.shape[0]
            for i in range(num_actions):
                desired_pos, desired_quat = to_pos_quat(actions_pose_mat[i])
                desired_cfg, success, _ = self.robot.model.manip_ik(
                    (desired_pos, desired_quat), q0=None
                )
                if success and desired_cfg is not None:
                    desired_pt = (
                        f"action_{i}",
                        self.robot.model.config_to_manip_command(desired_cfg),
                        bool(action_gripper[i]),
                    )
                    trajectory.append(desired_pt)
                else:
                    print(f"-> could not solve for skill; action_{i} unreachable")
                    return None
            # go back to initial pt with the gripper state same as last predicted action
            end_pt = ("end", joint_pos_pre, bool(action_gripper[-1]))
            trajectory.append(end_pt)
        return trajectory

    def _send_action_to_tf(self, action):
        """Helper function for visualizing the predicted grasps."""
        num_actions = action.shape[0]
        for i in range(num_actions):
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.child_frame_id = f"predicted_action_{i}"
            t.header.frame_id = "base_link"
            act_matrix = action[i]
            t.transform = ros_pose_to_transform(matrix_to_pose_msg(act_matrix))
            self.broadcaster.sendTransform(t)

    def try_executing_skill(
        self,
        combined_action: np.ndarray,
        wait_for_input: bool = False,
        p2p_motion: bool = False,
        trimesh_format: bool = True,
    ) -> bool:
        """Execute a predefined end-effector trajectory. Expected input dimension is NUM_WAYPOINTSx8,
        where each waypoint is: pos(3-val), ori(4-val), gripper(1-val)"""
        action_as_mat = []
        for act in combined_action:
            action_as_mat.append(
                to_matrix(act[:3], act[3:7], trimesh_format=trimesh_format)
            )
        action_as_mat = np.array(action_as_mat)
        action_as_mat = np.matmul(action_as_mat, self._robot_ee_to_grasp_offset)
        self._send_action_to_tf(action_as_mat)

        # Generate a plan
        trajectory = self.plan_for_skill(
            action_as_mat, combined_action[:, -1], p2p_motion=p2p_motion
        )

        if trajectory is None:
            print("Planning failed")
            return False

        for i, (name, waypoint, grasp) in enumerate(trajectory):
            self.robot.manip.goto_joint_positions(waypoint)
            if grasp == 1:
                self.robot.manip.close_gripper()
                if self.mode == "open":
                    rospy.sleep(0.5)
                self.mode = "close"
            elif grasp == 0:
                self.robot.manip.open_gripper()
                if self.mode == "close":
                    rospy.sleep(0.5)
                self.mode = "open"
            if wait_for_input:
                input(f"{i+1}) went to {name}")
            else:
                print(f"{i+1}) went to {name}")
        print(">>>--->> SKILL ATTEMPT COMPLETE <<---<<<")
        return True
