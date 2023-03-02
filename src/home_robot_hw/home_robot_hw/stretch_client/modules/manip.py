# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import rospy

from .abstract import AbstractControlModule, enforce_enabled

class StretchManipulationInterface(AbstractControlModule):
    def __init__(self, ros_client):
        self.ros_client = ros_client

    def _enable_hook(self) -> bool:
        """Called when interface is enabled."""
        # Switch interface mode & print messages
        result = self._pos_mode_service(TriggerRequest())
        rospy.loginfo(result.message)

        # Set manipulator params
        self._manipulator_params = ManipulatorBaseParams(
            se3_base=self._t_base_odom,
        )

        return result.success

    def _disable_hook(self) -> bool:
        """Called when interface is disabled."""
        return True

    # Interface methods

    def get_joint_state(self):
        with self._js_lock:
            return self.pos, self.vel, self.frc

    @enforce_enabled
    def goto(self, q, move_base=False, wait=True, max_wait_t=10.0, verbose=False):
        """some of these params are unsupported"""
        goal = self.config_to_ros_trajectory_goal(q)
        self.trajectory_client.send_goal(goal)
        if wait:
            #  Waiting for result seems to hang
            # self.trajectory_client.wait_for_result()
            print("waiting for result...")
            self.wait(q, max_wait_t, not move_base, verbose)
        return True

    @enforce_enabled
    def set_arm_joint_positions(self, joint_positions: List[float]):
        """
        list of robot arm joint positions:
            BASE_TRANSLATION = 0
            LIFT = 1
            ARM = 2
            WRIST_YAW = 3
            WRIST_PITCH = 4
            WRIST_ROLL = 5
        """
        assert len(joint_positions) == 6, "Joint position vector must be of length 6."

        # Preprocess base translation joint position (command is actually delta position)
        base_joint_pos_curr = self._compute_base_translation_pos()
        base_joint_pos_cmd = joint_positions[0] - base_joint_pos_curr

        # Construct and send command
        joint_goals = {
            ROS_BASE_TRANSLATION_JOINT: base_joint_pos_cmd,
            ROS_LIFT_JOINT: joint_positions[1],
            ROS_ARM_JOINT: joint_positions[2],
            ROS_WRIST_YAW: joint_positions[3],
            ROS_WRIST_PITCH: joint_positions[4],
            ROS_WRIST_ROLL: joint_positions[5],
        }

        self._send_ros_trajectory_goals(joint_goals)

        return True

    @enforce_enabled
    def set_ee_pose(
        self,
        pos: List[float],
        quat: Optional[List[float]] = None,
        relative: bool = False,
    ):
        """
        Does not rotate base.
        Cannot be used in navigation mode.
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
        q_raw = self.ik_solver.compute_ik(pos_rel, quat_rel, None)

        # Combine arm telescoping joints
        q_manip = np.zeros(6)
        q_manip[0] = q_raw[0]  # base X translation
        q_manip[1] = q_raw[1]  # lift
        q_manip[2] = np.sum(q_raw[2:6])  # squeeze arm telescoping joints into 1
        q_manip[3:6] = q_raw[6:9]  # yaw pitch roll

        # Execute joint command
        self.set_arm_joint_positions(q_manip)
