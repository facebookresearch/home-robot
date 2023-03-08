import numpy as np

from home_robot_hw.env.stretch_abstract_env import HelloStretchIdx

"""
Model as dumped by fixed-base URDF that following use:
Nb joints = 13 (nq=12,nv=12)
  Joint 0 universe: parent=0
  Joint 1 joint_head_pan: parent=0
  Joint 2 joint_head_tilt: parent=1
  Joint 3 joint_lift: parent=0
  Joint 4 joint_arm_l3: parent=3
  Joint 5 joint_arm_l2: parent=4
  Joint 6 joint_arm_l1: parent=5
  Joint 7 joint_arm_l0: parent=6
  Joint 8 joint_wrist_yaw: parent=7
  Joint 9 joint_wrist_pitch: parent=8
  Joint 10 joint_wrist_roll: parent=9
  Joint 11 joint_gripper_finger_left: parent=10
  Joint 12 joint_gripper_finger_right: parent=10
"""


def convert_pinocchio_pose_to_ros(pin):
    # TODO: this needs to be fixed
    joint_angles = np.zeros(11)
    joint_angles[HelloStretchIdx.LIFT] = pin[0]
    joint_angles[HelloStretchIdx.ARM] = pin[1] + pin[2] + pin[3] + pin[4]
    joint_angles[HelloStretchIdx.WRIST_YAW] = pin[5]
    joint_angles[HelloStretchIdx.WRIST_PITCH] = pin[6]
    joint_angles[HelloStretchIdx.WRIST_ROLL] = pin[7]
    return joint_angles


def ros_pose_to_pinocchio(joint_angles):
    pin_compatible_joints = np.zeros(9)
    pin_compatible_joints[0] = joint_angles[HelloStretchIdx.BASE_X]
    pin_compatible_joints[1] = joint_angles[HelloStretchIdx.LIFT]
    pin_compatible_joints[2] = pin_compatible_joints[3] = pin_compatible_joints[
        4
    ] = pin_compatible_joints[5] = (joint_angles[HelloStretchIdx.ARM] / 4)
    pin_compatible_joints[6] = joint_angles[HelloStretchIdx.WRIST_YAW]
    pin_compatible_joints[7] = joint_angles[HelloStretchIdx.WRIST_PITCH]
    pin_compatible_joints[8] = joint_angles[HelloStretchIdx.WRIST_ROLL]
    return pin_compatible_joints
