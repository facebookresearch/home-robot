from enum import Enum

from home_robot.agent.motion.stretch import HelloStretch, HelloStretchIdx


ROS_ARM_JOINTS = ["joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3"]
ROS_LIFT_JOINT = "joint_lift"
ROS_GRIPPER_FINGER = "joint_gripper_finger_left"
# ROS_GRIPPER_FINGER2 = "joint_gripper_finger_right"
ROS_HEAD_PAN = "joint_head_pan"
ROS_HEAD_TILT = "joint_head_tilt"
ROS_WRIST_YAW = "joint_wrist_yaw"
ROS_WRIST_PITCH = "joint_wrist_pitch"
ROS_WRIST_ROLL = "joint_wrist_roll"


ROS_TO_CONFIG = {
    ROS_LIFT_JOINT: HelloStretchIdx.LIFT,
    ROS_GRIPPER_FINGER: HelloStretchIdx.GRIPPER,
    # ROS_GRIPPER_FINGER2: HelloStretchIdx.GRIPPER,
    ROS_WRIST_YAW: HelloStretchIdx.WRIST_YAW,
    ROS_WRIST_PITCH: HelloStretchIdx.WRIST_PITCH,
    ROS_WRIST_ROLL: HelloStretchIdx.WRIST_ROLL,
    ROS_HEAD_PAN: HelloStretchIdx.HEAD_PAN,
    ROS_HEAD_TILT: HelloStretchIdx.HEAD_TILT,
}

CONFIG_TO_ROS = {}
for k, v in ROS_TO_CONFIG.items():
    if v not in CONFIG_TO_ROS:
        CONFIG_TO_ROS[v] = []
    CONFIG_TO_ROS[v].append(k)
CONFIG_TO_ROS[HelloStretchIdx.ARM] = ROS_ARM_JOINTS
# ROS_JOINT_NAMES += ROS_ARM_JOINTS

T_LOC_STABILIZE = 1.0


class ControlMode(Enum):
    IDLE = 0
    VELOCITY = 1
    NAVIGATION = 2
    MANIPULATION = 3
