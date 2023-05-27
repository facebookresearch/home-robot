from enum import Enum
from typing import Dict, List, Union

from home_robot.motion.stretch import HelloStretchIdx

ROS_ARM_JOINTS = ["joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3"]
ROS_LIFT_JOINT = "joint_lift"
ROS_GRIPPER_FINGER = "joint_gripper_finger_left"
# ROS_GRIPPER_FINGER2 = "joint_gripper_finger_right"
ROS_HEAD_PAN = "joint_head_pan"
ROS_HEAD_TILT = "joint_head_tilt"
ROS_WRIST_YAW = "joint_wrist_yaw"
ROS_WRIST_PITCH = "joint_wrist_pitch"
ROS_WRIST_ROLL = "joint_wrist_roll"


ROS_TO_CONFIG: Dict[str, HelloStretchIdx] = {
    ROS_LIFT_JOINT: HelloStretchIdx.LIFT,
    ROS_GRIPPER_FINGER: HelloStretchIdx.GRIPPER,
    # ROS_GRIPPER_FINGER2: HelloStretchIdx.GRIPPER,
    ROS_WRIST_YAW: HelloStretchIdx.WRIST_YAW,
    ROS_WRIST_PITCH: HelloStretchIdx.WRIST_PITCH,
    ROS_WRIST_ROLL: HelloStretchIdx.WRIST_ROLL,
    ROS_HEAD_PAN: HelloStretchIdx.HEAD_PAN,
    ROS_HEAD_TILT: HelloStretchIdx.HEAD_TILT,
}

CONFIG_TO_ROS: Dict[HelloStretchIdx, List[str]] = {}
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


REAL_WORLD_CATEGORIES = [
    "other",
    # objects
    "chair",
    "cup",
    "table",
    # receptacles
    "coffee_table",
    "sofa",
    "dining_table",
    "swivel_chair",
    "table",
    "tv_stand",
    "toilet",
    "balcony",
    "bookcase",
    "armchair",
    "swing_chair",
    "armoire",
    "kitchen_cabinet",
    "ottoman",
    "desk",
    "end_table",
    "nightstand",
    "chest_of_drawers",
    "storage_bench",
    "stool",
    "shower_stall",
    "chair",
    "console_table",
    "dining_area",
    "beanbag_chair",
    "easy_chair",
    "buffet",
    "l-shaped_couch",
    "sink_cabinet",
    "wall_shelf",
    "footstool",
    "washer",
    "cabinet",
    "bathtub",
    "rocking_chair",
    "hanging_cabinet",
    "flat_bench",
    "bar_stool",
    "shelving",
    "china_cabinet",
    "dressing_table",
    "hot_tub",
    "kitchen_island",
    "bar",
    "straight_chair",
    "bench",
    "air_hockey_table",
    "chaise_longue",
    "ladder_bookcase",
    "highchair",
    "wardrobe",
    "credenza",
    "swing_bench",
    "car",
    "gazebo",
    "serving_cart",
    "trunk",
    "shoe_rack,cabinet",
    "file",
    "medicine_chest",
    "washbasin",
    "daybed",
    "table-tennis_table",
    "sink_stand",
    "base_cabinet",
    "magazine_rack",
    "lectern",
    "shoe_rack",
    "foosball_table",
    "handcart",
    "conference_table",
    "step_stool",
    "mantel",
    "pool_table",
    "workbench",
    "plant_stand",
    "picnic_table",
    "dryer",
    "bathtub,shower_stall",
    "other",
]


# Subset for use with other code - just a minimal group of objects
SMALL_REAL_WORLD_CATEGORIES = [
    "other",
    "chair",
    "cup",
    "table",
    "other",
]
