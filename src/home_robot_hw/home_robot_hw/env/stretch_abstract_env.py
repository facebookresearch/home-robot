# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
import threading
import time
import timeit
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import actionlib
import numpy as np
import ros_numpy
import rospy
import sophus as sp
import tf2_ros
import trimesh.transformations as tra

# Import ROS messages and tools
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest
from trajectory_msgs.msg import JointTrajectoryPoint

import home_robot
import home_robot.core.abstract_env
from home_robot.core.interfaces import Action, Observations
from home_robot.core.robot import ControlMode
from home_robot.core.state import ManipulatorBaseParams
from home_robot.motion.stretch import HelloStretchIdx
from home_robot.utils.geometry import (
    posquat2sophus,
    sophus2xyt,
    xyt2sophus,
    xyt_base_to_global,
)
from home_robot_hw.constants import (
    CONFIG_TO_ROS,
    ROS_ARM_JOINTS,
    ROS_GRIPPER_FINGER,
    ROS_HEAD_PAN,
    ROS_HEAD_TILT,
    ROS_LIFT_JOINT,
    ROS_TO_CONFIG,
    ROS_WRIST_PITCH,
    ROS_WRIST_ROLL,
    ROS_WRIST_YAW,
    T_LOC_STABILIZE,
)
from home_robot_hw.remote import StretchClient
from home_robot_hw.ros.camera import RosCamera
from home_robot_hw.ros.utils import matrix_from_pose_msg, matrix_to_pose_msg
from home_robot_hw.ros.visualizer import Visualizer

MIN_DEPTH_REPLACEMENT_VALUE = 10000
MAX_DEPTH_REPLACEMENT_VALUE = 10001

BASE_X_IDX = HelloStretchIdx.BASE_X
BASE_Y_IDX = HelloStretchIdx.BASE_Y
BASE_THETA_IDX = HelloStretchIdx.BASE_THETA
LIFT_IDX = HelloStretchIdx.LIFT
ARM_IDX = HelloStretchIdx.ARM
GRIPPER_IDX = HelloStretchIdx.GRIPPER
WRIST_ROLL_IDX = HelloStretchIdx.WRIST_ROLL
WRIST_PITCH_IDX = HelloStretchIdx.WRIST_PITCH
WRIST_YAW_IDX = HelloStretchIdx.WRIST_YAW
HEAD_PAN_IDX = HelloStretchIdx.HEAD_PAN
HEAD_TILT_IDX = HelloStretchIdx.HEAD_TILT


class StretchEnv(home_robot.core.abstract_env.Env):
    """Defines a ROS-based interface to the real Stretch robot. Collect observations and command the robot."""

    # 3 for base position + rotation, 2 for lift + extension, 3 for rpy, 1 for gripper, 2 for head
    dof = 3 + 2 + 3 + 1 + 2
    min_depth_val = 0.1
    max_depth_val = 4.0
    goal_time_tolerance = 1.0

    exec_tol = np.array(
        [
            1e-3,
            1e-3,
            0.01,  # x y theta
            0.005,  # lift
            0.01,  # arm
            1.0,  # gripper - this never works
            # 0.015, 0.015, 0.015,  #wrist variables
            0.05,
            0.05,
            0.05,  # wrist variables
            0.1,
            0.1,  # head  and tilt
        ]
    )

    dist_tol = 1e-4
    theta_tol = 1e-3
    wait_time_step = 1e-3
    # This is an important value - used for determining if the robot has reached a goal position.
    # TODO: drop this or reduce it to something very small, use velocity + timesteps instead.
    msg_delay_t = 0.25
    block_spin_rate = 10

    base_link = "base_link"
    odom_link = "map"

    def __init__(
        self,
        init_cameras=True,
        depth_buffer_size=None,
        color_topic=None,
        depth_topic=None,
        dry_run=False,
    ):
        self.dry_run = dry_run

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def apply_action(
        self,
        action: Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ):
        pass

    @abstractmethod
    def get_observation(self) -> Observations:
        pass

    @property
    @abstractmethod
    def episode_over(self) -> bool:
        pass

    @abstractmethod
    def get_episode_metrics(self) -> Dict:
        pass

    @abstractmethod
    def get_robot(self) -> StretchClient:
        """Return a reference to the robot client"""
        pass


if __name__ == "__main__":
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_ros_test")
    print("Create ROS interface")
    rob = StretchEnv(init_cameras=True)
