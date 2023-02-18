# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import copy
import os
import pdb
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import actionlib
import numpy as np
import rospy
import sophus as sp
import trimesh.transformations as tra
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest
from trajectory_msgs.msg import JointTrajectoryPoint

import home_robot
from home_robot.agent.motion.ik import PybulletIKSolver
from home_robot.utils.geometry import (
    posquat2sophus,
    sophus2xyt,
    xyt2sophus,
    xyt_base_to_global,
)
from home_robot_hw.ros.path import get_package_path
from home_robot_hw.ros.utils import matrix_from_pose_msg, matrix_to_pose_msg

# IK solver configuration
PKG_PATH = os.path.dirname(home_robot.__file__)
URDF_PATH = "../../../assets/hab_stretch/urdf/planner_calibrated_manipulation_mode.urdf"
URDF_ABS_PATH = os.path.join(PKG_PATH, URDF_PATH)

EE_LINK_NAME = "link_straight_gripper"
CONTROLLED_JOINTS = [0, 3, 4, 5, 6, 7, 8, 9, 10]

# Joint names in the ROS joint trajectory server
ROS_BASE_TRANSLATION_JOINT = "translate_mobile_base"
ROS_ARM_JOINT = "joint_arm"
ROS_LIFT_JOINT = "joint_lift"
ROS_WRIST_YAW = "joint_wrist_yaw"
ROS_WRIST_PITCH = "joint_wrist_pitch"
ROS_WRIST_ROLL = "joint_wrist_roll"
ROS_GRIPPER_FINGER = "joint_gripper_finger_left"  # used to control entire gripper
ROS_HEAD_PAN = "joint_head_pan"
ROS_HEAD_TILT = "joint_head_tilt"

ROS_ARM_JOINTS_ACTUAL = ["joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3"]

# Parameters
T_LOC_STABILIZE = 1.0
T_GOAL_TIME_TOL = 1.0

STRETCH_GRIPPER_OPEN = 0.22
STRETCH_GRIPPER_CLOSE = -0.2


class ControlMode(Enum):
    IDLE = 0
    VELOCITY = 1
    NAVIGATION = 2
    MANIPULATION = 3


def limit_control_mode(valid_modes: List[ControlMode]):
    """Decorator for checking if a robot method is executed while the correct mode is present."""

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            curr_mode = self._robot_state.base_control_mode

            if curr_mode in valid_modes:
                return func(self, *args, **kwargs)
            else:
                rospy.logerr(
                    f"'{func.__name__}' is only available in the following modes: {valid_modes}"
                )
                rospy.logerr(f"Current mode is: {curr_mode}")
                return None

        return wrapper

    return decorator


@dataclass
class StretchRobotState:
    """
    Minimum representation of the state of the robot
    """

    base_control_mode: ControlMode

    last_base_update_timestamp: rospy.Time
    t_base_filtered: sp.SE3
    t_base_odom: sp.SE3

    last_joint_update_timestamp: rospy.Time
    q_lift: float
    q_arm: float
    q_wrist_yaw: float
    q_wrist_pitch: float
    q_wrist_roll: float
    q_gripper_finger: float
    q_head_pan: float
    q_head_tilt: float


@dataclass
class ManipulatorBaseParams:
    se3_base: sp.SE3


class LocalHelloRobot:
    """
    ROS interface for robot base control
    Currently only works with a local rosmaster
    """

    def __init__(self, init_node: bool = True, init_cameras: bool = False):
        self._robot_state = StretchRobotState(
            base_control_mode=ControlMode.IDLE,
            last_base_update_timestamp=rospy.Time(),
            t_base_filtered=sp.SE3(),
            t_base_odom=sp.SE3(),
            last_joint_update_timestamp=rospy.Time(),
            q_lift=0.0,
            q_arm=0.0,
            q_wrist_yaw=0.0,
            q_wrist_pitch=0.0,
            q_wrist_roll=0.0,
            q_gripper_finger=0.0,
            q_head_pan=0.0,
            q_head_tilt=0.0,
        )

        # Cameras
        if init_cameras:
            raise NotImplementedError
        else:
            self.rgb_cam, self.dpt_cam = None, None

        # Ros pubsub
        if init_node:
            rospy.init_node("user")

        self._goal_pub = rospy.Publisher("goto_controller/goal", Pose, queue_size=1)
        self._velocity_pub = rospy.Publisher("stretch/cmd_vel", Twist, queue_size=1)

        self._odom_sub = rospy.Subscriber(
            "odom",
            Odometry,
            self._odom_callback,
            queue_size=1,
        )
        self._base_state_sub = rospy.Subscriber(
            "state_estimator/pose_filtered",
            PoseStamped,
            self._base_state_callback,
            queue_size=1,
        )
        self._joint_state_sub = rospy.Subscriber(
            "stretch/joint_states",
            JointState,
            self._joint_state_callback,
            queue_size=1,
        )

        self._nav_mode_service = rospy.ServiceProxy(
            "switch_to_navigation_mode", Trigger
        )
        self._pos_mode_service = rospy.ServiceProxy("switch_to_position_mode", Trigger)

        self._goto_on_service = rospy.ServiceProxy("goto_controller/enable", Trigger)
        self._goto_off_service = rospy.ServiceProxy("goto_controller/disable", Trigger)
        self._set_yaw_service = rospy.ServiceProxy(
            "goto_controller/set_yaw_tracking", SetBool
        )

        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )

        # IK
        self.ik_solver = PybulletIKSolver(
            URDF_ABS_PATH, EE_LINK_NAME, CONTROLLED_JOINTS
        )

        # Initialize control mode & home robot
        self.switch_to_manipulation_mode()
        self.set_arm_joint_positions([0, 0.3, 0, 0, 0, 0])
        self.set_camera_pan_tilt(0, 0)
        self.close_gripper()
        self._robot_state.base_control_mode = ControlMode.IDLE

    # Getter interfaces
    def get_state(self):
        """
        base
            pose
                pos
                quat
            pose_se2
            twist_se2
        arm
            joint_positions
            ee
                pose
                    pos
                    quat
        head
            joint_positions
                pan
                tilt
            pose
                pos
                quat
        """
        robot_state = copy.copy(self._robot_state)
        output = defaultdict(dict)

        # Base state
        output["base"]["pose_se2"] = sophus2xyt(robot_state.t_base_filtered)
        output["base"]["twist_se2"] = np.zeros(3)

        # Manipulator states
        output["joint_positions"] = np.array(
            [
                self._compute_base_translation_pos(robot_state.t_base_odom),
                robot_state.q_lift,
                robot_state.q_arm,
                robot_state.q_wrist_yaw,
                robot_state.q_wrist_pitch,
                robot_state.q_wrist_roll,
            ]
        )

        # Head states
        output["head"]["pan"] = robot_state.q_head_pan
        output["head"]["tilt"] = robot_state.q_head_tilt

        return output

    def get_base_state(self):
        return self.get_state()["base"]

    def get_camera_image(self, filter_depth=True, compute_xyz=True):
        if self.rgb_cam is None or self.dpt_cam is None:
            rospy.logerr("Cameras not initialized!")
            return

        rgb = self.rgb_cam.get()
        if filter_depth:
            dpt = self.dpt_cam.get_filtered()
        else:
            dpt = self.dpt_cam.get()
        if compute_xyz:
            xyz = self.dpt_cam.depth_to_xyz(self.dpt_cam.fix_depth(dpt))
            imgs = [rgb, dpt, xyz]
        else:
            imgs = [rgb, dpt]
            xyz = None

        # Get xyz in base coords for later
        imgs = [np.rot90(np.fliplr(np.flipud(x))) for x in imgs]

        if xyz is not None:
            xyz = imgs[-1]
            H, W = rgb.shape[:2]
            xyz = xyz.reshape(-1, 3)

            # Rotate the sretch camera so that top of image is "up"
            R_stretch_camera = tra.euler_matrix(0, 0, -np.pi / 2)[:3, :3]
            xyz = xyz @ R_stretch_camera
            xyz = xyz.reshape(H, W, 3)
            imgs[-1] = xyz

        return imgs

    def get_joint_limits(self):
        """
        arm
            max
            min
        head
            pan
                max
                min
            tilt
                max
                min
        """
        raise NotImplementedError

    def get_ee_limits(self):
        """
        max
        min
        """
        raise NotImplementedError

    # Mode switching interfaces
    def switch_to_velocity_mode(self):
        result1 = self._nav_mode_service(TriggerRequest())
        result2 = self._goto_off_service(TriggerRequest())

        # Switch interface mode & print messages
        self._robot_state.base_control_mode = ControlMode.VELOCITY
        rospy.loginfo(result1.message)
        rospy.loginfo(result2.message)

        return result1.success and result2.success

    def switch_to_navigation_mode(self):
        result1 = self._nav_mode_service(TriggerRequest())
        result2 = self._goto_on_service(TriggerRequest())

        # Switch interface mode & print messages
        self._robot_state.base_control_mode = ControlMode.NAVIGATION
        rospy.loginfo(result1.message)
        rospy.loginfo(result2.message)

        return result1.success and result2.success

    def switch_to_manipulation_mode(self):
        result1 = self._pos_mode_service(TriggerRequest())
        result2 = self._goto_off_service(TriggerRequest())

        # Wait for navigation to stabilize
        rospy.sleep(T_LOC_STABILIZE)

        # Set manipulator params
        self._manipulator_params = ManipulatorBaseParams(
            se3_base=self._robot_state.t_base_odom,
        )

        # Switch interface mode & print messages
        self._robot_state.base_control_mode = ControlMode.MANIPULATION
        rospy.loginfo(result1.message)
        rospy.loginfo(result2.message)

        return result1.success and result2.success

    # Control interfaces
    @limit_control_mode([ControlMode.VELOCITY])
    def set_velocity(self, v, w):
        """
        Directly sets the linear and angular velocity of robot base.
        """
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self._velocity_pub.publish(msg)

    @limit_control_mode([ControlMode.NAVIGATION])
    def navigate_to(
        self,
        xyt: List[float],
        relative: bool = False,
        position_only: bool = False,
        avoid_obstacles: bool = False,
    ):
        """
        Cannot be used in manipulation mode.
        """
        # Parse inputs
        assert len(xyt) == 3, "Input goal location must be of length 3."

        if avoid_obstacles:
            raise NotImplementedError("Obstacle avoidance unavailable.")

        # Set yaw tracking
        self._set_yaw_service(SetBoolRequest(data=(not position_only)))

        # Compute absolute goal
        if relative:
            xyt_base = self.get_base_state()["pose_se2"]
            xyt_goal = xyt_base_to_global(xyt, xyt_base)
        else:
            xyt_goal = xyt

        # Set goal
        msg = matrix_to_pose_msg(xyt2sophus(xyt_goal).matrix())
        self._goal_pub.publish(msg)

    @limit_control_mode([ControlMode.MANIPULATION])
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

    @limit_control_mode([ControlMode.MANIPULATION])
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
        q_raw = self.ik_solver.compute_ik(pos_rel, quat_rel)

        # Combine arm telescoping joints
        q_manip = np.zeros(6)
        q_manip[0] = q_raw[0]  # base X translation
        q_manip[1] = q_raw[1]  # lift
        q_manip[2] = np.sum(q_raw[2:6])  # squeeze arm telescoping joints into 1
        q_manip[3:6] = q_raw[6:9]  # yaw pitch roll

        # Execute joint command
        self.set_arm_joint_positions(q_manip)

    @limit_control_mode(
        [
            ControlMode.VELOCITY,
            ControlMode.NAVIGATION,
            ControlMode.MANIPULATION,
        ]
    )
    def open_gripper(self):
        self._send_ros_trajectory_goals({ROS_GRIPPER_FINGER: STRETCH_GRIPPER_OPEN})

    @limit_control_mode(
        [
            ControlMode.VELOCITY,
            ControlMode.NAVIGATION,
            ControlMode.MANIPULATION,
        ]
    )
    def close_gripper(self):
        self._send_ros_trajectory_goals({ROS_GRIPPER_FINGER: STRETCH_GRIPPER_CLOSE})

    @limit_control_mode(
        [
            ControlMode.VELOCITY,
            ControlMode.NAVIGATION,
            ControlMode.MANIPULATION,
        ]
    )
    def set_camera_pan_tilt(
        self, pan: Optional[float] = None, tilt: Optional[float] = None
    ):
        joint_goals = {}
        if pan is not None:
            joint_goals[ROS_HEAD_PAN] = pan
        if tilt is not None:
            joint_goals[ROS_HEAD_TILT] = tilt

        self._send_ros_trajectory_goals(joint_goals)

    @limit_control_mode(
        [
            ControlMode.VELOCITY,
            ControlMode.NAVIGATION,
            ControlMode.MANIPULATION,
        ]
    )
    def set_camera_pose(self, pose_so3):
        raise NotImplementedError  # TODO

    @limit_control_mode([ControlMode.NAVIGATION])
    def navigate_to_camera_pose(self, pose_se3):
        # Compute base pose
        # Navigate to base pose
        # Perform camera pan/tilt
        raise NotImplementedError  # TODO

    # Helper functions
    def _send_ros_trajectory_goals(self, joint_goals: Dict[str, float]):
        # Preprocess arm joints (arm joints are actually 4 joints in one)
        if ROS_ARM_JOINT in joint_goals:
            arm_joint_goal = joint_goals.pop(ROS_ARM_JOINT)

            for arm_joint_name in ROS_ARM_JOINTS_ACTUAL:
                joint_goals[arm_joint_name] = arm_joint_goal / len(
                    ROS_ARM_JOINTS_ACTUAL
                )

        # Preprocess base translation joint (stretch_driver errors out if translation value is 0)
        if ROS_BASE_TRANSLATION_JOINT in joint_goals:
            if joint_goals[ROS_BASE_TRANSLATION_JOINT] == 0:
                joint_goals.pop(ROS_BASE_TRANSLATION_JOINT)

        # Parse input
        joint_names = []
        joint_values = []
        for name, val in joint_goals.items():
            joint_names.append(name)
            joint_values.append(val)

        # Construct goal positions
        point_msg = JointTrajectoryPoint()
        point_msg.positions = joint_values

        # Construct goal msg
        goal_msg = FollowJointTrajectoryGoal()
        goal_msg.goal_time_tolerance = rospy.Time(T_GOAL_TIME_TOL)
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.points = [point_msg]
        goal_msg.trajectory.header.stamp = rospy.Time.now()

        # Send goal
        self.trajectory_client.send_goal(goal_msg)

    def _compute_base_translation_pos(self, t_base=None):
        if self._robot_state.base_control_mode != ControlMode.MANIPULATION:
            return 0.0

        l0_pose = self._manipulator_params.se3_base
        l1_pose = self._robot_state.t_base_odom if t_base is None else t_base
        return (l0_pose.inverse() * l1_pose).translation()[0]

    # Subscriber callbacks
    def _odom_callback(self, msg: Odometry):
        self._robot_state.last_base_update_timestamp = msg.header.stamp
        self._robot_state.t_base_odom = sp.SE3(matrix_from_pose_msg(msg.pose.pose))

    def _base_state_callback(self, msg: PoseStamped):
        self._robot_state.last_base_update_timestamp = msg.header.stamp
        self._robot_state.t_base_filtered = sp.SE3(matrix_from_pose_msg(msg.pose))

    def _joint_state_callback(self, msg: JointState):
        self._robot_state.last_joint_update_timestamp = msg.header.stamp

        map_name2state = {
            ROS_LIFT_JOINT: "q_lift",
            ROS_WRIST_YAW: "wrist_yaw",
            ROS_WRIST_PITCH: "wrist_pitch",
            ROS_WRIST_ROLL: "wrist_roll",
            ROS_GRIPPER_FINGER: "gripper_finger",
            ROS_HEAD_PAN: "head_pan",
            ROS_HEAD_TILT: "head_tilt",
        }

        if ROS_ARM_JOINTS_ACTUAL[0] in msg.name:
            self._robot_state.q_arm = 0.0

        for name, pos in zip(msg.name, msg.position):
            # Arm telescoping joint: add displacement together
            if name in ROS_ARM_JOINTS_ACTUAL:
                self._robot_state.q_arm += pos

            # Normal joints
            elif name in map_name2state:
                setattr(self._robot_state, map_name2state[name], pos)


if __name__ == "__main__":
    # Launches an interactive terminal if file is directly run
    robot = LocalHelloRobot()

    import code

    code.interact(local=locals())
