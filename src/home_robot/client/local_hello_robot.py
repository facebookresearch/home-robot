from collections import defaultdict
from enum import Enum
import argparse
import pdb
import time
from typing import Optional, Iterable, List, Dict
from dataclasses import dataclass

import numpy as np
import sophus as sp
import rospy
from std_srvs.srv import Trigger, TriggerRequest
from std_srvs.srv import SetBool, SetBoolRequest
from geometry_msgs.msg import PoseStamped, Pose, Twist
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

from home_robot.utils.geometry import xyt2sophus, sophus2xyt, xyt_base_to_global
from home_robot.utils.geometry.ros import pose_sophus2ros, pose_ros2sophus


T_LOC_STABILIZE = 1.0
T_GOAL_TIME_TOL = 1.0


@dataclass
class ManipulatorBaseParams:
    xyt_base: np.ndarray
    se3_base: sp.SE3


class BaseControlMode(Enum):
    IDLE = 0
    VELOCITY = 1
    NAVIGATION = 2
    MANIPULATION = 3


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

STRETCH_GRIPPER_OPEN = 0.22
STRETCH_GRIPPER_CLOSE = -0.2


def limit_control_mode(valid_modes: List[BaseControlMode]):
    """Decorator for checking if a robot method is executed while the correct mode is present."""

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if self._control_mode in valid_modes:
                return func(self, *args, **kwargs)
            else:
                rospy.logwarn(
                    f"'{func.__name__}' is only available in the following modes: {valid_modes}"
                )
                rospy.logwarn(f"Current mode is: {self._control_mode}")
                return None

        return wrapper

    return decorator


class LocalHelloRobot:
    """
    ROS interface for robot base control
    Currently only works with a local rosmaster
    """

    def __init__(self, init_node: bool = True):
        self._base_state = None
        self._control_mode = BaseControlMode.IDLE
        self._manipulator_params = None

        # Ros pubsub
        if init_node:
            rospy.init_node("user")

        self._goal_pub = rospy.Publisher("goto_controller/goal", Pose, queue_size=1)
        self._velocity_pub = rospy.Publisher("stretch/cmd_vel", Twist, queue_size=1)

        self._state_sub = rospy.Subscriber(
            "state_estimator/pose_filtered",
            PoseStamped,
            self._state_callback,
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

        # Initialize control mode & home robot
        self.switch_to_manipulation_mode()
        self.close_gripper()
        self.set_arm_joint_positions([0.1, 0.3, 0, 0, 0, 0])
        self._control_mode = BaseControlMode.IDLE

    # Getter interfaces
    def get_robot_state(self):
        """
        Note: read poses from tf2 buffer

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
        robot_state = defaultdict(dict)

        # Base state
        robot_state["base"]["pose_se2"] = self._base_state
        robot_state["base"]["twist_se2"] = np.zeros(3)

        return robot_state

    def get_base_state(self):
        return self.get_robot_state["base"]

    def get_camera_image(self):
        """
        rgb, depth, xyz = self.robot.get_images()
        return rgb, depth
        """
        pass

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
        self._control_mode = BaseControlMode.VELOCITY
        rospy.loginfo(result1.message)
        rospy.loginfo(result2.message)

        return result1.success and result2.success

    def switch_to_navigation_mode(self):
        result1 = self._nav_mode_service(TriggerRequest())
        result2 = self._goto_on_service(TriggerRequest())

        # Switch interface mode & print messages
        self._control_mode = BaseControlMode.NAVIGATION
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
            xyt_base=self._base_state,
            se3_base=xyt2sophus(self._base_state),
        )

        # Switch interface mode & print messages
        self._control_mode = BaseControlMode.MANIPULATION
        rospy.loginfo(result1.message)
        rospy.loginfo(result2.message)

        return result1.success and result2.success

    # Control interfaces
    @limit_control_mode([BaseControlMode.VELOCITY])
    def set_velocity(self, v, w):
        """
        Directly sets the linear and angular velocity of robot base.
        """
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self._velocity_pub.publish(msg)

    @limit_control_mode([BaseControlMode.NAVIGATION])
    def navigate_to(
        self,
        xyt: Iterable[float],
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
        msg = pose_sophus2ros(xyt2sophus(xyt_goal))
        self._goal_pub.publish(msg)

    @limit_control_mode([BaseControlMode.MANIPULATION])
    def set_arm_joint_positions(self, joint_positions: Iterable[float]):
        """
        list of robot arm joint positions:
            BASE_TRANSLATION = 0
            LIFT = 1
            ARM = 2
            WRIST_ROLL = 3
            WRIST_PITCH = 4
            WRIST_YAW = 5
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
            ROS_WRIST_ROLL: joint_positions[3],
            ROS_WRIST_PITCH: joint_positions[4],
            ROS_WRIST_YAW: joint_positions[5],
        }

        self._send_ros_trajectory_goals(joint_goals)

        return True

    @limit_control_mode([BaseControlMode.MANIPULATION])
    def set_ee_pose(
        self,
        pos: Iterable[float],
        quat: Optional[Iterable[float]] = None,
        relative: bool = False,
    ):
        """
        Does not rotate base.
        Cannot be used in navigation mode.
        """
        # TODO: check pose
        raise NotImplementedError

    @limit_control_mode(
        [
            BaseControlMode.VELOCITY,
            BaseControlMode.NAVIGATION,
            BaseControlMode.MANIPULATION,
        ]
    )
    def open_gripper(self):
        self._send_ros_trajectory_goals({ROS_GRIPPER_FINGER: STRETCH_GRIPPER_OPEN})

    @limit_control_mode(
        [
            BaseControlMode.VELOCITY,
            BaseControlMode.NAVIGATION,
            BaseControlMode.MANIPULATION,
        ]
    )
    def close_gripper(self):
        self._send_ros_trajectory_goals({ROS_GRIPPER_FINGER: STRETCH_GRIPPER_CLOSE})

    @limit_control_mode(
        [
            BaseControlMode.VELOCITY,
            BaseControlMode.NAVIGATION,
            BaseControlMode.MANIPULATION,
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
            BaseControlMode.VELOCITY,
            BaseControlMode.NAVIGATION,
            BaseControlMode.MANIPULATION,
        ]
    )
    def set_camera_pose(self, pose_so3):
        raise NotImplementedError  # TODO

    @limit_control_mode([BaseControlMode.NAVIGATION])
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

    def _compute_base_translation_pos(self):
        l0_pose = self._manipulator_params.se3_base
        l1_pose = sophus2xyt(self._base_state)
        return (l0_pose.inverse() * l1_pose).translation()[0]

    # Subscriber callbacks
    def _state_callback(self, msg: PoseStamped):
        self._base_state = sophus2xyt(pose_ros2sophus(msg.pose))


if __name__ == "__main__":
    # Launches an interactive terminal if file is directly run
    robot = LocalHelloRobot()

    import code

    code.interact(local=locals())
