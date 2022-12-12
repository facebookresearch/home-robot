from collections import defaultdict
from enum import Enum
import argparse
import pdb
import logging
from typing import Optional, Iterable, List
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

log = logging.getLogger(__file__)


T_LOC_STABILIZE = 1


@dataclass
class ManipulatorBaseParams:
    xyt_base: np.ndarray
    se3_base: sp.SE3


class BaseControlMode(Enum):
    VELOCITY = 1
    NAVIGATION = 2
    MANIPULATION = 3


class HelloStretchIdx:
    BASE_X = 0
    BASE_Y = 1
    BASE_THETA = 2
    LIFT = 3
    ARM = 4
    GRIPPER = 5
    WRIST_ROLL = 6
    WRIST_PITCH = 7
    WRIST_YAW = 8
    HEAD_PAN = 9
    HEAD_TILT = 10


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


class LocalHelloRobot:
    """
    ROS interface for robot base control
    Currently only works with a local rosmaster
    """

    def __init__(self, init_node: bool = True):
        self._base_state = None

        # Ros interface from old home robot
        """
        self.robot = HelloStretchROSInterface(visualize_planner=False)
        self.robot.rgb_cam.wait_for_image()
        self.robot.dpt_cam.wait_for_image()
        """
        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )
        self.dof = 11
        self.ros_joint_names = []
        for i in range(3, self.dof):
            self.ros_joint_names += CONFIG_TO_ROS[i]

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
            "goto_controller/toggle_yaw_tracking", Trigger
        )

        # Initialize control mode
        self._control_mode = BaseControlMode.NAVIGATION

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

    # Mode switching ?
    def switch_to_velocity_mode(self):
        result1 = self._nav_mode_service(TriggerRequest())
        result2 = self._goto_off_service(TriggerRequest())

        # Switch interface mode & print messages
        self._control_mode = BaseControlMode.VELOCITY
        log.info(result1.message)
        log.info(result2.message)

        return result1.success and result2.success

    def switch_to_navigation_mode(self):
        result1 = self._nav_mode_service(TriggerRequest())
        result2 = self._goto_on_service(TriggerRequest())

        # Switch interface mode & print messages
        self._control_mode = BaseControlMode.NAVIGATION
        log.info(result1.message)
        log.info(result2.message)

        return result1.success and result2.success

    def switch_to_manipulation_mode(self):
        result1 = self._pos_mode_service(TriggerRequest())
        result2 = self._goto_off_service(TriggerRequest())

        # Wait for navigation to stabilize
        rospy.sleep(T_LOC_STABILIZE)

        # Set manipulator params TODO

        # Switch interface mode & print messages
        self._control_mode = BaseControlMode.MANIPULATION
        log.info(result1.message)
        log.info(result2.message)

        return result1.success and result2.success

    def _check_mode(self, func_name: str, valid_modes: List[BaseControlMode]) -> bool:
        if self._control_mode in valid_modes:
            return True
        else:
            log.warning(
                f"'{func_name}' is only available in the following modes: {valid_modes}"
            )
            return False

    # Control
    def set_velocity(self, v, w):
        """
        Directly sets the linear and angular velocity of robot base.
        """
        if not self._check_mode_valid("set_velocity", [BaseControlMode.VELOCITY]):
            return

        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self._velocity_pub.publish(msg)

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
        if not self._check_mode_valid("navigate_to", [BaseControlMode.NAVIGATION]):
            return

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

    def _config_to_ros_msg(self, q):
        """convert into a joint state message"""
        msg = JointTrajectoryPoint()
        msg.positions = [0.0] * len(self.ros_joint_names)
        idx = 0
        for i in range(3, self.dof):
            names = CONFIG_TO_ROS[i]
            for _ in names:
                # Only for arm - but this is a dumb way to check
                if "arm" in names[0]:
                    msg.positions[idx] = q[i] / len(names)
                else:
                    msg.positions[idx] = q[i]
                idx += 1
        return msg

    def _generate_ros_trajectory_goal(self, q):
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = self.ros_joint_names
        trajectory_goal.trajectory.points = [self._config_to_ros_msg(q)]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        return trajectory_goal

    def set_arm_joint_positions(self, joint_positions: Iterable[float]):
        """
        q: list of robot joint positions:
                BASE_X = 0
                BASE_Y = 1
                BASE_THETA = 2
                LIFT = 3
                ARM = 4
                GRIPPER = 5
                WRIST_ROLL = 6
                WRIST_PITCH = 7
                WRIST_YAW = 8
                HEAD_PAN = 9
                HEAD_TILT = 10

        TODO: new q
                BASE_THETA = 2
                LIFT = 3
                ARM = 4
                GRIPPER = 5
                WRIST_ROLL = 6
                WRIST_PITCH = 7
                WRIST_YAW = 8
        """
        assert len(joint_positions) == 6, "Joint position vector must be of length 6."
        q = [0] * 11
        q[3:9] = joint_positions
        goal = self._generate_ros_trajectory_goal(q)
        self.trajectory_client.send_goal(goal)

        return True

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
        if not self._check_mode_valid("set_ee_pose", [BaseControlMode.MANIPULATION]):
            return

        # TODO: check pose
        raise NotImplementedError

    def set_camera_pan_tilt(
        self, pan: Optional[float] = None, tilt: Optional[float] = None
    ):
        """
        if pan is not None:
            q[9] = pan
        if tilt is not None:
            q[10] = tilt
        self.robot.goto(q, move_base=False)
        """
        pass

    def navigate_to_camera_pose(self, pose_se3):
        # Compute base pose
        # Navigate to base pose
        # Perform camera pan/tilt
        pass  # TODO

    # Subscriber callbacks
    def _state_callback(self, msg: PoseStamped):
        self._base_state = sophus2xyt(pose_ros2sophus(msg.pose))


if __name__ == "__main__":
    # Launches an interactive terminal if file is directly run
    robot = LocalHelloRobot()

    import code

    code.interact(local=locals())
