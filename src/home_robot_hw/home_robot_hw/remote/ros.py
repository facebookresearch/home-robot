# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import threading
from typing import Dict, Optional

import actionlib
import numpy as np
import rospy
import sophus as sp
import tf
import tf2_ros
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest
from trajectory_msgs.msg import JointTrajectoryPoint

from home_robot.motion.stretch import HelloStretchIdx
from home_robot.utils.pose import to_matrix
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
)
from home_robot_hw.ros.camera import RosCamera
from home_robot_hw.ros.utils import matrix_from_pose_msg
from home_robot_hw.ros.visualizer import Visualizer

DEFAULT_COLOR_TOPIC = "/camera/color"
DEFAULT_DEPTH_TOPIC = "/camera/aligned_depth_to_color"


class StretchRosInterface:
    """Interface object with ROS topics and services"""

    goal_time_tolerance = 1.0
    msg_delay_t = 0.25

    # 3 for base position + rotation, 2 for lift + extension, 3 for rpy, 1 for gripper, 2 for head
    dof = 3 + 2 + 3 + 1 + 2

    # Joint names in the ROS joint trajectory server
    BASE_TRANSLATION_JOINT = "translate_mobile_base"
    ARM_JOINT = "joint_arm"
    LIFT_JOINT = "joint_lift"
    WRIST_YAW = "joint_wrist_yaw"
    WRIST_PITCH = "joint_wrist_pitch"
    WRIST_ROLL = "joint_wrist_roll"
    GRIPPER_FINGER = "joint_gripper_finger_left"  # used to control entire gripper
    HEAD_PAN = "joint_head_pan"
    HEAD_TILT = "joint_head_tilt"
    ARM_JOINTS_ACTUAL = ["joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3"]

    def __init__(
        self,
        init_cameras: bool = True,
        color_topic: Optional[str] = None,
        depth_topic: Optional[str] = None,
        depth_buffer_size: Optional[int] = None,
    ):
        # Initialize caches
        self.current_mode: Optional[str] = None

        self.pos = np.zeros(self.dof)
        self.vel = np.zeros(self.dof)
        self.frc = np.zeros(self.dof)

        self.se3_base_filtered: Optional[sp.SE3] = None
        self.se3_base_odom: Optional[sp.SE3] = None
        self.se3_camera_pose: Optional[sp.SE3] = None
        self.at_goal: bool = False

        self.last_odom_update_timestamp = rospy.Time(0)
        self.last_base_update_timestamp = rospy.Time(0)
        self.goal_reset_t = rospy.Time(0)

        # Create visualizers for pose information
        self.goal_visualizer = Visualizer("command_pose", rgba=[1.0, 0.0, 0.0, 0.5])
        self.curr_visualizer = Visualizer("current_pose", rgba=[0.0, 0.0, 1.0, 0.5])

        # Initialize ros communication
        self._create_pubs_subs()
        self._create_services()
        self._tf_listener = tf.TransformListener()

        self._ros_joint_names = []
        for i in range(3, self.dof):
            self._ros_joint_names += CONFIG_TO_ROS[i]

        # Initialize cameras
        self._color_topic = DEFAULT_COLOR_TOPIC if color_topic is None else color_topic
        self._depth_topic = DEFAULT_DEPTH_TOPIC if depth_topic is None else depth_topic
        self._depth_buffer_size = depth_buffer_size

        self.rgb_cam, self.dpt_cam = None, None
        if init_cameras:
            self._create_cameras(color_topic, depth_topic)
            self._wait_for_cameras()

    # Interfaces

    def get_joint_state(self):
        with self._js_lock:
            return self.pos, self.vel, self.frc

    def send_trajectory_goals(self, joint_goals: Dict[str, float]):
        # Preprocess arm joints (arm joints are actually 4 joints in one)
        if self.ARM_JOINT in joint_goals:
            arm_joint_goal = joint_goals.pop(self.ARM_JOINT)

            for arm_joint_name in self.ARM_JOINTS_ACTUAL:
                joint_goals[arm_joint_name] = arm_joint_goal / len(
                    self.ARM_JOINTS_ACTUAL
                )

        # Preprocess base translation joint (stretch_driver errors out if translation value is 0)
        if self.BASE_TRANSLATION_JOINT in joint_goals:
            if joint_goals[self.BASE_TRANSLATION_JOINT] == 0:
                joint_goals.pop(self.BASE_TRANSLATION_JOINT)

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
        goal_msg.goal_time_tolerance = rospy.Time(self.goal_time_tolerance)
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.points = [point_msg]
        goal_msg.trajectory.header.stamp = rospy.Time.now()

        # Send goal
        self.trajectory_client.send_goal(goal_msg)

    def wait_for_trajectory_action(self):
        self.trajectory_client.wait_for_result()

    def recent_depth_image(self, seconds):
        """Return true if we have up to date depth."""
        # Make sure we have a goal and our poses and depths are synced up - we need to have
        # received depth after we stopped moving
        if (
            self._goal_reset_t is not None
            and (rospy.Time.now() - self._goal_reset_t).to_sec() > self.msg_delay_t
        ):
            return (self.dpt_cam.get_time() - self._goal_reset_t).to_sec() > seconds
        else:
            return False

    def config_to_ros_trajectory_goal(self, q: np.ndarray) -> FollowJointTrajectoryGoal:
        """Create a joint trajectory goal to move the arm."""
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(self.goal_time_tolerance)
        trajectory_goal.trajectory.joint_names = self.ros_joint_names
        trajectory_goal.trajectory.points = [self._config_to_ros_msg(q)]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        return trajectory_goal

    def get_frame_pose(self, frame, base_frame=None, lookup_time=None):
        """look up a particular frame in base coords"""
        if lookup_time is None:
            lookup_time = rospy.Time(0)  # return most recent transform

        if base_frame is None:
            base_frame = "base_link"  # TODO configure base frame

        pose_mat = None
        transform_success = False
        while not transform_success:
            try:
                transform = self._tf_listener.lookupTransform(
                    base_frame, frame, lookup_time
                )
                pose_mat = to_matrix(*transform)
                transform_success = True
            except Exception:  # (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                pass

        return pose_mat

    # Helper functions

    def _create_services(self):
        """Create services to activate/deactive robot modes"""
        self.nav_mode_service = rospy.ServiceProxy("switch_to_navigation_mode", Trigger)
        self.pos_mode_service = rospy.ServiceProxy("switch_to_position_mode", Trigger)

        self.goto_on_service = rospy.ServiceProxy("goto_controller/enable", Trigger)
        self.goto_off_service = rospy.ServiceProxy("goto_controller/disable", Trigger)
        self.set_yaw_service = rospy.ServiceProxy(
            "goto_controller/set_yaw_tracking", SetBool
        )
        print("Wait for mode service...")
        self.pos_mode_service.wait_for_service()

    def _create_pubs_subs(self):
        """create ROS publishers and subscribers - only call once"""
        # Create the tf2 buffer first, used in camera init
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

        # Create command publishers
        self.goal_pub = rospy.Publisher("goto_controller/goal", Pose, queue_size=1)
        self.velocity_pub = rospy.Publisher("stretch/cmd_vel", Twist, queue_size=1)

        # Create subscribers
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
        self._camera_pose_sub = rospy.Subscriber(
            "camera_pose", PoseStamped, self._camera_pose_callback, queue_size=1
        )
        self._at_goal_sub = rospy.Subscriber(
            "goto_controller/at_goal", Bool, self._at_goal_callback, queue_size=10
        )
        self._mode_sub = rospy.Subscriber(
            "mode", String, self._mode_callback, queue_size=1
        )

        # Create trajectory client with which we can control the robot
        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )

        self._js_lock = (
            threading.Lock()
        )  # store latest joint state message - lock for access
        self._joint_state_subscriber = rospy.Subscriber(
            "stretch/joint_states", JointState, self._js_callback, queue_size=100
        )

        print("Waiting for trajectory server...")
        server_reached = self.trajectory_client.wait_for_server(
            timeout=rospy.Duration(30.0)
        )
        if not server_reached:
            print("ERROR: Failed to connect to arm action server.")
            rospy.signal_shutdown(
                "Unable to connect to arm action server. Timeout exceeded."
            )
            sys.exit()
        print("... connected to arm action server.")

        self.ros_joint_names = []
        for i in range(3, self.dof):
            self.ros_joint_names += CONFIG_TO_ROS[i]

    def _create_cameras(self, color_topic=None, depth_topic=None):
        if self.rgb_cam is not None or self.dpt_cam is not None:
            raise RuntimeError("Already created cameras")
        print("Creating cameras...")
        self.rgb_cam = RosCamera(self._color_topic)
        self.dpt_cam = RosCamera(self._depth_topic, buffer_size=self._depth_buffer_size)
        self.filter_depth = self._depth_buffer_size is not None

    def _wait_for_cameras(self):
        if self.rgb_cam is None or self.dpt_cam is None:
            raise RuntimeError("cameras not initialized")
        print("Waiting for rgb camera images...")
        self.rgb_cam.wait_for_image()
        print("Waiting for depth camera images...")
        self.dpt_cam.wait_for_image()
        print("..done.")
        print("rgb frame =", self.rgb_cam.get_frame())
        print("dpt frame =", self.dpt_cam.get_frame())
        if self.rgb_cam.get_frame() != self.dpt_cam.get_frame():
            raise RuntimeError("issue with camera setup; depth and rgb not aligned")

    def _config_to_ros_msg(self, q):
        """convert into a joint state message"""
        msg = JointTrajectoryPoint()
        msg.positions = [0.0] * len(self._ros_joint_names)
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

    # Rostopic callbacks

    def _at_goal_callback(self, msg):
        """Is the velocity controller done moving; is it at its goal?"""
        self.at_goal = msg.data
        if not self.at_goal:
            self._goal_reset_t = None
        elif self._goal_reset_t is None:
            self._goal_reset_t = rospy.Time.now()

    def _mode_callback(self, msg):
        """get position or navigation mode from stretch ros"""
        self._current_mode = msg.data

    def _odom_callback(self, msg: Odometry):
        """odometry callback"""
        self._last_odom_update_timestamp = msg.header.stamp
        self.se3_base_odom = sp.SE3(matrix_from_pose_msg(msg.pose.pose))

    def _base_state_callback(self, msg: PoseStamped):
        """base state updates from SLAM system"""
        self._last_base_update_timestamp = msg.header.stamp
        self.se3_base_filtered = sp.SE3(matrix_from_pose_msg(msg.pose))
        self.curr_visualizer(self.se3_base_filtered.matrix())

    def _camera_pose_callback(self, msg: PoseStamped):
        self._last_camera_update_timestamp = msg.header.stamp
        self.se3_camera_pose = sp.SE3(matrix_from_pose_msg(msg.pose))

    def _js_callback(self, msg):
        """Read in current joint information from ROS topics and update state"""
        # loop over all joint state info
        pos, vel, trq = np.zeros(self.dof), np.zeros(self.dof), np.zeros(self.dof)
        for name, p, v, e in zip(msg.name, msg.position, msg.velocity, msg.effort):
            # Check name etc
            if name in ROS_ARM_JOINTS:
                pos[HelloStretchIdx.ARM] += p
                vel[HelloStretchIdx.ARM] += v
                trq[HelloStretchIdx.ARM] += e
            elif name in ROS_TO_CONFIG:
                idx = ROS_TO_CONFIG[name]
                pos[idx] = p
                vel[idx] = v
                trq[idx] = e
        trq[HelloStretchIdx.ARM] /= 4
        with self._js_lock:
            self.pos, self.vel, self.frc = pos, vel, trq
