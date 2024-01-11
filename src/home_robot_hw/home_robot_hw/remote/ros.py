# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import threading
from typing import Dict, Optional

import actionlib
import numpy as np
import ros_numpy
import rospy
import sophus as sp
import tf2_ros
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import PointStamped, Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float32, String
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest
from trajectory_msgs.msg import JointTrajectoryPoint

from home_robot.motion.stretch import STRETCH_HEAD_CAMERA_ROTATIONS, HelloStretchIdx
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
from home_robot_hw.ros.lidar import RosLidar
from home_robot_hw.ros.utils import matrix_from_pose_msg
from home_robot_hw.ros.visualizer import Visualizer

DEFAULT_COLOR_TOPIC = "/camera/color"
DEFAULT_DEPTH_TOPIC = "/camera/aligned_depth_to_color"
DEFAULT_LIDAR_TOPIC = "/scan"


class StretchRosInterface:
    """Interface object with ROS topics and services"""

    # Base of the robot
    base_link = "base_link"

    goal_time_tolerance = 1.0
    msg_delay_t = 0.1

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
        init_lidar: bool = True,
        lidar_topic: Optional[str] = None,
        verbose: bool = False,
    ):
        # Verbosity for the ROS client
        self.verbose = verbose

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
        self._goal_reset_t = rospy.Time(0)

        # Create visualizers for pose information
        self.goal_visualizer = Visualizer("command_pose", rgba=[1.0, 0.0, 0.0, 0.5])
        self.curr_visualizer = Visualizer("current_pose", rgba=[0.0, 0.0, 1.0, 0.5])

        # Initialize ros communication
        self._safety_check()
        self._create_pubs_subs()
        self._create_services()

        self._ros_joint_names = []
        for i in range(3, self.dof):
            self._ros_joint_names += CONFIG_TO_ROS[i]

        # Initialize cameras
        self._color_topic = DEFAULT_COLOR_TOPIC if color_topic is None else color_topic
        self._depth_topic = DEFAULT_DEPTH_TOPIC if depth_topic is None else depth_topic
        self._lidar_topic = DEFAULT_LIDAR_TOPIC if lidar_topic is None else lidar_topic
        self._depth_buffer_size = depth_buffer_size

        self.rgb_cam, self.dpt_cam = None, None
        if init_cameras:
            self._create_cameras()
            self._wait_for_cameras()
        if init_lidar:
            self._lidar = RosLidar(self._lidar_topic)
            self._lidar.wait_for_scan()

    # Interfaces

    def get_joint_state(self):
        with self._js_lock:
            return self.pos, self.vel, self.frc

    def send_trajectory_goals(self, joint_goals: Dict[str, float], velocities=None):
        """Send trajectory goals to the robot. Goals are a dictionary of joint names and strings. Can optionally provide velicities as well."""

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
        if velocities is not None:
            point_msg.velocities = velocities

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

    def recent_depth_image(self, seconds, print_delay_timers: bool = False):
        """Return true if we have up to date depth."""
        # Make sure we have a goal and our poses and depths are synced up - we need to have
        # received depth after we stopped moving
        if print_delay_timers:
            print(
                " - 1",
                (rospy.Time.now() - self._goal_reset_t).to_sec(),
                self.msg_delay_t,
            )
            print(
                " - 2", (self.dpt_cam.get_time() - self._goal_reset_t).to_sec(), seconds
            )
        if (
            self._goal_reset_t is not None
            and (rospy.Time.now() - self._goal_reset_t).to_sec() > self.msg_delay_t
        ):
            return (self.dpt_cam.get_time() - self._goal_reset_t).to_sec() > seconds
        else:
            return False

    def config_to_ros_trajectory_goal(
        self, q: np.ndarray, dq: np.ndarray = None, ddq: np.ndarray = None
    ) -> FollowJointTrajectoryGoal:
        """Create a joint trajectory goal to move the arm."""
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(self.goal_time_tolerance)
        trajectory_goal.trajectory.joint_names = self.ros_joint_names
        trajectory_goal.trajectory.points = [self._config_to_ros_msg(q, dq, ddq)]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        return trajectory_goal

    # Helper functions

    def _safety_check(self, max_time: float = 30.0):
        """Make sure we can actually execute code on the robot as a quality of life measure"""
        run_stopped = rospy.wait_for_message("is_runstopped", Bool, timeout=max_time)
        if run_stopped is None:
            rospy.logwarn(
                "is_runstopped not received; you might have out of date stretch_ros"
            )
        elif run_stopped.data is True:
            rospy.logerr("Runstop is pressed! Cannot execute!")
            raise RuntimeError("Stretch is runstopped")
        calibrated = rospy.wait_for_message("is_calibrated", Bool, timeout=max_time)
        if calibrated is None:
            rospy.logwarn(
                "is_calibrated not received; you might have out of date stretch_ros"
            )
        elif calibrated.data is False:
            rospy.logwarn("Robot is not calibrated!")
        homed = rospy.wait_for_message("is_homed", Bool, timeout=max_time)
        if homed is None:
            rospy.logwarn(
                "is_homed not received; you might have out of date stretch_ros"
            )
        elif homed.data is False:
            rospy.logerr(
                "Robot is not homed! Cannot execute! Run stretch_robot_home.py"
            )
            raise RuntimeError("Stretch is not homed!")

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

        self.grasp_ready = None
        self.grasp_complete = None
        self.grasp_enable_pub = rospy.Publisher(
            "grasp_point/enable", Empty, queue_size=1
        )
        self.grasp_ready_sub = rospy.Subscriber(
            "grasp_point/ready", Empty, self._grasp_ready_callback
        )
        self.grasp_disable_pub = rospy.Publisher(
            "grasp_point/disable", Empty, queue_size=1
        )
        self.grasp_trigger_pub = rospy.Publisher(
            "grasp_point/trigger_grasp_point", PointStamped, queue_size=1
        )
        self.grasp_result_sub = rospy.Subscriber(
            "grasp_point/result", Float32, self._grasp_result_callback
        )

        self.place_ready = None
        self.place_complete = None
        self.location_above_surface_m = None
        self.place_enable_pub = rospy.Publisher(
            "place_point/enable", Float32, queue_size=1
        )
        self.place_ready_sub = rospy.Subscriber(
            "place_point/ready", Empty, self._place_ready_callback
        )
        self.place_disable_pub = rospy.Publisher(
            "place_point/disable", Empty, queue_size=1
        )
        self.place_trigger_pub = rospy.Publisher(
            "place_point/trigger_place_point", PointStamped, queue_size=1
        )
        self.place_result_sub = rospy.Subscriber(
            "place_point/result", Empty, self._place_result_callback
        )

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

    def _create_cameras(self):
        if self.rgb_cam is not None or self.dpt_cam is not None:
            raise RuntimeError("Already created cameras")
        print("Creating cameras...")
        self.rgb_cam = RosCamera(
            self._color_topic, rotations=STRETCH_HEAD_CAMERA_ROTATIONS
        )
        self.dpt_cam = RosCamera(
            self._depth_topic,
            rotations=STRETCH_HEAD_CAMERA_ROTATIONS,
            buffer_size=self._depth_buffer_size,
        )
        self.filter_depth = self._depth_buffer_size is not None

    def _wait_for_lidar(self):
        """wait until lidar has a message"""
        self._lidar.wait_for_scan()

    def _wait_for_cameras(self):
        if self.rgb_cam is None or self.dpt_cam is None:
            raise RuntimeError("cameras not initialized")
        print("Waiting for rgb camera images...")
        self.rgb_cam.wait_for_image()
        print("Waiting for depth camera images...")
        self.dpt_cam.wait_for_image()
        print("..done.")
        if self.verbose:
            print("rgb frame =", self.rgb_cam.get_frame())
            print("dpt frame =", self.dpt_cam.get_frame())
        if self.rgb_cam.get_frame() != self.dpt_cam.get_frame():
            raise RuntimeError("issue with camera setup; depth and rgb not aligned")

    def _config_to_ros_msg(self, q, dq=None, ddq=None):
        """convert into a joint state message"""
        msg = JointTrajectoryPoint()
        msg.positions = [0.0] * len(self._ros_joint_names)
        if dq is not None:
            msg.velocities = [0.0] * len(self._ros_joint_names)
        if ddq is not None:
            msg.accelerations = [0.0] * len(self._ros_joint_names)
        idx = 0
        for i in range(3, self.dof):
            names = CONFIG_TO_ROS[i]
            for _ in names:
                # Only for arm - but this is a dumb way to check
                if "arm" in names[0]:
                    msg.positions[idx] = q[i] / len(names)
                else:
                    msg.positions[idx] = q[i]
                if dq is not None:
                    msg.velocities[idx] = dq[i]
                if ddq is not None:
                    msg.accelerations[idx] = ddq[i]
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
        """camera pose from CameraPosePublisher, which reads from tf"""
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

    def get_frame_pose(self, frame, base_frame=None, lookup_time=None, timeout_s=None):
        """look up a particular frame in base coords (or some other coordinate frame)."""
        if lookup_time is None:
            lookup_time = rospy.Time(0)  # return most recent transform
        if timeout_s is None:
            timeout_ros = rospy.Duration(0.1)
        else:
            timeout_ros = rospy.Duration(timeout_s)
        if base_frame is None:
            base_frame = self.base_link
        try:
            stamped_transform = self.tf2_buffer.lookup_transform(
                base_frame, frame, lookup_time, timeout_ros
            )
            pose_mat = ros_numpy.numpify(stamped_transform.transform)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            print("!!! Lookup failed from", base_frame, "to", frame, "!!!")
            return None
        return pose_mat

    def _construct_single_joint_ros_goal(
        self, joint_name, position, goal_time_tolerance=1
    ):
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Duration(goal_time_tolerance)
        trajectory_goal.trajectory.joint_names = [
            joint_name,
        ]
        msg = JointTrajectoryPoint()
        msg.positions = [position]
        trajectory_goal.trajectory.points = [msg]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        return trajectory_goal

    def goto_x(self, x, wait=False, verbose=True):
        trajectory_goal = self._construct_single_joint_ros_goal(
            "translate_mobile_base", x
        )
        self.trajectory_client.send_goal(trajectory_goal)
        if wait:
            #  Waiting for result seems to hang
            self.trajectory_client.wait_for_result()
            # self.wait(q, max_wait_t, True, verbose)
            # print("-- TODO: wait for xy")
        return True

    def goto_theta(self, theta, wait=False, verbose=True):
        trajectory_goal = self._construct_single_joint_ros_goal(
            "rotate_mobile_base", theta
        )
        self.trajectory_client.send_goal(trajectory_goal)
        if wait:
            self.trajectory_client.wait_for_result()
            # self.wait(q, max_wait_t, True, verbose)
            # print("-- TODO: wait for theta")
        return True

    def goto_lift_position(self, delta_position, wait=False):
        # TODO spowers: utilize config_to_ros_trajectory_goal?
        success = False
        if abs(delta_position) > 0:  # self.exec_tol[HelloStretchIdx.LIFT]:
            position = self.pos[HelloStretchIdx.LIFT] + delta_position
            if position > 0.1 and position < 1:
                trajectory_goal = self._construct_single_joint_ros_goal(
                    "joint_lift", position
                )
                self.trajectory_client.send_goal(trajectory_goal)
                if wait:
                    self.trajectory_client.wait_for_result()
                    # self.wait(q, max_wait_t, True, verbose)
                    # print("-- TODO: wait for theta")
                success = True
        return success

    def goto_arm_position(self, delta_position, wait=False):
        if abs(delta_position) > 0:  # self.exec_tol[HelloStretchIdx.ARM]:
            position = self.pos[HelloStretchIdx.ARM] + delta_position
            trajectory_goal = self._construct_single_joint_ros_goal(
                "wrist_extension", position
            )
            self.trajectory_client.send_goal(trajectory_goal)
            if wait:
                self.trajectory_client.wait_for_result()
        return True

    def goto_wrist_yaw_position(self, delta_position, wait=False):
        if abs(delta_position) > 0:  # self.exec_tol[HelloStretchIdx.WRIST_YAW]:
            position = self.pos[HelloStretchIdx.WRIST_YAW] + delta_position
            trajectory_goal = self._construct_single_joint_ros_goal(
                "joint_wrist_yaw", position
            )
            self.trajectory_client.send_goal(trajectory_goal)
            if wait:
                self.trajectory_client.wait_for_result()
        return True

    def goto_wrist_roll_position(self, delta_position, wait=False):
        if abs(delta_position) > 0:  # self.exec_tol[HelloStretchIdx.WRIST_ROLL]:
            position = self.pos[HelloStretchIdx.WRIST_ROLL] + delta_position
            trajectory_goal = self._construct_single_joint_ros_goal(
                "joint_wrist_roll", position
            )
            self.trajectory_client.send_goal(trajectory_goal)
            if wait:
                self.trajectory_client.wait_for_result()
        return True

    def goto_wrist_pitch_position(self, delta_position, wait=False):
        if abs(delta_position) > 0:  # self.exec_tol[HelloStretchIdx.WRIST_PITCH]:
            position = self.pos[HelloStretchIdx.WRIST_PITCH] + delta_position
            trajectory_goal = self._construct_single_joint_ros_goal(
                "joint_wrist_pitch", position
            )
            self.trajectory_client.send_goal(trajectory_goal)
            if wait:
                self.trajectory_client.wait_for_result()
        return True

    def goto_gripper_position(self, delta_position, wait=False):
        if (
            abs(delta_position) > 0
        ):  # TODO controller seems to be commanding 0.05s.... self.exec_tol[HelloStretchIdx.GRIPPER]:  #0: #0.01:  # TODO: this is ...really high? (5?) self.exec_tol[HelloStretchIdx.GRIPPER]:
            position = self.pos[HelloStretchIdx.GRIPPER] + delta_position
            trajectory_goal = self._construct_single_joint_ros_goal(
                "joint_gripper_finger_left", position
            )
            self.trajectory_client.send_goal(trajectory_goal)
            if wait:
                self.trajectory_client.wait_for_result()
        return True

    def goto_head_pan_position(self, delta_position, wait=False):
        if abs(delta_position) > 0:  # self.exec_tol[HelloStretchIdx.HEAD_PAN]:
            position = self.pos[HelloStretchIdx.HEAD_PAN] + delta_position
            trajectory_goal = self._construct_single_joint_ros_goal(
                "joint_head_pan", position
            )
            self.trajectory_client.send_goal(trajectory_goal)
            if wait:
                self.trajectory_client.wait_for_result()
        return True

    def goto_head_tilt_position(self, delta_position, wait=False):
        if abs(delta_position) > 0:  # self.exec_tol[HelloStretchIdx.HEAD_TILT]:
            position = self.pos[HelloStretchIdx.HEAD_TILT] + delta_position
            trajectory_goal = self._construct_single_joint_ros_goal(
                "joint_head_tilt", position
            )
            self.trajectory_client.send_goal(trajectory_goal)
            if wait:
                self.trajectory_client.wait_for_result()
        return True

    def _interp(self, x1, x2, num_steps=10):
        diff = x2 - x1
        rng = np.arange(num_steps + 1) / num_steps
        rng = rng[:, None].repeat(3, axis=1)
        diff = diff[None].repeat(num_steps + 1, axis=0)
        x1 = x1[None].repeat(num_steps + 1, axis=0)
        return x1 + (rng * diff)

    def goto_wrist(self, roll, pitch, yaw, verbose=False, wait=False):
        """Separate out wrist commands from everything else"""
        q = self.pos
        r0, p0, y0 = (
            q[HelloStretchIdx.WRIST_ROLL],
            q[HelloStretchIdx.WRIST_PITCH],
            q[HelloStretchIdx.WRIST_YAW],
        )
        print("--------")
        print("roll", roll, "curr =", r0)
        print("pitch", pitch, "curr =", p0)
        print("yaw", yaw, "curr =", y0)
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Duration(1)
        trajectory_goal.trajectory.joint_names = [
            ROS_WRIST_ROLL,
            ROS_WRIST_PITCH,
            ROS_WRIST_YAW,
        ]
        pt = JointTrajectoryPoint()
        pt.positions = [roll, pitch, yaw]
        trajectory_goal.trajectory.points = [pt]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        self.trajectory_client.send_goal(trajectory_goal)
        if wait:
            self.trajectory_client.wait_for_result()

    def goto(self, q, move_base=False, wait=False, max_wait_t=10.0, verbose=False):
        """some of these params are unsupported"""
        goal = self.config_to_ros_trajectory_goal(q)
        self.trajectory_client.send_goal(goal)
        if wait:
            self.trajectory_client.wait_for_result()
        return True

    def _grasp_ready_callback(self, empty_msg):
        self.grasp_ready = True

    def _grasp_result_callback(self, float_msg):
        self.location_above_surface_m = float_msg.data
        self.grasp_complete = True

    def trigger_grasp(self, x, y, z):
        """Calls FUNMAP based grasping"""
        # 1. Enable the grasp node
        assert self.grasp_ready is None
        assert self.grasp_complete is None
        assert self.location_above_surface_m is None
        self.grasp_enable_pub.publish(Empty())
        self.place_disable_pub.publish(Empty())

        # 2. Wait until grasp node ready
        while self.grasp_ready is None:
            rospy.sleep(0.2)

        # 3. Call the trigger topic
        goal_point = PointStamped()
        goal_point.header.stamp = rospy.Time.now()
        goal_point.header.frame_id = "map"
        goal_point.point.x = x
        goal_point.point.y = y
        goal_point.point.z = z
        self.grasp_trigger_pub.publish(goal_point)

        # 4. Wait for grasp to complete
        print(" - Waiting for grasp to complete")
        while self.grasp_complete is None:
            rospy.sleep(0.2)
        assert self.location_above_surface_m is not None

        # 5. Disable the grasp node
        self.grasp_disable_pub.publish(Empty())

        self.grasp_ready = None
        self.grasp_complete = None
        return

    def _place_ready_callback(self, empty_msg):
        self.place_ready = True

    def _place_result_callback(self, msg):
        self.place_complete = True

    def trigger_placement(self, x, y, z):
        """Calls FUNMAP based placement"""
        # 1. Enable the place node
        assert self.place_ready is None
        assert self.place_complete is None
        assert self.location_above_surface_m is not None
        self.grasp_disable_pub.publish(Empty())
        msg = Float32()
        msg.data = self.location_above_surface_m
        self.place_enable_pub.publish(msg)

        # 2. Wait until place node ready
        while self.place_ready is None:
            rospy.sleep(0.2)

        # 3. Call the trigger topic
        goal_point = PointStamped()
        goal_point.header.stamp = rospy.Time.now()
        goal_point.header.frame_id = "map"
        goal_point.point.x = x
        goal_point.point.y = y
        goal_point.point.z = z
        self.place_trigger_pub.publish(goal_point)

        # 4. Wait for grasp to complete
        print(" - Waiting for place to complete")
        while self.place_complete is None:
            rospy.sleep(0.2)

        # 5. Disable the place node
        self.place_disable_pub.publish(Empty())

        self.location_above_surface_m = None
        self.place_ready = None
        self.place_complete = None
        return
