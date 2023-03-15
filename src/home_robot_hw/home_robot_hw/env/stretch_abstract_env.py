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
    ControlMode,
)
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
    ):
        """Create an interface into ROS execution here. This one needs to connect to:
            - joint_states to read current position
            - tf for SLAM
            - FollowJointTrajectory for arm motions

        Based on this code:
        https://github.com/hello-robot/stretch_ros/blob/master/hello_helpers/src/hello_helpers/hello_misc.py
        """
        # Deprecation warning
        rospy.logwarn(
            "'StretchEnv' is being deprecated. Use 'home_robot_hw.StretchClient' to interact with the robot."
        )

        self._base_control_mode = ControlMode.IDLE
        self._depth_buffer_size = depth_buffer_size
        self._reset_messages()
        self._create_pubs_subs()
        self.rgb_cam, self.dpt_cam = None, None
        if init_cameras:
            self._create_cameras(color_topic, depth_topic)

        self._create_services()
        print("... done.")
        self.wait_for_pose()
        if init_cameras:
            self.wait_for_cameras()

    def reset_state(self):
        self.pos = np.zeros(self.dof)
        self.vel = np.zeros(self.dof)
        self.frc = np.zeros(self.dof)

    def in_manipulation_mode(self):
        return self._base_control_mode == ControlMode.MANIPULATION

    def in_navigation_mode(self):
        return self._base_control_mode == ControlMode.NAVIGATION

    def _reset_messages(self):
        self._current_mode = None
        self._last_odom_update_timestamp = rospy.Time(0)
        self._last_base_update_timestamp = rospy.Time(0)
        self._t_base_filtered = None
        self._t_base_odom = None
        self._at_goal = False
        self._goal_reset_t = rospy.Time(0)

    def in_position_mode(self):
        """is the robot in position mode"""
        return self._current_mode == "position"

    def at_goal(self) -> bool:
        """Returns true if the agent is currently at its goal location"""
        if (
            self._goal_reset_t is not None
            and (rospy.Time.now() - self._goal_reset_t).to_sec() > self.msg_delay_t
        ):
            return self._at_goal
        else:
            return False

    def _at_goal_callback(self, msg):
        """Is the velocity controller done moving; is it at its goal?"""
        self._at_goal = msg.data
        if not self._at_goal:
            self._goal_reset_t = None
        elif self._goal_reset_t is None:
            self._goal_reset_t = rospy.Time.now()

    def _mode_callback(self, msg):
        """get position or navigation mode from stretch ros"""
        self._current_mode = msg.data

    def _odom_callback(self, msg: Odometry):
        """odometry callback"""
        self._last_odom_update_timestamp = msg.header.stamp
        self._t_base_odom = sp.SE3(matrix_from_pose_msg(msg.pose.pose))

    def _base_state_callback(self, msg: PoseStamped):
        """base state updates from SLAM system"""
        self._last_base_update_timestamp = msg.header.stamp
        self._t_base_filtered = sp.SE3(matrix_from_pose_msg(msg.pose))
        self.curr_visualizer(self._t_base_filtered.matrix())

    def _camera_pose_callback(self, msg: PoseStamped):
        self._last_camera_update_timestamp = msg.header.stamp
        self._t_camera_pose = sp.SE3(matrix_from_pose_msg(msg.pose))

    def get_base_pose(self):
        """get the latest base pose from sensors"""
        return sophus2xyt(self._t_base_filtered)

    def get_base_pose_matrix(self):
        """get matrix version of the base pose"""
        return self._t_base_filtered.matrix()

    def get_camera_pose_matrix(self, rotated=False):
        """get matrix version of the camera pose"""
        mat = self._t_camera_pose.matrix()
        if rotated:
            # If we are using the rotated versions of the images
            return mat @ tra.euler_matrix(0, np.pi / 2, 0)
        else:
            return mat

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

    def get_joint_state(self):
        with self._js_lock:
            return self.pos, self.vel, self.frc

    def _create_cameras(self, color_topic=None, depth_topic=None):
        if self.rgb_cam is not None or self.dpt_cam is not None:
            raise RuntimeError("Already created cameras")
        if color_topic is None:
            color_topic = "/camera/color"
        if depth_topic is None:
            depth_topic = "/camera/aligned_depth_to_color"
        print("Creating cameras...")
        self.rgb_cam = RosCamera(color_topic)
        self.dpt_cam = RosCamera(depth_topic, buffer_size=self._depth_buffer_size)
        self.filter_depth = self._depth_buffer_size is not None

    def wait_for_cameras(self):
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

    def wait_for_pose(self):
        """wait until we have an accurate pose estimate"""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self._t_base_filtered is not None:
                break
            rate.sleep()

    def _create_pubs_subs(self):
        """create ROS publishers and subscribers - only call once"""
        # Store latest joint state message - lock for access
        self._js_lock = threading.Lock()

        # Create visualizers for pose information
        self.goal_visualizer = Visualizer("command_pose", rgba=[1.0, 0.0, 0.0, 0.5])
        self.curr_visualizer = Visualizer("current_pose", rgba=[0.0, 0.0, 1.0, 0.5])

        self._at_goal_sub = rospy.Subscriber(
            "goto_controller/at_goal", Bool, self._at_goal_callback, queue_size=10
        )
        self._mode_sub = rospy.Subscriber(
            "mode", String, self._mode_callback, queue_size=1
        )
        # Create the tf2 buffer first, used in camera init
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

        # Create command publishers
        self._goal_pub = rospy.Publisher("goto_controller/goal", Pose, queue_size=1)
        self._velocity_pub = rospy.Publisher("stretch/cmd_vel", Twist, queue_size=1)

        # Create trajectory client with which we can control the robot
        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )
        self._joint_state_subscriber = rospy.Subscriber(
            "stretch/joint_states", JointState, self._js_callback, queue_size=100
        )
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
        self.reset_state()

    def _create_services(self):
        """Create services to activate/deactive robot modes"""
        self._nav_mode_service = rospy.ServiceProxy(
            "switch_to_navigation_mode", Trigger
        )
        self._pos_mode_service = rospy.ServiceProxy("switch_to_position_mode", Trigger)

        self._goto_on_service = rospy.ServiceProxy("goto_controller/enable", Trigger)
        self._goto_off_service = rospy.ServiceProxy("goto_controller/disable", Trigger)
        self._set_yaw_service = rospy.ServiceProxy(
            "goto_controller/set_yaw_tracking", SetBool
        )
        print("Wait for mode service...")
        self._pos_mode_service.wait_for_service()

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

    def config_to_ros_trajectory_goal(self, q: np.ndarray) -> FollowJointTrajectoryGoal:
        """Create a joint trajectory goal to move the arm."""
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(self.goal_time_tolerance)
        trajectory_goal.trajectory.joint_names = self.ros_joint_names
        trajectory_goal.trajectory.points = [self.config_to_ros_msg(q)]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        return trajectory_goal

    def config_to_ros_msg(self, q):
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

    def wait(self, q1, max_wait_t=10.0, no_base=False, verbose=False):
        """helper function to wait until we reach a position"""
        t0 = timeit.default_timer()
        while (timeit.default_timer() - t0) < max_wait_t:
            # update and get pose metrics
            q0, dq0 = self.update()
            err = np.abs(q1 - q0)
            if no_base:
                err[:3] = 0.0
            dt = timeit.default_timer() - t0
            if verbose:
                print("goal =", q1)
                print(dt, err < self.exec_tol)
                self.pretty_print(err)
            if np.all(err < self.exec_tol):
                return True
            time.sleep(self.wait_time_step)
        return False

    def update(self):
        """Return the full joint-state of the robot"""
        # Return a full state for the robot
        pos = self.get_base_pose()
        if pos is not None:
            x, y, theta = pos
        else:
            x, y, theta = 0.0, 0.0, 0.0
        with self._js_lock:
            pos, vel = self.pos.copy(), self.vel.copy()
        pos[:3] = np.array([x, y, theta])
        return pos, vel

    def pretty_print(self, q):
        print("-" * 20)
        print("lift:      ", q[LIFT_IDX])
        print("arm:       ", q[ARM_IDX])
        print("gripper:   ", q[GRIPPER_IDX])
        print("wrist yaw: ", q[WRIST_YAW_IDX])
        print("wrist pitch:", q[WRIST_PITCH_IDX])
        print("wrist roll: ", q[WRIST_ROLL_IDX])
        print("head pan:   ", q[HEAD_PAN_IDX])
        print("head tilt:   ", q[HEAD_TILT_IDX])
        print("-" * 20)

    # Mode switching interfaces
    def switch_to_velocity_mode(self):
        result1 = self._nav_mode_service(TriggerRequest())
        result2 = self._goto_off_service(TriggerRequest())

        # Switch interface mode & print messages
        # TODO - switch control mode in robot state
        self._base_control_mode = ControlMode.VELOCITY
        rospy.loginfo(result1.message)
        rospy.loginfo(result2.message)

        return result1.success and result2.success

    def switch_to_navigation_mode(self):
        """switch stretch to navigation control"""
        if not self.in_navigation_mode():
            result1 = self._nav_mode_service(TriggerRequest())
        else:
            result1 = None
        result2 = self._goto_on_service(TriggerRequest())

        # Switch interface mode & print messages
        self._base_control_mode = ControlMode.NAVIGATION
        if result1 is not None:
            rospy.loginfo(result1.message)
            nav_mode_success = result1.success
        else:
            nav_mode_success = True
        rospy.loginfo(result2.message)

        return nav_mode_success and result2.success

    def switch_to_manipulation_mode(self):
        result1 = self._pos_mode_service(TriggerRequest())
        result2 = self._goto_off_service(TriggerRequest())

        # Wait for navigation to stabilize
        rospy.sleep(T_LOC_STABILIZE)

        # Set manipulator params
        self._manipulator_params = ManipulatorBaseParams(
            se3_base=self._t_base_odom,
        )

        # Switch interface mode & print messages
        self._base_control_mode = ControlMode.MANIPULATION
        rospy.loginfo(result1.message)
        rospy.loginfo(result2.message)

        return result1.success and result2.success

    def process_depth(self, depth):
        depth[depth < self.min_depth_val] = MIN_DEPTH_REPLACEMENT_VALUE
        depth[depth > self.max_depth_val] = MAX_DEPTH_REPLACEMENT_VALUE
        return depth

    def get_images(self, compute_xyz=False, rotate_images=True):
        """helper logic to get images from the robot's camera feed"""
        rgb = self.rgb_cam.get()
        if self.filter_depth:
            dpt = self.dpt_cam.get_filtered()
        else:
            dpt = self.process_depth(self.dpt_cam.get())
        if compute_xyz:
            xyz = self.dpt_cam.depth_to_xyz(self.dpt_cam.fix_depth(dpt))
            imgs = [rgb, dpt, xyz]
        else:
            imgs = [rgb, dpt]
            xyz = None

        if rotate_images:
            # Get xyz in base coords for later
            # TODO: replace with the new util function
            imgs = [np.rot90(np.fliplr(np.flipud(x))) for x in imgs]

        if xyz is not None:
            xyz = imgs[-1]
            H, W = rgb.shape[:2]
            xyz = xyz.reshape(-1, 3)

            if rotate_images:
                # Rotate the stretch camera so that top of image is "up"
                R_stretch_camera = tra.euler_matrix(0, 0, -np.pi / 2)[:3, :3]
                xyz = xyz @ R_stretch_camera
                xyz = xyz.reshape(H, W, 3)
                imgs[-1] = xyz

        return imgs

    def get_pose(
        self,
        frame: str,
        base_frame: Optional[str] = None,
        lookup_time: Optional[float] = None,
        timeout_s: Optional[float] = None,
    ) -> np.ndarray:
        """look up a particular frame in base coords"""
        if lookup_time is None:
            lookup_time = rospy.Time(0)  # return most recent transform
        if timeout_s is None:
            timeout_ros = rospy.Duration(0.1)
        else:
            timeout_ros = rospy.Duration(timeout_s)
        if base_frame is None:
            base_frame = self.odom_link
        try:
            stamped_transform = self.tf2_buffer.lookup_transform(
                base_frame, frame, lookup_time, timeout_ros
            )
            pose_mat = ros_numpy.numpify(stamped_transform.transform)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            print(
                "!!! Lookup failed from",
                self.base_link,
                "to",
                self.odom_link,
                f"!!!. Exception: {e}",
            )
            return None
        return pose_mat

    # Control interfaces
    def set_velocity(self, v, w):
        """
        Directly sets the linear and angular velocity of robot base.
        """
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self._velocity_pub.publish(msg)

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

    def navigate_to(
        self,
        xyt: Iterable[float],
        relative: bool = False,
        position_only: bool = False,
        avoid_obstacles: bool = False,
        blocking: bool = False,
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
            xyt_base = sophus2xyt(self._t_base_filtered)
            xyt_goal = xyt_base_to_global(xyt, xyt_base)
        else:
            xyt_goal = xyt

        # Clear self.at_goal
        self._at_goal = False
        self._goal_reset_t = None

        # Set goal
        goal_matrix = xyt2sophus(xyt_goal).matrix()
        self.goal_visualizer(goal_matrix)
        msg = matrix_to_pose_msg(goal_matrix)
        self._goal_pub.publish(msg)

        if blocking:
            rospy.sleep(self.msg_delay_t)
            rate = rospy.Rate(self.block_spin_rate)
            while not rospy.is_shutdown():
                # Verify that we are at goal and perception is synchronized with pose
                if self.at_goal() and self.recent_depth_image(self.msg_delay_t):
                    break
                else:
                    rate.sleep()
            # TODO: this should be unnecessary
            # TODO: add this back in if we are having trouble building maps
            # Make sure that depth and position are synchonized
            # rospy.sleep(self.msg_delay_t * 5)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def apply_action(self, action: Action, info: Optional[Dict[str, Any]] = None):
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


if __name__ == "__main__":
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_ros_test")
    print("Create ROS interface")
    rob = StretchEnv(init_cameras=True)
