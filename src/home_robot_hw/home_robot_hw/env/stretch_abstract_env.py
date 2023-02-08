from abc import abstractmethod
from typing import Any, Dict, Optional

import actionlib
import home_robot
import home_robot.core.abstract_env
import numpy as np
import rospy
import sophus as sp
import tf2_ros
import threading

# Import ROS messages and tools
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryGoal
from geometry_msgs.msg import PoseStamped, Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerRequest
from std_srvs.srv import SetBool, SetBoolRequest
from trajectory_msgs.msg import JointTrajectoryPoint

from home_robot.utils.geometry import (
    xyt2sophus,
    sophus2xyt,
    xyt_base_to_global,
    posquat2sophus,
)

from home_robot.core.interfaces import Action, Observations
from home_robot.agent.motion.robot import HelloStretchIdx
from home_robot_hw.ros.camera import RosCamera
from home_robot_hw.constants import (ROS_ARM_JOINTS, ROS_LIFT_JOINT, ROS_GRIPPER_FINGER, ROS_HEAD_PAN, ROS_HEAD_TILT, ROS_WRIST_ROLL, ROS_WRIST_YAW, ROS_WRIST_PITCH, ROS_GRIPPER_FINGER, ROS_TO_CONFIG, CONFIG_TO_ROS)
from home_robot_hw.ros.utils import matrix_from_pose_msg, matrix_to_pose_msg


MIN_DEPTH_REPLACEMENT_VALUE = 10000
MAX_DEPTH_REPLACEMENT_VALUE = 10001


class StretchEnv(home_robot.core.abstract_env.Env):
    """ Defines a ROS-based interface to the real Stretch robot. Collect observations and command the robot."""

    # 3 for base position + rotation, 2 for lift + extension, 3 for rpy, 1 for gripper, 2 for head
    dof = 3 + 2 + 3 + 1 + 2
    min_depth_val = 0.1
    max_depth_val = 4.0

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

        self._depth_buffer_size = depth_buffer_size
        self._create_pubs_subs()
        self.rgb_cam, self.dpt_cam = None, None
        if init_cameras:
            self._create_cameras(color_topic, depth_topic)
        self._create_services()
        self._reset_messages()
        print("... done.")
        if init_cameras:
            self.wait_for_cameras()

    def _reset_messages(self):
        self._current_mode = None
        self._last_odom_update_timestamp = rospy.Time(0)
        self._last_base_update_timestamp = rospy.Time(0)
        self._t_base_filtered = None
        self._t_base_odom = None

    def in_position_mode(self):
        """ is the robot in position mode """
        return self._current_mode == "position"

    def _mode_callback(self, msg):
        """ get position or navigation mode from stretch ros """
        self._current_mode = msg.data

    def _odom_callback(self, msg: Odometry):
        """ odometry callback """
        self._last_odom_update_timestamp = msg.header.stamp
        self._t_base_odom = sp.SE3(matrix_from_pose_msg(msg.pose.pose))

    def _base_state_callback(self, msg: PoseStamped):
        """ base state updates from SLAM system """
        self._last_base_update_timestamp = msg.header.stamp
        self._t_base_filtered = sp.SE3(matrix_from_pose_msg(msg.pose))

    def get_base_pose(self):
        """ get the latest base pose from sensors """
        return self._t_base_filtered

    def _js_callback(self, msg):
        """ Read in current joint information from ROS topics and update state """
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

    def _create_cameras(self, color_topic=None, depth_topic=None):
        if self.rgb_cam is not None or self.dpt_cam is not None:
            raise RuntimeError('Already created cameras')
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
            raise RuntimeError('cameras not initialized')
        print("Waiting for rgb camera images...")
        self.rgb_cam.wait_for_image()
        print("Waiting for depth camera images...")
        self.dpt_cam.wait_for_image()
        print("..done.")
        print("rgb frame =", self.rgb_cam.get_frame())
        print("dpt frame =", self.dpt_cam.get_frame())
        if self.rgb_cam.get_frame() != self.dpt_cam.get_frame():
            raise RuntimeError("issue with camera setup; depth and rgb not aligned")

    def _create_pubs_subs(self):
        """ create ROS publishers and subscribers - only call once """
        # Store latest joint state message - lock for access
        self._js_lock = threading.Lock()
        self.mode = ""
        self._mode_sub = rospy.Subscriber("mode", String, self._mode_callback, queue_size=1)
        # Create the tf2 buffer first, used in camera init
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)
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
 
        print("Waiting for trajectory server...")
        server_reached = self.trajectory_client.wait_for_server(
            timeout=rospy.Duration(30.0)
        )
        print("... connected.")
        if not server_reached:
            print("ERROR: Failed to connect to arm action server.")
            rospy.signal_shutdown(
                "Unable to connect to arm action server. Timeout exceeded."
            )
            sys.exit()

    def _create_services(self):
        """ Create services to activate/deactive robot modes """
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
        """ switch stretch to navigation control """
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


if __name__ == '__main__':
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_ros_test")
    print("Create ROS interface")
    rob = StretchEnv(init_cameras=True)
   
