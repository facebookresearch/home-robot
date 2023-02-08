from abc import abstractmethod
from typing import Any, Dict, Optional

import home_robot
import home_robot.core.abstract_env


class StretchEnv(home_robot.core.abstract_env.Env):
    """ Defines a ROS-based interface to the real Stretch robot. Collect observations and command the robot."""

    def __init__(
        self,
        model=None,
        visualize_planner=False,
        root=".",
        init_cameras=True,
        depth_buffer_size=None,
        urdf_path=None,
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

        if urdf_path is None:
            # By default try to use ROS to find the URDF
            urdf_path = get_urdf_dir()

        # No hardware interface here for the ROS code
        if model is None:
            model = HelloStretch(
                visualize=visualize_planner, root=root, urdf_path=urdf_path
            )
        self.model = model  # This is the model
        self.dof = model.dof

        # Create the tf2 buffer first, used in camera init
        self.tf2_buffer = tf2_ros.Buffer()

        if color_topic is None:
            color_topic = "/camera/color"
        if depth_topic is None:
            depth_topic = "/camera/aligned_depth_to_color"

        self._create_services()
        if init_cameras:
            print("Creating cameras...")
            self.rgb_cam = RosCamera(color_topic)
            self.dpt_cam = RosCamera(depth_topic, buffer_size=depth_buffer_size)
            self.filter_depth = depth_buffer_size is not None
            print("Waiting for rgb camera images...")
            self.rgb_cam.wait_for_image()
            print("Waiting for depth camera images...")
            self.dpt_cam.wait_for_image()
            print("..done.")
            print("rgb frame =", self.rgb_cam.get_frame())
            print("dpt frame =", self.dpt_cam.get_frame())
            if self.rgb_cam.get_frame() != self.dpt_cam.get_frame():
                raise RuntimeError("issue with camera setup; depth and rgb not aligned")
        else:
            self.rgb_cam, self.dpt_cam = None, None

        # Store latest joint state message - lock for access
        self._js_lock = threading.Lock()
        self.mode = ""
        self._mode_pub = rospy.Subscriber("mode", String, self._mode_cb, queue_size=1)
        rospy.sleep(0.5)
        print("... done.")
        if not self.in_position_mode():
            print("Switching to position mode...")
            print(self.switch_to_position())

        # ROS stuff
        self.trajectory_client = actionlib.SimpleActionClient(
            "/stretch_controller/follow_joint_trajectory", FollowJointTrajectoryAction
        )
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)
        self.joint_state_subscriber = rospy.Subscriber(
            "stretch/joint_states", JointState, self._js_cb, queue_size=100
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

        self.ros_joint_names = []
        for i in range(3, self.dof):
            self.ros_joint_names += CONFIG_TO_ROS[i]
        self.reset_state()

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


if __name__ == '__main__':
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_ros_test")
    print("Create ROS interface")
    rob = StretchEnv(visualize_planner=False, init_cameras=True)
   
