# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import rospy


class StretchRosInterface:
    def __init__(self):
        self._create_pubs_subs()
        self._create_services()

    def _create_services(self):
        """Create services to activate/deactive robot modes"""
        self.nav_mode_service = rospy.ServiceProxy(
            "switch_to_navigation_mode", Trigger
        )
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
        # Store latest joint state message - lock for access
        self._js_lock = threading.Lock()

        # Create visualizers for pose information
        self.goal_visualizer = Visualizer("command_pose", rgba=[1., 0., 0., 0.5])
        self.curr_visualizer = Visualizer("current_pose", rgba=[0., 0., 1., 0.5])

        self._at_goal_sub = rospy.Subscriber(
            "goto_controller/at_goal",
            Bool,
            self._at_goal_callback,
            queue_size=10)
        self._mode_sub = rospy.Subscriber(
            "mode", String, self._mode_callback, queue_size=1
        )
        # Create the tf2 buffer first, used in camera init
        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

        # Create command publishers
        self.goal_pub = rospy.Publisher("goto_controller/goal", Pose, queue_size=1)
        self.velocity_pub = rospy.Publisher("stretch/cmd_vel", Twist, queue_size=1)

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
        self._camera_pose_sub = rospy.Subscriber("camera_pose", PoseStamped, self._camera_pose_callback, queue_size=1)

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

    def send_ros_trajectory_goals(self, joint_goals: Dict[str, float]):
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
