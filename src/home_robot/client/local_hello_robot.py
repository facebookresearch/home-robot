from collections import defaultdict
import argparse
import pdb
from typing import Optional, Iterable

import numpy as np

import rospy
from std_srvs.srv import Trigger, TriggerRequest
from std_srvs.srv import SetBool, SetBoolRequest
from geometry_msgs.msg import PoseStamped, Pose, Twist

from home_robot.utils.geometry import xyt2sophus, sophus2xyt, xyt_base_to_global
from home_robot.utils.geometry.ros import pose_sophus2ros, pose_ros2sophus
from home_robot.hw.ros.stretch_ros import HelloStretchROSInterface


class LocalHelloRobot:
    """
    ROS interface for robot base control
    Currently only works with a local rosmaster
    """

    def __init__(self, init_node: bool = True):
        self._base_state = None

        # Ros interface from old home robot
        self.robot = HelloStretchROSInterface(visualize_planner=False)
        self.robot.rgb_cam.wait_for_image()
        self.robot.dpt_cam.wait_for_image()

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
            "/switch_to_navigation_mode", Trigger
        )
        self._pos_mode_service = rospy.ServiceProxy("/switch_to_position_mode", Trigger)
        self._goto_on_service = rospy.ServiceProxy("goto_controller/enable", Trigger)
        self._goto_off_service = rospy.ServiceProxy("goto_controller/disable", Trigger)
        self._set_yaw_service = rospy.ServiceProxy(
            "goto_controller/toggle_yaw_tracking", Trigger
        )

    # ==========================================
    # Old API
    def set_nav_mode(self):
        """
        Switches to navigation mode.
        Robot always tries to move to goal in nav mode.
        """
        result = self._nav_mode_service(TriggerRequest())
        self._goto_on_service(TriggerRequest())
        print(result.message)

    def set_pos_mode(self):
        """
        Switches to position mode.
        """
        result = self._pos_mode_service(TriggerRequest())
        print(result.message)
        result = self._goto_off_service(TriggerRequest())
        print(result.message)

    def set_yaw_tracking(self, value: bool = True):
        """
        Turns yaw tracking on/off.
        Robot only tries to reach the xy position of goal if off.
        """
        result = self._set_yaw_service(SetBoolRequest(data=value))
        print(result.message)
        return result.success

    def get_base_state(self):
        """
        Returns base location in the form of [x, y, rz].
        """
        return self._base_state

    def set_goal(self, xyt):
        """
        Sets the goal for the goto controller.
        """
        msg = pose_sophus2ros(xyt2sophus(xyt))
        self._goal_pub.publish(msg)

    def set_velocity(self, v, w):
        """
        Directly sets the linear and angular velocity of robot base.
        Command gets overwritten immediately if goto controller is on.
        """
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self._velocity_pub.publish(msg)

    # ==========================================
    # New interface
    def get_robot_state(self):
        """
        Note: read poses from tf2 buffer

        base
            pose_se2
            twist_se2
        arm
            joint_positions
            ee
                pose
                    base
                        pos
                        quat
                    world
                        pos
                        quat
        head
            joint_positions
                pan
                tilt
            pose
                base
                    pos
                    quat
                world
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
        rgb, depth, xyz = self.robot.get_images()
        return rgb, depth

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
    def switch_to_navigation_mode(self):
        result1 = self._nav_mode_service(TriggerRequest())
        print(result1.message)
        result2 = self._goto_on_service(TriggerRequest())
        print(result2.message)
        return result1.success and result2.success

    def switch_to_manipulation_mode(self):
        result1 = self._pos_mode_service(TriggerRequest())
        print(result1.message)
        result2 = self._goto_off_service(TriggerRequest())
        print(result2.message)
        return result1.success and result2.success

    # Control
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
        # TODO: check mode
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
        """
        assert len(joint_positions) == 6, "Joint position vector must be of length 6."
        q = self.robot.update()
        q[3:9] = joint_positions
        self.robot.goto(q, move_base=False)

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

    def set_camera_pose(
        self, pan: Optional[float] = None, tilt: Optional[float] = None
    ):
        q = self.robot.update()
        if pan is not None:
            q[9] = pan
        if tilt is not None:
            q[10] = tilt
        self.robot.goto(q, move_base=False)

    # Subscriber callbacks
    def _state_callback(self, msg: PoseStamped):
        self._base_state = sophus2xyt(pose_ros2sophus(msg.pose))


if __name__ == "__main__":
    # Launches an interactive terminal if file is directly run
    robot = LocalHelloRobot()

    import code

    code.interact(local=locals())
