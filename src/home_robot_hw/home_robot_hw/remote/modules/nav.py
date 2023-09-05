# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Iterable

import rospy
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBoolRequest, TriggerRequest

from home_robot.motion.robot import Robot
from home_robot.utils.geometry import sophus2xyt, xyt2sophus, xyt_base_to_global
from home_robot_hw.constants import T_LOC_STABILIZE
from home_robot_hw.ros.utils import matrix_to_pose_msg

from .abstract import AbstractControlModule, enforce_enabled


class StretchNavigationClient(AbstractControlModule):
    block_spin_rate = 10

    def __init__(self, ros_client, robot_model: Robot):
        super().__init__()

        self._ros_client = ros_client
        self._robot_model = robot_model
        self._wait_for_pose()

    # Enable / disable

    def _enable_hook(self) -> bool:
        """Called when interface is enabled."""
        result = self._ros_client.nav_mode_service(TriggerRequest())
        rospy.loginfo(result.message)
        return result.success

    def _disable_hook(self) -> bool:
        """Called when interface is disabled."""
        result = self._ros_client.goto_off_service(TriggerRequest())
        rospy.sleep(T_LOC_STABILIZE)  # wait for robot movement to stop
        return result.success

    # Interface methods

    def get_base_pose(self):
        """get the latest base pose from sensors"""
        return sophus2xyt(self._ros_client.se3_base_filtered)

    def at_goal(self) -> bool:
        """Returns true if the agent is currently at its goal location"""
        if (
            self._ros_client._goal_reset_t is not None
            and (rospy.Time.now() - self._ros_client._goal_reset_t).to_sec()
            > self._ros_client.msg_delay_t
        ):
            return self._ros_client.at_goal
        else:
            return False

    @enforce_enabled
    def set_velocity(self, v, w):
        """
        Directly sets the linear and angular velocity of robot base.
        """
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w

        self._ros_client.goto_off_service(TriggerRequest())
        self._ros_client.velocity_pub.publish(msg)

    @enforce_enabled
    def navigate_to(
        self,
        xyt: Iterable[float],
        relative: bool = False,
        position_only: bool = False,
        avoid_obstacles: bool = False,
        blocking: bool = True,
    ):
        """
        Cannot be used in manipulation mode.
        """
        # Parse inputs
        assert len(xyt) == 3, "Input goal location must be of length 3."

        if avoid_obstacles:
            raise NotImplementedError("Obstacle avoidance unavailable.")

        # Set yaw tracking
        self._ros_client.set_yaw_service(SetBoolRequest(data=(not position_only)))

        # Compute absolute goal
        if relative:
            xyt_base = sophus2xyt(self._ros_client.se3_base_filtered)
            xyt_goal = xyt_base_to_global(xyt, xyt_base)
        else:
            xyt_goal = xyt

        # Clear self.at_goal
        self._ros_client.at_goal = False
        self._ros_client.goal_reset_t = None

        # Set goal
        goal_matrix = xyt2sophus(xyt_goal).matrix()
        self._ros_client.goal_visualizer(goal_matrix)
        msg = matrix_to_pose_msg(goal_matrix)

        self._ros_client.goto_on_service(TriggerRequest())
        self._ros_client.goal_pub.publish(msg)

        self._register_wait(self._wait_for_goal_reached)
        if blocking:
            self.wait()

    @enforce_enabled
    def home(self):
        self.navigate_to([0.0, 0.0, 0.0], blocking=True)

    # Helper methods

    def _wait_for_pose(self):
        """wait until we have an accurate pose estimate"""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self._ros_client.se3_base_filtered is not None:
                break
            rate.sleep()

    def _wait_for_goal_reached(self):
        """Wait until goal is reached"""
        rospy.sleep(self._ros_client.msg_delay_t)
        rate = rospy.Rate(self.block_spin_rate)
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            t1 = rospy.Time.now()
            print(
                "...waited for controller",
                (t1 - t0).to_sec(),
                "is at goal =",
                self.at_goal(),
            )
            # Verify that we are at goal and perception is synchronized with pose
            if self.at_goal() and self._ros_client.recent_depth_image(
                self._ros_client.msg_delay_t
            ):
                break
            else:
                rate.sleep()
        # TODO: this should be unnecessary
        # TODO: add this back in if we are having trouble building maps
        # Make sure that depth and position are synchronized
        # rospy.sleep(self.msg_delay_t * 5)
