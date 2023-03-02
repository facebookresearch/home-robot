# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import rospy

from home_robot_hw.constants import T_LOC_STABILIZE

from .abstract import AbstractControlModule, enforce_enabled

class StretchNavigationInterface(AbstractControlModule):
    def __init__(self, ros_client):
        self.ros_client = ros_client

    # Enable / disable

    def _enable_hook(self) -> bool:
        if not self.in_navigation_mode():
            result1 = self._nav_mode_service(TriggerRequest())
        result2 = self._goto_on_service(TriggerRequest())

        # Switch interface mode & print messages
        rospy.loginfo(result1.message)
        rospy.loginfo(result2.message)

        return result1.success and result2.success

    def _disable_hook(self) -> bool:
        rospy.sleep(T_LOC_STABILIZE)  # wait for robot movement to stop
        return True

    # Interface methods

    def get_pose(self):
        """get the latest base pose from sensors"""
        return sophus2xyt(self._t_base_filtered)

    def at_goal(self) -> bool:
        """Returns true if the agent is currently at its goal location"""
        if self._goal_reset_t is not None and (rospy.Time.now() - self._goal_reset_t).to_sec() > self.msg_delay_t:
            return self._at_goal
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
        self._velocity_pub.publish(msg)


    @enforce_enabled
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