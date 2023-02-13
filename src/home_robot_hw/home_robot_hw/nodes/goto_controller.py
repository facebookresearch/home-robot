#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import List, Optional
import threading

import numpy as np
import sophus as sp
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from std_srvs.srv import SetBool, SetBoolResponse
from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry

from home_robot.control.goto_controller import GotoVelocityController
from home_robot.utils.geometry import xyt_global_to_base, sophus2xyt, xyt2sophus
from home_robot_hw.ros.utils import matrix_from_pose_msg
from home_robot_hw.ros.visualizer import Visualizer


log = logging.getLogger(__name__)

CONTROL_HZ = 20


class GotoVelocityControllerNode:
    """
    Self-contained controller module for moving a diff drive robot to a target goal.
    Target goal is update-able at any given instant.
    """

    def __init__(
        self,
        hz: float,
        odom_only_feedback: bool = True,
        config_name: str = "noplan_velocity_hw",
    ):
        self.hz = hz
        self.odom_only = odom_only_feedback

        # Control module
        self.controller = GotoVelocityController(config_name)

        # Initialize
        self.xyt_goal: Optional[np.ndarray] = None

        self.active = False
        self.track_yaw = True

        # Visualizations
        self.goal_visualizer = Visualizer("goto_controller/goal_abs")

    def _pose_update_callback(self, msg: PoseStamped):
        if not self.odom_only:
            pose_sp = sp.SE3(matrix_from_pose_msg(msg.pose))
            self.controller.update_pose_feedback(pose_sp)

    def _odom_update_callback(self, msg: Odometry):
        if self.odom_only:
            pose_sp = sp.SE3(matrix_from_pose_msg(msg.pose.pose))
            self.controller.update_pose_feedback(pose_sp)

    def _goal_update_callback(self, msg: Pose):
        pose_sp = sp.SE3(matrix_from_pose_msg(msg))

        """
        if self.odom_only:
            # Project absolute goal from current odometry reading
            pose_delta = xyt2sophus(self.xyt_loc_odom).inverse() * pose_sp
            pose_goal = xyt2sophus(self.xyt_loc_odom) * pose_delta
        else:
            # Assign absolute goal directly
            pose_goal = pose_sp
        """

        pose_goal = pose_sp

        self.controller.update_goal(sophus2xyt(pose_goal))
        self.xyt_goal = self.controller.xyt_goal

        # Visualize
        self.goal_visualizer(pose_goal.matrix())

    def _enable_service(self, request):
        self.active = True
        return TriggerResponse(
            success=True,
            message=f"Goto controller is now RUNNING",
        )

    def _disable_service(self, request):
        self.active = False
        return TriggerResponse(
            success=True,
            message=f"Goto controller is now STOPPED",
        )

    def _set_yaw_tracking_service(self, request: SetBool):
        track_yaw = request.data

        self.controller.set_yaw_tracking(track_yaw)

        status_str = "ON" if self.track_yaw else "OFF"
        return SetBoolResponse(
            success=True,
            message=f"Yaw tracking is now {status_str}",
        )

    def _set_velocity(self, v_m, w_r):
        cmd = Twist()
        cmd.linear.x = v_m
        cmd.angular.z = w_r
        self.vel_command_pub.publish(cmd)

    def _run_control_loop(self):
        rate = rospy.Rate(self.hz)

        while not rospy.is_shutdown():
            if self.active and self.xyt_goal is not None:
                # Compute control
                v_cmd, w_cmd = self.controller.compute_control()

                # Command robot
                self._set_velocity(v_cmd, w_cmd)

            # Spin
            rate.sleep()

    def main(self):
        # ROS comms
        rospy.init_node("goto_controller")

        self.vel_command_pub = rospy.Publisher("stretch/cmd_vel", Twist, queue_size=1)

        rospy.Subscriber(
            "state_estimator/pose_filtered",
            PoseStamped,
            self._pose_update_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "odom",
            Odometry,
            self._odom_update_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "goto_controller/goal", Pose, self._goal_update_callback, queue_size=1
        )

        rospy.Service("goto_controller/enable", Trigger, self._enable_service)
        rospy.Service("goto_controller/disable", Trigger, self._disable_service)
        rospy.Service(
            "goto_controller/set_yaw_tracking", SetBool, self._set_yaw_tracking_service
        )

        # Run controller
        log.info("Goto Controller launched.")
        self._run_control_loop()


if __name__ == "__main__":
    GotoVelocityControllerNode(CONTROL_HZ).main()
