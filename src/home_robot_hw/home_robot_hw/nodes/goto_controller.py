#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import threading
from typing import List, Optional

import numpy as np
import rospy
import sophus as sp
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32
from std_srvs.srv import (
    SetBool,
    SetBoolResponse,
    Trigger,
    TriggerRequest,
    TriggerResponse,
)

from home_robot.control.goto_controller import GotoVelocityController
from home_robot.utils.config import get_control_config
from home_robot.utils.geometry import sophus2xyt, xyt2sophus
from home_robot_hw.ros.utils import matrix_from_pose_msg
from home_robot_hw.ros.visualizer import Visualizer

log = logging.getLogger(__name__)

CONTROL_HZ = 20
VEL_THRESHOlD = 0.001
RVEL_THRESHOLD = 0.005
DEBUG_CONTROL_LOOP = False


class GotoVelocityControllerNode:
    """
    Self-contained controller module for moving a diff drive robot to a target goal.
    Target goal is update-able at any given instant.
    """

    # How long should the controller report done before we're actually confident that we're done?
    done_t = rospy.Duration(0.1)

    def __init__(
        self,
        hz: float,
        odom_only_feedback: bool = False,
        config_name: str = "noplan_velocity_hw",
    ):
        self.hz = hz
        self.odom_only = odom_only_feedback

        # Control module
        controller_cfg = get_control_config(config_name)
        self.controller = GotoVelocityController(controller_cfg)
        # Update the velocity and acceleration configs from the file
        self.controller.update_velocity_profile(
            controller_cfg.v_max,
            controller_cfg.w_max,
            controller_cfg.acc_lin,
            controller_cfg.acc_ang,
        )

        # Initialize
        self.vel_odom: Optional[np.ndarray] = None
        self.xyt_filtered: Optional[np.ndarray] = None
        self.xyt_goal: Optional[np.ndarray] = None

        self.active = False
        self.is_done = True
        self.controller_finished = True
        self.done_since = rospy.Time(0)
        self.track_yaw = True
        self.goal_set_t = rospy.Time(0)

        # Visualizations
        self.goal_visualizer = Visualizer("goto_controller/goal_abs")

    def _set_v_max(self, msg):
        self.controller.update_velocity_profile(v_max=msg.data)

    def _set_w_max(self, msg):
        self.controller.update_velocity_profile(w_max=msg.data)

    def _set_acc_lin(self, msg):
        self.controller.update_velocity_profile(acc_lin=msg.data)

    def _set_acc_ang(self, msg):
        self.controller.update_velocity_profile(acc_ang=msg.data)

    def _pose_update_callback(self, msg: PoseStamped):
        pose_sp = sp.SE3(matrix_from_pose_msg(msg.pose))
        self.xyt_filtered = sophus2xyt(pose_sp)
        if not self.odom_only:
            self.controller.update_pose_feedback(self.xyt_filtered)

    def _odom_update_callback(self, msg: Odometry):
        pose_sp = sp.SE3(matrix_from_pose_msg(msg.pose.pose))
        self.vel_odom = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])
        if self.odom_only:
            self.controller.update_pose_feedback(sophus2xyt(pose_sp))

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

        if self.active:
            pose_goal = pose_sp

            self.controller.update_goal(sophus2xyt(pose_goal))
            self.xyt_goal = self.controller.xyt_goal

            self.is_done = False
            self.goal_set_t = rospy.Time.now()
            self.controller_finished = False

            # Visualize
            self.goal_visualizer(pose_goal.matrix())

        # Do not update goal if controller is not active (prevents _enable_service to suddenly start moving the robot)
        else:
            log.warn("Received a goal while NOT active. Goal is not updated.")

    def _enable_service(self, request: TriggerRequest) -> TriggerResponse:
        """activates the controller and acks activation request"""
        self.xyt_goal = None
        self.active = True
        return TriggerResponse(
            success=True,
            message="Goto controller is now RUNNING",
        )

    def _disable_service(self, request: TriggerRequest) -> TriggerResponse:
        """disables the controller and acks deactivation request"""
        self.active = False
        self.xyt_goal = None
        return TriggerResponse(
            success=True,
            message="Goto controller is now STOPPED",
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
                self.is_done = False
                v_cmd, w_cmd = self.controller.compute_control()
                done = self.controller.is_done()

                # Compute timeout
                time_since_goal_set = (rospy.Time.now() - self.goal_set_t).to_sec()
                if self.controller.timeout(time_since_goal_set):
                    done = True
                    v_cmd, w_cmd = 0, 0

                # Check if actually done (velocity = 0)
                if done and self.vel_odom is not None:
                    if (
                        self.vel_odom[0] < VEL_THRESHOlD
                        and self.vel_odom[1] < RVEL_THRESHOLD
                    ):
                        if not self.controller_finished:
                            self.controller_finished = True
                            self.done_since = rospy.Time.now()
                        elif (
                            self.controller_finished
                            and (rospy.Time.now() - self.done_since) > self.done_t
                        ):
                            self.is_done = True
                    else:
                        self.controller_finished = False
                        self.done_since = rospy.Time(0)

                    if DEBUG_CONTROL_LOOP:
                        print(
                            "done =",
                            done,
                            "vel =",
                            self.vel_odom,
                            "controller done =",
                            self.controller_finished,
                            "is done =",
                            self.is_done,
                        )

                # Command robot
                self._set_velocity(v_cmd, w_cmd)
                self.at_goal_pub.publish(self.is_done)

                if self.is_done:
                    self.active = False
                    self.xyt_goal = None

            # Spin
            rate.sleep()

    def main(self):
        # ROS comms
        rospy.init_node("goto_controller")

        self.vel_command_pub = rospy.Publisher("stretch/cmd_vel", Twist, queue_size=1)
        self.at_goal_pub = rospy.Publisher(
            "goto_controller/at_goal", Bool, queue_size=1
        )

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

        # Create individual subscribers
        rospy.Subscriber("goto_controller/v_max", Float32, self._set_v_max)
        rospy.Subscriber("goto_controller/w_max", Float32, self._set_w_max)
        rospy.Subscriber("goto_controller/acc_lin", Float32, self._set_acc_lin)
        rospy.Subscriber("goto_controller/acc_ang", Float32, self._set_acc_ang)

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
