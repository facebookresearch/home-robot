#!/usr/bin/env python3

"""
Copyright (c) 2011, Willow Garage, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Willow Garage, Inc. nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA, OR PROFITS OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import rospy
import copy
import os
import numpy as np

# Our imports
from home_robot.hw.ros.stretch_ros import HelloStretchROSInterface
from home_robot.hw.ros.path import get_package_path
from home_robot.hw.ros.utils import to_normalized_quaternion_msg, matrix_from_pose_msg
from home_robot.agent.motion.robot import (
    PLANNER_STRETCH_URDF,
    STRETCH_TO_GRASP,
    STRETCH_HOME_Q,
    STRETCH_PREGRASP_Q,
    HelloStretchIdx,
)
from home_robot.utils.data_tools.writer import DataWriter
from home_robot.utils.pose import to_matrix

from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from visualization_msgs.msg import *
from geometry_msgs.msg import Point, Quaternion
from tf.broadcaster import TransformBroadcaster
import tf.transformations as tra

from random import random
from math import sin
from threading import Lock


class InteractiveMarkerManager(object):
    """Wrap interactive marker script into a class for simplicity and cleanliness"""

    def __init__(self, robot, allow_base_motion=False):
        """Takes robot as an argument - in case we want to do IK and actually move the robot
        to different positions interactively.

        In the future this should use Austin's API"""
        self.robot = robot
        self.allow_base_motion = allow_base_motion
        self.model = self.robot.get_model()

        # Set up some teleop tools
        self.menu_handler = MenuHandler()
        self.menu_handler.insert("Stow the arm", callback=self._cb_stow)
        self.menu_handler.insert("Raise the arm", callback=self._cb_raise)
        self.menu_handler.insert("Look straight", callback=self._cb_look_straight)
        self.menu_handler.insert("Look forward", callback=self._cb_look_front)
        self.menu_handler.insert("Look at gripper", callback=self._cb_look_at_ee)
        self.menu_handler.insert("Open gripper", callback=self._cb_open_ee)
        self.menu_handler.insert("Close gripper", callback=self._cb_close_ee)
        self.menu_handler.insert("Go To Marker", callback=self._cb_move_to_marker)

        self.br = TransformBroadcaster()
        self._pose_lock = Lock()
        self._cmd_lock = Lock()

        # Track the pose for where we're currently commanding the robot
        self.pose = None

        self.server = InteractiveMarkerServer("demo_control")
        rate = rospy.Rate(10)

        pose_mat = None
        while not rospy.is_shutdown() and pose_mat is None:
            print("getting pose...")
            rate.sleep()
            pose_mat = rob.get_pose(
                frame="link_straight_gripper", base_frame="base_link"
            )
            pose_mat = pose_mat @ STRETCH_TO_GRASP

        position = Point(*pose_mat[:3, 3])
        orientation = Quaternion(*tra.quaternion_from_matrix(pose_mat))
        self.make6DofMarker(
            False, InteractiveMarkerControl.NONE, position, orientation, True
        )
        self.server.applyChanges()

    def check_switch_to_position_mode(self):
        """only switch if necessary"""
        if not self.robot.in_position_mode():
            self.switch_to_position_mode()

    def _cb_move_to_marker(self, feedback):
        with self._cmd_lock:
            self.check_switch_to_position_mode()
            q0, _ = self.robot.update()
            with self._pose_lock:
                if self.pose is None:
                    return
                q = self.model.lift_arm_ik_from_matrix(self.pose, q0)
                print("Attempting to move...")
                print(self.pose)
                print("q =", q)
            if q is not None:
                self.robot.goto(q, move_base=False, wait=False)

    def _cb_open_ee(self, msg):
        print("Opening the gripper")
        with self._cmd_lock:
            q, _ = self.robot.update()
            q = self.model.update_gripper(q, open=True)
            self.robot.goto(q, move_base=False, wait=False)

    def _cb_close_ee(self, msg):
        print("Closing the gripper")
        with self._cmd_lock:
            q, _ = self.robot.update()
            q = self.model.update_gripper(q, open=False)
            self.robot.goto(q, move_base=False, wait=False)

    def _cb_stow(self, msg):
        print("Stowing the robot arm")
        with self._cmd_lock:
            q, _ = self.robot.update()
            q[HelloStretchIdx.ARM] = 0
            self.robot.goto(q, move_base=False, wait=False)
            rospy.sleep(1.0)
            self.robot.goto(STRETCH_HOME_Q, move_base=False, wait=False)

    def _cb_raise(self, msg):
        print("Raising the robot arm")
        with self._cmd_lock:
            q, _ = self.robot.update()
            q[HelloStretchIdx.ARM] = 0
            q[HelloStretchIdx.LIFT] = 0.5
            self.robot.goto(q, move_base=False, wait=False)
            rospy.sleep(1.0)
            q = STRETCH_PREGRASP_Q.copy()
            q = self.model.update_look_at_ee(q)
            self.robot.goto(q, move_base=False, wait=False)

    def _cb_look_at_ee(self, msg):
        print("Looking at the arm ee")
        with self._cmd_lock:
            q, _ = self.robot.update()
            q = self.model.update_look_at_ee(q)
            self.robot.goto(q, move_base=False, wait=False)

    def _cb_look_straight(self, msg):
        print("Looking straight ahead")
        with self._cmd_lock:
            q, _ = self.robot.update()
            q = self.model.update_look_ahead(q)
            self.robot.goto(q, move_base=False, wait=False)

    def _cb_look_front(self, msg):
        print("Looking at the front")
        with self._cmd_lock:
            q, _ = self.robot.update()
            q = self.model.update_look_front(q)
            self.robot.goto(q, move_base=False, wait=False)

    #####################################################################
    # Utils from willow garage
    def processFeedback(self, feedback):
        s = "Feedback from marker '" + feedback.marker_name
        s += "' / control '" + feedback.control_name + "'"

        mp = ""
        if feedback.mouse_point_valid:
            mp = " at " + str(feedback.mouse_point.x)
            mp += ", " + str(feedback.mouse_point.y)
            mp += ", " + str(feedback.mouse_point.z)
            mp += " in frame " + feedback.header.frame_id

        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            rospy.loginfo(s + ": button click" + mp + ".")
        elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
            rospy.loginfo(
                s + ": menu item " + str(feedback.menu_entry_id) + " clicked" + mp + "."
            )
        elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            rospy.loginfo(s + ": pose changed")
            pose = matrix_from_pose_msg(feedback.pose)
            if self.verbose:
                print("\nMarker moved to:")
                print(pose)
            with self._pose_lock:
                self.pose = pose
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
            rospy.loginfo(s + ": mouse down" + mp + ".")
        elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            rospy.loginfo(s + ": mouse up" + mp + ".")
        self.server.applyChanges()

    #####################################################################
    # Marker Creation

    def make6DofMarker(
        self,
        fixed,
        interaction_mode,
        position,
        orientation,
        show_6dof=False,
        allow_base_motion=False,
    ):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.pose.position = position
        int_marker.pose.orientation = orientation
        int_marker.scale = 0.25

        int_marker.name = "simple_6dof"
        int_marker.description = "Simple 6-DOF Control"

        if fixed:
            int_marker.name += "_fixed"
            int_marker.description += "\n(fixed orientation)"

        if interaction_mode != InteractiveMarkerControl.NONE:
            control_modes_dict = {
                InteractiveMarkerControl.MOVE_3D: "MOVE_3D",
                InteractiveMarkerControl.ROTATE_3D: "ROTATE_3D",
                InteractiveMarkerControl.MOVE_ROTATE_3D: "MOVE_ROTATE_3D",
            }
            int_marker.name += "_" + control_modes_dict[interaction_mode]
            int_marker.description = "3D Control"
            if show_6dof:
                int_marker.description += " + 6-DOF controls"
            int_marker.description += "\n" + control_modes_dict[interaction_mode]

        if show_6dof:
            control = InteractiveMarkerControl()
            control.orientation = to_normalized_quaternion_msg(1, 1, 0, 0)
            control.name = "rotate_x"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation = to_normalized_quaternion_msg(1, 1, 0, 0)
            control.name = "move_x"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation = to_normalized_quaternion_msg(1, 0, 1, 0)
            control.name = "rotate_z"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation = to_normalized_quaternion_msg(1, 0, 1, 0)
            control.name = "move_z"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation = to_normalized_quaternion_msg(1, 0, 0, 1)
            control.name = "rotate_y"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            # NOTE: If you want to add move-y, you can do it here.
            # But for a marker in the base coordinates of the robot, this might not make sense,
            # unless we allow the robot to "strafe."
            control = InteractiveMarkerControl()
            control.orientation = to_normalized_quaternion_msg(1, 0, 0, 1)
            control.name = "move_y"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

        self.server.insert(int_marker, self.processFeedback)
        self.menu_handler.apply(self.server, int_marker.name)


if __name__ == "__main__":
    rospy.init_node("basic_controls")

    # Create robot interface
    planner_urdf = os.path.join(get_package_path(), "..", PLANNER_STRETCH_URDF)
    rob = HelloStretchROSInterface(
        visualize_planner=False, init_cameras=False, urdf_path=planner_urdf
    )
    manager = InteractiveMarkerManager(rob)
    rospy.spin()
