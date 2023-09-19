# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState, Joy

from home_robot_hw.remote import StretchClient
from home_robot_hw.remote.ros import StretchRosInterface
from home_robot_hw.teleop.stretch_xbox_controller_teleop import (
    manage_base,
    manage_end_of_arm,
    manage_head,
    manage_lift_arm,
    set_use_dex_wrist_mapping,
)


class StretchXboxController(object):
    def __init__(
        self,
        model,
        on_first_joystick_input=None,
        start_button_callback=None,
        back_button_callback=None,
    ):
        self._robot_client = StretchClient()
        self._robot_client.switch_to_manipulation_mode()
        self._robot = StretchRosInterface(init_cameras=False, depth_buffer_size=1)
        self._on_first_joystick_input = on_first_joystick_input
        self._joystick_subscriber = rospy.Subscriber(
            "joy", Joy, self._joystick_callback, queue_size=1
        )
        self._start_button_callback = start_button_callback
        self._back_button_callback = back_button_callback

        self._extend_arm_timer = None
        self._lift_arm_timer = None
        self._wrist_yaw_timer = None
        self._wrist_roll_timer = None
        self._wrist_pitch_timer = None
        self._gripper_timer = None
        self._head_pan_timer = None
        self._head_tilt_timer = None
        self._move_base_timer = None

        self._dpad_controls_camera = False  # #True
        set_use_dex_wrist_mapping(
            not self._dpad_controls_camera
        )  # So we can get roll and pitch

        self._base_motion_off = False

    def _convert_joy_msg_to_xbox_state(self, joy_msg):
        state = {
            "left_stick_x": -joy_msg.axes[0],
            "left_stick_y": joy_msg.axes[1],
            "left_trigger_pulled": joy_msg.axes[2],
            "right_stick_x": -joy_msg.axes[
                3
            ],  # Negative is more intuitive for where I typically sit relative to the robot ... (TODO)
            "right_stick_y": joy_msg.axes[4],
            "right_trigger_pulled": joy_msg.axes[5],
            # These get used as booleans. TODO: threshold currently somewhat arbitrary
            "right_pad_pressed": joy_msg.axes[6] < -0.5,
            "left_pad_pressed": joy_msg.axes[6] > 0.5,
            "top_pad_pressed": joy_msg.axes[7] > 0.5,
            "bottom_pad_pressed": joy_msg.axes[7] < -0.5,
            "bottom_button_pressed": joy_msg.buttons[0],
            "right_button_pressed": joy_msg.buttons[1],
            "left_button_pressed": joy_msg.buttons[2],
            "top_button_pressed": joy_msg.buttons[3],
            "left_shoulder_button_pressed": joy_msg.buttons[4],
            "right_shoulder_button_pressed": joy_msg.buttons[5],
            "back_button_pressed": joy_msg.buttons[6],  # TODO: check axes
            "start_button_pressed": joy_msg.buttons[7],
        }  # TODO: check axes
        # TODO: start_button_pressed not in original output

        """state = {'middle_led_ring_button_pressed': self.middle_led_ring_button.pressed,
                 'left_stick_x': self.left_stick.x,
                 'left_stick_y': self.left_stick.y,
                 'right_stick_x': self.right_stick.x,
                 'right_stick_y': self.right_stick.y,
                 'left_stick_button_pressed': self.left_stick_button.pressed,
                 'right_stick_button_pressed': self.right_stick_button.pressed,
                 'bottom_button_pressed': self.bottom_button.pressed,
                 'top_button_pressed': self.top_button.pressed,
                 'left_button_pressed': self.left_button.pressed,
                 'right_button_pressed': self.right_button.pressed,
                 'left_shoulder_button_pressed': self.left_shoulder_button.pressed,
                 'right_shoulder_button_pressed': self.right_shoulder_button.pressed,
                 'select_button_pressed': self.select_button.pressed,
                 'start_button_pressed': self.start_button.pressed,
                 'left_trigger_pulled': self.left_trigger.pulled,
                 'right_trigger_pulled': self.right_trigger.pulled,
                 'bottom_pad_pressed': self.bottom_pad.pressed,
                 'top_pad_pressed': self.top_pad.pressed,
                 'left_pad_pressed': self.left_pad.pressed,
                 'right_pad_pressed': self.right_pad.pressed}"""
        return state

    def _set_mode(self) -> None:
        """If the robot is not in position mode, make sure that it is."""
        if not self._robot_client.in_navigation_mode():
            print("--> Switching to navigation mode")
            self._robot_client.switch_to_navigation_mode()

    def _create_arm_extension_loop(self, controller_state):
        arm_scale = 0.2  # TODO: better config/less hacky

        def callback(event):
            # Re-run the manager because it uses globals to accumulate speed
            converted_lift_command, converted_arm_command = manage_lift_arm(
                robot=None, controller_state=controller_state
            )
            self._robot_client._ros_client.goto_arm_position(
                arm_scale * converted_arm_command[0], wait=True
            )

        return callback

    def _create_lift_arm_loop(self, controller_state):
        lift_scale = 0.2  # TODO: better config/less hacky

        def callback(event):
            # Re-run the manager because it uses globals to accumulate speed
            converted_lift_command, converted_arm_command = manage_lift_arm(
                robot=None, controller_state=controller_state
            )
            self._robot_client._ros_client.goto_lift_position(
                lift_scale * converted_lift_command[0], wait=True
            )

        return callback

    def _create_move_base_loop(self, controller_state):
        rot_scale = 0.10
        trans_scale = 0.5

        def callback(event):
            # Re-run the manager because it uses globals to accumulate speed
            translation_command, rotation_command = manage_base(
                robot=None, controller_state=controller_state
            )
            # Execute the commands
            if translation_command is not None:
                self._set_mode()
                self._robot_client.nav.set_velocity(
                    trans_scale * translation_command[1], 0
                )

            if rotation_command is not None:
                self._set_mode()
                self._robot_client.nav.set_velocity(0, rot_scale * rotation_command[1])

        return callback

    def _create_wrist_yaw_loop(self, controller_state):
        def callback(event):
            # Re-run the manager because it uses globals to accumulate speed
            (
                wrist_yaw_command,
                wrist_roll_command,
                wrist_pitch_command,
                gripper_command,
            ) = manage_end_of_arm(robot=None, controller_state=controller_state)
            if wrist_yaw_command[0] != 0:
                self._robot_client._ros_client.goto_wrist_yaw_position(
                    wrist_yaw_command[0], wait=True
                )

        return callback

    def _create_wrist_roll_loop(self, controller_state):
        def callback(event):
            # Re-run the manager because it uses globals to accumulate speed
            (
                wrist_yaw_command,
                wrist_roll_command,
                wrist_pitch_command,
                gripper_command,
            ) = manage_end_of_arm(robot=None, controller_state=controller_state)
            if wrist_roll_command[0] != 0:
                self._robot_client._ros_client.goto_wrist_roll_position(
                    wrist_roll_command[0], wait=True
                )

        return callback

    def _create_wrist_pitch_loop(self, controller_state):
        def callback(event):
            # Re-run the manager because it uses globals to accumulate speed
            (
                wrist_yaw_command,
                wrist_roll_command,
                wrist_pitch_command,
                gripper_command,
            ) = manage_end_of_arm(robot=None, controller_state=controller_state)
            if wrist_pitch_command[0] != 0:
                self._robot_client._ros_client.goto_wrist_pitch_position(
                    wrist_pitch_command[0], wait=True
                )

        return callback

    def _create_gripper_loop(self, controller_state):
        def callback(event):
            (
                wrist_yaw_command,
                wrist_roll_command,
                wrist_pitch_command,
                gripper_command,
            ) = manage_end_of_arm(robot=None, controller_state=controller_state)
            self._robot_client._ros_client.goto_gripper_position(
                gripper_command[0], wait=True
            )

        return callback

    def _create_head_pan_loop(self, controller_state):
        def callback(event):
            head_pan_command, head_tilt_command = manage_head(
                robot=None, controller_state=controller_state
            )
            self._robot_client._ros_client.goto_head_pan_position(
                head_pan_command[0], wait=True
            )

        return callback

    def _create_head_tilt_loop(self, controller_state):
        def callback(event):
            head_pan_command, head_tilt_command = manage_head(
                robot=None, controller_state=controller_state
            )
            self._robot_client._ros_client.goto_head_tilt_position(
                head_tilt_command[0], wait=True
            )

        return callback

    def _joystick_callback(self, msg):  # noqa
        callback_hz = 30

        if self._on_first_joystick_input is not None:
            self._on_first_joystick_input()

        # Cancel all in-progress looped actions (they'll trigger again here if relevant)
        if self._extend_arm_timer is not None:
            self._extend_arm_timer.shutdown()

        if self._lift_arm_timer is not None:
            self._lift_arm_timer.shutdown()

        if self._wrist_yaw_timer is not None:
            self._wrist_yaw_timer.shutdown()

        if self._wrist_roll_timer is not None:
            self._wrist_roll_timer.shutdown()

        if self._wrist_pitch_timer is not None:
            self._wrist_pitch_timer.shutdown()

        if self._gripper_timer is not None:
            self._gripper_timer.shutdown()

        if self._head_pan_timer is not None:
            self._head_pan_timer.shutdown()

        if self._head_tilt_timer is not None:
            self._head_tilt_timer.shutdown()

        if self._move_base_timer is not None:
            self._move_base_timer.shutdown()

        # Get the new relevant commands
        controller_state = self._convert_joy_msg_to_xbox_state(msg)

        if controller_state["left_trigger_pulled"] < -0.5:
            self._dpad_controls_camera = not self._dpad_controls_camera
            set_use_dex_wrist_mapping(not self._dpad_controls_camera)
            print(
                f"Dpad controlling camera? {self._dpad_controls_camera} (If false, controls the wrist)"
            )

        translation_command, rotation_command = None, None
        if not self._base_motion_off:
            translation_command, rotation_command = manage_base(
                robot=None, controller_state=controller_state
            )

        converted_lift_command, converted_arm_command = manage_lift_arm(
            robot=None, controller_state=controller_state
        )
        (
            wrist_yaw_command,
            wrist_roll_command,
            wrist_pitch_command,
            gripper_command,
        ) = manage_end_of_arm(robot=None, controller_state=controller_state)
        head_pan_command, head_tilt_command = None, None

        if self._dpad_controls_camera:
            head_pan_command, head_tilt_command = manage_head(
                robot=None, controller_state=controller_state
            )

        # Execute the commands
        if translation_command is not None:
            self._set_mode()
            self._robot_client.nav.set_velocity(translation_command[1], 0)

        if rotation_command is not None:
            self._set_mode()
            self._robot_client.nav.set_velocity(0, rotation_command[1])

        # These are in loops because it feels more natural to hold the button in these cases rather than press it repeatedly
        # Since the callback only fires when there is a state change for these, we have to intentionally loop them
        # to achieve the desired effect.

        if translation_command or rotation_command is not None:
            self._set_mode()
            self._move_base_timer = rospy.Timer(
                rospy.Duration(1 / callback_hz),
                self._create_move_base_loop(controller_state),
                oneshot=False,
            )
        if converted_lift_command is not None:
            self._lift_arm_timer = rospy.Timer(
                rospy.Duration(1 / callback_hz),
                self._create_lift_arm_loop(controller_state),
                oneshot=False,
            )

        if converted_arm_command is not None:
            self._extend_arm_timer = rospy.Timer(
                rospy.Duration(1 / callback_hz),
                self._create_arm_extension_loop(controller_state),
                oneshot=False,
            )

        if wrist_yaw_command is not None:
            self._wrist_yaw_timer = rospy.Timer(
                rospy.Duration(1 / callback_hz),
                self._create_wrist_yaw_loop(controller_state),
                oneshot=False,
            )

        if wrist_roll_command is not None:
            self._wrist_roll_timer = rospy.Timer(
                rospy.Duration(1 / callback_hz),
                self._create_wrist_roll_loop(controller_state),
                oneshot=False,
            )

        if wrist_pitch_command is not None:
            self._wrist_pitch_timer = rospy.Timer(
                rospy.Duration(1 / callback_hz),
                self._create_wrist_pitch_loop(controller_state),
                oneshot=False,
            )

        if gripper_command is not None:
            self._gripper_timer = rospy.Timer(
                rospy.Duration(1 / callback_hz),
                self._create_gripper_loop(controller_state),
                oneshot=False,
            )

        if head_pan_command is not None:
            self._robot_client._ros_client.goto_head_pan_position(head_pan_command[0])

        if head_tilt_command is not None:
            self._robot_client._ros_client.goto_head_tilt_position(head_tilt_command[0])

        if (
            controller_state["start_button_pressed"]
            and self._start_button_callback is not None
        ):
            self._start_button_callback()

        if (
            controller_state["back_button_pressed"]
            and self._back_button_callback is not None
        ):
            self._back_button_callback()
