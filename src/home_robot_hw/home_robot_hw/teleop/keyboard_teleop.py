# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time
from enum import Enum

import numpy as np
from pynput import keyboard as kb
from scipy.spatial.transform import Rotation as R

from home_robot_hw.remote import StretchClient

HZ_DEFAULT = 15

# Movement params
BASE_VEL_MAX = 0.20  # 6 * v_max + w_max <= 1.8  (computed from max wheel vel & vel diff required for w)
BASE_RVEL_MAX = 0.45
EE_DIFF = 0.03
WRIST_DIFF = 0.1


class ControlMode(Enum):
    NAV = 0
    MANIP = 1


class RobotController:
    def __init__(self, hz=HZ_DEFAULT):
        # Params
        self.dt = 1.0 / hz

        # Robot
        print("Connecting to robot...")
        self.robot = StretchClient()
        self.robot.switch_to_navigation_mode()
        print("Connected.")

        # Keyboard
        self.key_states = {}

        # Controller states
        self.alive = True
        self.mode = ControlMode.NAV

    def on_press(self, key):
        print(key)
        if key == kb.key.esc:
            self.alive = False
            return False  # returning False from a callback stops listener
        elif key == kb.key.space:
            self._set_mode((self.mode + 1 % 2))
        else:
            self.key_states[key] = 1

    def on_release(self, key):
        if key in self.key_states:
            self.key_states[key] = 0

    def run(self):
        print(
            "(+[__]o) Teleoperation started. ^o^"
            "-------------------------------------\n"
            "Enter:\t Switch modes\n"
            "Esc:\t End teleoperation\n"
            "\n"
            "== Navigation mode commands ==\n"
            "Up:\t Move forwards\n"
            "Down:\t Move backwards\n"
            "Left:\t Rotate counterclockwise\n"
            "Right:\t Rotate clockwise\n"
            "\n"
            "== Manipulation mode commands ==\n"
            "W:\t Raise arm upwards\n"
            "A:\t Strafe left\n"
            "S:\t Lower arm downwards\n"
            "D:\t Strafe right\n"
            "I:\t Extend arm\n"
            "J:\t Rotate gripper counterclockwise\n"
            "K:\t Retract arm\n"
            "L:\t Rotate gripper clockwise\n"
            "Space:\t Close gripper (when held)\n"
            "-------------------------------------\n"
        )
        self._set_mode(ControlMode.NAV)

        while self.alive:
            # Base
            if self.mode == ControlMode.NAV:
                vel = self._compute_net_command(kb.Key.up, kb.Key.down, BASE_VEL_MAX)
                rvel = self._compute_net_command(
                    kb.Key.left, kb.Key.right, BASE_RVEL_MAX
                )

                # Command robot
                print(vel, rvel)
                if vel or rvel:
                    self.robot.nav.set_velocity(vel, rvel)

            # Manipulator
            elif self.mode == ControlMode.MANIP:
                x = self._compute_net_command(
                    kb.KeyCode.from_char("a"), kb.KeyCode.from_char("d"), EE_DIFF
                )
                z = self._compute_net_command(
                    kb.KeyCode.from_char("w"), kb.KeyCode.from_char("s"), EE_DIFF
                )
                y = self._compute_net_command(
                    kb.KeyCode.from_char("k"), kb.KeyCode.from_char("i"), EE_DIFF
                )
                rz = self._compute_net_command(
                    kb.KeyCode.from_char("k"), kb.KeyCode.from_char("i"), WRIST_DIFF
                )

                # Command robot
                if x or y or z or rz:
                    self.robot.manip.goto_ee_pose(
                        [x, y, z],
                        R.from_rotvec([0, 0, rz]).as_quat(),
                        relative=True,
                        blocking=False,
                    )

            # Spin
            time.sleep(self.dt)

    def _compute_net_command(self, pos_key, neg_key, intensity):
        pos_val = self.key_states[pos_key] if pos_key in self.key_states else 0
        neg_val = self.key_states[neg_key] if neg_key in self.key_states else 0
        return intensity * (pos_val - neg_val)

    def _set_mode(self, mode):
        self.mode = mode
        if mode == ControlMode.NAV:
            self.robot.switch_to_navigation_mode()
            self.robot.head.look_ahead()
            print("IN NAVIGATION MODE")
        elif mode == ControlMode.MANIP:
            self.robot.switch_to_manipulation_mode()
            self.robot.head.look_at_ee()
            print("IN MANIPULATION MODE")


def run_teleop():
    robot_controller = RobotController()
    listener = kb.Listener(
        on_press=robot_controller.on_press,
        on_release=robot_controller.on_release,
        suppress=True,  # suppress terminal outputs
    )

    # Start teleop
    listener.start()
    robot_controller.run()

    # Cleanup
    listener.join()
    print("(+[__]o) Teleoperation ended. =_=")


if __name__ == "__main__":
    run_teleop()
