#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import rospy
from std_srvs.srv import Trigger


class SwitchToPositionMode(object):
    def __init__(self):
        self.switch = rospy.ServiceProxy("switch_to_position_mode", Trigger)
        print("Waiting for mode service...")
        self.switch.wait_for_service()
        print("Switching to position...", self.switch())

    def __call__(self):
        print("Switching to position...", self.switch())


if __name__ == "__main__":
    SwitchToPositionMode()
