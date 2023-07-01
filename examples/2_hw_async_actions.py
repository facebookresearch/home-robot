# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time

import numpy as np

from home_robot.motion.stretch import STRETCH_HOME_Q, HelloStretchKinematics
from home_robot_hw.remote import StretchClient

if __name__ == "__main__":
    robot = StretchClient()

    # Async navigation
    robot.switch_to_navigation_mode()

    robot.nav.navigate_to([1, 0, 0], blocking=False)
    time.sleep(2)
    robot.nav.navigate_to([0.2, 0.5, 0], blocking=False)  # update goal
    robot.nav.wait()  # wait for nav actions to complete

    # Async nav and head motion
    robot.nav.navigate_to([0, 0, 0], blocking=False)
    robot.head.look_at_ee()
    robot.wait()  # wait for all actions to complete

    # Async manip and head motion
    robot.switch_to_manipulation_mode()

    pos_desired = np.array([0.2, -0.2, 0.4])
    robot.manip.goto_ee_pose(pos_desired, relative=True, blocking=False)
    robot.head.look_at_ee()
