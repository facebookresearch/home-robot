# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Optional

import numpy as np
import rospy

import home_robot
from home_robot_hw.env.stretch_abstract_env import StretchEnv


class StretchSimpleNavEnv(StretchEnv):
    """Simple environment to move robot forward by 0.25m"""

    def __init__(self, forward_step=0.25, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.forward_step = forward_step  # in meters
        self.reset()

    def reset(self) -> None:
        """overriding reset function to get current pose of robot"""
        self._current_pose = self.get_base_pose()

    def apply_action(self) -> None:
        """
        method creates a 3-valued array which stores an SE2 navigation goal
            x, y, theta
        x is stored as forward_step and the robot is asked to move
        relative to current position
        """
        continuous_action = np.zeros(3)
        continuous_action[0] = self.forward_step
        if not self.in_navigation_mode():
            self.switch_to_navigation_mode()
        self.navigate_to(continuous_action, relative=True)
        print("-------")
        print(continuous_action)
        rospy.sleep(5.0)

    def episode_over(self) -> None:
        pass

    def get_episode_metrics(self) -> None:
        pass

    def get_observation(self) -> None:
        pass


if __name__ == "__main__":
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_nav_test")
    print("Create ROS interface")
    env = StretchSimpleNavEnv()

    env.apply_action()
    rospy.sleep(1.0)
    print("Confirm the robot has moved forward")
