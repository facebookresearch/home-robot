from typing import Any, Dict, Optional

import numpy as np
import rospy

import home_robot
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.utils.geometry import obs2xyt, sophus2obs
from home_robot_hw.env.stretch_abstract_env import StretchEnv

class StretchSimpleNavEnv(StretchEnv):
    """Create a detic-based object nav environment"""

    def __init__(
        self, config=None, forward_step=0.25, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.forward_step = forward_step  # in meters
        self.reset()

    def reset(self):
        self._current_pose = self.get_base_pose()

    def apply_action(self):
        continuous_action = np.zeros(3)
        continuous_action[0] = 0.25
        if continuous_action is not None:
            if not self.in_navigation_mode():
                self.switch_to_navigation_mode()
            self.navigate_to(continuous_action, relative=True)
        print("-------")
        print(continuous_action)
        rospy.sleep(5.0)

    def episode_over(self):
        return True

    def get_episode_metrics(self):
        return True

    def get_observation(self):
        return False


if __name__ == "__main__":
    # Create the robot
    print("--------------")
    print("Start example - hardware using ROS")
    rospy.init_node("hello_stretch_nav_test")
    print("Create ROS interface")
    env = StretchSimpleNavEnv()

    env.reset()
    env.apply_action()
    rospy.sleep(1.0)
    print("Confirm the robot has moved forward")
