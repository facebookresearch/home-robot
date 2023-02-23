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

    def apply_action(self, action: Action, info: Optional[Dict[str, Any]] = None):
        if self.visualizer is not None:
            self.visualizer.visualize(**info)
        continuous_action = np.zeros(3)
        if action == DiscreteNavigationAction.MOVE_FORWARD:
            print("FORWARD")
            continuous_action[0] = self.forward_step
        elif action == DiscreteNavigationAction.TURN_RIGHT:
            print("TURN RIGHT")
            continuous_action[2] = -self.rotate_step
        elif action == DiscreteNavigationAction.TURN_LEFT:
            print("TURN LEFT")
            continuous_action[2] = self.rotate_step
        else:
            # Do nothing if "stop"
            # continuous_action = None
            # if not self.in_manipulation_mode():
            #     self.switch_to_manipulation_mode()
            pass

        if continuous_action is not None:
            if not self.in_navigation_mode():
                self.switch_to_navigation_mode()
            self.navigate_to(continuous_action, relative=True)
        print("-------")
        print(action)
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
    rob = StretchSimpleNavEnv()
    rob.switch_to_navigation_mode()

    # Debug the observation space
    import matplotlib.pyplot as plt

    while not rospy.is_shutdown():

        while not rospy.is_shutdown():
            cmd = None
            try:
                cmd = input("Enter a number 0-3:")
                cmd = DiscreteNavigationAction(int(cmd))
            except ValueError:
                cmd = None
            if cmd is not None:
                break
        rob.apply_action(cmd)

        obs = rob.get_observation()
        rgb, depth = obs.rgb, obs.depth
        # xyt = obs2xyt(obs.base_pose)

        # Add a visualiztion for debugging
        depth[depth > 5] = 0
        plt.subplot(121)
        plt.imshow(rgb)
        plt.subplot(122)
        plt.imshow(depth)
        # plt.subplot(133); plt.imshow(obs.semantic

        print()
        print("----------------")
        print("values:")
        print("RGB =", np.unique(rgb))
        print("Depth =", np.unique(depth))
        # print("XY =", xyt[:2])
        # print("Yaw=", xyt[-1])
        print("Compass =", obs.compass)
        print("Gps =", obs.gps)
        plt.show()


if False:
    observations = []
    obs = rob.get_observation()
    observations.append(obs)

    xyt = np.zeros(3)
    xyt[2] = obs.compass
    xyt[:2] = obs.gps
    # xyt = obs2xyt(obs.base_pose)
    xyt[0] += 0.1
    # rob.navigate_to(xyt)
    rob.rotate(0.2)
    rospy.sleep(10.0)
    obs = rob.get_observation()
    observations.append(obs)

    xyt[0] = 0
    # rob.navigate_to(xyt)
    rob.rotate(-0.2)
    rospy.sleep(10.0)
    obs = rob.get_observation()
    observations.append(obs)

    for obs in observations:
        rgb, depth = obs.rgb, obs.depth
        # xyt = obs2xyt(obs.base_pose)

        # Add a visualiztion for debugging
        depth[depth > 5] = 0
        plt.subplot(121)
        plt.imshow(rgb)
        plt.subplot(122)
        plt.imshow(depth)
        # plt.subplot(133); plt.imshow(obs.semantic

        print()
        print("----------------")
        print("values:")
        print("RGB =", np.unique(rgb))
        print("Depth =", np.unique(depth))
        # print("XY =", xyt[:2])
        # print("Yaw=", xyt[-1])
        print("Compass =", obs.compass)
        print("Gps =", obs.gps)
        plt.show()

