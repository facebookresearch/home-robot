#!/usr/bin/env python
import rospy
import numpy as np
from time import sleep

from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.utils.config import get_config
from home_robot_hw.env.simple_navigation_env import StretchSimpleNavEnv

if __name__ == "__main__":
    rospy.init_node("eval_episode_stretch_objectnav")
    env = StretchSimpleNavEnv()

    env.reset()

    action = np.zeros(3)
    action[0] = 0.25
    if not env.in_navigation_mode():
        env.switch_to_navigation_mode()
    env.navigate_to(action, relative=True)
    input('---')
    sleep(2.0)

    # t = 0
    # while not env.episode_over:
        # t += 1
        # print("STEP =", t)
        # obs = env.get_observation()
        # action, info = agent.act(obs)
        # env.apply_action(action, info=info)
        # input("press enter for next action")

    # print(env.get_episode_metrics())

