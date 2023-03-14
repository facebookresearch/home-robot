#!/usr/bin/env python
import rospy

from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot.utils.config import get_config
from home_robot_hw.env.stretch_object_nav_env import StretchObjectNavEnv

if __name__ == "__main__":
    config_path = "configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.EXP_NAME = "debug"
    config.freeze()

    rospy.init_node("eval_episode_stretch_objectnav")
    agent = ObjectNavAgent(config=config)
    env = StretchObjectNavEnv(config=config)
    env.goto(STRETCH_HOME_Q)

    agent.reset()
    env.reset()

    t = 0
    while not env.episode_over:
        t += 1
        obs = env.get_observation()
        _, info = agent.act(obs)
        if t < 24:
            action = DiscreteNavigationAction.TURN_RIGHT
        else:
            action = DiscreteNavigationAction.STOP
        print("STEP =", t, "ACTION =", action)
        env.apply_action(action, info=info)

    print(env.get_episode_metrics())
