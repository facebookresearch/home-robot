#!/usr/bin/env python
import rospy
from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
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

    rospy.init_node('eval_episode_stretch_objectnav')
    agent = ObjectNavAgent(config=config)
    # env = HabitatObjectNavEnv(Env(config=config.TASK_CONFIG), config=config)
    env = StretchObjectNavEnv()

    agent.reset()
    env.reset()

    t = 0
    while not env.episode_over:
        t += 1
        print(t)
        obs = env.get_observation()
        action, info = agent.act(obs)
        env.apply_action(action, info=info)

    print(env.get_episode_metrics())
