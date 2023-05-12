import rospy
from slap_manipulation.agents.language_ovmm_agent import LangAgent

from home_robot_hw.env.stretch_pick_and_place_env import StretchPickandPlaceEnv


def main():
    TASK = "bring me a cup from the table"
    rospy.init_node("eval_episode_lang_ovmm")

    env = StretchPickandPlaceEnv()
    agent = LangAgent()
    robot = env.get_robot()

    agent.reset()
    obs = env.reset()

    t = 0
    while not env.episode_over:
        t += 1
        print("STEP =", t)
        obs = env.get_observation()
        steps = agent.get_steps(TASK)
        for step in steps:
            action, info = agent.act(obs, step)
            obs = env.apply_action(action, info=info)
            if "complex_action" in info and info["complex_action"]:
                while agent.is_busy():
                    action, info = agent.act(obs, step)
                    obs = env.apply_action(action, info=info)
