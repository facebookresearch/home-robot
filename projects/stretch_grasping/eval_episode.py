#!/usr/bin/env python
import click
import rospy

from home_robot.agent.hierarchical.pick_and_place_agent import PickAndPlaceAgent
from home_robot.motion.stretch import STRETCH_HOME_Q
from home_robot.utils.config import get_config
from home_robot_hw.env.stretch_pick_and_place_env import StretchPickandPlaceEnv


@click.command()
@click.option("--test_pick", default=False, is_flag=True)
def main(test_pick=False):
    config_path = "projects/stretch_grasping/configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.EXP_NAME = "debug"
    config.freeze()

    rospy.init_node("eval_episode_stretch_objectnav")
    agent = PickAndPlaceAgent(
        config=config, skip_find_object=test_pick, skip_place=test_pick
    )
    env = StretchPickandPlaceEnv(config=config, test_grasping=test_pick)

    agent.reset()
    env.reset("bathtub", "cup", "chair")

    t = 0
    while not env.episode_over:
        t += 1
        print("STEP =", t)
        obs = env.get_observation()
        action, info = agent.act(obs)
        env.apply_action(action, info=info)

    print(env.get_episode_metrics())


if __name__ == "__main__":
    main()
