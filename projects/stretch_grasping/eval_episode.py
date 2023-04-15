#!/usr/bin/env python
from typing import Optional, Tuple

import click
import rospy
from config_utils import get_config

# BELOW WILL BE REMOVED!!!!
from habitat_baselines.config.default import _BASELINES_CFG_DIR
from habitat_baselines.config.default import get_config as get_habitat_config
from omegaconf import DictConfig, OmegaConf


def get_config(
    path: str, opts: Optional[list] = None, configs_dir: str = _BASELINES_CFG_DIR
) -> Tuple[DictConfig, str]:
    config = get_habitat_config(path, overrides=opts, configs_dir=configs_dir)
    return config, ""


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

    # TODO: WILL BE REMOVED!!!!!
    habitat_config, config_str = get_config("rearrange/modular_nav.yaml")
    OmegaConf.set_readonly(config, True)

    config = DictConfig({**config, **habitat_config})
    rospy.init_node("eval_episode_stretch_objectnav")
    agent = PickAndPlaceAgent(
        config=config, skip_find_object=test_pick, skip_place=test_pick
    )
    env = StretchPickandPlaceEnv(config=config, test_grasping=test_pick)

    agent.reset()
    env.reset("table", "cup", "chair")

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
