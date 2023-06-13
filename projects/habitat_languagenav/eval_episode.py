import argparse
import json
import sys
from pathlib import Path

# TODO Install home_robot, home_robot_sim and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot_sim"),
)

from config_utils import get_config
from habitat.core.env import Env

from home_robot.agent.languagenav_agent.languagenav_agent import LanguageNavAgent
from home_robot_sim.env.habitat_languagenav_env.habitat_languagenav_env import (
    HabitatLanguageNavEnv,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="languagenav/modular_languagenav_hm3d.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/habitat_languagenav/configs/agent/hm3d_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    print("Arguments:")
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    print("-" * 100)

    config = get_config(args.habitat_config_path, args.baseline_config_path)

    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.habitat.dataset.split = "val"
    config.EXP_NAME = "debug"

    agent = LanguageNavAgent(config=config)
    habitat_env = Env(config)
    env = HabitatLanguageNavEnv(habitat_env, config=config)

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
