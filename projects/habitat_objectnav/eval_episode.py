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

from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot_sim.env.habitat_objectnav_env.habitat_objectnav_env import (
    HabitatObjectNavEnv,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="objectnav/modular_objectnav_hm3d.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/habitat_objectnav/configs/agent/hm3d_eval.yaml",
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

    agent = ObjectNavAgent(config=config, device_id=0)
    env = HabitatObjectNavEnv(Env(config=config), config=config)

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
