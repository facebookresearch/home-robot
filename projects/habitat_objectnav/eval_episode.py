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
from omegaconf import  OmegaConf, DictConfig

if __name__ == "__main__":
    config_path = "rearrange/modular_nav.yaml"
    config, config_str = get_config(config_path)
    OmegaConf.set_readonly(config, False)

    config.habitat_baselines.num_environments = 1
    OmegaConf.set_struct(config.habitat_baselines,  False)
    config.habitat_baselines.print_images = 1
    config.habitat.dataset.split = "val"
    config.habitat_baselines.exp_name = "debug"

    OmegaConf.set_readonly(config, True)
    baseline_config = OmegaConf.load('projects/habitat_objectnav/configs/agent/floorplanner_eval.yaml')
    config = DictConfig({**config, **baseline_config})

    agent = ObjectNavAgent(config=config)
    env = HabitatObjectNavEnv(Env(config=config.habitat), config=config)

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
