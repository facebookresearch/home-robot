from pathlib import Path
import sys

# TODO Install home_robot, home_robot_sim and remove this
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent.parent / "src/home_robot"
    ),
)
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent.parent / "src/home_robot_sim"
    ),
)

from habitat.core.env import Env

from config_utils import get_config
from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot_sim.env.habitat_objectnav_env.habitat_objectnav_env import HabitatObjectNavEnv


if __name__ == "__main__":
    config_path = "configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.TASK_CONFIG.DATASET.SPLIT = "val"
    config.EXP_NAME = "debug"
    config.freeze()

    agent = ObjectNavAgent(config=config)
    env = HabitatObjectNavEnv(Env(config=config.TASK_CONFIG), config=config)

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
