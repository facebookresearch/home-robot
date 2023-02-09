from pathlib import Path
import sys

from habitat.core.env import Env

from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observation
from home_robot_sim.env.habitat_objectnav_env.habitat_objectnav_env import (
    HabitatObjectNavEnv,
)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "example/habitat_objectnav"),
)
from config_utils import get_config

TEST_NUM_STEPS = 3
CONFIG_PATH = "floorplanner_eval_test.yaml"


class TestAgent(Agent):
    def reset(self):
        pass

    def act(self, obs):
        assert type(obs) is Observation
        return DiscreteNavigationAction.FORWARD, {}


def test_objectnav_env():
    # Parse configuration
    config_path = "configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 0
    config.TASK_CONFIG.DATASET.SPLIT = "val"
    config.EXP_NAME = "test"
    config.freeze()

    # Initialize agent & env
    agent = TestAgent()
    env = HabitatObjectNavEnv(Env(config=config.TASK_CONFIG), config=config)

    agent.reset()
    env.reset()

    # Run simulation for a few steps
    t = 0
    for _ in range(TEST_NUM_STEPS):
        t += 1
        print(t)
        obs = env.get_observation()
        action, info = agent.act(obs)
        env.apply_action(action, info=info)


if __name__ == "__main__":
    test_objectnav_env()
