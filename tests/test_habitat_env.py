from pathlib import Path
import sys
import pytest
from subprocess import Popen

import habitat
from habitat.config.default import Config

from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot_sim.env.habitat_objectnav_env.habitat_objectnav_env import (
    HabitatObjectNavEnv,
)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "example/habitat_objectnav"),
)

TEST_NUM_STEPS = 3
CONFIG_NAME = "floorplanner_eval_test"


class DummyTestAgent(Agent):
    def reset(self):
        pass

    def act(self, obs):
        assert type(obs) is Observations
        return DiscreteNavigationAction.FORWARD, {}


def test_objectnav_env():
    config = Config()
    config.merge_from_file("configs/test_agent.yaml")
    task_config = Config()
    task_config.merge_from_other_cfg(habitat.config.default._C)
    task_config.merge_from_file("configs/test_task.yaml")
    config.TASK_CONFIG = task_config
    config.freeze()

    # Initialize agent & env
    agent = DummyTestAgent()
    env = HabitatObjectNavEnv(habitat.Env(config=config.TASK_CONFIG), config=config)

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
    # git clone https://github.com/facebookresearch/habitat-sim
    # python habitat-sim/src_python/habitat_sim/utils/datasets_download.py --uids mp3d_example_scene --data-path data/
    # mkdir data/scene_datasets/mp3d_example/mp3d
    # mv data/scene_datasets/mp3d_example/17DRP5sb8fy data/scene_datasets/mp3d_example/mp3d
    test_objectnav_env()
