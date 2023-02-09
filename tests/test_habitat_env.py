from pathlib import Path
import sys
import pytest
from subprocess import Popen

from hydra import compose, initialize
import habitat

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


class TestAgent(Agent):
    def reset(self):
        pass

    def act(self, obs):
        assert type(obs) is Observations
        return DiscreteNavigationAction.FORWARD, {}


@pytest.fixture
def download_dataset():
    p_sim = Popen(
        [
            "python",
            "-m",
            "habitat_sim.utils.datasets_download",
            "--uids" "habitat_test_pointnav_dataset" "--data-path",
            "data/",
        ]
    )
    p_sim.wait()


def test_objectnav_env(download_dataset):
    # Parse configuration
    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name=CONFIG_NAME)

    # Initialize agent & env
    agent = TestAgent()
    env = HabitatObjectNavEnv(
        habitat.Env(
            config=habitat.get_config(
                "benchmark/nav/objectnav/objectnav_hm3d_with_semantic.yaml"
            )
        ),
        config=cfg,
    )

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
