# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import json
import sys
from pathlib import Path

from config_utils import get_config
from omegaconf import DictConfig, OmegaConf

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot_sim"),
)
from evaluator import OVMMEvaluator
from habitat import make_dataset
from habitat.core.environments import get_env_class
from habitat.utils.gym_definitions import _get_env_name

from home_robot.agent.ovmm_agent.ovmm_llm_agent import OvmmLLMAgent
from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import (
    HabitatOpenVocabManipEnv,
)


def create_ovmm_env_fn(config):
    """Create habitat environment using configsand wrap HabitatOpenVocabManipEnv around it. This function is used by VectorEnv for creating the individual environments"""
    habitat_config = config.habitat
    dataset = make_dataset(habitat_config.dataset.type, config=habitat_config.dataset)
    env_class_name = _get_env_name(config)
    env_class = get_env_class(env_class_name)
    habitat_env = env_class(config=habitat_config, dataset=dataset)
    habitat_env.seed(habitat_config.seed)
    env = HabitatOpenVocabManipEnv(habitat_env, config, dataset=dataset)
    return env


class MemoryCollector(OVMMEvaluator):
    """Class for creating vectorized environments, evaluating OpenVocabManipAgent on an episode dataset and returning metrics"""

    def __init__(self, config):
        super().__init__(config)

    def collect(self, num_episodes_per_env=1):
        """Return a dict mapping each timestep to the clip features of detected objects in the current frame"""
        self._init_envs(
            config=self.config, is_eval=True, make_env_fn=create_ovmm_env_fn
        )
        agent = OvmmLLMAgent(
            config=self.config,
            obs_spaces=self.envs.observation_spaces,
            action_spaces=self.envs.orig_action_spaces,
        )
        self._eval(
            agent,
            self.envs,
            num_episodes_per_env=num_episodes_per_env,
        )
        return agent.memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/agent/memory_collection_single_episode.yaml",
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

    print("Configs:")
    config, config_str = get_config(args.habitat_config_path, opts=args.opts)
    baseline_config = OmegaConf.load(args.baseline_config_path)
    config = DictConfig({**config, **baseline_config})
    collector = MemoryCollector(config)

    result = collector.collect(
        num_episodes_per_env=config.EVAL_VECTORIZED.num_episodes_per_env,
    )
    with open("datadump/memory_data.json", "w") as json_file:
        json.dump(result, json_file)
