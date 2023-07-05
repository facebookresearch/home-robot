# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os

from config_utils import get_habitat_config, get_ovmm_baseline_config
from evaluator import OVMMEvaluator
from omegaconf import DictConfig, OmegaConf

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def merge_configs(habitat_config, baseline_config):
    config = DictConfig({**habitat_config, **baseline_config})

    visualize = config.VISUALIZE or config.PRINT_IMAGES
    if not visualize:
        if "robot_third_rgb" in config.habitat.gym.obs_keys:
            config.habitat.gym.obs_keys.remove("robot_third_rgb")
        if "third_rgb_sensor" in config.habitat.simulator.agents.main_agent.sim_sensors:
            config.habitat.simulator.agents.main_agent.sim_sensors.pop(
                "third_rgb_sensor"
            )

    episode_ids_range = config.habitat.dataset.episode_indices_range
    if episode_ids_range is not None:
        config.EXP_NAME = os.path.join(
            config.EXP_NAME, f"{episode_ids_range[0]}_{episode_ids_range[1]}"
        )

    OmegaConf.set_readonly(config, True)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, default="local", choices=["local", "remote"]
    )
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/agent/hssd_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "overrides",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    # get habitat config
    habitat_config_path = os.environ.get(
        "CHALLENGE_CONFIG_FILE", args.habitat_config_path
    )
    habitat_config, _ = get_habitat_config(
        habitat_config_path, overrides=args.overrides
    )

    # get baseline config
    baseline_config = get_ovmm_baseline_config(args.baseline_config_path)

    # merge habitat and baseline configs
    eval_config = merge_configs(habitat_config, baseline_config)

    # create agent
    agent = OpenVocabManipAgent(eval_config)

    # create evaluator
    evaluator = OVMMEvaluator(eval_config)
    evaluator.eval(
        agent,
        num_episodes_per_env=eval_config.EVAL_VECTORIZED.num_episodes_per_env,
    )
