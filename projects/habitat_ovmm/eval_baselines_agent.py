# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os

from evaluator import OVMMEvaluator
from utils.config_utils import (
    get_habitat_config,
    get_omega_config,
    merge_configs,
)

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot.agent.ovmm_agent.random_agent import RandomAgent

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation_type",
        type=str,
        choices=["local", "local_vectorized", "remote"],
        default="local",
    )
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/agent/heuristic_agent.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--env_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/agent/hssd_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="baseline",
        choices=["baseline", "random"],
        help="Agent to evaluate",
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
    baseline_config = get_omega_config(args.baseline_config_path)

    # get env config
    env_config = get_omega_config(args.env_config_path)

    # merge habitat, eval config and baseline configs
    agent_config, env_config = merge_configs(habitat_config, baseline_config, env_config)

    # create agent
    if args.agent_type == "random":
        agent = RandomAgent(agent_config)
    else:
        agent = OpenVocabManipAgent(agent_config)

    # create evaluator
    evaluator = OVMMEvaluator(env_config)

    # evaluate agent
    metrics = evaluator.evaluate(
        agent=agent,
        evaluation_type=args.evaluation_type,
        num_episodes=args.num_episodes,
    )
    print(metrics)
