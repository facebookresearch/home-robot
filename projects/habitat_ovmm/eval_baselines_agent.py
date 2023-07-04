# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os

from config_utils import get_habitat_config, get_ovmm_baseline_config, merge_configs
from evaluator import OVMMEvaluator

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


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

    # vectorized_local_evaluate example
    # evaluator.vectorized_local_evaluate(
    #     agent,
    #     num_episodes_per_env=eval_config.EVAL_VECTORIZED.num_episodes_per_env,
    # )

    # standard evaluate example
    evaluator.evaluate(agent, remote=args.evaluation == "remote", num_episodes=1)
