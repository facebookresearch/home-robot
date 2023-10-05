# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os

from evaluator import OVMMEvaluator
from utils.config_utils import (
    create_agent_config,
    create_env_config,
    get_habitat_config,
    get_omega_config,
)

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot.agent.ovmm_agent.random_agent import RandomAgent
from home_robot.agent.ovmm_agent.vlm_agent import VLMAgent
from home_robot.agent.ovmm_agent.vlm_exploration_agent import VLMExplorationAgent

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
        default="projects/vlm_planning/configs/agent/heuristic_instance_tracking_agent.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--env_config_path",
        type=str,
        default="projects/vlm_planning/configs/env/hssd_demo.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="baseline",
        choices=["baseline", "explore"],
        help="Agent to evaluate",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Specify any task in natural language",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=None,
        help="Max agent action step",
    )

    parser.add_argument(
        "--data_collection",
        type=bool,
        default=False,
        help="If is data collection or not",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Data collection save directory",
    )

    parser.add_argument(
        "--episode_min",
        type=int,
        default=None,
        help="Min idx of episode",
    )

    parser.add_argument(
        "--episode_max",
        type=int,
        default=None,
        help="Max idx of episode",
    )

    parser.add_argument(
        "--cfg-path",
        default="src/home_robot/home_robot/perception/detection/minigpt4/MiniGPT-4/eval_configs/ovmm_test.yaml",
        help="path to configuration file.",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=1, help="specify the gpu to load the model."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "overrides",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    parsed_args = parser.parse_args()

    episode_idx_list = []
    for idx in range(parsed_args.episode_min, parsed_args.episode_max):
        episode_idx_list.append(idx)
    parsed_args.overrides.append("habitat.dataset.episode_ids=" + str(episode_idx_list))

    # get habitat config
    habitat_config, _ = get_habitat_config(
        parsed_args.habitat_config_path, overrides=parsed_args.overrides
    )

    # get baseline config
    baseline_config = get_omega_config(parsed_args.baseline_config_path)

    # get env config
    env_config = get_omega_config(parsed_args.env_config_path)

    # merge habitat and env config to create env config
    env_config = create_env_config(
        habitat_config, env_config, evaluation_type=parsed_args.evaluation_type
    )

    # merge env config and baseline config to create agent config
    agent_config = create_agent_config(env_config, baseline_config)
    # create agent
    if parsed_args.agent_type == "explore":
        agent = VLMExplorationAgent(config=agent_config, args=parsed_args)
    else:
        agent = VLMAgent(config=agent_config, args=parsed_args)

    # create evaluator
    evaluator = OVMMEvaluator(
        env_config,
        save_instance_memory=parsed_args.data_collection,
        save_dir=parsed_args.save_dir,
    )

    # evaluate agent
    metrics = evaluator.evaluate(
        agent=agent,
        evaluation_type=parsed_args.evaluation_type,
        num_episodes=parsed_args.num_episodes,
    )
    print("Metrics:\n", metrics)
