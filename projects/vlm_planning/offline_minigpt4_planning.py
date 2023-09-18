# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pickle

from utils.config_utils import (
    create_agent_config,
    create_env_config,
    get_habitat_config,
    get_omega_config,
)

# from evaluator import OVMMEvaluator
from home_robot.agent.ovmm_agent.vlm_agent import VLMAgent


def main():
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
        "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
    )

    parser.add_argument(
        "--instance_memory",
        default=None,
        help="collected instance memory filename",
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
    with open(parsed_args.instance_memory, "rb") as f:
        instance_memory = pickle.load(f)
    agent = VLMAgent(config=agent_config, args=parsed_args)

    while True:
        world_representation = agent.get_obj_centric_world_representation(
            external_instance_memory=instance_memory
        )
        print(
            "Saving the object crops (as world represenation) into crops_for_planning/ ..."
        )
        task = input("task: ")
        agent.set_task(task)
        plan = agent.ask_vlm_for_plan(world_representation)
        print("Plan: " + str(plan))
        print(
            "(Don't forget to check the crops_for_planning/ folder to see if the above plan makes sense to you)"
        )
        input("Hit enter if the check is over and you want to continue testing")


if __name__ == "__main__":
    main()
