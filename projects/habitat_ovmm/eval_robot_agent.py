# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from utils.config_utils import (
    create_agent_config,
    create_env_config,
    get_habitat_config,
    get_omega_config,
)

from home_robot.agent.multitask import get_parameters
from home_robot.agent.multitask.robot_agent import RobotAgent
from home_robot.perception import create_semantic_sensor
from home_robot.utils.rpc import get_vlm_rpc_stub
from home_robot_sim.ovmm_sim_client import OvmmSimClient, SimGraspPlanner
from home_robot_sim.utils.env_utils import create_ovmm_env_fn

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--env_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/env/hssd_demo_robot_agent.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--agent_parameters",
        type=str,
        default="src/home_robot_sim/configs/default.yaml",
        help="path to parameters file for agent",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="GPU device id",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=5,
        help="rate?",
    )
    parser.add_argument(
        "--vlm_server_addr",
        default="cortex-robot-elb-57c549656770fe85.elb.us-east-1.amazonaws.com",
        help="ip address or domain name of vlm server.",
    )
    parser.add_argument(
        "--vlm_server_port",
        default="50054",
        help="port of vlm server.",
    )
    parser.add_argument(
        "--manual_wait",
        type=bool,
        default=False,
        help="manual_wait?",
    )
    parser.add_argument(
        "--navigate_home",
        type=bool,
        default=False,
        help="manual_wait?",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="verbose output",
    )
    parser.add_argument(
        "--show_intermediate_maps",
        type=bool,
        default=True,
        help="verbose output",
    )

    parser.add_argument(
        "overrides",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()

    # get habitat config
    habitat_config, _ = get_habitat_config(
        args.habitat_config_path, overrides=args.overrides
    )

    # get env config
    env_config = get_omega_config(args.env_config_path)

    # merge habitat and env config to create env config
    env_config = create_env_config(habitat_config, env_config, evaluation_type="local")

    logger.info("Creating OVMM simulation environment")
    env = create_ovmm_env_fn(env_config)

    robot = OvmmSimClient(sim_env=env, is_stretch_robot=True)

    print("- Create semantic sensor based on detic")
    config, semantic_sensor = create_semantic_sensor(
        device_id=args.device_id, verbose=args.verbose
    )

    grasp_client = SimGraspPlanner(robot)

    parameters = get_parameters(args.agent_parameters)
    print(parameters)
    object_to_find, location_to_place = robot.get_task_obs()

    stub = get_vlm_rpc_stub(
        vlm_server_addr=args.vlm_server_addr, vlm_server_port=args.vlm_server_port
    )

    demo = RobotAgent(
        robot, semantic_sensor, parameters, rpc_stub=stub, grasp_client=grasp_client
    )
    demo.start(goal=object_to_find, visualize_map_at_start=args.show_intermediate_maps)

    matches = demo.get_found_instances_by_class(object_to_find)

    # demo.robot.navigate_to([-0.1, 0, 0], relative=True)
    # demo.update()
    # import numpy as np

    # demo.robot.navigate_to([0, 0, np.pi / 4], relative=True)
    # demo.update()
    # demo.robot.navigate_to([0, 0, np.pi / 4], relative=True)
    # demo.update()
    breakpoint()
    print("rotate in place for a bit")
    demo.rotate_in_place(steps=12)

    demo.run_exploration(
        args.rate,
        args.manual_wait,
        explore_iter=parameters["exploration_steps"],
        task_goal=object_to_find,
        go_home_at_end=args.navigate_home,
    )

    print("Done collecting data (exploration).")
    matches = demo.get_found_instances_by_class(object_to_find)
    print("-> Found", len(matches), f"instances of class {object_to_find}.")

    demo.execute_vlm_plan()
    print(f"- Move to any instance of {object_to_find}")
    try:
        smtai = demo.move_to_any_instance(matches)
        if not smtai:
            print("Moving to instance failed!")
        else:
            print(f"- Grasp {object_to_find} using FUNMAP")
            res = demo.grasp(object_goal=object_to_find)
            print(f"- Grasp result: {res}")

            matches = demo.get_found_instances_by_class(location_to_place)
            if len(matches) == 0:
                print(f"!!! No location {location_to_place} found. Exploring !!!")
                demo.run_exploration(
                    rate,
                    manual_wait,
                    explore_iter=explore_iter,
                    task_goal=location_to_place,
                    go_home_at_end=navigate_home,
                )

            print(f"- Move to any instance of {location_to_place}")
            smtai2 = demo.move_to_any_instance(matches)
            if not smtai2:
                print(f"Going to instance of {location_to_place} failed!")
            else:
                print(f"- Placing on {location_to_place} using FUNMAP")
                if not no_manip:
                    # run_grasping(
                    #    robot,
                    #    semantic_sensor,
                    #    to_grasp=None,
                    #    to_place=location_to_place,
                    # )
                    pass
    except RuntimeError as e:
        raise (e)
    finally:
        demo.voxel_map.write_to_pickle("test.pkl")

    # breakpoint()

    # # create evaluator
    # evaluator = OVMMEvaluator(env_config)

    # # evaluate agent
    # metrics = evaluator.evaluate(
    #     agent=agent,
    #     evaluation_type=args.evaluation_type,
    #     num_episodes=args.num_episodes,
    # )
    # print("Metrics:\n", metrics)
