# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import math
import os
import pickle
import random
import shutil
import sys
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d
import torch
from atomicwrites import atomic_write
from loguru import logger

import home_robot.utils.planar as nc
from examples.demo_utils.mock_agent import MockSpotDemoAgent

# Simple IO tool for robot agents
from home_robot.agent.multitask.robot_agent import publish_obs
from home_robot.mapping.voxel import SparseVoxelMap  # Aggregate 3d information
from home_robot.mapping.voxel import (  # Sample positions in free space for our robot to move to
    SparseVoxelMapNavigationSpace,
)
from home_robot.motion import ConfigurationSpace, Planner, PlanResult
from home_robot.motion.rrt_connect import RRTConnect
from home_robot.motion.shortcut import Shortcut
from home_robot.motion.spot import (  # Just saves the Spot robot footprint for kinematic planning; This should be changed in the future
    SimpleSpotKinematics,
)
from home_robot.perception.encoders import ClipEncoder
from home_robot.perception.wrapper import create_semantic_sensor
from home_robot.utils.config import Config, get_config, load_config
from home_robot.utils.demo_chat import (
    DemoChat,
    start_demo_ui_server,
    stop_demo_ui_server,
)
from home_robot.utils.geometry import xyt_global_to_base
from home_robot.utils.point_cloud import numpy_to_pcd
from home_robot.utils.rpc import (
    get_obj_centric_world_representation,
    get_output_from_world_representation,
    get_vlm_rpc_stub,
    parse_pick_and_place_plan,
)
from home_robot.utils.threading import Interval
from home_robot.utils.visualization import get_x_and_y_from_path
from home_robot_spot import SpotClient, VoxelMapSubscriber
from home_robot_spot.grasp_env import GraspController
from home_robot_spot.spot_demo_agent import SpotDemoAgent


# def main(dock: Optional[int] = 549):
def main(dock: Optional[int] = None, args=None):
    """Runs the demo. Will explore a certain amount then do pick and place.

    Args:
        dock(int): id of dock to return to at the end.
        args: arguments from argparser
    """
    level = logger.level("DEMO", no=38, color="<yellow>", icon="🤖")
    print(f"{level=}")
    start_time = time.time()
    logger.info("Starting demo at {}", start_time)
    data: Dict[str, List[str]] = {}
    if args.enable_vlm == 1:
        stub = get_vlm_rpc_stub(args.vlm_server_addr, args.vlm_server_port)
        # channel = grpc.insecure_channel(
        #    f"{args.vlm_server_addr}:{args.vlm_server_port}"
        # )
        # stub = AgentgRPCStub(channel)
    else:
        # No vlm to use, just default behavior
        stub = None

    # TODO add this to config
    spot_config = get_config("src/home_robot_spot/configs/default_config.yaml")[0]
    if args.location == "pit":
        parameters = get_config("src/home_robot_spot/configs/parameters.yaml")[0]
    elif args.location == "fre":
        parameters = get_config("src/home_robot_spot/configs/parameters_fre.yaml")[0]
    else:
        logger.critical(
            f"Location {args.location} is invalid, please enter a valid location"
        )

    print("-" * 8, "PARAMETERS", "-" * 8)
    print(parameters)

    timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"
    path = os.path.expanduser(f"data/hw_exps/spot/{timestamp}")
    logger.add(f"{path}/{timestamp}.log", backtrace=True, diagnose=True)
    os.makedirs(path, exist_ok=True)
    logger.info("Saving viz data to {}", path)
    if args.mock_agent:
        demo = MockSpotDemoAgent(parameters, spot_config, dock, path)
    else:
        demo = SpotDemoAgent(parameters, spot_config, dock, path)
    spot = demo.spot
    voxel_map = demo.voxel_map
    semantic_sensor = demo.semantic_sensor
    navigation_space = demo.navigation_space
    start = None
    goal = None
    # TODO add desktop password here maybe via config
    os.system("echo 'batman1234' | sudo -S  kill -9 $(lsof -t -i:8901)")
    logger.info("killed old UI port")
    try:
        start_demo_ui_server()

        # Turn on the robot using the client above
        spot.start()
        demo.start()
        logger.success("Spot started")
        logger.info("Sleep 1s")
        time.sleep(0.5)
        x0, y0, theta0 = spot.current_position
        logger.info(f"Start exploring from {x0=}, {y0=}, {theta0=}")

        # Start thread to update voxel map
        if parameters["use_async_subscriber"]:
            voxel_map_subscriber = VoxelMapSubscriber(spot, voxel_map, semantic_sensor)
            voxel_map_subscriber.start()
        else:
            demo.update()

        # logger.critical("Not running explore")
        demo.rotate_in_place()
        demo.run_teleop_data()

        logger.info("Exploration complete!")
        demo.run_task(stub, center=np.array([x0, y0, theta0]), data=data)

    except Exception as e:
        logger.critical("Exception caught: {}", e)
        raise e

    finally:
        stop_demo_ui_server()
        demo.finish()
        if parameters["write_data"]:
            demo.voxel_map.write_to_pickle(f"{path}/spot_observations.pkl")
            if start is None:
                start = demo.spot.current_position
            if voxel_map.get_instances() is not None:
                if demo.should_visualize():
                    pc_xyz, pc_rgb = voxel_map.show(
                        backend="open3d", instances=False, orig=np.zeros(3)
                    )
                else:
                    pc_xyz, pc_rgb = voxel_map.get_xyz_rgb()
                pcd_filename = f"{path}/spot_output_{timestamp}.pcd"
                pkl_filename = f"{path}/spot_output_{timestamp}.pkl"
                logger.info("Writing data...")
                # Create pointcloud
                if len(pcd_filename) > 0:
                    pcd = numpy_to_pcd(pc_xyz, pc_rgb / 255)
                    open3d.io.write_point_cloud(pcd_filename, pcd)
                    logger.info(f"wrote pcd to {pcd_filename}")
                if len(pkl_filename) > 0:
                    voxel_map.write_to_pickle_add_data(pkl_filename, data)
                    logger.info(f"wrote pkl to {pkl_filename}")

                # TODO dont repeat this code
                obstacles, explored = voxel_map.get_2d_map()
                img = (10 * obstacles) + explored
                if start is not None:
                    navigation_space.draw_state_on_grid(img, start, weight=5)
                if goal is not None:
                    navigation_space.draw_state_on_grid(img, goal, weight=5)
                plt.imshow(img)
                if demo.should_visualize():
                    plt.show()
                plt.imsave(f"{path}/exploration_step_final.png", img)

        spot.navigate_to(np.array([x0, y0, theta0]))
        time.sleep(0.5)
        logger.warning("Safely stop the robot...")
        spot.spot.close_gripper()
        logger.info("Robot sit down")
        spot.spot.sit()
        spot.spot.power_off()
        spot.stop()
        end_time = time.time()
        elapsed_time = end_time - start_time
        emin = int(elapsed_time // 60)
        esec = int(elapsed_time % 60)
        logger.success("Demo finished at {}", end_time)
        logger.success(f"Elapsed time: {emin} mins {esec} secs")
        message = f"Elapsed time: {emin} mins {esec} secs"
        with open(f"{path}/elapsed_time.txt", "w") as f:
            f.write(message)
            f.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--enable_vlm",
        default=00,
        type=int,
        help="Enable loading Minigpt4. 1 == use vlm, 0 == test without vlm",
    )
    parser.add_argument(
        "--task",
        default="find a green bottle",
        help="Specify any task in natural language for VLM",
    )
    # AWS IP Address cortex-robot-elb-57c549656770fe85.elb.us-east-1.amazonaws.com
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
        "--gpu-id", type=int, default=1, help="specify the gpu to load the model."
    )
    parser.add_argument(
        "--planning_times",
        default=1,
        help="Num of times of calling VLM for inference -- might be useful for long context length",
    )
    parser.add_argument(
        "--location",
        "-l",
        default="pit",
        help="location of the spot (fre or pit)",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="For minigpt4 configs: override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--mock_agent",
        "-m",
        default=False,
        action="store_true",
        help="Use a mock agent instead of the real one",
    )
    args = parser.parse_args()
    main(args=args)
