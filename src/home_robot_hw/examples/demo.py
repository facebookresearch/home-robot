# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import sys
import time
import timeit
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import open3d
import rospy
import torch
from PIL import Image

# Mapping and perception
import home_robot.utils.depth as du
from home_robot.agent.multitask import get_parameters
from home_robot.agent.multitask.robot_agent import RobotAgent
from home_robot.perception import create_semantic_sensor

# Import planning tools for exploration
from home_robot.perception.encoders import ClipEncoder

# Chat and UI tools
from home_robot.utils.point_cloud import numpy_to_pcd, show_point_cloud
from home_robot.utils.visualization import get_x_and_y_from_path
from home_robot_hw.remote import StretchClient
from home_robot_hw.ros.grasp_helper import GraspClient as RosGraspClient
from home_robot_hw.ros.visualizer import Visualizer
from home_robot_hw.utils.grasping import GraspPlanner


def do_manipulation_test(demo, object_to_find, location_to_place):
    """Run a quick manipulation test, picking and placing right in front of the robot."""
    print("- Switch to manipulation mode")
    demo.robot.switch_to_manipulation_mode()
    time.sleep(1.0)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        print(f"- Try to grasp {object_to_find}")
        # result = demo.grasp(object_goal=object_to_find)
        # print(f"{result=}")
        # if not result:
        #    continue
        print(f"- Try to place {object_to_find} on {location_to_place}")
        result = demo.place(object_goal=location_to_place)
        print(f"{result=}")
        rate.sleep()


@click.command()
@click.option("--rate", default=5, type=int)
@click.option("--visualize", default=False, is_flag=True)
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--output-filename", default="stretch_output", type=str)
@click.option("--show-intermediate-maps", default=False, is_flag=True)
@click.option("--show-final-map", default=False, is_flag=True)
@click.option("--show-paths", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--test-grasping", default=False, is_flag=True)
@click.option("--explore-iter", default=-1)
@click.option("--navigate-home", default=False, is_flag=True)
@click.option("--force-explore", default=False, is_flag=True)
@click.option("--no-manip", default=False, is_flag=True)
@click.option(
    "--input-path",
    type=click.Path(),
    default="output.pkl",
    help="Input path with default value 'output.npy'",
)
@click.option("--use-vlm", default=False, is_flag=True, help="use remote vlm to plan")
@click.option("--vlm-server-addr", default="127.0.0.1")
@click.option("--vlm-server-port", default="50054")
@click.option(
    "--write-instance-images",
    default=False,
    is_flag=True,
    help="write out images of every object we found",
)
@click.option("--parameter-file", default="src/home_robot_hw/configs/default.yaml")
def main(
    rate,
    visualize,
    manual_wait,
    output_filename,
    navigate_home: bool = True,
    device_id: int = 0,
    verbose: bool = True,
    show_intermediate_maps: bool = False,
    show_final_map: bool = False,
    show_paths: bool = False,
    random_goals: bool = True,
    test_grasping: bool = False,
    force_explore: bool = False,
    no_manip: bool = False,
    explore_iter: int = 10,
    use_vlm: bool = False,
    vlm_server_addr: str = "127.0.0.1",
    vlm_server_port: str = "50054",
    write_instance_images: bool = False,
    parameter_file: str = "src/home_robot_hw/configs/default.yaml",
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        show_intermediate_maps(bool): show maps as we explore
        show_final_map(bool): show the final 3d map after moving around and mapping the world
        show_paths(bool): display paths after planning
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    output_pcd_filename = output_filename + "_" + formatted_datetime + ".pcd"
    output_pkl_filename = output_filename + "_" + formatted_datetime + ".pkl"

    print("- Load parameters")
    parameters = get_parameters(parameter_file)
    print(parameters)

    stub = None
    if use_vlm:
        if parameters.get("vlm_option", "rpc") == "rpc":
            try:
                from home_robot.utils.rpc import get_vlm_rpc_stub
            except KeyError:
                print(
                    "Environment not configured for RPC connection! Needs $ACCEL_CORTEX to be set."
                )
            stub = get_vlm_rpc_stub(vlm_server_addr, vlm_server_port)

    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    print("- Connect to Stretch")
    robot = StretchClient()
    robot.nav.navigate_to([0, 0, 0])

    if explore_iter >= 0:
        parameters["exploration_steps"] = explore_iter
    object_to_find, location_to_place = parameters.get_task_goals()

    print("- Create semantic sensor based on detic")
    config, semantic_sensor = create_semantic_sensor(
        device_id=device_id, verbose=verbose
    )

    print("- Start robot agent with data collection")
    grasp_client = GraspPlanner(robot, env=None, semantic_sensor=semantic_sensor)
    demo = RobotAgent(
        robot, semantic_sensor, parameters, rpc_stub=stub, grasp_client=grasp_client
    )
    demo.start(goal=object_to_find, visualize_map_at_start=show_intermediate_maps)
    if object_to_find is not None:
        print(f"\nSearch for {object_to_find} and {location_to_place}")
        matches = demo.get_found_instances_by_class(object_to_find)
        print(f"Currently {len(matches)} matches for {object_to_find}.")
    else:
        matches = []

    # Run grasping test - just grab whatever is in front of the robot
    if test_grasping:
        do_manipulation_test(demo, object_to_find, location_to_place)
        return

    if parameters["in_place_rotation_steps"] > 0:
        demo.rotate_in_place(
            steps=parameters["in_place_rotation_steps"],
            visualize=False,  # show_intermediate_maps,
        )

    # Run the actual procedure
    try:
        if len(matches) == 0 or force_explore:
            print(f"Exploring for {object_to_find}, {location_to_place}...")
            demo.run_exploration(
                rate,
                manual_wait,
                explore_iter=parameters["exploration_steps"],
                task_goal=object_to_find,
                go_home_at_end=navigate_home,
                visualize=show_intermediate_maps,
            )
        print("Done collecting data.")
        matches = demo.get_found_instances_by_class(object_to_find)
        print("-> Found", len(matches), f"instances of class {object_to_find}.")

        if use_vlm:
            print("!!!!!!!!!!!!!!!!!!!!!")
            print("Query the VLM.")
            print(f"VLM's response: {demo.get_plan_from_vlm()}")
            input(
                "# TODO: execute the above plan (seems like we are not doing it right now)"
            )

        if len(matches) == 0:
            print("No matching objects. We're done here.")
        else:
            # Look at all of our instances - choose and move to one
            print(f"- Move to any instance of {object_to_find}")
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
    except Exception as e:
        raise (e)
    finally:
        if show_final_map:
            pc_xyz, pc_rgb = demo.voxel_map.show()
            # TODO: Segfaults here for some reason
            # obstacles, explored = demo.voxel_map.get_2d_map()
            # plt.subplot(1, 2, 1)
            # plt.imshow(obstacles)
            # plt.subplot(1, 2, 2)
            # plt.imshow(explored)
            # plt.show()
        else:
            pc_xyz, pc_rgb = demo.voxel_map.get_xyz_rgb()

        # Create pointcloud and write it out
        if len(output_pcd_filename) > 0:
            print(f"Write pcd to {output_pcd_filename}...")
            pcd = numpy_to_pcd(pc_xyz, pc_rgb / 255)
            open3d.io.write_point_cloud(output_pcd_filename, pcd)
        if len(output_pkl_filename) > 0:
            print(f"Write pkl to {output_pkl_filename}...")
            demo.voxel_map.write_to_pickle(output_pkl_filename)

        if write_instance_images:
            demo.save_instance_images(".")

        demo.go_home()
        demo.finish()
        rospy.signal_shutdown("done")


if __name__ == "__main__":
    main()
