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

# Mapping and perception
import home_robot.utils.depth as du
from home_robot.agent.multitask.robot_agent import RobotAgent
from home_robot.agent.ovmm_agent import create_semantic_sensor
from home_robot.mapping import SparseVoxelMap, SparseVoxelMapNavigationSpace

# Import planning tools for exploration
from home_robot.perception.encoders import ClipEncoder

# Other tools
from home_robot.utils.config import get_config, load_config

# Chat and UI tools
from home_robot.utils.point_cloud import numpy_to_pcd, show_point_cloud
from home_robot.utils.rpc import get_vlm_rpc_stub
from home_robot.utils.visualization import get_x_and_y_from_path
from home_robot_hw.remote import StretchClient
from home_robot_hw.ros.grasp_helper import GraspClient as RosGraspClient
from home_robot_hw.ros.visualizer import Visualizer


def run_grasping(
    robot: StretchClient, semantic_sensor, to_grasp="cup", to_place="chair"
):
    """Start running grasping code here"""
    robot.switch_to_manipulation_mode()
    robot.move_to_demo_pregrasp_posture()
    rospy.sleep(2)

    def within(x, y):
        return (
            x >= 0
            and x < obs.semantic.shape[0]
            and y >= 0
            and y < obs.semantic.shape[1]
        )

    if to_grasp is not None:
        ### GRASPING ROUTINE
        # Get observations from the robot
        obs = robot.get_observation()
        # Predict masks
        obs = semantic_sensor.predict(obs)

        print(f"Try to grasp {to_grasp}:")
        to_grasp_oid = None
        for oid in np.unique(obs.semantic):
            if oid == 0:
                continue
            cid, classname = semantic_sensor.current_vocabulary.map_goal_id(oid)
            print(f"- {oid} {cid} = {classname}")
            if classname == to_grasp:
                to_grasp_oid = oid

        x, y = np.mean(np.where(obs.semantic == to_grasp_oid), axis=1)
        if not within(x, y):
            print("WARN: to_grasp object not within valid semantic map bounds")
            return
        x = int(x)
        y = int(y)

        c_x, c_y, c_z = obs.xyz[x, y]
        c_pt = np.array([c_x, c_y, c_z, 1.0])
        m_pt = obs.camera_pose @ c_pt
        m_x, m_y, m_z, _ = m_pt

        print(f"- Execute grasp at {m_x=}, {m_y=}, {m_z=}.")
        robot._ros_client.trigger_grasp(m_x, m_y, m_z)
        robot.switch_to_manipulation_mode()
        robot.move_to_demo_pregrasp_posture()
        print(" - Done grasping!")

    if to_place is not None:
        ### PLACEMENT ROUTINE
        # Get observations from the robot
        obs = robot.get_observation()
        # Predict masks
        obs = semantic_sensor.predict(obs)

        to_place_oid = None
        for oid in np.unique(obs.semantic):
            if oid == 0:
                continue
            cid, classname = semantic_sensor.current_vocabulary.map_goal_id(oid)
            print(f"- {oid} {cid} = {classname}")
            if classname == to_place:
                to_place_oid = oid

        x, y = np.mean(np.where(obs.semantic == to_place_oid), axis=1)
        if not within(x, y):
            print("WARN: to_place object not within valid semantic map bounds")
            return
        x = int(x)
        y = int(y)

        c_x, c_y, c_z = obs.xyz[x, y]
        c_pt = np.array([c_x, c_y, c_z, 1.0])
        m_pt = obs.camera_pose @ c_pt
        m_x, m_y, m_z, _ = m_pt

        print(f"- Execute place at {m_x=}, {m_y=}, {m_z=}.")
        robot._ros_client.trigger_placement(m_x, m_y, m_z)
        robot.switch_to_manipulation_mode()
        robot.move_to_demo_pregrasp_posture()
        rospy.sleep(2)
        print(" - Done placing!")


def get_task_goals(parameters: Dict[str, Any]) -> Tuple[str, str]:
    """Helper for extracting task information"""
    if "object_to_find" in parameters:
        object_to_find = parameters["object_to_find"]
        if len(object_to_find) == 0:
            object_to_find = None
    else:
        object_to_find = None
    if "location_to_place" in parameters:
        location_to_place = parameters["location_to_place"]
        if len(location_to_place) == 0:
            location_to_place = None
    else:
        location_to_place = None
    return object_to_find, location_to_place


@click.command()
@click.option("--rate", default=5, type=int)
@click.option("--visualize", default=False, is_flag=True)
@click.option("--manual_wait", default=False, is_flag=True)
@click.option("--output-filename", default="stretch_output", type=str)
@click.option("--show-intermediate-maps", default=False, is_flag=True)
@click.option("--show-final-map", default=False, is_flag=True)
@click.option("--show-paths", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--test-grasping", default=False, is_flag=True)
@click.option("--explore-iter", default=20)
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

    if use_vlm:
        stub = get_vlm_rpc_stub(vlm_server_addr, vlm_server_port)
    else:
        stub = None

    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    print("- Connect to Stretch")
    robot = StretchClient()
    robot.nav.navigate_to([0, 0, 0])

    print("- Load parameters")
    parameters = get_config("src/home_robot_hw/configs/default.yaml")[0]
    print(parameters)
    object_to_find, location_to_place = get_task_goals(parameters)

    print("- Create semantic sensor based on detic")
    config, semantic_sensor = create_semantic_sensor(device_id, verbose)

    # Run grasping test - just grab whatever is in front of the robot
    if test_grasping:
        run_grasping(
            robot,
            semantic_sensor,
            to_grasp=object_to_find,
            to_place=location_to_place,
        )
        rospy.signal_shutdown("done")
        return

    print("- Start robot agent with data collection")
    demo = RobotAgent(robot, semantic_sensor, parameters, rpc_stub=stub)
    demo.start(goal=object_to_find, visualize_map_at_start=show_intermediate_maps)
    if object_to_find is not None:
        print(f"\nSearch for {object_to_find} and {location_to_place}")
        matches = demo.get_found_instances_by_class(object_to_find)
        print(f"Currently {len(matches)} matches for {object_to_find}.")
    else:
        matches = []

    try:
        if len(matches) == 0 or force_explore:
            print(f"Exploring for {object_to_find}, {location_to_place}...")
            demo.run_exploration(
                rate,
                manual_wait,
                explore_iter=parameters["exploration_steps"],
                task_goal=object_to_find,
                go_home_at_end=navigate_home,
            )
        print("Done collecting data.")
        matches = demo.get_found_instances_by_class(object_to_find)
        print("-> Found", len(matches), f"instances of class {object_to_find}.")
        # demo.voxel_map.show(orig=np.zeros(3))

        if stub is not None:
            print("!!!!!!!!!!!!!!!!!!!!!")
            print("Query the LLM.")
            print(demo.get_plan_from_vlm())

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
                if not no_manip:
                    run_grasping(
                        robot, semantic_sensor, to_grasp=object_to_find, to_place=None
                    )

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
                        run_grasping(
                            robot,
                            semantic_sensor,
                            to_grasp=None,
                            to_place=location_to_place,
                        )
    except Exception as e:
        raise (e)
    finally:
        if show_final_map:
            pc_xyz, pc_rgb = demo.voxel_map.show()
            obstacles, explored = demo.voxel_map.get_2d_map()
            plt.subplot(1, 2, 1)
            plt.imshow(obstacles)
            plt.subplot(1, 2, 2)
            plt.imshow(explored)
            plt.show()
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

        from PIL import Image

        for i, instance in enumerate(demo.voxel_map.get_instances()):
            for j, view in enumerate(instance.instance_views):
                image = Image.fromarray(view.cropped_image.byte().cpu().numpy())
                image.save(f"instance{i}_view{j}.png")

        demo.go_home()
        demo.finish()
        rospy.signal_shutdown("done")


if __name__ == "__main__":
    main()
