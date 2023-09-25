# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import timeit
from pathlib import Path
from typing import Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import open3d
import rospy
import torch

import home_robot.utils.depth as du
from home_robot.agent.ovmm_agent import create_semantic_sensor
from home_robot.mapping import SparseVoxelMap, SparseVoxelMapNavigationSpace
from home_robot.mapping.voxel import plan_to_frontier

# Import planning tools for exploration
from home_robot.motion.rrt_connect import RRTConnect
from home_robot.motion.shortcut import Shortcut
from home_robot.motion.stretch import HelloStretchKinematics
from home_robot.utils.config import load_config
from home_robot.utils.geometry import xyt2sophus
from home_robot.utils.image import Camera
from home_robot.utils.point_cloud import numpy_to_pcd, pcd_to_numpy, show_point_cloud
from home_robot.utils.pose import convert_pose_habitat_to_opencv, to_pos_quat
from home_robot.utils.visualization import get_x_and_y_from_path
from home_robot_hw.env.stretch_pick_and_place_env import StretchPickandPlaceEnv
from home_robot_hw.remote import StretchClient
from home_robot_hw.ros.grasp_helper import GraspClient as RosGraspClient
from home_robot_hw.ros.visualizer import Visualizer
from home_robot_hw.utils.collector import RosMapDataCollector


def run_exploration(
    collector: RosMapDataCollector,
    robot: StretchClient,
    rate: int = 10,
    manual_wait: bool = False,
    explore_iter: int = 3,
    try_to_plan_iter: int = 10,
    dry_run: bool = False,
    random_goals: bool = False,
    visualize: bool = False,
):
    """Go through exploration. We use the voxel_grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

    Args:
        visualize(bool): true if we should do intermediate debug visualizations"""
    rate = rospy.Rate(rate)

    # Create planning space
    space = collector.get_planning_space()

    # Create a simple motion planner
    planner = Shortcut(RRTConnect(space, space.is_valid))

    print("Go to (0, 0, 0) to start with...")
    robot.nav.navigate_to([0, 0, 0])

    # Explore some number of times
    for i in range(explore_iter):
        print("\n" * 2)
        print("-" * 20, i, "-" * 20)
        start = robot.get_base_pose()
        start_is_valid = space.is_valid(start)
        # if start is not valid move backwards a bit
        if not start_is_valid:
            print("Start not valid. back up a bit.")
            robot.nav.navigate_to([-0.1, 0, 0], relative=True)
            continue
        print("       Start:", start)
        # sample a goal
        if random_goals:
            goal = next(space.sample_random_frontier()).cpu().numpy()
        else:
            res = plan_to_frontier(
                start,
                planner,
                space,
                collector.voxel_map,
                try_to_plan_iter=try_to_plan_iter,
                visualize=visualize,
            )
        if visualize:
            # After doing everything
            collector.show(orig=show_goal)

        # if it fails, skip; else, execute a trajectory to this position
        if res.success:
            print("Full plan:")
            for i, pt in enumerate(res.trajectory):
                print("-", i, pt.state)
            if not dry_run:
                robot.nav.execute_trajectory([pt.state for pt in res.trajectory])

        # Append latest observations
        collector.step()
        if manual_wait:
            input("... press enter ...")

    # Finally - plan back to (0,0,0)
    print("Go back to (0, 0, 0) to finish...")
    start = robot.get_base_pose()
    goal = np.array([0, 0, 0])
    res = planner.plan(start, goal)
    # if it fails, skip; else, execute a trajectory to this position
    if res.success:
        print("Full plan to home:")
        for i, pt in enumerate(res.trajectory):
            print("-", i, pt.state)
        if not dry_run:
            robot.nav.execute_trajectory([pt.state for pt in res.trajectory])
    else:
        print("WARNING: planning to home failed!")


def collect_data(
    rate,
    visualize,
    manual_wait,
    pcd_filename,
    pkl_filename,
    voxel_size: float = 0.01,
    device_id: int = 0,
    verbose: bool = True,
    visualize_map_at_start: bool = False,
    visualize_map: bool = False,
    blocking: bool = True,
    **kwargs,
):
    """Collect data from a Stretch robot. Robot will move through a preset trajectory, stopping repeatedly."""

    print("- Connect to Stretch")
    robot = StretchClient()

    config, semantic_sensor = create_semantic_sensor(device_id, verbose)

    print("- Start ROS data collector")
    collector = RosMapDataCollector(
        robot, semantic_sensor, visualize, voxel_size=voxel_size
    )

    # Tuck the arm away
    print("Sending arm to  home...")
    robot.switch_to_manipulation_mode()

    robot.move_to_nav_posture()
    robot.head.look_close(blocking=False)
    print("... done.")

    # Move the robot
    robot.switch_to_navigation_mode()
    collector.step(visualize_map=visualize_map_at_start)  # Append latest observations
    run_exploration(collector, robot, rate, manual_wait)

    print("Done collecting data.")
    robot.nav.navigate_to((0, 0, 0))
    pc_xyz, pc_rgb = collector.show()

    if visualize_map:
        import matplotlib.pyplot as plt

        obstacles, explored = collector.get_2d_map()

        plt.subplot(1, 2, 1)
        plt.imshow(obstacles)
        plt.subplot(1, 2, 2)
        plt.imshow(explored)
        plt.show()

    # Create pointcloud
    if len(pcd_filename) > 0:
        pcd = numpy_to_pcd(pc_xyz, pc_rgb / 255)
        open3d.io.write_point_cloud(pcd_filename, pcd)
    if len(pkl_filename) > 0:
        collector.voxel_map.write_to_pickle(pkl_filename)

    rospy.signal_shutdown("done")


def run_grasping():
    pass


@click.command()
@click.option("--rate", default=5, type=int)
@click.option("--visualize", default=False, is_flag=True)
@click.option("--manual_wait", default=False, is_flag=True)
@click.option("--output-pcd-filename", default="output.ply", type=str)
@click.option("--output-pkl-filename", default="output.pkl", type=str)
@click.option("--show-maps", default=False, is_flag=True)
@click.option("--show-paths", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--test-grasping", default=False, is_flag=True)
@click.option(
    "--input-path",
    type=click.Path(),
    default="output.pkl",
    help="Input path with default value 'output.npy'",
)
def main(
    rate,
    visualize,
    manual_wait,
    output_pcd_filename,
    output_pkl_filename,
    run_explore: bool = False,
    input_path: str = ".",
    voxel_size: float = 0.01,
    device_id: int = 0,
    verbose: bool = True,
    show_maps: bool = False,
    show_paths: bool = False,
    random_goals: bool = True,
    test_grasping: bool = False,
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        run_explore(bool): should sample frontier points and path to them; on robot will go there.
        show_maps(bool): show 2d maps
        show_paths(bool): display paths after planning
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """
    click.echo(f"Using input path: {input_path}")

    if test_grasping:
        run_grasping()

    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    collect_data(
        rate,
        visualize,
        manual_wait,
        output_pcd_filename,
        output_pkl_filename,
        voxel_size,
        device_id,
        verbose,
        **kwargs,
    )


if __name__ == "__main__":
    main()
