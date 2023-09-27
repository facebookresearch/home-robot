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
    explore_iter: int = 20,
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
        print("-" * 20, i + 1, "/", explore_iter, "-" * 20)
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


def run_grasping(robot: StretchClient, semantic_sensor):
    """Start running grasping code here"""
    robot.switch_to_manipulation_mode()
    robot.move_to_manip_posture()
    robot.manip.goto_joint_positions(
        [
            0.0,  # base x
            0.6,  # Lift
            0.01,  # Arm
            0,  # Roll
            -1.5,  # Pitch
            0,  # Yaw
        ]
    )

    # Get observations from the robot
    obs = robot.get_observation()
    # Predict masks
    obs = semantic_sensor.predict(obs)

    for oid in np.unique(obs.semantic):
        if oid == 0:
            continue
        cid, classname = semantic_sensor.current_vocabulary.map_goal_id(oid)
        print(f"- {oid} {cid} = {classname}")

    # plt.subplot(131)
    # plt.imshow(obs.rgb)
    # plt.subplot(132)
    # plt.imshow(obs.xyz)
    # plt.subplot(133)
    # plt.imshow(obs.semantic)
    # plt.show()

    # show_point_cloud(obs.xyz, obs.rgb / 255, orig=np.zeros(3))
    # breakpoint()


@click.command()
@click.option("--rate", default=5, type=int)
@click.option("--visualize", default=False, is_flag=True)
@click.option("--manual_wait", default=False, is_flag=True)
@click.option("--output-pcd-filename", default="output.ply", type=str)
@click.option("--output-pkl-filename", default="output.pkl", type=str)
@click.option("--show-intermediate_map", default=False, is_flag=True)
@click.option("--show-final-map", default=False, is_flag=True)
@click.option("--show-paths", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--test-grasping", default=False, is_flag=True)
@click.option("--explore-iter", default=20)
@click.option("--navigate-home", default=False, is_flag=True)
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
    navigate_home: bool = True,
    input_path: str = ".",
    voxel_size: float = 0.01,
    device_id: int = 0,
    verbose: bool = True,
    show_intermediate_maps: bool = False,
    show_final_map: bool = False,
    show_paths: bool = False,
    random_goals: bool = True,
    test_grasping: bool = False,
    explore_iter: int = 10,
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        run_explore(bool): should sample frontier points and path to them; on robot will go there.
        show_intermediate_maps(bool): show maps as we explore
        show_final_map(bool): show the final 3d map after moving around and mapping the world
        show_paths(bool): display paths after planning
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """
    click.echo(f"Using input path: {input_path}")

    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    print("- Connect to Stretch")
    robot = StretchClient()

    print("- Create semantic sensor based on detic")
    config, semantic_sensor = create_semantic_sensor(device_id, verbose)

    # Run grasping test - just grab whatever is in front of the robot
    if test_grasping:
        run_grasping(robot, semantic_sensor)
        rospy.signal_shutdown("done")
        return

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
    collector.step(visualize_map=show_intermediate_maps)  # Append latest observations
    run_exploration(collector, robot, rate, manual_wait, explore_iter=explore_iter)

    print("Done collecting data.")
    if navigate_home:
        robot.nav.navigate_to((0, 0, 0))
    if show_final_map:
        pc_xyz, pc_rgb = collector.show()

    if show_maps:
        import matplotlib.pyplot as plt

        obstacles, explored = collector.get_2d_map()

        plt.subplot(1, 2, 1)
        plt.imshow(obstacles)
        plt.subplot(1, 2, 2)
        plt.imshow(explored)
        plt.show()

    # Create pointcloud
    if len(output_pcd_filename) > 0:
        pcd = numpy_to_pcd(pc_xyz, pc_rgb / 255)
        open3d.io.write_point_cloud(output_pcd_filename, pcd)
    if len(output_pkl_filename) > 0:
        collector.voxel_map.write_to_pickle(output_pkl_filename)

    rospy.signal_shutdown("done")


if __name__ == "__main__":
    main()
