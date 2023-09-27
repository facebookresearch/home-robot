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

# Mapping and perception
import home_robot.utils.depth as du
from home_robot.agent.ovmm_agent import create_semantic_sensor
from home_robot.mapping import SparseVoxelMap, SparseVoxelMapNavigationSpace
from home_robot.mapping.semantic.instance_tracking_modules import Instance
from home_robot.mapping.voxel import plan_to_frontier

# Import planning tools for exploration
from home_robot.motion.rrt_connect import RRTConnect
from home_robot.motion.shortcut import Shortcut
from home_robot.motion.stretch import HelloStretchKinematics
from home_robot.perception.encoders import ClipEncoder

# Other tools
from home_robot.utils.config import load_config
from home_robot.utils.point_cloud import numpy_to_pcd, show_point_cloud
from home_robot.utils.visualization import get_x_and_y_from_path
from home_robot_hw.remote import StretchClient
from home_robot_hw.ros.grasp_helper import GraspClient as RosGraspClient
from home_robot_hw.ros.visualizer import Visualizer
from home_robot_hw.utils.collector import RosMapDataCollector


class DemoAgent:
    def __init__(
        self, robot: StretchClient, collector: RosMapDataCollector, semantic_sensor
    ):
        self.robot = robot
        self.collector = collector
        self.voxel_map = self.collector.voxel_map
        self.robot_model = self.collector.robot_model
        self.encoder = self.collector.encoder
        self.semantic_sensor = semantic_sensor
        self.normalize_embeddings = True

        # Create planning space
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            self.robot_model,
            step_size=0.1,
            dilate_frontier_size=12,  # 0.6 meters back from every edge
            dilate_obstacle_size=6,
        )

        # Create a simple motion planner
        self.planner = Shortcut(RRTConnect(self.space, self.space.is_valid))

    def start(self, goal: Optional[str] = None, visualize_map_at_start: bool = False):
        # Tuck the arm away
        print("Sending arm to  home...")
        self.robot.switch_to_manipulation_mode()

        self.robot.move_to_nav_posture()
        self.robot.head.look_close(blocking=False)
        print("... done.")

        # Move the robot into navigation mode
        self.robot.switch_to_navigation_mode()
        self.collector.step(
            visualize_map=visualize_map_at_start
        )  # Append latest observations
        return self.check_if_found_goal(goal)

    def encode_text(self, text: str):
        """Helper function for getting text embeddings"""
        emb = self.encoder.encode_text(text)
        if self.normalize_embeddings:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def check_if_found_goal(self, goal: str, debug: bool = False) -> Optional[Instance]:
        """Check to see if goal is in our instance memory or not."""

        instances = self.voxel_map.get_instances()
        goal_emb = self.encode_text(goal)
        if debug:
            neg1_emb = self.encode_text("the color purple")
            neg2_emb = self.encode_text("a blank white wall")
        for instance in instances:
            if debug:
                print("# views =", len(instance.instance_views))
                print("    cls =", instance.category_id)
            # TODO: remove debug code when not needed for visualization
            # instance._show_point_cloud_open3d()
            img_emb = instance.get_image_embedding(
                aggregation_method="mean", normalize=self.normalize_embeddings
            )
            goal_score = torch.matmul(goal_emb, img_emb)
            if debug:
                neg1_score = torch.matmul(neg1_emb, img_emb)
                neg2_score = torch.matmul(neg2_emb, img_emb)
                print("scores =", goal_score, neg1_score, neg2_score)

    def run_exploration(
        rate: int = 10,
        manual_wait: bool = False,
        explore_iter: int = 3,
        try_to_plan_iter: int = 10,
        dry_run: bool = False,
        random_goals: bool = False,
        visualize: bool = False,
        task_goal: str = None,
        go_home_at_end: bool = False,
    ) -> Optional[Instance]:
        """Go through exploration. We use the voxel_grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

        Args:
            visualize(bool): true if we should do intermediate debug visualizations"""
        rate = rospy.Rate(rate)

        print("Go to (0, 0, 0) to start with...")
        self.robot.nav.navigate_to([0, 0, 0])

        # Explore some number of times
        for i in range(explore_iter):
            print("\n" * 2)
            print("-" * 20, i + 1, "/", explore_iter, "-" * 20)
            start = self.robot.get_base_pose()
            start_is_valid = self.space.is_valid(start)
            # if start is not valid move backwards a bit
            if not start_is_valid:
                print("Start not valid. back up a bit.")
                self.robot.nav.navigate_to([-0.1, 0, 0], relative=True)
                continue
            print("       Start:", start)
            # sample a goal
            if random_goals:
                goal = next(self.space.sample_random_frontier()).cpu().numpy()
            else:
                res = plan_to_frontier(
                    start,
                    self.planner,
                    self.space,
                    self.voxel_map,
                    try_to_plan_iter=try_to_plan_iter,
                    visualize=visualize,
                )
            if visualize:
                # After doing everything
                self.collector.show(orig=show_goal)

            # if it fails, skip; else, execute a trajectory to this position
            if res.success:
                print("Full plan:")
                for i, pt in enumerate(res.trajectory):
                    print("-", i, pt.state)
                if not dry_run:
                    self.robot.nav.execute_trajectory(
                        [pt.state for pt in res.trajectory]
                    )

            # Append latest observations
            self.collector.step()
            if manual_wait:
                input("... press enter ...")

            if task_goal is not None:
                res = self.check_if_found_goal(task_goal)
                if res is not None:
                    print("!!! GOAL FOUND! Done exploration. !!!")
                    break

        if go_home_at_end:
            # Finally - plan back to (0,0,0)
            print("Go back to (0, 0, 0) to finish...")
            start = self.robot.get_base_pose()
            goal = np.array([0, 0, 0])
            res = self.planner.plan(start, goal)
            # if it fails, skip; else, execute a trajectory to this position
            if res.success:
                print("Full plan to home:")
                for i, pt in enumerate(res.trajectory):
                    print("-", i, pt.state)
                if not dry_run:
                    self.robot.nav.execute_trajectory(
                        [pt.state for pt in res.trajectory]
                    )
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
@click.option("--show-intermediate-maps", default=False, is_flag=True)
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
    encoder = ClipEncoder("ViT-B/32")
    collector = RosMapDataCollector(
        robot,
        semantic_sensor,
        visualize,
        voxel_size=voxel_size,
        obs_min_height=0.1,
        obs_max_height=1.8,
        obs_min_density=5,
        encoder=encoder,
    )

    object_to_find = "cup"
    demo = DemoAgent(robot, collector, semantic_sensor)
    demo.start(goal=object_to_find, visualize_map_at_start=show_intermediate_maps)

    res = demo.run_exploration(
        collector,
        robot,
        rate,
        manual_wait,
        explore_iter=explore_iter,
        task_goal=object_to_find,
        go_home_at_end=navigate_home,
    )

    print("Done collecting data.")

    if show_final_map:
        pc_xyz, pc_rgb = collector.show()

        import matplotlib.pyplot as plt

        obstacles, explored = collector.get_2d_map()

        plt.subplot(1, 2, 1)
        plt.imshow(obstacles)
        plt.subplot(1, 2, 2)
        plt.imshow(explored)
        plt.show()

    # Create pointcloud
    if len(output_pcd_filename) > 0:
        print(f"Write pcd to {output_pcd_filename}...")
        pcd = numpy_to_pcd(pc_xyz, pc_rgb / 255)
        open3d.io.write_point_cloud(output_pcd_filename, pcd)
    if len(output_pkl_filename) > 0:
        print(f"Write pkl to {output_pkl_filename}...")
        collector.voxel_map.write_to_pickle(output_pkl_filename)

    rospy.signal_shutdown("done")


if __name__ == "__main__":
    main()
