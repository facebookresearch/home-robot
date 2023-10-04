# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import sys
import time
import timeit
from pathlib import Path
from typing import List, Optional, Tuple

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
    """Basic demo code. Collects everything that we need to make this work."""

    def __init__(
        self,
        robot: StretchClient,
        semantic_sensor,
        visualize: bool = False,
    ):
        self.robot = robot
        self.semantic_sensor = semantic_sensor
        self.normalize_embeddings = True
        self.pos_err_threshold = 0.15
        self.rot_err_threshold = 0.3

        self.encoder = ClipEncoder("ViT-B/32")
        # Wrapper for SparseVoxelMap which connects to ROS
        self.collector = RosMapDataCollector(
            self.robot,
            self.semantic_sensor,
            visualize,
            voxel_size=0.02,
            obs_min_height=0.1,
            obs_max_height=1.8,
            obs_min_density=10,  # This many points makes it an obstacle
            pad_obstacles=1,  # Add this many units (voxel_size) to the area around obstacles
            local_radius=0.15,
            encoder=self.encoder,
        )
        self.voxel_map = self.collector.voxel_map
        self.robot_model = self.collector.robot_model

        # Create planning space
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            self.robot_model,
            step_size=0.1,
            dilate_frontier_size=12,  # 0.6 meters back from every edge = 12 * 0.02 = 0.24
            dilate_obstacle_size=2,
        )

        # Dictionary storing attempts to visit each object
        self._object_attempts = {}

        # Create a simple motion planner
        self.planner = Shortcut(RRTConnect(self.space, self.space.is_valid))

    def move_to_any_instance(self, matches: List[Tuple[int, Instance]]):
        """Check instances and find one we can move to"""

        self.robot.move_to_nav_posture()
        start = self.robot.get_base_pose()
        start_is_valid = self.space.is_valid(start)
        start_is_valid_retries = 5
        while not start_is_valid and start_is_valid_retries > 0:
            print(f"Start {start} is not valid. back up a bit.")
            self.robot.nav.navigate_to([-0.1, 0, 0], relative=True)
            # Get the current position in case we are still invalid
            start = self.robot.get_base_pose()
            start_is_valid = self.space.is_valid(start)
            start_is_valid_retries -= 1
        res = None

        # Just terminate here - motion planning issues apparently!
        if not start_is_valid:
            raise RuntimeError("Invalid start state!")

        # Find and move to one of these
        for i, match in matches:
            print("Checking instance", i)
            # TODO: this is a bad name for this variable
            for j, view in enumerate(match.instance_views):
                print(i, j, view.cam_to_world)
            print("at =", match.bounds)
            res = None
            mask = self.voxel_map.mask_from_bounds(match.bounds)
            for goal in self.space.sample_near_mask(mask, radius_m=0.7):
                goal = goal.cpu().numpy()
                print("       Start:", start)
                print("Sampled Goal:", goal)
                show_goal = np.zeros(3)
                show_goal[:2] = goal[:2]
                goal_is_valid = self.space.is_valid(goal)
                print("Start is valid:", start_is_valid)
                print(" Goal is valid:", goal_is_valid)
                if not goal_is_valid:
                    print(" -> resample goal.")
                    continue

                # plan to the sampled goal
                res = self.planner.plan(start, goal)
                print("Found plan:", res.success)
                if res.success:
                    break
            else:
                # TODO: remove debug code
                # breakpoint()
                print("-> could not plan to instance", i)
                if i not in self._object_attempts:
                    self._object_attempts[i] = 1
                else:
                    self._object_attempts[i] += 1
            if res is not None and res.success:
                break

        if res is not None and res.success:
            # Now move to this location
            print("Full plan to object:")
            for i, pt in enumerate(res.trajectory):
                print("-", i, pt.state)
            self.robot.nav.execute_trajectory(
                [pt.state for pt in res.trajectory],
                pos_err_threshold=self.pos_err_threshold,
                rot_err_threshold=self.rot_err_threshold,
            )
            time.sleep(1.0)
            self.robot.nav.navigate_to([0, 0, np.pi / 2], relative=True)
            self.robot.move_to_manip_posture()
            return True

        return False

    def print_found_classes(self, goal: Optional[str] = None):
        """Helper. print out what we have found according to detic."""
        instances = self.voxel_map.get_instances()
        if goal is not None:
            print(f"Looking for {goal}.")
        print("So far, we have found these classes:")
        for i, instance in enumerate(instances):
            oid = int(instance.category_id.item())
            name = self.semantic_sensor.get_class_name_for_id(oid)
            print(i, name, instance.score)

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
        self.print_found_classes(goal)
        return self.get_found_instances_by_class(goal)

    def encode_text(self, text: str):
        """Helper function for getting text embeddings"""
        emb = self.encoder.encode_text(text)
        if self.normalize_embeddings:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def get_found_instances_by_class(
        self, goal: str, threshold: int = 0, debug: bool = False
    ) -> Optional[List[Tuple[int, Instance]]]:
        """Check to see if goal is in our instance memory or not. Return a list of everything with the correct class."""
        matching_instances = []
        instances = self.voxel_map.get_instances()
        for i, instance in enumerate(instances):
            oid = int(instance.category_id.item())
            name = self.semantic_sensor.get_class_name_for_id(oid)
            if name.lower() == goal.lower():
                matching_instances.append((i, instance))
        return self.filter_matches(matching_instances, threshold=threshold)

    def filter_matches(self, matches: List[Tuple[int, Instance]], threshold: int = 1):
        """return only things we have not tried {threshold} times"""
        filtered_matches = []
        for i, instance in matches:
            if i not in self._object_attempts or self._object_attempts[i] < threshold:
                filtered_matches.append((i, instance))
        return filtered_matches

    def go_home(self):
        """Simple helper function to send the robot home safely after a trial."""
        print("Go back to (0, 0, 0) to finish...")
        print("- change posture and switch to navigation mode")
        self.robot.move_to_nav_posture()
        self.robot.head.look_close(blocking=False)
        self.robot.switch_to_navigation_mode()

        print("- try to motion plan there")
        start = self.robot.get_base_pose()
        goal = np.array([0, 0, 0])
        res = self.planner.plan(start, goal)
        # if it fails, skip; else, execute a trajectory to this position
        if res.success:
            print("- executing full plan to home!")
            self.robot.nav.execute_trajectory([pt.state for pt in res.trajectory])
        else:
            print("Can't go home!")

    def choose_best_goal_instance(self, goal: str, debug: bool = False) -> Instance:
        instances = self.voxel_map.get_instances()
        goal_emb = self.encode_text(goal)
        if debug:
            neg1_emb = self.encode_text("the color purple")
            neg2_emb = self.encode_text("a blank white wall")
        best_instance = None
        best_score = -float("Inf")
        for instance in instances:
            if debug:
                print("# views =", len(instance.instance_views))
                print("    cls =", instance.category_id)
            # TODO: remove debug code when not needed for visualization
            # instance._show_point_cloud_open3d()
            img_emb = instance.get_image_embedding(
                aggregation_method="mean", normalize=self.normalize_embeddings
            )
            goal_score = torch.matmul(goal_emb, img_emb).item()
            if debug:
                neg1_score = torch.matmul(neg1_emb, img_emb).item()
                neg2_score = torch.matmul(neg2_emb, img_emb).item()
                print("scores =", goal_score, neg1_score, neg2_score)
            if goal_score > best_score:
                best_instance = instance
                best_score = goal_score
        return best_instance

    def run_exploration(
        self,
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
        self.robot.move_to_nav_posture()

        print("Go to (0, 0, 0) to start with...")
        self.robot.nav.navigate_to([0, 0, 0])

        # Explore some number of times
        matches = []
        for i in range(explore_iter):
            print("\n" * 2)
            print("-" * 20, i + 1, "/", explore_iter, "-" * 20)
            self.print_found_classes(task_goal)
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
                print("Plan successful!")
                # print("Full plan:")
                # for i, pt in enumerate(res.trajectory):
                #     print("-", i, pt.state)
                if not dry_run:
                    self.robot.nav.execute_trajectory(
                        [pt.state for pt in res.trajectory],
                        pos_err_threshold=self.pos_err_threshold,
                        rot_err_threshold=self.rot_err_threshold,
                    )

            # Append latest observations
            self.collector.step()
            if manual_wait:
                input("... press enter ...")

            if task_goal is not None:
                matches = self.get_found_instances_by_class(task_goal)
                if len(matches) > 0:
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

        return matches


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
@click.option("--object-to-find", default="bottle", type=str)
@click.option("--location-to-place", default="chair", type=str)
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
    object_to_find: str = "bottle",
    location_to_place: str = "chair",
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

    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    print("- Connect to Stretch")
    robot = StretchClient()
    robot.nav.navigate_to([0, 0, 0])

    print("- Create semantic sensor based on detic")
    config, semantic_sensor = create_semantic_sensor(device_id, verbose)

    # Run grasping test - just grab whatever is in front of the robot
    if test_grasping:
        run_grasping(
            robot, semantic_sensor, to_grasp=object_to_find, to_place=location_to_place
        )
        rospy.signal_shutdown("done")
        return

    print("- Start ROS data collector")
    demo = DemoAgent(robot, semantic_sensor, visualize=visualize)
    demo.start(goal=object_to_find, visualize_map_at_start=show_intermediate_maps)
    print(f"\nSearch for {object_to_find} and {location_to_place}")
    matches = demo.get_found_instances_by_class(object_to_find)
    print(f"Currently {len(matches)} matches for {object_to_find}.")

    if len(matches) == 0 or force_explore:
        print(f"Exploring for {object_to_find}, {location_to_place}...")
        demo.run_exploration(
            rate,
            manual_wait,
            explore_iter=explore_iter,
            task_goal=object_to_find,
            go_home_at_end=navigate_home,
        )
    print("Done collecting data.")
    matches = demo.get_found_instances_by_class(object_to_find)
    print("-> Found", len(matches), f"instances of class {object_to_find}.")
    # demo.voxel_map.show(orig=np.zeros(3))

    # Look at all of our instances - choose and move to one
    print(f"- Move to any instance of {object_to_find}")
    smtai = demo.move_to_any_instance(matches)
    if not smtai:
        print("Moving to instance failed!")
    else:
        print(f"- Grasp {object_to_find} using FUNMAP")
        if not no_manip:
            run_grasping(robot, semantic_sensor, to_grasp=object_to_find, to_place=None)

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
                    robot, semantic_sensor, to_grasp=None, to_place=location_to_place
                )

    if show_final_map:
        pc_xyz, pc_rgb = demo.collector.show()

        import matplotlib.pyplot as plt

        obstacles, explored = demo.collector.get_2d_map()

        plt.subplot(1, 2, 1)
        plt.imshow(obstacles)
        plt.subplot(1, 2, 2)
        plt.imshow(explored)
        plt.show()
    else:
        pc_xyz, pc_rgb = demo.collector.get_xyz_rgb()

    # Create pointcloud and write it out
    if len(output_pcd_filename) > 0:
        print(f"Write pcd to {output_pcd_filename}...")
        pcd = numpy_to_pcd(pc_xyz, pc_rgb / 255)
        open3d.io.write_point_cloud(output_pcd_filename, pcd)
    if len(output_pkl_filename) > 0:
        print(f"Write pkl to {output_pkl_filename}...")
        demo.collector.voxel_map.write_to_pickle(output_pkl_filename)

    demo.go_home()
    rospy.signal_shutdown("done")


if __name__ == "__main__":
    main()
