# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pickle
import random
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from home_robot.agent.multitask import get_parameters
from home_robot.agent.multitask.robot_agent import RobotAgent
from home_robot.core.interfaces import Observations
from home_robot.mapping import SparseVoxelMap, SparseVoxelMapNavigationSpace
from home_robot.mapping.voxel import plan_to_frontier
from home_robot.perception import create_semantic_sensor
from home_robot.utils.dummy_stretch_client import DummyStretchClient
from home_robot.utils.geometry import xyt_global_to_base


def plan_to_deltas(xyt0, plan):
    tol = 1e-6
    for i, node in enumerate(plan.trajectory):
        xyt1 = node.state
        dxyt = xyt_global_to_base(xyt1, xyt0)
        print((i + 1), "/", len(plan.trajectory), xyt1, "diff =", dxyt)
        nonzero = np.abs(dxyt) > tol
        assert np.sum(nonzero) <= 1, "only one value should change in the trajectory"
        xyt0 = xyt1


def add_raw_obs_to_voxel_map(obs_history, voxel_map):
    key_obs = []
    num_obs = len(obs_history["rgb"])

    for obs_id in range(num_obs):
        key_obs.append(
            Observations(
                rgb=obs_history["rgb"][obs_id].numpy(),
                gps=obs_history["base_poses"][obs_id][:2].numpy(),
                compass=[obs_history["base_poses"][obs_id][2].numpy()],
                xyz=obs_history["xyz"][obs_id].numpy(),
                depth=obs_history["depth"][obs_id].numpy(),
                camera_pose=obs_history["camera_poses"][obs_id].numpy(),
                camera_K=obs_history["camera_K"][obs_id].numpy(),
            )
        )
    config, semantic_sensor = create_semantic_sensor()
    voxel_map.reset()
    key_obs = key_obs[::2]  # TODO: set frame skip param in config
    # key_obs = key_obs[35:45]
    for idx, obs in enumerate(key_obs):
        # image_array = np.array(obs.rgb, dtype=np.uint8)
        # image = Image.fromarray(image_array)
        # image.show()
        obs = semantic_sensor.predict(obs)
        voxel_map.add_obs(obs)
    print(len(voxel_map.get_instances()))
    return voxel_map


@click.command()
@click.option(
    "--input-path",
    "-i",
    type=click.Path(),
    default="output.pkl",
    help="Input path with default value 'output.npy'",
)
@click.option(
    "--config-path",
    "-c",
    type=click.Path(),
    default="src/home_robot_hw/configs/default.yaml",
    help="Path to planner config.",
)
@click.option(
    "--frame",
    "-f",
    type=int,
    default=-1,
    help="number of frames to read",
)
@click.option("--show-svm", "-s", type=bool, is_flag=True, default=False)
@click.option("--pkl-is-svm", "-p", type=bool, is_flag=True, default=False)
@click.option("--test-planning", type=bool, is_flag=True, default=False)
@click.option("--test-sampling", type=bool, is_flag=True, default=False)
@click.option("--test-vlm", type=bool, is_flag=True, default=False)
@click.option("--show-instances", type=bool, is_flag=True, default=False)
@click.option("--query", "-q", type=str, default="")
@click.option("--pkl-not-obs", "-p", type=bool, is_flag=True, default=False)
def main(
    input_path,
    config_path,
    voxel_size: float = 0.01,
    show_maps: bool = True,
    pkl_is_svm: bool = True,
    pkl_not_obs: bool = False,
    test_planning: bool = False,
    test_sampling: bool = False,
    test_vlm: bool = False,
    frame: int = -1,
    show_svm: bool = False,
    try_to_plan_iter: int = 10,
    show_instances: bool = False,
    query: str = "",
):
    """Simple script to load a voxel map"""
    input_path = Path(input_path)
    print("Loading:", input_path)
    if pkl_is_svm:
        with input_path.open("rb") as f:
            loaded_voxel_map = pickle.load(f)
        if frame >= 0:
            raise RuntimeError(
                "cannot pass a target frame if in SVM mode; the whole map will be loaded instead."
            )
    else:
        loaded_voxel_map = None

    dummy_robot = DummyStretchClient()
    if len(config_path) > 0:
        print("- Load parameters")
        parameters = get_parameters(config_path)
        print(parameters)
        agent = RobotAgent(
            dummy_robot,
            None,
            parameters,
            rpc_stub=None,
            grasp_client=None,
            voxel_map=loaded_voxel_map,
        )
        voxel_map = agent.voxel_map

        if pkl_not_obs:
            print(
                "Reading from pkl file that doesn't include homerobot observations..."
            )
            obs_history = pickle.load(input_path.open("rb"))
            voxel_map = add_raw_obs_to_voxel_map(obs_history, voxel_map)

        elif not pkl_is_svm:
            print("Reading from pkl file of raw observations...")
            voxel_map.read_from_pickle(input_path, num_frames=frame)
    else:
        agent = None
        voxel_map = SparseVoxelMap(resolution=voxel_size)

    # TODO: read this from file or something
    # x0 = np.array([0, 0, 0])
    x0 = np.array([0, 0, 0])
    # x0 = np.array([2.6091852, 3.2328937, 0.8379814])
    # x0 = np.array([3.1000001, 0.0, 4.2857614])
    # x0 = np.array([0.0, -0.0, 1.5707968])
    # x0 = np.array([1.1499997, -0.60000074, -1.4168407])
    start_xyz = [x0[0], x0[1], 0]

    if agent is not None:
        print("Agent loaded:", agent)
        # Display with agent overlay
        space = agent.get_navigation_space()

        if show_svm:
            # x0 = np.array([0, 0, 0])
            footprint = dummy_robot.get_robot_model().get_footprint()
            print(f"{x0} valid = {space.is_valid(x0)}")
            voxel_map.show(
                instances=show_instances, orig=start_xyz, xyt=x0, footprint=footprint
            )
            # TODO: remove debug visualization code
            # x1 = np.array([0, 0, np.pi / 4])
            # print(f"{x1} valid = {space.is_valid(x1)}")
            # voxel_map.show(instances=show_instances, orig=start_xyz, xyt=x1, footprint=footprint)
            # x2 = np.array([0.5, 0.5, np.pi / 4])
            # print(f"{x2} valid = {space.is_valid(x2)}")
            # voxel_map.show(instances=show_instances, orig=start_xyz, xyt=x2, footprint=footprint)

        obstacles, explored = voxel_map.get_2d_map(debug=False)
        frontier, outside, traversible = space.get_frontier()

        plt.subplot(2, 2, 1)
        plt.imshow(explored)
        plt.axis("off")
        plt.title("Explored")

        plt.subplot(2, 2, 2)
        plt.imshow(obstacles)
        plt.axis("off")
        plt.title("Obstacles")

        plt.subplot(2, 2, 3)
        plt.imshow(frontier.cpu().numpy())
        plt.axis("off")
        plt.title("Frontier")

        plt.subplot(2, 2, 4)
        plt.imshow(traversible.cpu().numpy())
        plt.axis("off")
        plt.title("Traversible")

        plt.show()

        if test_planning:

            print("--- Sampling goals ---")
            start_is_valid = space.is_valid(x0, verbose=True, debug=False)
            if not start_is_valid:
                print("you need to manually set the start pose to be valid")
                return

            # Get frontier sampler
            sampler = space.sample_closest_frontier(
                x0, verbose=False, min_dist=0.1, step_dist=0.1
            )
            planner = agent.planner

            print(f"Closest frontier to {x0}:")
            start = x0
            for i, goal in enumerate(sampler):
                if goal is None:
                    # No more positions to sample
                    break

                np.random.seed(0)
                random.seed(0)

                print()
                print()
                print("-" * 20)
                res = planner.plan(start, goal.cpu().numpy())
                print("start =", start)
                print("goal =", goal.cpu().numpy())
                print(i, "sampled", goal, "success =", res.success)
                if res.success:
                    plan_to_deltas(x0, res)
                # Try to plan
                # res = plan_to_frontier(
                #     start,
                #    planner,
                #    space,
                #    voxel_map,
                #    try_to_plan_iter=try_to_plan_iter,
                #    visualize=False,
                # )
                # print("Planning result:", res)

            print("... done sampling frontier points.")
        if test_sampling:
            # Plan to an instance
            # Query the instances by something first
            if len(query) == 0:
                query = input("Enter a query: ")
            matches = agent.get_ranked_instances(query)
            print("Found", len(matches), "matches for query", query)
            for score, i, instance in matches:
                print(f"Try to plan to instance {i} with score {score}")
                res = agent.plan_to_instance(instance, x0, verbose=False, radius_m=0.3)
                if show_instances:
                    plt.imshow(instance.get_best_view().get_image())
                    plt.title(f"Instance {i} with score {score}")
                    plt.axis("off")
                    plt.show()
                print(" - Plan result:", res.success)
                if res.success:
                    print(" - Plan length:", len(res.trajectory))
                    break
            if res is not None and res.success:
                print("Plan found:")
                for i, node in enumerate(res.trajectory):
                    print(i, "/", len(res.trajectory), node.state)
                footprint = dummy_robot.get_robot_model().get_footprint()
                sampled_xyt = res.trajectory[-1].state
                xyz = np.array([sampled_xyt[0], sampled_xyt[1], 0.1])
                # Display the sampled goal location that we can reach
                voxel_map.show(
                    instances=show_instances,
                    orig=xyz,
                    xyt=sampled_xyt,
                    footprint=footprint,
                )

        if test_vlm:
            start_is_valid = space.is_valid(x0, verbose=True, debug=False)
            if not start_is_valid:
                print("you need to manually set the start pose to be valid")
                return
            while True:
                try:
                    agent.get_plan_from_vlm(current_pose=x0, show_plan=True)
                except KeyboardInterrupt:
                    break


if __name__ == "__main__":
    """run the test script."""
    main()
