# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import sys
import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d
from PIL import Image

from home_robot.agent.ovmm_agent import (
    OvmmPerception,
    build_vocab_from_category_map,
    read_category_map_file,
)
from home_robot.mapping.voxel import SparseVoxelMap  # Aggregate 3d information
from home_robot.mapping.voxel import (  # Sample positions in free space for our robot to move to
    SparseVoxelMapNavigationSpace,
)
from home_robot.motion import ConfigurationSpace, Planner, PlanResult
from home_robot.motion.rrt_connect import RRTConnect
from home_robot.motion.shortcut import Shortcut
from home_robot.motion.spot import (  # Just saves the Spot robot footprint for kinematic planning
    SimpleSpotKinematics,
)
from home_robot.utils.config import get_config, load_config
from home_robot.utils.point_cloud import numpy_to_pcd
from home_robot.utils.visualization import get_x_and_y_from_path
from home_robot_spot import SpotClient, VoxelMapSubscriber
from home_robot_spot.grasp_env import GraspController


def plan_to_frontier(
    start: np.ndarray,
    planner: Planner,
    space: ConfigurationSpace,
    voxel_map: SparseVoxelMap,
    visualize: bool = False,
    try_to_plan_iter: int = 10,
) -> PlanResult:
    # extract goal using fmm planner
    tries = 0
    failed = False
    res = None
    start_is_valid = space.is_valid(start)
    print("\n----------- Planning to frontier -----------")
    print("Start is valid:", start_is_valid)
    if not start_is_valid:
        return PlanResult(False, reason="invalid start state")
    for goal in space.sample_closest_frontier(start, step_dist=1.5, min_dist=1.0):
        if goal is None:
            failed = True
            break
        goal = goal.cpu().numpy()
        print("Sampled Goal:", goal)
        show_goal = np.zeros(3)
        show_goal[:2] = goal[:2]
        goal_is_valid = space.is_valid(goal)
        print("Start is valid:", start_is_valid)
        print(" Goal is valid:", goal_is_valid)
        if not goal_is_valid:
            print(" -> resample goal.")
            continue
        # plan to the sampled goal
        res = planner.plan(start, goal)
        print("Found plan:", res.success)
        if visualize:
            obstacles, explored = voxel_map.get_2d_map()
            img = (10 * obstacles) + explored
            space.draw_state_on_grid(img, start, weight=5)
            space.draw_state_on_grid(img, goal, weight=5)
            plt.imshow(img)
            if res.success:
                path = voxel_map.plan_to_grid_coords(res)
                x, y = get_x_and_y_from_path(path)
                plt.plot(y, x)
                plt.show()
        if res.success:
            break
        else:
            if visualize:
                plt.show()
            tries += 1
            if tries >= try_to_plan_iter:
                failed = True
                break
            continue
    else:
        print(" ------ no valid goals found!")
        failed = True
    if failed:
        print(" ------ sampling and planning failed! Might be no where left to go.")
        return PlanResult(False, reason="planning to frontier failed")
    return res


def navigate_to_an_instance(
    spot, voxel_map, planner, instance_id, visualize=False, n_sample=10
):
    instances = voxel_map.get_instances()
    instance = instances[instance_id]

    # TODO this can be random
    view = instance.instance_views[-1]
    goal_position = np.asarray(view.pose)
    import pdb

    pdb.set_trace()
    print(goal_position)
    spot.navigate_to(goal_position, blocking=True)

    if visualize:
        cropped_image = view.cropped_image
        plt.imshow(cropped_image)
        plt.show()
        plt.imsave(f"instance_{instance_id}.png", cropped_image)

    return True


def get_obj_centric_world_representation(instance_memory, max_context_length):
    crops = []
    for global_id, instance in enumerate(instance_memory):
        instance_crops = instance.instance_views
        crops.append((global_id, random.sample(instance_crops, 1)[0].cropped_image))
    # TODO: the model currenly can only handle 20 crops
    if len(crops) > max_context_length:
        print(
            "\nWarning: this version of minigpt4 can only handle limited size of crops -- sampling a subset of crops from the instance memory..."
        )
        crops = random.sample(crops, max_context_length)
    import shutil

    debug_path = "crops_for_planning/"
    shutil.rmtree(debug_path, ignore_errors=True)
    os.mkdir(debug_path)
    ret = []
    for id, crop in enumerate(crops):
        Image.fromarray(crop[1], "RGB").save(
            debug_path + str(id) + "_" + str(crop[0]) + ".png"
        )
        ret.append(str(id) + "_" + str(crop[0]) + ".png")
    return ret


# def main(dock: Optional[int] = 549):
def main(dock: Optional[int] = None, args=None):
    if args.enable_vlm == 1:
        sys.path.append(
            "src/home_robot/home_robot/perception/detection/minigpt4/MiniGPT-4/"
        )
        from minigpt4_example import Predictor

        # load VLM
        vlm = Predictor(args)
        print("VLM planner initialized")

        # set task
        print("Reset the agent task to " + args.task)

    # TODO add this to config
    spot_config = get_config("src/home_robot_spot/configs/default_config.yaml")[0]

    # TODO move these parameters to config
    parameters = {
        "step_size": 2.0,  # (originally .1, we can make it all the way to 2 maybe actually)
        "visualize": False,
        "exploration_steps": 15,
        # Voxel map
        "obs_min_height": 0.5,  # Originally .1, floor appears noisy in the 3d map of freemont so we're being super conservative
        "obs_max_height": 1.8,  # Originally 1.8, spot is shorter than stretch tho
        "obs_min_density": 25,  # Originally 10, making it bigger because theres a bunch on noise
        "voxel_size": 0.05,
        "local_radius": 0.75,  # Can probably be bigger than original (.15)
        # Frontier
        "min_size": 5,  # Can probably be bigger than original (10)
        "max_size": 20,  # Can probably be bigger than original (10)
        # Other parameters tuned (footprint is a third of the real robot size)
        "use_async_subscriber": False,
    }

    # Create voxel map
    voxel_map = SparseVoxelMap(
        resolution=parameters["voxel_size"],
        local_radius=parameters["local_radius"],
        obs_min_height=parameters["obs_min_height"],
        obs_max_height=parameters["obs_max_height"],
        obs_min_density=parameters["obs_min_density"],
    )

    # Create kinematic model (very basic for now - just a footprint)
    robot_model = SimpleSpotKinematics()

    # Create navigation space example
    navigation_space = SparseVoxelMapNavigationSpace(
        voxel_map=voxel_map,
        robot=robot_model,
        step_size=parameters["step_size"],
        rotation_step_size=4.0,
        dilate_frontier_size=5,
        dilate_obstacle_size=0,
    )
    print(" - Created navigation space and environment")
    print(f"   {navigation_space=}")

    # Create segmentation sensor and load config. Returns config from file, as well as a OvmmPerception object that can be used to label scenes.
    print("- Loading configuration")
    config = load_config(visualize=False)

    print("- Create and load vocabulary and perception model")
    semantic_sensor = OvmmPerception(config, 0, True, module="detic")
    obj_name_to_id, rec_name_to_id = read_category_map_file(
        config.ENVIRONMENT.category_map_file
    )
    vocab = build_vocab_from_category_map(obj_name_to_id, rec_name_to_id)
    semantic_sensor.update_vocabulary_list(vocab, 0)
    semantic_sensor.set_vocabulary(0)

    planner = Shortcut(RRTConnect(navigation_space, navigation_space.is_valid))

    spot = SpotClient(config=spot_config, dock_id=dock, use_midas=False)
    try:
        # Turn on the robot using the client above
        spot.start()

        print("Go to (0, 0, 0) to start with...")
        spot.navigate_to([0, 0, 0], blocking=True)
        print("Sleep 1s")
        time.sleep(1)
        print("Start exploring")

        # Start thread to update voxel map
        if parameters["use_async_subscriber"]:
            voxel_map_subscriber = VoxelMapSubscriber(spot, voxel_map, semantic_sensor)
            voxel_map_subscriber.start()
        else:
            # Alternately, update synchronously
            obs = spot.get_rgbd_obs()
            obs = semantic_sensor.predict(obs)
            # TODO: remove debug code
            print(obs.gps, obs.compass)
            voxel_map.add_obs(obs, xyz_frame="world")
            voxel_map.show()

        # Well do a 360 degree turn to get some observations (this helps debug the robot)
        # for _ in range(4):
        #     spot.move_base(0, .5)
        #     time.sleep(5)

        for step in range(int(parameters["exploration_steps"])):

            print()
            print("-" * 8, step + 1, "/", int(parameters["exploration_steps"]), "-" * 8)

            # Get current position and goal
            start = spot.current_relative_position
            goal = None
            print("Start xyt:", start)
            start_is_valid = navigation_space.is_valid(start)
            print("Start is valid:", start_is_valid)
            print("Start is safe:", voxel_map.xyt_is_safe(start))

            # TODO do something is start is not valid
            if not start_is_valid:
                print("!!!!!!!!")
                break

            explore_methodical = False
            if explore_methodical:
                print("Generating the next closest frontier point...")
                res = plan_to_frontier(start, planner, navigation_space, voxel_map)
                if not res.success:
                    print(res.reason)
                    break
            else:
                print("picking a random frontier point and trying to move there...")
                # Sample a goal in the frontier (TODO change to closest frontier)
                goal = next(
                    navigation_space.sample_random_frontier(
                        min_size=parameters["min_size"], max_size=parameters["max_size"]
                    )
                )
                goal = goal.cpu().numpy()
                goal_is_valid = navigation_space.is_valid(goal)
                print(
                    f" Goal is valid: {goal_is_valid}",
                )
                if not goal_is_valid:
                    # really we should sample a new goal
                    continue
                #  Build plan
                res = planner.plan(start, goal)
                print(goal)
                print("Res success:", res.success)

            # TODO this trajectory is really ineficient, we can interpolate smarter
            for i, node in enumerate(res.trajectory):
                print(" - go to", i, "xyt =", node.state)
                spot.navigate_to(node.state, blocking=True)

            if not parameters["use_async_subscriber"]:
                print("Synchronous obs update")
                import pdb

                pdb.set_trace()
                obs = spot.get_rgbd_obs()
                obs = semantic_sensor.predict(obs)
                voxel_map.add_obs(obs, xyz_frame="world")

            if step % 1 == 0 and parameters["visualize"]:
                if parameters["use_async_subscriber"]:
                    print(
                        "Observations processed for the map so far: ",
                        voxel_map_subscriber.current_obs,
                    )
                robot_center = np.zeros(3)
                robot_center[:2] = spot.current_relative_position[:2]
                voxel_map.show(backend="open3d", orig=robot_center, instances=True)

                obstacles, explored = voxel_map.get_2d_map()
                img = (10 * obstacles) + explored
                start_unnormalized = spot.unnormalize_gps_compass(start)
                goal_unnormalized = spot.unnormalize_gps_compass(goal)

                navigation_space.draw_state_on_grid(img, start_unnormalized, weight=5)
                navigation_space.draw_state_on_grid(img, goal_unnormalized, weight=5)
                plt.imshow(img)
                plt.show()
                plt.imsave(f"exploration_step_{step}.png", img)

        print("Exploration complete!")
        robot_center = np.zeros(3)
        robot_center[:2] = spot.current_relative_position[:2]
        voxel_map.show(backend="open3d", orig=robot_center, instances=True)
        instances = voxel_map.get_instances()
        success = False
        while not success:
            if args.enable_vlm == 1:
                # get world_representation for planning
                while True:
                    world_representation = get_obj_centric_world_representation(
                        instances, args.context_length
                    )
                    # ask vlm for plan
                    task = input("please type any task you want the robot to do: ")
                    sample = vlm.prepare_sample(task, world_representation)
                    plan = vlm.evaluate(sample)
                    print(plan)
                    execute = input(
                        "do you want to execute (replan otherwise)? (y/n): "
                    )
                    if "y" in execute:
                        current_high_level_action = plan.split("; ")[0]
                        instance_id = int(
                            world_representation[
                                int(
                                    current_high_level_action.split("(")[1]
                                    .split(")")[0]
                                    .split(", ")[0]
                                    .split("_")[1]
                                )
                            ]
                            .split(".")[0]
                            .split("_")[1]
                        )
                        break
            else:
                # Navigating to a cup or bottle
                for i, each_instance in enumerate(instances):
                    if vocab.goal_id_to_goal_name[
                        int(each_instance.category_id.item())
                    ] in ["bottle", "cup"]:
                        instance_id = i
                        break

            # for debug
            spot.navigate_to([0, 0, 0], blocking=True)

            print("Navigating to instance ")
            print(f"Instance id: {instance_id}")
            success = navigate_to_an_instance(
                spot, voxel_map, planner, instance_id, visualize=parameters["visualize"]
            )
            print(f"Success: {success}")

            # try to pick up this instance
            if success:
                object_category_name = vocab.goal_id_to_goal_name[
                    int(instances[instance_id].category_id.item())
                ]
                gaze = GraspController(
                    config=spot_config,
                    spot=spot.spot,
                    objects=[[object_category_name]],
                    confidence=0.1,
                    show_img=False,
                    top_grasp=False,
                    hor_grasp=True,
                )
                spot.spot.open_gripper()
                time.sleep(1)
                print("Resetting environment...")
                success = gaze.gaze_and_grasp()
                time.sleep(2)

    except Exception as e:
        print("Exception caught:")
        print(e)
        raise e

    finally:
        print("Writing data...")
        pc_xyz, pc_rgb = voxel_map.show(
            backend="open3d", instances=False, orig=np.zeros(3)
        )
        pcd_filename = "spot_output.pcd"
        pkl_filename = "spot_output.pkl"

        # Create pointcloud
        if len(pcd_filename) > 0:
            pcd = numpy_to_pcd(pc_xyz, pc_rgb / 255)
            open3d.io.write_point_cloud(pcd_filename, pcd)
            print(f"... wrote pcd to {pcd_filename}")
        if len(pkl_filename) > 0:
            voxel_map.write_to_pickle(pkl_filename)
            print(f"... wrote pkl to {pkl_filename}")

        # TODO dont repeat this code
        obstacles, explored = voxel_map.get_2d_map()
        img = (10 * obstacles) + explored
        if start is not None:
            start_unnormalized = spot.unnormalize_gps_compass(start)
            navigation_space.draw_state_on_grid(img, start_unnormalized, weight=5)
        if goal is not None:
            goal_unnormalized = spot.unnormalize_gps_compass(goal)
            navigation_space.draw_state_on_grid(img, goal_unnormalized, weight=5)
        plt.imshow(img)
        plt.show()
        plt.imsave("exploration_step_final.png", img)

        print("Safely stop the robot...")
        spot.spot.open_gripper()
        spot.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--enable_vlm",
        default=0,
        type=int,
        help="Enable loading Minigpt4",
    )
    parser.add_argument(
        "--task",
        default="find a green bottle",
        help="Specify any task in natural language for VLM",
    )
    parser.add_argument(
        "--cfg-path",
        default="src/home_robot/home_robot/perception/detection/minigpt4/MiniGPT-4/eval_configs/ovmm_test.yaml",
        help="path to configuration file.",
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
        "--context_length",
        default=20,
        help="Maximum number of images the vlm can reason about",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="For minigpt4 configs: override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    main(args=args)
