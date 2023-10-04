# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import os
import pickle
import random
import sys
import time
import shutil
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d
import torch
from atomicwrites import atomic_write
from PIL import Image

import home_robot_spot.nav_client as nc
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
from home_robot.utils.geometry import xyt_global_to_base
from home_robot.utils.point_cloud import numpy_to_pcd
from home_robot.utils.visualization import get_x_and_y_from_path
from home_robot_spot import SpotClient, VoxelMapSubscriber
from home_robot_spot.grasp_env import GraspController

def goto(spot: SpotClient, planner: Planner, goal):
    """Send the spot to the correct location."""
    start = spot.current_position

    #  Build plan
    res = planner.plan(start, goal)
    print(goal)
    print("Res success:", res.success)

    # Move to the next location
    if res.success:
        print("- follow the plan to goal")
        spot.execute_plan(res)
    else:
        print("- just go ahead and try it anyway")
        spot.navigate_to(goal)
    return res

# NOTE: this requires 'pip install atomicwrites'
def publish_obs(model: SparseVoxelMapNavigationSpace, path: str):
    timestep = len(model.voxel_map.observations) - 1
    with atomic_write(f"{path}/{timestep}.pkl", mode="wb") as f:
        model_obs = model.voxel_map.observations[-1]
        print(f"Saving observation to pickle file...{f'{timestep}.pkl'}")
        pickle.dump(
            dict(
                rgb=model_obs.rgb.cpu().detach(),
                depth=model_obs.depth.cpu().detach(),
                instance_image=model_obs.instance.cpu().detach(),
                instance_classes=model_obs.instance_classes.cpu().detach(),
                instance_scores=model_obs.instance_scores.cpu().detach(),
                camera_pose=model_obs.camera_pose.cpu().detach(),
                camera_K=model_obs.camera_K.cpu().detach(),
                xyz_frame=model_obs.xyz_frame,
            ),
            f,
        )
    print(" > Done saving observation to pickle file.")

def plan_to_frontier(
    start: np.ndarray,
    planner: Planner,
    space: ConfigurationSpace,
    voxel_map: SparseVoxelMap,
    visualize: bool = False,
    try_to_plan_iter: int = 10,
    debug: bool = False,
) -> PlanResult:
    """Find frontier point to move to."""
    # extract goal using fmm planner
    tries = 0
    failed = False
    res = None
    start_is_valid = space.is_valid(start)
    print("\n----------- Planning to frontier -----------")
    print("Start is valid:", start_is_valid)
    if not start_is_valid:
        return PlanResult(False, reason="invalid start state")
    for goal in space.sample_closest_frontier(
        start,
        step_dist=0.5,
        min_dist=0,
        debug=debug,
        verbose=True,
    ):
        if goal is None:
            failed = True
            break
        goal = goal.cpu().numpy()
        print("Start:", start)
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
        print("Planning...")
        res = planner.plan(start, goal, verbose=True)
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
    spot,
    voxel_map,
    planner,
    instance_id,
    visualize=False,
    n_sample=10,
    should_plan: bool = True,
):
    """Navigate to a specific object instance"""
    instances = voxel_map.get_instances()
    instance = instances[instance_id]

    # TODO this can be random
    view = instance.instance_views[-1]
    goal_position = np.asarray(view.pose)
    start = spot.current_position

    print(goal_position)
    if should_plan:
        res = planner.plan(start, goal_position)
        print(goal_position)
        print("Res success:", res.success)
        if res.success:
            spot.execute_plan(res)
        else:
            print("!!! PLANNING FAILED !!!")
    else:
        spot.navigate_to(goal_position, blocking=True)

    if visualize:
        cropped_image = view.cropped_image
        plt.imshow(cropped_image)
        plt.show()
        plt.imsave(f"instance_{instance_id}.png", cropped_image)

    return True

def place_in_an_instance(
    instance_id, spot, voxel_map, place_height=0.3, place_rotation=[0, np.pi / 2, 0]
):
    # Parameters for the placing function from the pointcloud
    ground_normal = torch.tensor([0.0, 0.0, 1])
    nbr_dist = 0.15
    residual_thresh = 0.03

    # Get the pointcloud of the instance
    pc_xyz = voxel_map.get_instances()[instance_id].point_cloud

    # get the location (in global coordinates) of the placeable location
    location, area_prop = nc.find_placeable_location(
        pc_xyz, ground_normal, nbr_dist, residual_thresh
    )

    # Navigate close to that location
    # TODO solve the system of equations to get k such that the distance is .75 meters

    instance_pose = voxel_map.get_instances()[instance_id].instance_views[-1].pose
    vr = np.array([instance_pose[0], instance_pose[1]])
    vp = np.asarray(location[:2])
    k = 1 - (1 / (np.linalg.norm(vp - vr)))
    vf = vr + (vp - vr) * k
    spot.navigate_to(np.array([vf[0], vf[1], instance_pose[2]]), blocking=True)

    # Transform placing position to body frame coordinates
    x, y, yaw = spot.spot.get_xy_yaw()
    local_xyt = xyt_global_to_base(location, np.array([x, y, yaw]))

    # z is the height of the receptacle minus the height of spot + the desired delta for placing
    z = location[2] - spot.spot.body.z + place_height
    local_xyz = np.array([local_xyt[0], local_xyt[1], z])
    rotations = np.array([0, 0, 0])

    # Now we place
    spot.spot.move_gripper_to_point(local_xyz, rotations)
    time.sleep(2)
    spot.spot.rotate_gripper_with_delta(wrist_roll=np.pi / 2)
    spot.spot.open_gripper()

    # reset arm
    time.sleep(1)
    spot.reset_arm()


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
    data = {}
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
    parameters = get_config("src/home_robot_spot/configs/parameters.yaml")[0]
    # Create voxel map
    voxel_map = SparseVoxelMap(
        resolution=parameters["voxel_size"],
        local_radius=parameters["local_radius"],
        obs_min_height=parameters["obs_min_height"],
        obs_max_height=parameters["obs_max_height"],
        obs_min_density=parameters["obs_min_density"],
        smooth_kernel_size=parameters["smooth_kernel_size"],
    )

    # Create kinematic model (very basic for now - just a footprint)
    robot_model = SimpleSpotKinematics()

    # Create navigation space example
    navigation_space = SparseVoxelMapNavigationSpace(
        voxel_map=voxel_map,
        robot=robot_model,
        step_size=parameters["step_size"],
        rotation_step_size=parameters["rotation_step_size"],
        dilate_frontier_size=parameters["dilate_frontier_size"],
        dilate_obstacle_size=parameters["dilate_obstacle_size"],
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

    spot = SpotClient(
        config=spot_config, dock_id=dock, use_midas=parameters["use_midas"], use_zero_depth=parameters['use_zero_depth']
    )
    try:
        # Turn on the robot using the client above
        spot.start()

        print("Sleep 1s")
        time.sleep(1)
        print("Start exploring!")
        x0, y0, theta0 = spot.current_position
        spot.navigate_to([x0, y0, theta0], blocking=True)

        # Start thread to update voxel map
        if parameters["use_async_subscriber"]:
            voxel_map_subscriber = VoxelMapSubscriber(spot, voxel_map, semantic_sensor)
            voxel_map_subscriber.start()
        else:
            # Alternately, update synchronously
            time.sleep(1.5)
            obs = spot.get_rgbd_obs()
            obs = semantic_sensor.predict(obs)
            # TODO: remove debug code
            print(obs.gps, obs.compass)
            voxel_map.add_obs(obs, xyz_frame="world")

        # Do a 360 degree turn to get some observations (this helps debug the robot)
        for i in range(8):
            spot.navigate_to([x0, y0, theta0 + (i + 1) * np.pi / 4], blocking=True)
            if not parameters["use_async_subscriber"]:
                time.sleep(1.5)
                obs = spot.get_rgbd_obs()
                obs = semantic_sensor.predict(obs)
                voxel_map.add_obs(obs, xyz_frame="world")
                print("-", i + 1, "-")
                print("Camera pose =", obs.camera_pose[:3, 3].cpu().numpy())
                print("Base pose =", obs.gps, obs.compass)

        voxel_map.show()
        for step in range(int(parameters["exploration_steps"])):

            print()
            print("-" * 8, step + 1, "/", int(parameters["exploration_steps"]), "-" * 8)

            # Get current position and goal
            start = spot.current_position
            goal = None
            print("Start xyt:", start)
            start_is_valid = navigation_space.is_valid(start)
            print("Start is valid:", start_is_valid)
            print("Start is safe:", voxel_map.xyt_is_safe(start))

            # TODO do something is start is not valid
            if not start_is_valid:
                print("!!!!!!!!"*10)
                print("Start is not valid, exiting exploration...")
                # Move a little bit backwards
                spot.move_base(-0.5,0.5)
                #break

            if parameters["explore_methodical"]:
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

            # Move to the next location
            spot.execute_plan(res)

            if not parameters["use_async_subscriber"]:
                print("Synchronous obs update")
                time.sleep(1.5)
                obs = spot.get_rgbd_obs()
                print("- Observed from coordinates:", obs.gps, obs.compass)
                obs = semantic_sensor.predict(obs)
                timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"
                path = f"{os.environ['HOME_ROBOT_ROOT']}/viz_data/{timestamp}"
                os.makedirs(path, exist_ok=True)
                publish_obs(navigation_space, path, step)
                voxel_map.add_obs(obs, xyz_frame="world")

            if step % 1 == 0 and parameters["visualize"]:
                if parameters["use_async_subscriber"]:
                    print(
                        "Observations processed for the map so far: ",
                        voxel_map_subscriber.current_obs,
                    )
                robot_center = np.zeros(3)
                robot_center[:2] = spot.current_position[:2]
                voxel_map.show(backend="open3d", orig=robot_center, instances=True)

                obstacles, explored = voxel_map.get_2d_map()
                img = (10 * obstacles) + explored
                # start_unnormalized = spot.unnormalize_gps_compass(start)
                navigation_space.draw_state_on_grid(img, start, weight=5)
                if goal is not None:
                    # goal_unnormalized = spot.unnormalize_gps_compass(goal)
                    navigation_space.draw_state_on_grid(img, goal, weight=5)

                plt.imshow(img)
                plt.show()
                plt.imsave(f"exploration_step_{step}.png", img)

        print("Exploration complete!")
        robot_center = np.zeros(3)
        robot_center[:2] = spot.current_position[:2]
        voxel_map.show(backend="open3d", orig=robot_center, instances=True)
        instances = voxel_map.get_instances()
        blacklist = {}
        while True:
            # for debug, sending the robot back to original position
            goto(spot, planner, np.array([x0, y0, theta0]))
            success = False
            pick_instance_id = None
            place_instance_id = None
            if args.enable_vlm == 1:
                # get world_representation for planning
                while True:
                    world_representation = get_obj_centric_world_representation(
                        instances, args.context_length
                    )
                    # ask vlm for plan
                    task = input("please type any task you want the robot to do: ")
                    # task is the prompt, save it
                    data["prompt"] = task
                    sample = vlm.prepare_sample(task, world_representation)
                    plan = vlm.evaluate(sample)
                    print(plan)

                    execute = input(
                        "do you want to execute (replan otherwise)? (y/n): "
                    )
                    if "y" in execute:
                        # now it is hacky to get two instance ids TODO: make it more general for all actions
                        # get pick instance id
                        current_high_level_action = plan.split("; ")[0]
                        pick_instance_id = int(
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
                        if len(plan.split(': ')) > 2:
                            # get place instance id
                            current_high_level_action = plan.split("; ")[2]
                            place_instance_id = int(
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
                            print("place_instance_id", place_instance_id)                   
                        break
            if not pick_instance_id:
                # Navigating to a cup or bottle
                for i, each_instance in enumerate(instances):
                    if vocab.goal_id_to_goal_name[
                        int(each_instance.category_id.item())
                    ] in ["bottle", "cup"]:
                        pick_instance_id = i
                        break
            if not place_instance_id:   
                for i, each_instance in enumerate(instances):
                    if vocab.goal_id_to_goal_name[
                        int(each_instance.category_id.item())
                    ] in ["chair"]:
                        place_instance_id = i
                        break

            if pick_instance_id is None or place_instance_id is None:
                print("No instances found!")
                success = False
            else:
                print("Navigating to instance ")
                print(f"Instance id: {pick_instance_id}")
                success = navigate_to_an_instance(
                    spot,
                    voxel_map,
                    planner,
                    pick_instance_id,
                    visualize=parameters["visualize"],
                )
                print(f"Success: {success}")

                # # try to pick up this instance
                # if success:
                
                # TODO: change the grasp API to be able to grasp from the point cloud / mask of the instance
                # currently it will fail if there are two instances of the same category sitting close to each other
                object_category_name = vocab.goal_id_to_goal_name[
                    int(instances[pick_instance_id].category_id.item())
                ]
                opt = input(f"Grasping {object_category_name}..., y/n?: ")
                if opt == 'n':
                    blacklist[pick_instance_id] = instances[pick_instance_id]
                    del instances[pick_instance_id]
                    continue
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
                if success:
                    # navigate to the place instance
                    print("Navigating to instance ")
                    print(f"Instance id: {place_instance_id}")
                    success = navigate_to_an_instance(
                        spot,
                        voxel_map,
                        planner,
                        place_instance_id,
                        visualize=parameters["visualize"],
                    )

                    breakpoint()
                    place_in_an_instance(place_instance_id, spot, voxel_map, place_height=0.15)
                '''
                # PLACING 

                # Put here the instance to place
                instance = 2

                # Parameters for the placing function from the pointcloud
                ground_normal = torch.tensor([0.0, 0.0, 1])
                nbr_dist = .15
                residual_thresh = 0.03

                # Get the pointcloud of the instance
                pc_xyz = voxel_map.get_instances()[instance].point_cloud

                # get the location (in global coordinates) of the placeable location
                location, area_prop = nc.find_placeable_location(pc_xyz, ground_normal, nbr_dist, residual_thresh)

                # Navigate close to that location
                instance_pose = voxel_map.get_instances()[instance].instance_views[-1].pose
                vr = np.array([instance_pose[0], instance_pose[1]])
                vp = location[:2]
                vf = vr + (vp - vr) * 0.5
                spot.navigate_to(np.array([vf[0], vf[1], instance_pose[2]]), blocking=True)

                # Transform placing position to local coordinates
                x,y,yaw = spot.get_xy_yaw()
                local_xyt = xyt_global_to_base(location, np.array([x,y,yaw]))
                local_xyz = np.array([local_xyt[0], local_xyt[1], location[2]])
                rotations = np.array([0, np.pi/2, 0])
                spot.spot.move_gripper_to_point(local_xyz, rotation, blocking=True)

                pc_xyz, _, _, _ = voxel_map.voxel_pcd.get_pointcloud()
                pc_xyz, pc_rgb = voxel_map.show(backend="open3d", instances=False, orig=np.zeros(3)) 

                instance = 1
                navigate_to_an_instance(spot, voxel_map, planner, instance, True)
                ground_normal = torch.tensor([0.0, 0.0, 1])
                nbr_dist = .15
                residual_thresh = 0.03
                pc_xyz = voxel_map.get_instances()[instance].point_cloud
                location, area_prop = nc.find_placeable_location(pc_xyz, ground_normal, nbr_dist, residual_thresh)
                ans = spot.navigate_to(np.array([location[0], location[1], 0.0]), blocking=True)
                print("location:", location)

                # Now transforming from base to world coordinates
                l

                # visualize pointcloud and add location as red
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(pc_xyz)
                pcd.colors = open3d.utility.Vector3dVector(pc_rgb)
                pcd.colors[location] = [1, 0, 0]
                open3d.visualization.draw_geometries([pcd])


                # TODO> Navigate to that point
                # TODO VISUALIZE THAT POINT
                # ransform point to base coordinates
                # Move armjoint with ik to x,y,z+.02
                '''
                # pick = gaze.get_pick_location()
                # spot.spot.set_arm_joint_positions(pick, travel_time=1)
                # time.sleep(1)
                # spot.spot.open_gripper()
                # time.sleep(2)
                if success:
                    print("Successfully grasped the object!")
                    # exit out of loop without killing script
                    break

    except Exception as e:
        print("Exception caught:")
        print(e)
        raise e

    finally:
        if parameters["write_data"]:
            print("Writing data...")
            pc_xyz, pc_rgb = voxel_map.show(
                backend="open3d", instances=False, orig=np.zeros(3)
            )
            timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"
            pcd_filename = f"spot_output_{timestamp}.pcd"
            pkl_filename = f"spot_output_{timestamp}.pkl"

            # Create pointcloud
            if len(pcd_filename) > 0:
                pcd = numpy_to_pcd(pc_xyz, pc_rgb / 255)
                open3d.io.write_point_cloud(pcd_filename, pcd)
                print(f"... wrote pcd to {pcd_filename}")
            if len(pkl_filename) > 0:
                voxel_map.write_to_pickle_add_data(pkl_filename, data)
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
        default=10,
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
