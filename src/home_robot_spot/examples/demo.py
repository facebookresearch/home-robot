# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import os
import pickle
import random
import shutil
import sys
import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d
import torch
from atomicwrites import atomic_write
from loguru import logger
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
from home_robot.perception.encoders import ClipEncoder
from home_robot.utils.config import get_config, load_config
from home_robot.utils.geometry import xyt_global_to_base
from home_robot.utils.point_cloud import numpy_to_pcd
from home_robot.utils.visualization import get_x_and_y_from_path
from home_robot_spot import SpotClient, VoxelMapSubscriber
from home_robot_spot.grasp_env import GraspController

## Temporary hack until we make accel-cortex pip installable
# sys.path.append("path to accel-cortext base folder")
print("Make sure path to accel-cortex base folder is set")
sys.path.append(os.path.expanduser("~/src/accel-cortex"))
import grpc
import src.rpc
import task_rpc_env_pb2
from src.utils.observations import ObjectImage, Observations, ProtoConverter
from task_rpc_env_pb2_grpc import AgentgRPCStub


# NOTE: this requires 'pip install atomicwrites'
def publish_obs(model: SparseVoxelMapNavigationSpace, path: str):
    timestep = len(model.voxel_map.observations) - 1
    with atomic_write(f"{path}/{timestep}.pkl", mode="wb") as f:
        instances = model.voxel_map.get_instances()
        model_obs = model.voxel_map.observations[-1]
        if len(instances) > 0:
            bounds, names = zip(*[(v.bounds, v.category_id) for v in instances])
            bounds = torch.stack(bounds, dim=0)
            names = torch.stack(names, dim=0).unsqueeze(-1)
        else:
            bounds = torch.zeros(0, 3, 2)
            names = torch.zeros(0, 1)
        logger.info(f"Saving observation to pickle file...{f'{path}/{timestep}.pkl'}")
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
                box_bounds=bounds,
                box_names=names,
            ),
            f,
        )
    # logger.success("Done saving observation to pickle file.")


def get_close(instance_id, spot, voxel_map, dist=0.25):
    # Parameters for the placing function from the pointcloud
    ground_normal = torch.tensor([0.0, 0.0, 1])
    nbr_dist = 0.15
    residual_thresh = 0.03

    # # Get the pointcloud of the instance
    pc_xyz = voxel_map.get_instances()[instance_id].point_cloud

    # # get the location (in global coordinates) of the placeable location
    location, area_prop = nc.find_placeable_location(
        pc_xyz, ground_normal, nbr_dist, residual_thresh
    )

    # # Navigate close to that location
    # # TODO solve the system of equations to get k such that the distance is .75 meters

    instance_pose = voxel_map.get_instances()[instance_id].instance_views[-1].pose
    vr = np.array([instance_pose[0], instance_pose[1]])
    vp = np.asarray(location[:2])
    k = 1 - (dist / (np.linalg.norm(vp - vr)))
    vf = vr + (vp - vr) * k
    return instance_pose, location, vf


def place_in_an_instance(
    instance_pose,
    location,
    vf,
    spot,
    place_height=0.3,
    place_rotation=[0, np.pi / 2, 0],
):
    # TODO: Check if vf is correct
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
    time.sleep(0.5)
    spot.reset_arm()


def get_obj_centric_world_representation(instance_memory, max_context_length):
    """Get version that LLM can handle - convert images into torch if not already"""
    obs = Observations(object_images=[])
    for global_id, instance in enumerate(instance_memory):
        instance_crops = instance.instance_views
        crop = random.sample(instance_crops, 1)[0].cropped_image
        if isinstance(crop, np.ndarray):
            crop = torch.from_numpy(crop)
        obs.object_images.append(
            ObjectImage(
                crop_id=global_id,
                image=crop,
            )
        )
    # TODO: the model currenly can only handle 20 crops
    if len(obs.object_images) > max_context_length:
        logger.warning(
            "\nWarning: this version of minigpt4 can only handle limited size of crops -- sampling a subset of crops from the instance memory..."
        )
        obs.object_images = random.sample(obs.object_images, max_context_length)

    return obs


class SpotDemoAgent:
    def __init__(
        self, parameters, spot_config, dock: Optional[int] = None, path: str = None
    ):
        self.parameters = parameters
        self.encoder = ClipEncoder(self.parameters["clip"])
        self.voxel_map = SparseVoxelMap(
            resolution=parameters["voxel_size"],
            local_radius=parameters["local_radius"],
            obs_min_height=parameters["obs_min_height"],
            obs_max_height=parameters["obs_max_height"],
            obs_min_density=parameters["obs_min_density"],
            smooth_kernel_size=parameters["smooth_kernel_size"],
            encoder=self.encoder,
        )
        logger.info("Created SparseVoxelMap")
        # Create kinematic model (very basic for now - just a footprint)
        self.robot_model = SimpleSpotKinematics()
        # Create navigation space example
        self.navigation_space = SparseVoxelMapNavigationSpace(
            voxel_map=self.voxel_map,
            robot=self.robot_model,
            step_size=parameters["step_size"],
            rotation_step_size=parameters["rotation_step_size"],
            dilate_frontier_size=parameters["dilate_frontier_size"],
            dilate_obstacle_size=parameters["dilate_obstacle_size"],
        )
        logger.log("DEMO", "Created navigation space and environment")
        logger.log("DEMO", f"{self.navigation_space}")

        # Create segmentation sensor and load config. Returns config from file, as well as a OvmmPerception object that can be used to label scenes.
        logger.log("DEMO", "Loading configuration")
        config = load_config(visualize=False)

        print("- Create and load vocabulary and perception model")
        self.semantic_sensor = OvmmPerception(config, 0, True, module="detic")
        obj_name_to_id, rec_name_to_id = read_category_map_file(
            config.ENVIRONMENT.category_map_file
        )
        self.vocab = build_vocab_from_category_map(obj_name_to_id, rec_name_to_id)
        self.semantic_sensor.update_vocabulary_list(self.vocab, 0)
        self.semantic_sensor.set_vocabulary(0)

        self.planner = Shortcut(
            RRTConnect(self.navigation_space, self.navigation_space.is_valid)
        )
        self.spot = SpotClient(
            config=spot_config,
            dock_id=dock,
            use_midas=parameters["use_midas"],
            use_zero_depth=parameters["use_zero_depth"],
        )
        self.spot_config = spot_config
        self.path = path

    def plan_to_frontier(
        self,
        start: Optional[np.ndarray] = None,
        visualize: bool = False,
        try_to_plan_iter: int = 10,
        debug: bool = False,
    ) -> PlanResult:
        """Find frontier point to move to."""
        # extract goal using fmm planner
        tries = 0
        failed = False
        res = None
        start = self.spot.current_position
        start_is_valid = self.navigation_space.is_valid(start)
        logger.log("DEMO", "\n----------- Planning to frontier -----------")
        if not start_is_valid:
            logger.error("Start is valid: {}", start_is_valid)
            self.spot.navigate_to([-0.25, 0, 0], relative=True)
            return PlanResult(False, reason="invalid start state")
        else:
            logger.success("Start is valid: {}", start_is_valid)

        for goal in self.navigation_space.sample_closest_frontier(
            start,
            step_dist=self.parameters["frontier_step_dist"],
            min_dist=self.parameters["frontier_min_dist"],
            debug=debug,
            verbose=True,
        ):
            if goal is None:
                failed = True
                break
            goal = goal.cpu().numpy()
            logger.info("Start: {}", start)
            logger.info("Sampled Goal: {}", goal)
            show_goal = np.zeros(3)
            show_goal[:2] = goal[:2]
            goal_is_valid = self.navigation_space.is_valid(goal)
            if start_is_valid:
                logger.success("Start is valid: {}", start_is_valid)
            else:
                raise RuntimeError(f"Start is not valid: {start_is_valid}, {start=}")
            if goal_is_valid:
                logger.success("Goal is valid: {}", goal_is_valid)
            if not goal_is_valid:
                logger.error("Goal is valid: {}, Resampling goal", goal_is_valid)
                continue
            # plan to the sampled goal
            logger.log("DEMO", "Planning...")
            res = self.planner.plan(start, goal, verbose=True)
            if res.success:
                logger.success("Found plan: {}", res.success)
                for i, node in enumerate(res.trajectory):
                    logger.info(f"{i}, {node.state}")
            else:
                logger.error("Found plan: {}", res.success)
            if visualize:
                obstacles, explored = self.voxel_map.get_2d_map()
                img = (10 * obstacles) + explored
                self.navigation_space.draw_state_on_grid(img, start, weight=5)
                self.navigation_space.draw_state_on_grid(img, goal, weight=5)
                plt.imshow(img)
                if res.success:
                    path = self.voxel_map.plan_to_grid_coords(res)
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

    def sample_random_frontier(self) -> np.ndarray:
        """Get a random frontier point"""
        goal = next(
            self.navigation_space.sample_random_frontier(
                min_size=self.parameters["min_size"],
                max_size=self.parameters["max_size"],
            )
        )
        goal = goal.cpu().numpy()
        return goal

    def rotate_in_place(self):
        # Do a 360 degree turn to get some observations (this helps debug the robot)
        logger.info("Rotate in place")
        x0, y0, theta0 = self.spot.current_position
        for i in range(8):
            self.spot.navigate_to([x0, y0, theta0 + (i + 1) * np.pi / 4], blocking=True)
            if not self.parameters["use_async_subscriber"]:
                self.update()

    def update(self, step=0):
        time.sleep(0.1)
        obs = self.spot.get_rgbd_obs()
        print("Observed from coordinates:", obs.gps, obs.compass)
        obs = self.semantic_sensor.predict(obs)
        self.voxel_map.add_obs(obs, xyz_frame="world")
        filename = f"{self.path}/viz_data/"
        os.makedirs(filename, exist_ok=True)
        publish_obs(self.navigation_space, filename)

    def visualize(self, start, goal, step):
        """Update visualization for demo"""
        if self.parameters["use_async_subscriber"]:
            print(
                "Observations processed for the map so far: ",
                self.voxel_map_subscriber.current_obs,
            )
        robot_center = np.zeros(3)
        robot_center[:2] = self.spot.current_position[:2]
        self.voxel_map.show(backend="open3d", orig=robot_center, instances=True)

        obstacles, explored = self.voxel_map.get_2d_map()
        img = (10 * obstacles) + explored
        # start_unnormalized = spot.unnormalize_gps_compass(start)
        self.navigation_space.draw_state_on_grid(img, start, weight=5)
        if goal is not None:
            # goal_unnormalized = spot.unnormalize_gps_compass(goal)
            self.navigation_space.draw_state_on_grid(img, goal, weight=5)

        plt.imshow(img)
        plt.show()
        plt.imsave(f"exploration_step_{step}.png", img)

    def navigate_to_an_instance(
        self,
        instance_id,
        visualize=False,
        n_sample=10,
        should_plan: bool = True,
    ):
        """Navigate to a specific object instance"""
        instances = self.voxel_map.get_instances()
        instance = instances[instance_id]

        # TODO this can be random
        view = instance.instance_views[-1]
        goal_position = np.asarray(view.pose)
        start = self.spot.current_position

        print(f"{goal_position=}")
        print(f"{start=}")
        print(f"{instance.bounds=}")
        if should_plan:
            start_is_valid = self.navigation_space.is_valid(start)
            goal_is_valid = self.navigation_space.is_valid(goal_position)
            print(f"{start_is_valid=}")
            print(f"{goal_is_valid=}")
            res = self.planner.plan(start, goal_position)
            if res.success:
                logger.success("Res success: {}", res.success)
                self.spot.execute_plan(
                    res,
                    pos_err_threshold=self.parameters["trajectory_pos_err_threshold"],
                    rot_err_threshold=self.parameters["trajectory_rot_err_threshold"],
                )
            else:
                logger.error("Res success: {}, !!!PLANNING FAILED!!!", res.success)
                should_plan = False
        if not should_plan:
            logger.info("Navigating to goal position: {}", goal_position)
            self.spot.navigate_to(goal_position, blocking=True)

        if visualize:
            cropped_image = view.cropped_image
            plt.imshow(cropped_image)
            plt.show()
            plt.imsave(f"instance_{instance_id}.png", cropped_image)

        return True

    def goto(self, goal: np.ndarray):
        """Send the spot to the correct location."""
        start = self.spot.current_position

        #  Build plan
        res = self.planner.plan(start, goal)
        logger.info("[demo.goto] Goal: {}", goal)
        if res:
            logger.success("[demo.goto] Res success: {}", res.success)
        else:
            logger.error("[demo.goto] Res success: {}", res.success)
        # print("Res success:", res.success)
        # Move to the next location
        if res.success:
            logger.info("[demo.goto] Following the plan to goal")
            self.spot.execute_plan(
                res,
                pos_err_threshold=self.parameters["trajectory_pos_err_threshold"],
                rot_err_threshold=self.parameters["trajectory_rot_err_threshold"],
                verbose=True,
            )
        else:
            logger.warning("[demo.goto] Just go ahead and try it anyway")
            self.spot.navigate_to(goal)
        return res

    def run_task(self, stub, center, data):
        """Actually use VLM to perform task

        Args:
            stub: VLM connection if available
            center: 3d point x y theta in se(2)
            data: used by vlm I guess"""

        robot_center = np.zeros(3)
        robot_center[:2] = self.spot.current_position[:2]
        self.voxel_map.show(backend="open3d", orig=robot_center, instances=True)
        instances = self.voxel_map.get_instances()
        blacklist = []
        while True:
            # for debug, sending the robot back to original position
            self.goto(center)
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
                    output = stub.stream_act_on_observations(
                        ProtoConverter.wrap_obs_iterator(
                            episode_id=random.randint(1, 1000000),
                            obs=world_representation,
                            goal=task,
                        )
                    )
                    plan = output.action
                    print(plan)

                    execute = input(
                        "do you want to execute (replan otherwise)? (y/n): "
                    )
                    if "y" in execute:
                        # now it is hacky to get two instance ids TODO: make it more general for all actions
                        # get pick instance id
                        current_high_level_action = plan.split("; ")[0]
                        pick_instance_id = int(
                            world_representation.object_images[
                                int(
                                    current_high_level_action.split("(")[1]
                                    .split(")")[0]
                                    .split(", ")[0]
                                    .split("_")[1]
                                )
                            ].crop_id
                        )
                        if len(plan.split(": ")) > 2:
                            # get place instance id
                            current_high_level_action = plan.split("; ")[2]
                            place_instance_id = int(
                                world_representation.object_images[
                                    int(
                                        current_high_level_action.split("(")[1]
                                        .split(")")[0]
                                        .split(", ")[0]
                                        .split("_")[1]
                                    )
                                ].crop_id
                            )
                            print("place_instance_id", place_instance_id)
                        break
            if not pick_instance_id:
                # Navigating to a cup or bottle
                for i, each_instance in enumerate(instances):
                    if (
                        self.vocab.goal_id_to_goal_name[
                            int(each_instance.category_id.item())
                        ]
                        in self.parameters["pick_categories"]
                    ):
                        pick_instance_id = i
                        break
            if not place_instance_id:
                for i, each_instance in enumerate(instances):
                    if (
                        self.vocab.goal_id_to_goal_name[
                            int(each_instance.category_id.item())
                        ]
                        in self.parameters["place_categories"]
                    ):
                        place_instance_id = i
                        break

            if pick_instance_id is None or place_instance_id is None:
                print("No instances found!")
                success = False
                # TODO add all the items here
                objects = {}
                for i in range(len(instances)):
                    objects[
                        str(
                            (
                                self.vocab.goal_id_to_goal_name[
                                    int(instances[i].category_id.item())
                                ]
                            )
                        )
                    ] = i
                print(objects)
                # TODO: Add better handling
                if pick_instance_id is None:
                    new_id = input(
                        "enter a new instance to pick up from the list above: "
                    )
                    if isinstance(new_id, int):
                        pick_instance_id = new_id
                        break
                if place_instance_id is None:
                    new_id = input(
                        "enter a new instance to place from the list above: "
                    )
                    if isinstance(new_id, int):
                        place_instance_id = new_id
                        break
                success = False
            else:
                print("Navigating to instance ")
                print(f"Instance id: {pick_instance_id}")
                success = self.navigate_to_an_instance(
                    pick_instance_id,
                    visualize=self.parameters["visualize"],
                )
                print(f"Success: {success}")

                # # try to pick up this instance
                # if success:

                # TODO: change the grasp API to be able to grasp from the point cloud / mask of the instance
                # currently it will fail if there are two instances of the same category sitting close to each other
                object_category_name = self.vocab.goal_id_to_goal_name[
                    int(instances[pick_instance_id].category_id.item())
                ]
                opt = input(f"Grasping {object_category_name}..., y/n?: ")
                if opt == "n":
                    blacklist.append(pick_instance_id)
                    del instances[pick_instance_id]
                    continue
                gaze = GraspController(
                    config=self.spot_config,
                    spot=self.spot.spot,
                    objects=[[object_category_name]],
                    confidence=0.1,
                    show_img=False,
                    top_grasp=False,
                    hor_grasp=True,
                )
                self.spot.open_gripper()
                time.sleep(0.5)
                logger.log("DEMO", "Resetting environment...")
                # TODO: have a better way to reset the environment

                obj_pose = instances[pick_instance_id].instance_views[-1].pose
                xy = np.array([obj_pose[0], obj_pose[1]])
                curr_pose = self.spot.current_position
                vr = np.array([curr_pose[0], curr_pose[1]])
                distance = np.linalg.norm(xy + vr)
                if distance > 2.0:
                    instance_pose, location, vf = get_close(
                        pick_instance_id, self.spot, self.voxel_map
                    )
                    logger.info("Navigating closer to the object")
                    self.spot.navigate_to(
                        np.array([vf[0], vf[1], instance_pose[2]]), blocking=True
                    )
                time.sleep(0.5)
                success = gaze.gaze_and_grasp()
                time.sleep(0.5)
                if success:
                    # TODO: @JAY make placing cleaner
                    # navigate to the place instance
                    print("Navigating to instance ")
                    print(f"Instance id: {place_instance_id}")
                    success = self.navigate_to_an_instance(
                        place_instance_id,
                        visualize=self.parameters["visualize"],
                    )
                    place_location = self.vocab.goal_id_to_goal_name[
                        int(instances[place_instance_id].category_id.item())
                    ]
                    instance_pose, location, vf = get_close(
                        place_instance_id, self.spot, self.voxel_map, dist=0.5
                    )
                    logger.info(
                        "placing {object} at {place}",
                        object=object_category_name,
                        place=place_location,
                    )
                    place_in_an_instance(instance_pose, location, vf, self.spot)
                """
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
                """
                if success:
                    logger.success("Successfully grasped the object!")
                    self.goto(center)
                    # exit out of loop without killing script
                    break


# def main(dock: Optional[int] = 549):
def main(dock: Optional[int] = None, args=None):
    level = logger.level("DEMO", no=38, color="<yellow>", icon="🤖")
    print(f"{level=}")
    data: Dict[str, List[str]] = {}
    if args.enable_vlm == 1:
        channel = grpc.insecure_channel(
            f"{args.vlm_server_addr}:{args.vlm_server_port}"
        )
        stub = AgentgRPCStub(channel)
    else:
        # No vlm to use, just default behavior
        stub = None

    # TODO add this to config
    spot_config = get_config("src/home_robot_spot/configs/default_config.yaml")[0]
    if args.location == "pit":
        parameters = get_config("src/home_robot_spot/configs/parameters.yaml")[0]
    if args.location == 'fre':
         parameters = get_config("src/home_robot_spot/configs/parameters_fre.yaml")[0]
    else:
        logger.critical(f"Location {args.location} is invalid, please enter a valid location")
    timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"
    path = os.path.expanduser(f"data/hw_exps/spot/{timestamp}")
    logger.add(f"{path}/{timestamp}.log", backtrace=True, diagnose=True)
    os.makedirs(path, exist_ok=True)
    logger.info("Saving viz data to {}", path)
    demo = SpotDemoAgent(parameters, spot_config, dock, path)
    spot = demo.spot
    voxel_map = demo.voxel_map
    semantic_sensor = demo.semantic_sensor
    navigation_space = demo.navigation_space
    planner = demo.planner
    start = None
    goal = None

    try:
        # Turn on the robot using the client above
        spot.start()
        logger.success("Spot started")
        logger.info("Sleep 1s")
        time.sleep(0.5)
        logger.info("Start exploring!")
        x0, y0, theta0 = spot.current_position

        # Start thread to update voxel map
        if parameters["use_async_subscriber"]:
            voxel_map_subscriber = VoxelMapSubscriber(spot, voxel_map, semantic_sensor)
            voxel_map_subscriber.start()
        else:
            demo.update()

        demo.rotate_in_place()

        voxel_map.show()
        for step in range(int(parameters["exploration_steps"])):
            # logger.log("DEMO", "\n----------- Step {} -----------", step + 1)
            print(
                "-" * 20, step + 1, "/", int(parameters["exploration_steps"]), "-" * 20
            )

            # Get current position and goal
            start = spot.current_position
            goal = None
            logger.info("Start xyt: {}", start)
            start_is_valid = navigation_space.is_valid(start)
            if start_is_valid:
                logger.success("Start is valid: {}", start_is_valid)
            else:
                # TODO do something is start is not valid
                logger.error("!!!!!!!! INVALID START POSITION !!!!!!")
                spot.navigate_to([-0.25, 0, 0], relative=True)
                continue
            logger.info("Start is safe: {}", voxel_map.xyt_is_safe(start))

            if parameters["explore_methodical"]:
                logger.info("Generating the next closest frontier point...")
                res = demo.plan_to_frontier()
                if not res.success:
                    logger.warning(res.reason)
                    logger.debug("Switching to random exploration")
                    goal = next(
                        navigation_space.sample_random_frontier(
                            min_size=parameters["min_size"],
                            max_size=parameters["max_size"],
                        )
                    )
                    goal = goal.cpu().numpy()
                    goal_is_valid = navigation_space.is_valid(goal)
                    logger.info(f" Goal is valid: {goal_is_valid}")
                    if not goal_is_valid:
                        # really we should sample a new goal
                        continue

                    #  Build plan
                    res = planner.plan(start, goal)
                    logger.info(goal)
                    if res.success:
                        logger.success("Res success: {}", res.success)
                    else:
                        logger.error("Res success: {}", res.success)
            else:
                logger.info(
                    "picking a random frontier point and trying to move there..."
                )
                # Sample a goal in the frontier (TODO change to closest frontier)
                goal = demo.sample_random_frontier()
                goal_is_valid = navigation_space.is_valid(goal)
                logger.info(
                    f" Goal is valid: {goal_is_valid}",
                )
                if not goal_is_valid:
                    # really we should sample a new goal
                    continue

                #  Build plan
                res = planner.plan(start, goal)
                logger.info(goal)
                if res.success:
                    logger.success("Res success: {}", res.success)
                else:
                    logger.error("Res success: {}", res.success)

            if res.success:
                spot.execute_plan(
                    res,
                    pos_err_threshold=parameters["trajectory_pos_err_threshold"],
                    rot_err_threshold=parameters["trajectory_rot_err_threshold"],
                )
            else:
                logger.warning("Just go ahead and try it anyway")
                spot.navigate_to(goal)

            if not parameters["use_async_subscriber"]:
                demo.update(step + 1)

            if step % 1 == 0 and parameters["visualize"]:
                demo.visualize(start, goal, step)

        logger.info("Exploration complete!")
        demo.run_task(stub, center=np.array([x0, y0, theta0]), data=data)

    except Exception as e:
        logger.critical("Exception caught: {}", e)
        raise e

    finally:
        if parameters["write_data"]:
            if start is None:
                start = demo.spot.current_position
            if voxel_map.get_instances() is not None:
                pc_xyz, pc_rgb = voxel_map.show(
                    backend="open3d", instances=False, orig=np.zeros(3)
                )
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--enable_vlm",
        default=1,
        type=int,
        help="Enable loading Minigpt4. 1 == use vlm, 0 == test without vlm",
    )
    parser.add_argument(
        "--task",
        default="find a green bottle",
        help="Specify any task in natural language for VLM",
    )
    parser.add_argument(
        "--vlm_server_addr",
        default="localhost",
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
        "--context_length",
        default=20,
        help="Maximum number of images the vlm can reason about",
    )
    parser.add_argument(
        "--location", "-l"
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
    args = parser.parse_args()
    main(args=args)
