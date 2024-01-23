# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import math
import os
import pickle
import random
import shutil
import sys
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d
import torch
from atomicwrites import atomic_write
from loguru import logger

import home_robot.utils.planar as nc
#from examples.demo_utils.mock_agent import MockSpotDemoAgent

# Simple IO tool for robot agents
from home_robot.agent.multitask.robot_agent import RobotAgent, publish_obs
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
from home_robot.perception import create_semantic_sensor
from home_robot.perception.encoders import ClipEncoder
from home_robot.utils.config import Config, get_config, load_config
from home_robot.utils.demo_chat import (
    DemoChat,
    start_demo_ui_server,
    stop_demo_ui_server,
)
from home_robot.utils.geometry import xyt_global_to_base
from home_robot.utils.point_cloud import numpy_to_pcd
from home_robot.utils.rpc import (
    get_obj_centric_world_representation,
    get_output_from_world_representation,
    get_vlm_rpc_stub,
    parse_pick_and_place_plan,
)
from home_robot.utils.threading import Interval
from home_robot.utils.visualization import get_x_and_y_from_path
from home_robot_spot import SpotClient, VoxelMapSubscriber
from home_robot_spot.grasp_env import GraspController


class SpotDemoAgent(RobotAgent):
    """Demo agent for use in Spot experiments. Extends the base robot demo agent. Work in progress code."""

    def __init__(
        self,
        parameters: Dict[str, Any],
        spot_config: Config,
        dock: Optional[int] = None,
        path: str = None,
    ):
        self.spot_config = spot_config
        self.path = path
        self.current_state = "WAITING"
        self.obs_count = 0
        self.parameters = parameters
        if self.parameters["encoder"] == "clip":
            self.encoder = ClipEncoder(self.parameters["clip"])
        else:
            raise NotImplementedError(
                f"unsupported encoder {self.parameters['encoder']}"
            )
        self.vis_folder = f"{self.path}/viz_data/"
        self.voxel_map = SparseVoxelMap(
            resolution=parameters["voxel_size"],
            local_radius=parameters["local_radius"],
            obs_min_height=parameters["obs_min_height"],
            obs_max_height=parameters["obs_max_height"],
            min_depth=parameters["min_depth"],
            max_depth=parameters["max_depth"],
            obs_min_density=parameters["obs_min_density"],
            smooth_kernel_size=parameters["smooth_kernel_size"],
            encoder=self.encoder,
        )
        logger.info("Created SparseVoxelMap")
        self.step = 0
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

        logger.info("- Create and load vocabulary and perception model")
        _, self.semantic_sensor = create_semantic_sensor(
            device=0,
            verbose=True,
            module="detic",
            module_kwargs={"confidence_threshold": 0.6},
            category_map_file=self.parameters["category_map_file"],
            config=config,
        )
        self.vocab = self.semantic_sensor.current_vocabulary

        os.makedirs(f"{self.path}/viz_data", exist_ok=True)
        os.makedirs(f"{self.path}/viz_data/instances/", exist_ok=True)
        with atomic_write(f"{self.path}/viz_data/vocab_dict.pkl", mode="wb") as f:
            pickle.dump(self.semantic_sensor.seg_id_to_name, f)

        self.planner = Shortcut(
            RRTConnect(self.navigation_space, self.navigation_space.is_valid),
            shortcut_iter=self.parameters["shortcut_iter"],
        )
        self.spot = SpotClient(
            config=spot_config,
            dock_id=dock,
            use_midas=parameters["use_midas"],
            use_zero_depth=parameters["use_zero_depth"],
        )

        # Create grasp object
        self.gaze = GraspController(
            config=self.spot_config,
            spot=self.spot.get_bd_spot_client(),
            objects=None,
            confidence=0.1,
            show_img=True,
            top_grasp=False,
            hor_grasp=True,
        )

        # if parameters["chat"]:
        #     self.chat = DemoChat(f"{self.path}/viz_data/demo_chat.json")
        #     if self.parameters["limited_obs_publish_sleep"] > 0:
        #         self._publisher = Interval(
        #             self.publish_limited_obs,
        #             sleep_time=self.parameters["limited_obs_publish_sleep"],
        #         )
        #     else:
        #         self._publisher = None
        # else:
        self.chat = None
        self._publisher = None

    def start(self):
        if self._publisher is not None:
            self._publisher.start()

    def finish(self):
        if self._publisher is not None:
            print("- Stopping publisher...")
            self._publisher.cancel()
        print("... Done.")

    def set_objects_for_grasping(self, objects: List[List[str]]):
        """Set the objects used for grasping"""
        self.gaze.set_objects(objects)

    def backup_from_invalid_state(self):
        """Helper function to get the robot unstuck (it is too close to geometry)"""
        self.spot.navigate_to([-0.25, 0, 0], relative=True, blocking=True)

    def should_visualize(self) -> bool:
        """Returns true if we are expected to do visualizations in this script (not externally)"""
        return self.parameters["visualize"]

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
            self.backup_from_invalid_state()
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
        old_state = self.current_state
        self.current_state = "SCANNING"
        x0, y0, theta0 = self.spot.current_position
        for i in range(8):
            self.spot.navigate_to([x0, y0, theta0 + (i + 1) * np.pi / 4], blocking=True)
            if not self.parameters["use_async_subscriber"]:
                self.update()

        # Should we display after spinning? If visualize is true we will
        if self.should_visualize():
            self.voxel_map.show()
        self.current_state = old_state

    def say(self, msg: str):
        """Provide input either on the command line or via chat client"""
        if self.chat is not None:
            self.chat.output(msg)
        else:
            print(msg)

    def ask(self, msg: str) -> str:
        """Receive input from the user either via the command line or something else"""
        if self.chat is not None:
            return self.chat.input(msg)
        else:
            return input(msg)

    def update(self):
        """Update sensor measurements"""
        time.sleep(1)
        obs = self.spot.get_rgbd_obs()
        logger.info("Observed from coordinates:", obs.gps, obs.compass)
        obs = self.semantic_sensor.predict(obs)
        self.voxel_map.add_obs(obs, xyz_frame="world")
        self.publish_full_obs()
        # os.makedirs(self.vis_folder, exist_ok=True)
        # self.obs_count += 1
        # publish_obs(
        #     self.navigation_space, self.vis_folder, self.current_state, self.obs_count, instance_id=None
        # )
        self.step += 1

    def publish_full_obs(self, instance_id=None):
        os.makedirs(self.vis_folder, exist_ok=True)
        self.obs_count += 1
        publish_obs(
            self.navigation_space,
            self.vis_folder,
            self.current_state,
            self.obs_count,
            instance_id,
        )

    def publish_limited_obs(self):
        """Used to send a small update to the remote UI"""
        logger.trace(f"{self.current_state=}")
        if self.current_state == "WAITING":
            return True
        obs = self.spot.get_rgbd_obs()
        self.obs_count += 1
        with atomic_write(f"{self.vis_folder}/{self.obs_count}.pkl", mode="wb") as f:
            logger.trace(
                f"Saving limited observation to pickle file: {f'{self.path}/viz_data/{self.obs_count}.pkl'}"
            )
            pickle.dump(
                dict(
                    obs=obs,
                    limited_obs=True,
                    current_state=self.current_state,
                ),
                f,
            )
        return True

    def visualize(self, start, goal, step=0):
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

    def place_in_an_instance(
        self,
        instance_id: int,
        instance_pose,
        location,
        vf,
        place_height=0.3,
        place_rotation=[0, np.pi / 2, 0],
    ):
        """Move to a position to place in an environment."""
        # TODO: Check if vf is correct
        # if self.parameters["use_get_close"]:
        old_state = self.current_state
        self.current_state = "PLACE"
        self.spot.navigate_to(instance_pose, blocking=True)
        # Compute distance
        dxy = location[:2] - instance_pose[:2]
        theta = math.atan2(dxy[1], dxy[0])
        print(f"Now rotate towards placement location with {theta=}...")
        self.spot.navigate_to(
            np.array([instance_pose[0], instance_pose[1], theta]), blocking=True
        )
        dist_to_place = np.linalg.norm(dxy)
        dist_to_move = max(0, dist_to_place - self.parameters["place_offset"])
        print(f"Moving {dist_to_move} closer to {location}...")
        self.spot.navigate_to(
            np.array([dist_to_move, 0, 0]), relative=True, blocking=True
        )
        time.sleep(0.1)
        # Transform placing position to body frame coordinates
        local_xyt = xyt_global_to_base(location, self.spot.current_position)

        # z is the height of the receptacle minus the height of spot + the desired delta for placing
        z = location[2] - self.spot.spot.body.z + place_height
        local_xyz = np.array([local_xyt[0], local_xyt[1], z])
        rotations = np.array([0, 0, 0])
        local_xyz[0] += self.parameters["gripper_offset_x"]

        # Now we place
        self.spot.spot.move_gripper_to_point(local_xyz, rotations)
        # arm = self.spot.spot.get_arm_joint_positions()
        # arm[-1] = place_rotation[-1]
        # arm[-2] = place_rotation[0]
        # self.spot.spot.set_arm_joint_positions(arm, travel_time=1.5)
        time.sleep(2)
        self.spot.spot.open_gripper()

        # reset arm
        time.sleep(0.5)
        self.spot.reset_arm()
        self.current_state = old_state

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
        view = instance.get_best_view(metric=self.parameters["best_view_metric"])
        goal_position = np.asarray(view.pose)
        start = self.spot.current_position
        start_is_valid = self.navigation_space.is_valid(start)
        old_state = self.current_state
        self.current_state = "NAV_TO_INSTANCE"

        print("\n----- NAVIGATE TO THE RIGHT INSTANCE ------")
        print("Start is valid:", start_is_valid)
        print(f"{goal_position=}")
        print(f"{start=}")
        print(f"{instance.bounds=}")
        if should_plan and start_is_valid:
            # TODO: this is a bad name for this variable
            print("listing all views for your convenience:")
            for j, view in enumerate(instance.instance_views):
                print(j, view.cam_to_world)
            res = None
            mask = self.voxel_map.mask_from_bounds(instance.bounds)
            for goal in self.navigation_space.sample_near_mask(
                mask, radius_m=self.parameters["pick_place_radius"]
            ):
                goal = goal.cpu().numpy()
                print("       Start:", start)
                print("Sampled Goal:", goal)
                show_goal = np.zeros(3)
                show_goal[:2] = goal[:2]
                goal_is_valid = self.navigation_space.is_valid(goal)
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

            if res is not None and res.success:
                logger.success("Res success: {}", res.success)
                self.spot.execute_plan(
                    res,
                    pos_err_threshold=self.parameters["trajectory_pos_err_threshold"],
                    rot_err_threshold=self.parameters["trajectory_rot_err_threshold"],
                    per_step_timeout=self.parameters["trajectory_per_step_timeout"],
                    verbose=False,
                )
                goal_position = goal
            else:
                logger.error("Res success: {}, !!!PLANNING FAILED!!!", res)
                should_plan = False

        # Finally, navigate to the final position
        logger.info(
            "Navigating to goal position: {}, start = {}",
            goal_position,
            self.spot.current_position,
        )
        self.spot.navigate_to(goal_position, blocking=True)
        logger.info(
            "Navigating to goal position: {}, reached = {}",
            goal_position,
            self.spot.current_position,
        )
        logger.info(f"Used motion planning to find gaze location: {should_plan}")

        if visualize:
            cropped_image = view.cropped_image
            plt.imshow(cropped_image)
            plt.show()
            plt.imsave(
                f"instance_{instance_id}.png", cropped_image.cpu().numpy() / 255.0
            )

        logger.info("Wait for a second and then try to grasp/place!")
        time.sleep(1.0)
        self.current_state = old_state
        return True

    def goto(self, goal: np.ndarray):
        """Send the spot to the correct location from wherever it is. Try to plan there."""
        start = self.spot.current_position

        #  Build plan
        res = self.planner.plan(start, goal)
        logger.info("[demo.goto] Goal: {} From: {}", goal, start)
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
                verbose=False,
            )
        else:
            logger.warning("[demo.goto] Just go ahead and try it anyway")
            self.spot.navigate_to(goal)
        return res

    def get_pose_for_best_view(self, instance_id: int) -> torch.Tensor:
        """Get the best view for a particular object by whatever metric we use, and return the associated pose (as an xyt)"""
        instances = self.voxel_map.get_instances()
        return (
            instances[instance_id]
            .get_best_view(metric=self.parameters["best_view_metric"])
            .pose
        )

    def get_close(self, instance_id, dist=0.25):
        """Compute a nearer location to {instance_id} to move to and go there.

        Returns:
            instance_pose (torch.Tensor): the view we want to start at
            location (np.ndarray): xyz to place at if we do that
            vf: viewpoint made closer by {dist}"""
        # Parameters for the placing function from the pointcloud
        ground_normal = torch.tensor([0.0, 0.0, 1])
        nbr_dist = self.parameters["nbr_dist"]
        residual_thresh = self.parameters["residual_thresh"]

        # # Get the pointcloud of the instance
        pc_xyz = self.voxel_map.get_instances()[instance_id].point_cloud

        # # get the location (in global coordinates) of the placeable location
        location, area_prop = nc.find_placeable_location(
            pc_xyz, ground_normal, nbr_dist, residual_thresh
        )

        # Add parameter to improve placement performance
        if self.parameters["force_place_at_center"]:
            # We will still use the height from the navigation client
            mean_xyz = pc_xyz.mean(dim=0)
            location[0] = mean_xyz[0]
            location[1] = mean_xyz[1]

        # # Navigate close to that location
        # # TODO solve the system of equations to get k such that the distance is .75 meters
        instance_pose = self.get_pose_for_best_view(instance_id)
        vr = np.array([instance_pose[0], instance_pose[1]])
        vp = np.asarray(location[:2])
        k = 1 - (dist / (np.linalg.norm(vp - vr)))
        vf = vr + (vp - vr) * k
        return instance_pose, location, vf

    def get_language_task(self):
        if "command" in self.parameters:
            return self.parameters["command"]
        else:
            return self.ask(
                "Let's try something else! What other task would you like the robot to perform?"
            )

        #   "Not sure I can do that. Please type any other task you want the robot to do:"

    def confirm_plan(self, plan: str):
        print(f"Received plan: {plan}")
        if "confirm_plan" not in self.parameters or self.parameters["confirm_plan"]:
            execute = self.chat.input(
                "Do you want to execute (replan otherwise)? (y/n): "
            )
            return execute[0].lower() == "y"
        else:
            if plan[:7] == "explore":
                print("Currently we do not explore! Explore more to start with!")
                return False
            return True

    def run_task(self, stub, center, data):
        """Actually use VLM to perform task

        Args:
            stub: VLM connection if available
            center: 3d point x y theta in se(2)
            data: used by vlm I guess"""

        robot_center = np.zeros(3)
        robot_center[:2] = self.spot.current_position[:2]
        if self.should_visualize():
            self.voxel_map.show(backend="open3d", orig=robot_center, instances=True)
        instances = self.voxel_map.get_instances()
        blacklist = []
        while True:
            # for debug, sending the robot back to original position
            self.goto(center)
            success = False
            pick_instance_id = None
            place_instance_id = None
            if stub is not None:
                # get world_representation for planning
                while True:
                    # self.navigate_to_an_instance(
                    #    instance_id=0, should_plan=self.parameters["plan_to_instance"]
                    # )
                    world_representation = get_obj_centric_world_representation(
                        instances,
                        self.parameters["context_length"],
                        self.parameters["sample_strategy"],
                    )
                    if self.parameters["our_vlm"]:
                        import torchvision.transforms as transforms

                        # Specify the desired size
                        desired_size = (256, 256)

                        # Create the resize transform
                        resize_transform = transforms.Resize(desired_size)

                        # Iterate over the object images and resize them
                        for obj_image in world_representation.object_images:
                            obj_image.image = torch.permute(obj_image.image, (2, 0, 1))
                            obj_image.image = resize_transform(obj_image.image)
                            obj_image.image = torch.permute(obj_image.image, (1, 2, 0))
                            obj_image.image /= 255
                            print(obj_image.image.shape)
                    c = 0
                    for c, obj_image in enumerate(world_representation.object_images):
                        cv2.imwrite(
                            f"debug/obj_image_{c}.png", np.asarray(obj_image.image)
                        )
                    # task is the prompt, save it
                    data["prompt"] = self.get_language_task()
                    logger.info(f'User Command: {data["prompt"]}.')
                    output = get_output_from_world_representation(
                        stub=stub,
                        world_representation=world_representation,
                        goal=data["prompt"],
                    )
                    plan = output.action
                    logger.info(f"Received plan: {plan}")
                    if self.confirm_plan(plan):
                        pick_instance_id, place_instance_id = parse_pick_and_place_plan(
                            world_representation, plan
                        )
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
                    new_id = self.chat.input(
                        "enter a new instance to pick up from the list above: "
                    )
                    if isinstance(new_id, int):
                        pick_instance_id = new_id
                        break
                if place_instance_id is None:
                    new_id = self.chat.input(
                        "enter a new instance to place from the list above: "
                    )
                    if isinstance(new_id, int):
                        place_instance_id = new_id
                        break
                success = True
            else:
                self.say("Navigating to instance ")
                self.say(f"Instance id: {pick_instance_id}")
                self.publish_full_obs(pick_instance_id)
                success = self.navigate_to_an_instance(
                    pick_instance_id,
                    visualize=self.should_visualize(),
                    should_plan=self.parameters["plan_to_instance"],
                )
                self.say(f"Success: {success}")
                if self.parameters["find_only"]:
                    obj_pose = self.get_pose_for_best_view(pick_instance_id)
                    xy = np.array([obj_pose[0], obj_pose[1]])
                    curr_pose = self.spot.current_position
                    vr = np.array([curr_pose[0], curr_pose[1]])
                    distance = np.linalg.norm(xy - vr)
                    instance_pose, location, vf = self.get_close(pick_instance_id)
                    logger.info("Navigating closer to the object")
                    self.spot.navigate_to(
                        np.array(
                            [
                                vf[0],
                                vf[1],
                                instance_pose[2] + self.parameters["place_offset"],
                            ]
                        ),
                        blocking=True,
                    )
                    time.sleep(1)
                    logger.success("Tried navigating to close to the object!")
                    rgb = np.asarray(
                        instances[pick_instance_id].instance_views[0].cropped_image
                    )[:, :, ::-1]
                    cv2.imwrite("pick_object_instance.png", rgb)
                    logger.success("At the pick instance, looking at the object")
                    if place_instance_id is not None:
                        rgb = np.asarray(
                            instances[place_instance_id].instance_views[0].cropped_image
                        )[:, :, ::-1]
                        cv2.imwrite("place_instance.png", rgb)
                        logger.info("Navigating to place instance now")
                        obj_pose = self.get_pose_for_best_view(pick_instance_id)
                        xy = np.array([obj_pose[0], obj_pose[1]])
                        curr_pose = self.spot.current_position
                        vr = np.array([curr_pose[0], curr_pose[1]])
                        # Compute a distance for debugging and info
                        distance = np.linalg.norm(xy - vr)
                        instance_pose, location, vf = self.get_close(pick_instance_id)
                        logger.info(f"Navigating closer to the object: {distance}")
                        success = self.spot.navigate_to(
                            np.array(
                                [
                                    vf[0],
                                    vf[1],
                                    instance_pose[2] + self.parameters["place_offset"],
                                ]
                            ),
                            blocking=True,
                        )
                        time.sleep(1)
                    if success:
                        break
                    continue

                success = self._pick(
                    instances, blacklist, pick_instance_id, place_instance_id
                )

                if success:
                    logger.success("Successfully grasped the object!")
                    self.goto(center)
                    # exit out of loop without killing script
                    break
                else:
                    # Go back to look position
                    self.gaze.reset_to_look()

        # At the end, go back to where we started
        self.goto(center)

    def _pick(
        self, instances, blacklist, pick_instance_id: int, place_instance_id: int
    ) -> bool:
        """Try to pick and place an instance"""
        # TODO: change the grasp API to be able to grasp from the point cloud / mask of the instance
        # currently it will fail if there are two instances of the same category sitting close to each other
        object_category_name = self.vocab.goal_id_to_goal_name[
            int(instances[pick_instance_id].category_id.item())
        ]
        if self.parameters["verify_before_grasp"]:
            opt = self.ask(f"Grasping {object_category_name}..., y/n?: ")
        else:
            opt = "y"
        if opt == "n":
            blacklist.append(pick_instance_id)
            del instances[pick_instance_id]
            return False

        logger.info(f"Grasping: {object_category_name}")
        self.set_objects_for_grasping([[object_category_name]])
        self.spot.open_gripper()
        time.sleep(0.5)

        logger.log("DEMO", "Resetting environment...")
        # TODO: have a better way to reset the environment
        obj_pose = self.get_pose_for_best_view(pick_instance_id)
        xy = np.array([obj_pose[0], obj_pose[1]])
        curr_pose = self.spot.current_position
        vr = np.array([curr_pose[0], curr_pose[1]])
        distance = np.linalg.norm(xy - vr)

        # Try to get closer to the object
        if distance > 2.0 and self.parameters["use_get_close"]:
            instance_pose, location, vf = self.get_close(pick_instance_id)
            logger.info("Navigating closer to the object")
            self.spot.navigate_to(
                np.array(
                    [
                        vf[0],
                        vf[1],
                        instance_pose[2] + self.parameters["place_offset"],
                    ]
                ),
                blocking=True,
            )
        time.sleep(0.5)
        success = self.gaze.gaze_and_grasp(
            finish_sweep_before_deciding=self.parameters["finish_grasping"]
        )
        time.sleep(0.5)
        if success:
            # TODO: @JAY make placing cleaner
            # navigate to the place instance
            print("Navigating to instance ")
            print(f"Instance id: {place_instance_id}")
            self.publish_full_obs(place_instance_id)
            success = self.navigate_to_an_instance(
                place_instance_id,
                visualize=self.should_visualize(),
                should_plan=self.parameters["plan_to_instance"],
            )
            print(f"navigated to place {success=}")
            place_location = self.vocab.goal_id_to_goal_name[
                int(instances[place_instance_id].category_id.item())
            ]
            # Get close to the instance after we nvagate
            instance_pose, location, vf = self.get_close(place_instance_id, dist=0.5)
            if not self.parameters["use_get_close"]:
                vf = instance_pose
            # Now we can try to actually place at the target location
            logger.info(
                "placing {object} at {place}",
                object=object_category_name,
                place=place_location,
            )
            rot = self.gaze.get_pick_location()
            self.place_in_an_instance(
                place_instance_id,
                instance_pose,
                location,
                vf,
                place_rotation=rot,
                place_height=self.parameters["place_height"],
            )

        return success

    def run_explore(self):
        """Run exploration in different environments. Will explore until there's nothing else to find."""
        # Track the number of times exploration failed
        explore_failures = 0
        old_state = self.current_state
        self.current_state = "EXPLORE"
        for exploration_step in range(int(self.parameters["exploration_steps"])):
            # logger.log("DEMO", "\n----------- Step {} -----------", step + 1)
            print(
                "-" * 20,
                exploration_step + 1,
                "/",
                int(self.parameters["exploration_steps"]),
                "-" * 20,
            )

            # Get current position and goal
            start = self.spot.current_position
            goal = None
            logger.info("Start xyt: {}", start)
            start_is_valid = self.navigation_space.is_valid(start)
            if start_is_valid:
                logger.success("Start is valid: {}", start_is_valid)
            else:
                # TODO do something is start is not valid
                logger.error("!!!!!!!! INVALID START POSITION !!!!!!")
                self.backup_from_invalid_state()
                continue

            logger.info("Start is safe: {}", self.voxel_map.xyt_is_safe(start))

            if self.parameters["explore_methodical"]:
                logger.info("Generating the next closest frontier point...")

                # Sampling along the frontier
                if explore_failures <= self.parameters["max_explore_failures"]:
                    res = self.plan_to_frontier()
                else:
                    res = None

                # Handle the case where we could not get to nearby frontier
                if res is not None and res.success:
                    explore_failures = 0
                else:
                    explore_failures += 1
                    logger.warning("Exploration failed: " + str(res.reason))
                    if explore_failures > self.parameters["max_explore_failures"]:
                        logger.debug("Switching to random exploration")
                        goal = next(
                            self.navigation_space.sample_random_frontier(
                                min_size=self.parameters["min_size"],
                                max_size=self.parameters["max_size"],
                            )
                        )
                        if goal is None:
                            # Nowhere to go
                            logger.info("Done exploration!")
                            return

                        goal = goal.cpu().numpy()
                        goal_is_valid = self.navigation_space.is_valid(goal)
                        logger.info(f" Goal is valid: {goal_is_valid}")
                        if not goal_is_valid:
                            # really we should sample a new goal
                            continue

                        #  Build plan
                        res = self.planner.plan(start, goal)
                        logger.info(goal)
                        if res.success:
                            logger.success("Plan success: {}", res.success)
                            explore_failures = 0
                        else:
                            logger.error("Plan success: {}", res.success)
                            logger.error("Failed to plan to the frontier.")
            else:
                logger.info(
                    "picking a random frontier point and trying to move there..."
                )
                # Sample a goal in the frontier (TODO change to closest frontier)
                goal = self.sample_random_frontier()
                if goal is None:
                    logger.info("Done exploration!")
                    return

                goal_is_valid = self.navigation_space.is_valid(goal)
                logger.info(
                    f" Goal is valid: {goal_is_valid}",
                )
                if not goal_is_valid:
                    # really we should sample a new goal
                    continue

                #  Build plan
                res = self.planner.plan(start, goal)
                logger.info(goal)
                if res.success:
                    logger.success("Res success: {}", res.success)
                else:
                    logger.error("Res success: {}", res.success)

            if res.success:
                self.spot.execute_plan(
                    res,
                    pos_err_threshold=self.parameters["trajectory_pos_err_threshold"],
                    rot_err_threshold=self.parameters["trajectory_rot_err_threshold"],
                    per_step_timeout=self.parameters["trajectory_per_step_timeout"],
                    verbose=False,
                )
            elif goal is not None and len(goal) > 0:
                logger.warning("Just go ahead and try it anyway")
                self.spot.navigate_to(goal)

            if not self.parameters["use_async_subscriber"]:
                # Synchronous updates
                self.update()

            if self.step % 1 == 0 and self.should_visualize():
                self.visualize(start, goal)
        self.current_state = old_state

    def run_teleop_data(self):
        """Run exploration in different environments. Will explore until there's nothing else to find."""
        # Track the number of times exploration failed
        explore_failures = 0
        old_state = self.current_state
        self.current_state = "EXPLORE"
        for exploration_step in range(int(self.parameters["exploration_steps"])):
            # logger.log("DEMO", "\n----------- Step {} -----------", step + 1)
            print(
                "-" * 20,
                exploration_step + 1,
                "/",
                int(self.parameters["exploration_steps"]),
                "-" * 20,
            )

            # Get current position and goal
            start = self.spot.current_position
            goal = None
            logger.info("Start xyt: {}", start)
            start_is_valid = self.navigation_space.is_valid(start)
            if start_is_valid:
                logger.success("Start is valid: {}", start_is_valid)
            else:
                # TODO do something is start is not valid
                logger.error("!!!!!!!! INVALID START POSITION !!!!!!")
                self.backup_from_invalid_state()
                continue

            logger.info("Start is safe: {}", self.voxel_map.xyt_is_safe(start))

            if self.parameters["explore_methodical"]:
                logger.info("Generating the next closest frontier point...")

                # Sampling along the frontier
                if explore_failures <= self.parameters["max_explore_failures"]:
                    #! TODO: Check if this is the exploration, if yes -> replace with keyboard teleop
                    linear = input("enter linear: ")
                    angular = input("enter angular: ")
                    self.spot.move_base(float(linear), float(angular))
                    # res = self.plan_to_frontier()
            if not self.parameters["use_async_subscriber"]:
                # Synchronous updates
                self.update()

            if self.step % 1 == 0 and self.should_visualize():
                self.visualize(start, goal)
        self.current_state = old_state
