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
from typing import Any, Dict, List, Optional

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
from home_robot.utils.config import Config, get_config, load_config
from home_robot.utils.geometry import xyt_global_to_base
from home_robot.utils.point_cloud import numpy_to_pcd
from home_robot.utils.visualization import get_x_and_y_from_path
from home_robot_spot import SpotClient, VoxelMapSubscriber
from home_robot_spot.grasp_env import GraspController

## Temporary hack until we make accel-cortex pip installable
print("Make sure path to accel-cortex base folder is set")
sys.path.append(os.path.expanduser(os.environ["ACCEL_CORTEX"]))
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
            scores = torch.tensor([ins.score for ins in instances])
            embeds = (
                torch.stack(
                    [
                        ins.get_image_embedding(aggregation_method="mean")
                        for ins in instances
                    ]
                )
                .cpu()
                .detach()
            )
        else:
            bounds = torch.zeros(0, 3, 2)
            names = torch.zeros(0, 1)
            scores = torch.zeros(
                0,
            )
            embeds = torch.zeros(0, 512)

        # Map
        obstacles, explored = model.get_2d_map()
        map_im = obstacles.int() + explored.int()

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
                box_scores=scores,
                box_embeddings=embeds,
                map_im=map_im.cpu().detach(),
            ),
            f,
        )
    # logger.success("Done saving observation to pickle file.")


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
                image=crop.contiguous(),
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
        self,
        parameters: Dict[str, Any],
        spot_config: Config,
        dock: Optional[int] = None,
        path: str = None,
    ):
        self.parameters = parameters
        self.encoder = ClipEncoder(self.parameters["clip"])
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
        with atomic_write(f"{self.path}/viz_data/vocab_dict.pkl", mode="wb") as f:
            pickle.dump(self.semantic_sensor.segmenter.seg_id_to_name, f)

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
        self.spot_config = spot_config
        self.path = path

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

    def set_objects_for_grasping(self, objects: List[List[str]]):
        """Set the objects used for grasping"""
        self.gaze.set_objects(objects)

    def backup_from_invalid_state(self):
        """Helper function to get the robot unstuck (it is too close to geometry)"""
        self.spot.navigate_to([-0.25, 0, 0], relative=True, blocking=True)

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
        x0, y0, theta0 = self.spot.current_position
        for i in range(8):
            self.spot.navigate_to([x0, y0, theta0 + (i + 1) * np.pi / 4], blocking=True)
            if not self.parameters["use_async_subscriber"]:
                self.update()

        # Should we display after spinning? If visualize is true we will
        if self.parameters["visualize"]:
            self.voxel_map.show()

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

    def place_in_an_instance(
        self,
        instance_pose,
        location,
        vf,
        place_height=0.3,
        place_rotation=[0, np.pi / 2, 0],
    ):
        """Move to a position to place in an environment."""
        # TODO: Check if vf is correct
        self.spot.navigate_to(np.array([vf[0], vf[1], instance_pose[2]]), blocking=True)

        # Transform placing position to body frame coordinates
        x, y, yaw = self.spot.spot.get_xy_yaw()
        local_xyt = xyt_global_to_base(location, np.array([x, y, yaw]))

        # z is the height of the receptacle minus the height of spot + the desired delta for placing
        z = location[2] - self.spot.spot.body.z + place_height
        local_xyz = np.array([local_xyt[0], local_xyt[1], z])
        rotations = np.array([0, 0, 0])

        # Now we place
        self.spot.spot.move_gripper_to_point(local_xyz, rotations)
        time.sleep(2)
        arm = self.spot.spot.get_arm_joint_positions()
        arm[-1] = place_rotation[-1]
        arm[-2] = place_rotation[0]
        self.spot.spot.set_arm_joint_positions(arm, travel_time=1.5)
        time.sleep(2)
        self.spot.spot.open_gripper()

        # reset arm
        time.sleep(0.5)
        self.spot.reset_arm()

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
        start_is_valid = self.navigation_space.is_valid(start)

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

            if res.success:
                logger.success("Res success: {}", res.success)
                self.spot.execute_plan(
                    res,
                    pos_err_threshold=self.parameters["trajectory_pos_err_threshold"],
                    rot_err_threshold=self.parameters["trajectory_rot_err_threshold"],
                    per_step_timeout=self.parameters["trajectory_per_step_timeout"],
                )
                goal_position = goal
            else:
                logger.error("Res success: {}, !!!PLANNING FAILED!!!", res.success)
        # Finally, navigate to the final position
        logger.info(
            "Navigating to goal position: {}, start = {}",
            goal_position,
            self.spot.current_position,
        )
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
            return input("please type any task you want the robot to do: ")

    def confirm_plan(self, plan: str):
        print(f"Received plan: {plan}")
        if "confirm_plan" not in self.parameters or self.parameters["confirm_plan"]:
            execute = input("Do you want to execute (replan otherwise)? (y/n): ")
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
                    self.navigate_to_an_instance(
                        instance_id=0, should_plan=self.parameters["plan_to_instance"]
                    )
                    world_representation = get_obj_centric_world_representation(
                        instances, args.context_length
                    )
                    # task is the prompt, save it
                    data["prompt"] = self.get_language_task()
                    output = stub.stream_act_on_observations(
                        ProtoConverter.wrap_obs_iterator(
                            episode_id=random.randint(1, 1000000),
                            obs=world_representation,
                            goal=data["prompt"],
                        )
                    )
                    plan = output.action
                    if self.confirm_plan(plan):
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
                success = True
            else:
                print("Navigating to instance ")
                print(f"Instance id: {pick_instance_id}")
                success = self.navigate_to_an_instance(
                    pick_instance_id,
                    visualize=self.parameters["visualize"],
                    should_plan=self.parameters["plan_to_instance"],
                )
                print(f"Success: {success}")

                # # try to pick up this instance
                # if success:

                # TODO: change the grasp API to be able to grasp from the point cloud / mask of the instance
                # currently it will fail if there are two instances of the same category sitting close to each other
                object_category_name = self.vocab.goal_id_to_goal_name[
                    int(instances[pick_instance_id].category_id.item())
                ]
                if self.parameters["verify_before_grasp"]:
                    opt = input(f"Grasping {object_category_name}..., y/n?: ")
                else:
                    opt = "y"
                if opt == "n":
                    blacklist.append(pick_instance_id)
                    del instances[pick_instance_id]
                    continue
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
                        np.array([vf[0], vf[1], instance_pose[2]]), blocking=True
                    )
                time.sleep(0.5)
                success = self.gaze.gaze_and_grasp(finish_sweep_before_deciding=False)
                time.sleep(0.5)
                if success:
                    # TODO: @JAY make placing cleaner
                    # navigate to the place instance
                    print("Navigating to instance ")
                    print(f"Instance id: {place_instance_id}")
                    success = self.navigate_to_an_instance(
                        place_instance_id,
                        visualize=self.parameters["visualize"],
                        should_plan=self.parameters["plan_to_instance"],
                    )
                    print(f"navigated to place {success=}")
                    place_location = self.vocab.goal_id_to_goal_name[
                        int(instances[place_instance_id].category_id.item())
                    ]
                    # Get close to the instance after we nvagate
                    instance_pose, location, vf = self.get_close(
                        place_instance_id, dist=0.5
                    )
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
                        instance_pose, location, vf, place_rotation=rot
                    )

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
                else:
                    # Go back to look position
                    self.gaze.reset_to_look()

    def run_explore(self):
        """Run exploration in different environments. Will explore until there's nothing else to find."""
        # Track the number of times exploration failed
        explore_failures = 0
        for step in range(int(self.parameters["exploration_steps"])):
            # logger.log("DEMO", "\n----------- Step {} -----------", step + 1)
            print(
                "-" * 20,
                step + 1,
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
                res = self.plan_to_frontier()
                if res.success:
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
                            logger.success("Res success: {}", res.success)
                        else:
                            logger.error("Res success: {}", res.success)
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
                self.update(step + 1)

            if step % 1 == 0 and self.parameters["visualize"]:
                self.visualize(start, goal, step)


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
    elif args.location == "fre":
        parameters = get_config("src/home_robot_spot/configs/parameters_fre.yaml")[0]
    else:
        logger.critical(
            f"Location {args.location} is invalid, please enter a valid location"
        )
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
        demo.run_explore()

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
        default=00,
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
        "--location",
        "-l",
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
