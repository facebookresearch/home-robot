# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from atomicwrites import atomic_write
from loguru import logger
from PIL import Image

from home_robot.agent.multitask import Parameters
from home_robot.core.robot import GraspClient, RobotClient
from home_robot.mapping.instance import Instance
from home_robot.mapping.voxel import (
    SparseVoxelMap,
    SparseVoxelMapNavigationSpace,
    plan_to_frontier,
)
from home_robot.motion import PlanResult, RRTConnect, Shortcut
from home_robot.perception.encoders import get_encoder
from home_robot.utils.demo_chat import (
    DemoChat,
    start_demo_ui_server,
    stop_demo_ui_server,
)
from home_robot.utils.threading import Interval


def publish_obs(
    model: SparseVoxelMapNavigationSpace,
    path: str,
    state: str,
    timestep: int,
    target_id: Dict[str, int] = None,
):
    """publish observation for use by the UI"""
    # NOTE: this requires 'pip install atomicwrites'
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
        obstacles, explored = model.voxel_map.get_2d_map()
        obstacles = obstacles.int()
        explored = explored.int()

        logger.info(f"Saving observation to pickle file: {f'{path}/{timestep}.pkl'}")
        pickle.dump(
            dict(
                limited_obs=False,
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
                obstacles=obstacles.cpu().detach(),
                explored=explored.cpu().detach(),
                current_state=state,
                target_id=target_id,
            ),
            f,
        )

    # Print out all the instances we have seen
    for i, instance in enumerate(model.voxel_map.get_instances()):
        for j, view in enumerate(instance.instance_views):
            filename = f"{path}/instances/instance{i}_view{j}.png"
            if not os.path.exists(filename):
                image = Image.fromarray(view.cropped_image.byte().cpu().numpy())
                image.save(filename)


class RobotAgent:
    """Basic demo code. Collects everything that we need to make this work."""

    def __init__(
        self,
        robot: RobotClient,
        semantic_sensor,
        parameters: Dict[str, Any],
        grasp_client: Optional[GraspClient] = None,
        rpc_stub=None,
    ):
        if isinstance(parameters, Dict):
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise RuntimeError(f"parameters of unsupported type: {type(parameters)}")
        self.robot = robot
        self.rpc_stub = rpc_stub
        self.grasp_client = grasp_client

        self.semantic_sensor = semantic_sensor
        self.normalize_embeddings = True
        self.pos_err_threshold = parameters["trajectory_pos_err_threshold"]
        self.rot_err_threshold = parameters["trajectory_rot_err_threshold"]
        self.current_state = "WAITING"
        self.encoder = get_encoder(parameters["encoder"], parameters["encoder_args"])
        self.obs_count = 0
        self.guarantee_instance_is_reachable = (
            parameters.guarantee_instance_is_reachable
        )

        # Wrapper for SparseVoxelMap which connects to ROS
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

        # Create planning space
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            self.robot.get_robot_model(),
            step_size=parameters["step_size"],
            rotation_step_size=parameters["rotation_step_size"],
            dilate_frontier_size=parameters[
                "dilate_frontier_size"
            ],  # 0.6 meters back from every edge = 12 * 0.02 = 0.24
            dilate_obstacle_size=parameters["dilate_obstacle_size"],
        )

        # Dictionary storing attempts to visit each object
        self._object_attempts = {}
        self._cached_plans = {}

        # Create a simple motion planner
        self.planner = Shortcut(RRTConnect(self.space, self.space.is_valid))

        timestamp = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"
        # f"data/hw_exps/{self.parameters['name']}/{timestamp}"
        self.path = os.path.expanduser(f"data/hw_exps/spot/{timestamp}")
        print(f"Writing logs to {self.path}")
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(f"{self.path}/viz_data", exist_ok=True)
        with atomic_write(f"{self.path}/viz_data/vocab_dict.pkl", mode="wb") as f:
            pickle.dump(self.semantic_sensor.seg_id_to_name, f)

        if parameters["start_ui_server"]:
            start_demo_ui_server()
        if parameters["chat"]:
            self.chat = DemoChat(f"{self.path}/demo_chat.json")
            if self.parameters["limited_obs_publish_sleep"] > 0:
                self._publisher = Interval(
                    self.publish_limited_obs,
                    sleep_time=self.parameters["limited_obs_publish_sleep"],
                )
            else:
                self._publisher = None
        else:
            self.chat = None
            self._publisher = None

    def place(self, object_goal: Optional[str] = None, **kwargs) -> bool:
        """Try to place an object."""
        if not self.robot.in_manipulation_mode():
            self.robot.switch_to_manipulation_mode()
        if self.grasp_client is None:
            logger.warn("Tried to place without providing a grasp client.")
            return False
        return self.grasp_client.try_placing(object_goal=object_goal, **kwargs)

    def grasp(self, object_goal: Optional[str] = None, **kwargs) -> bool:
        """Try to grasp a potentially specified object."""
        # Put the robot in manipulation mode
        if not self.robot.in_manipulation_mode():
            self.robot.switch_to_manipulation_mode()
        if self.grasp_client is None:
            logger.warn("Tried to grasp without providing a grasp client.")
            return False
        return self.grasp_client.try_grasping(object_goal=object_goal, **kwargs)

    def rotate_in_place(self, steps: int = 12):
        """Simple helper to rotate in place"""
        step_size = 2 * np.pi / steps
        i = 0
        while i < steps:
            self.robot.navigate_to([0, 0, step_size], relative=True, blocking=True)
            # TODO remove debug code
            # print(i, self.robot.get_base_pose())
            self.update()
            if self.robot.last_motion_failed():
                # We have a problem!
                self.robot.navigate_to([-0.1, 0, 0], relative=True, blocking=True)
                i = 0
            else:
                i += 1

    def get_plan_from_vlm(self):
        """This is a connection to a remote thing for getting language commands"""
        assert self.rpc_stub is not None, "must have RPC stub to connect to remote VLM"
        # This is not a very stable import
        # So we guard it under this part where it's necessary
        from home_robot.utils.rpc import (
            get_obj_centric_world_representation,
            get_output_from_world_representation,
        )

        instances = self.voxel_map.get_instances()
        world_representation = get_obj_centric_world_representation(
            instances, self.parameters["vlm_context_length"]
        )
        output = get_output_from_world_representation(
            self.rpc_stub, world_representation, self.get_command()
        )
        return output

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

    def get_command(self):
        if "command" in self.parameters:
            return self.parameters["command"]
        else:
            return self.ask("please type any task you want the robot to do: ")

    def __del__(self):
        """Clean up at the end if possible"""
        self.finish()

    def finish(self):
        """Clean up at the end if possible"""
        print("Finishing different processes:")
        if self.parameters["start_ui_server"]:
            print("- Stopping UI server...")
            stop_demo_ui_server()
        if self._publisher is not None:
            print("- Stopping publisher...")
            self._publisher.cancel()
        print("... Done.")

    def publish_limited_obs(self):
        obs = self.robot.get_observation()
        self.obs_count += 1
        with atomic_write(f"{self.path}/viz_data/{self.obs_count}.pkl", mode="wb") as f:
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

    def update(self, visualize_map=False):
        """Step the data collector. Get a single observation of the world. Remove bad points, such as those from too far or too near the camera. Update the 3d world representation."""
        obs = self.robot.get_observation()
        self.obs_count += 1
        obs_count = self.obs_count
        # Semantic prediction
        obs = self.semantic_sensor.predict(obs)

        # Add observation - helper function will unpack it
        self.voxel_map.add_obs(obs)
        if visualize_map:
            # Now draw 2d
            self.voxel_map.get_2d_map(debug=True)

        # Send message to user interface
        if self.chat is not None:
            publish_obs(self.space, self.path, self.current_state, obs_count)

    def plan_to_instance(
        self,
        instance: Instance,
        start: np.ndarray,
        verbose: bool = True,
        instance_id: int = -1,
    ) -> PlanResult:
        """Move to a specific instance. Goes until a motion plan is found.

        Args:
            instance(Instance): an object in the world
            verbose(bool): extra info is printed
            instance_ind(int): if >= 0 we will try to use this to retrieve stored plans"""

        res = None
        if verbose:
            for j, view in enumerate(instance.instance_views):
                print(f"- instance {instance_id} view {j} at {view.cam_to_world}")

        start_is_valid = self.space.is_valid(start)
        if not start_is_valid:
            return PlanResult(success=False, reason="invalid start state")

        mask = self.voxel_map.mask_from_bounds(instance.bounds)
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
            if instance_id >= 0 and instance_id in self._cached_plans:
                if verbose:
                    print(f"- retrieving cached plan for {instance_id}")
                res = self._cached_plans[instance_id]
            else:
                res = self.planner.plan(start, goal)
            print("Found plan:", res.success)
            if res.success:
                break
        if res is None:
            return PlanResult(success=False, reason="no valid plans found")
        return res

    def move_to_any_instance(self, matches: List[Tuple[int, Instance]]):
        """Check instances and find one we can move to"""
        self.current_state = "NAV_TO_INSTANCE"
        self.robot.move_to_nav_posture()
        start = self.robot.get_base_pose()
        start_is_valid = self.space.is_valid(start)
        start_is_valid_retries = 5
        while not start_is_valid and start_is_valid_retries > 0:
            print(f"Start {start} is not valid. back up a bit.")
            self.robot.navigate_to([-0.1, 0, 0], relative=True)
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
            res = self.plan_to_instance(match, start, instance_id=i)
            if res is not None and res.success:
                break
            else:
                # TODO: remove debug code
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
            self.robot.execute_trajectory(
                [pt.state for pt in res.trajectory],
                pos_err_threshold=self.pos_err_threshold,
                rot_err_threshold=self.rot_err_threshold,
            )
            time.sleep(1.0)
            self.robot.navigate_to([0, 0, np.pi / 2], relative=True)
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
        if self._publisher is not None:
            self._publisher.start()
        # Tuck the arm away
        print("Sending arm to  home...")
        self.robot.switch_to_manipulation_mode()

        self.robot.move_to_nav_posture()
        # self.robot.head.look_close(blocking=False)
        print("... done.")

        # Move the robot into navigation mode
        self.robot.switch_to_navigation_mode()
        self.update(visualize_map=visualize_map_at_start)  # Append latest observations
        self.print_found_classes(goal)
        return self.get_found_instances_by_class(goal)

    def encode_text(self, text: str):
        """Helper function for getting text embeddings"""
        emb = self.encoder.encode_text(text)
        if self.normalize_embeddings:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def get_found_instances_by_class(
        self, goal: Optional[str], threshold: int = 0, debug: bool = False
    ) -> List[Tuple[int, Instance]]:
        """Check to see if goal is in our instance memory or not. Return a list of everything with the correct class.

        Parameters:
            goal(str): optional name of the object we want to find
            threshold(int): number of object attempts we are allowed to do for this object
            debug(bool): print debug info

        Returns:
            instance_id(int): a unique int identifying this instance
            instance(Instance): information about a particular object we believe exists"""
        matching_instances = []
        if goal is None:
            # No goal means no matches
            return []
        instances = self.voxel_map.get_instances()
        for i, instance in enumerate(instances):
            oid = int(instance.category_id.item())
            name = self.semantic_sensor.get_class_name_for_id(oid)
            if name.lower() == goal.lower():
                matching_instances.append((i, instance))
        return self.filter_matches(matching_instances, threshold=threshold)

    def get_reachable_instances_by_class(
        self, goal: Optional[str], threshold: int = 0, debug: bool = False
    ) -> List[Tuple[int, Instance]]:
        """See if we can reach dilated object masks for different objects.

        Parameters:
            goal(str): optional name of the object we want to find
            threshold(int): number of object attempts we are allowed to do for this object
            debug(bool): print debug info

        Returns list of tuples with two members:
            instance_id(int): a unique int identifying this instance
            instance(Instance): information about a particular object we believe exists"""
        matches = self.get_found_instances_by_class(
            goal=goal, threshold=threshold, debug=debug
        )
        reachable_matches = []
        self._cached_plans = {}
        start = self.robot.get_base_pose()
        for i, instance in matches:
            # compute its mask
            # see if this mask's area is explored and reachable from the current robot
            if self.guarantee_instance_is_reachable:
                res = self.plan_to_instance(instance, start, instance_id=i)
                self._cached_plans[i] = res
                if res.success:
                    reachable_matches.append(instance)
            else:
                reachable_matches.append(instance)
        return reachable_matches

    def filter_matches(
        self, matches: List[Tuple[int, Instance]], threshold: int = 1
    ) -> Tuple[int, Instance]:
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
        self.current_state = "NAV_TO_HOME"
        # self.robot.move_to_nav_posture()
        # self.robot.head.look_close(blocking=False)
        self.robot.switch_to_navigation_mode()

        print("- try to motion plan there")
        start = self.robot.get_base_pose()
        goal = np.array([0, 0, 0])
        res = self.planner.plan(start, goal)
        # if it fails, skip; else, execute a trajectory to this position
        if res.success:
            print("- executing full plan to home!")
            self.robot.execute_trajectory([pt.state for pt in res.trajectory])
            print("Done!")
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
        show_goal: bool = False,
    ) -> Optional[Instance]:
        """Go through exploration. We use the voxel_grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

        Args:
            visualize(bool): true if we should do intermediate debug visualizations"""
        self.current_state = "EXPLORE"
        self.robot.move_to_nav_posture()

        print("Go to (0, 0, 0) to start with...")
        self.robot.navigate_to([0, 0, 0])

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
                self.robot.navigate_to([-0.1, 0, 0], relative=True)
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
                self.voxel_map.show(orig=show_goal)

            # if it fails, skip; else, execute a trajectory to this position
            if res.success:
                print("Plan successful!")
                if not dry_run:
                    self.robot.execute_trajectory(
                        [pt.state for pt in res.trajectory],
                        pos_err_threshold=self.pos_err_threshold,
                        rot_err_threshold=self.rot_err_threshold,
                    )
            if self.robot.last_motion_failed():
                print("!!!!!!!!!!!!!!!!!!!!!!")
                print("ROBOT IS STUCK! Move back!")
                r = np.random.randint(3)
                if r == 0:
                    self.robot.navigate_to([-0.1, 0, 0], relative=True, blocking=True)
                elif r == 1:
                    self.robot.navigate_to(
                        [0, 0, np.pi / 4], relative=True, blocking=True
                    )
                elif r == 2:
                    self.robot.navigate_to(
                        [0, 0, -np.pi / 4], relative=True, blocking=True
                    )

            # Append latest observations
            self.update()
            if manual_wait:
                input("... press enter ...")

            if task_goal is not None:
                matches = self.get_reachable_instances_by_class(task_goal)
                if len(matches) > 0:
                    print("!!! GOAL FOUND! Done exploration. !!!")
                    break

        if go_home_at_end:
            self.current_state = "NAV_TO_HOME"
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
                    self.robot.execute_trajectory([pt.state for pt in res.trajectory])
            else:
                print("WARNING: planning to home failed!")
        return matches
