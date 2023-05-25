import json
import os
import random
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import habitat
import numpy as np
import torch
from habitat.core.environments import GymHabitatEnv
from habitat.core.simulator import Observations
from torch import Tensor

import home_robot
from home_robot.core.interfaces import (
    ContinuousFullBodyAction,
    ContinuousNavigationAction,
    DiscreteNavigationAction,
)
from home_robot.motion.stretch import (
    STRETCH_ARM_EXTENSION,
    STRETCH_ARM_LIFT,
    STRETCH_NAVIGATION_Q,
    STRETCH_PREGRASP_Q,
    map_joint_q_state_to_action_space,
)
from home_robot.utils.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
)
from home_robot_sim.env.habitat_abstract_env import HabitatEnv
from home_robot_sim.env.habitat_objectnav_env.constants import (
    RearrangeBasicCategories,
    RearrangeDETICCategories,
)
from home_robot_sim.env.habitat_objectnav_env.visualizer import Visualizer


class SimJointActionIndex(IntEnum):
    """
    Enum representing the indices of different joints in the action space.
    """

    # TODO: This needs to be common between sim and real as they share the same API for actions
    ARM = 0  # A single value is used to control the extension
    LIFT = 1
    WRIST_YAW = 2
    WRIST_PITCH = 3
    WRIST_ROLL = 4
    HEAD_PAN = 5
    HEAD_TILT = 6


class HabitatOpenVocabManipEnv(HabitatEnv):
    semantic_category_mapping: Union[RearrangeBasicCategories, RearrangeDETICCategories]
    joints_dof = 7

    def __init__(self, habitat_env: habitat.core.env.Env, config, dataset):
        super().__init__(habitat_env)
        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        self._dataset = dataset
        self.visualize = config.VISUALIZE or config.PRINT_IMAGES
        if self.visualize:
            self.visualizer = Visualizer(config, dataset)
        self.goal_type = config.habitat.task.goal_type
        self.episodes_data_path = config.habitat.dataset.data_path
        self.video_dir = config.habitat_baselines.video_dir
        self.max_forward = (
            config.habitat.task.actions.base_velocity.max_displacement_along_axis
        )
        self.max_turn_degrees = (
            config.habitat.task.actions.base_velocity.max_turn_degrees
        )
        self.max_joints_delta = (
            config.habitat.task.actions.arm_action.delta_pos_limit
        )  # for normalizing arm delta
        self.max_turn = (
            self.max_turn_degrees / 180 * np.pi
        )  # for normalizing turn angle
        self.discrete_forward = (
            config.ENVIRONMENT.forward
        )  # amount the agent can move in a discrete step
        self.discrete_turn_degrees = (
            config.ENVIRONMENT.turn_angle
        )  # amount the agent turns in a discrete turn
        self.joints_mask = np.array(
            config.habitat.task.actions.arm_action.arm_joint_mask
        )  # mask specifying which arm joints are to be set
        self.config = config

        self._obj_name_to_id_mapping = self._dataset.obj_category_to_obj_category_id
        self._rec_name_to_id_mapping = self._dataset.recep_category_to_recep_category_id
        self._obj_id_to_name_mapping = {
            k: v for v, k in self._obj_name_to_id_mapping.items()
        }
        self._rec_id_to_name_mapping = {
            k: v for v, k in self._rec_name_to_id_mapping.items()
        }

        if self.ground_truth_semantics:
            self.semantic_category_mapping = RearrangeBasicCategories()
            with open(config.ENVIRONMENT.receptacle_sensor_id_file) as f:
                self._receptacle_sensor_name_to_id = json.load(f)
        else:
            # combining objs and recep IDs into one mapping
            self.obj_rec_combined_mapping = {}
            for i in range(
                len(self._obj_id_to_name_mapping) + len(self._rec_id_to_name_mapping)
            ):
                if i < len(self._obj_id_to_name_mapping):
                    self.obj_rec_combined_mapping[i + 1] = self._obj_id_to_name_mapping[
                        i
                    ]
                else:
                    self.obj_rec_combined_mapping[i + 1] = self._rec_id_to_name_mapping[
                        i - len(self._obj_id_to_name_mapping)
                    ]
            self.semantic_category_mapping = RearrangeDETICCategories(
                # TODO (Arun): add this back when we move to v3 dataset
                # self.obj_rec_combined_mapping
            )

        if not self.ground_truth_semantics:
            from home_robot.perception.detection.detic.detic_perception import (
                DeticPerception,
            )

            # TODO Specify confidence threshold as a parameter
            gpu_device_id = self.config.habitat.simulator.habitat_sim_v0.gpu_device_id
            self.segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary=",".join(
                    ["."] + list(self.obj_rec_combined_mapping.values()) + ["other"]
                ),
                sem_gpu_id=gpu_device_id,
            )
        self._last_habitat_obs = None

    def get_current_episode(self):
        if isinstance(self.habitat_env, GymHabitatEnv):
            return self.habitat_env.current_episode()
        else:
            return self.habitat_env.current_episode

    def set_vis_dir(self):
        scene_id = self.get_current_episode().scene_id.split("/")[-1].split(".")[0]
        episode_id = self.get_current_episode().episode_id
        self.visualizer.set_vis_dir(scene_id=scene_id, episode_id=episode_id)

    def reset(self):
        habitat_obs = self.habitat_env.reset()
        self._last_habitat_obs = habitat_obs
        self.semantic_category_mapping.reset_instance_id_to_category_id(
            self.habitat_env
        )
        self._last_obs = self._preprocess_obs(habitat_obs)
        if self.visualize:
            self.visualizer.reset()
            self.set_vis_dir()
        return self._last_obs

    def convert_pose_to_real_world_axis(self, hab_pose):
        """Update axis convention of habitat pose to match the real-world axis convention"""
        hab_pose[[0, 1, 2]] = hab_pose[[2, 0, 1]]
        hab_pose[:, [0, 1, 2]] = hab_pose[:, [2, 0, 1]]
        return hab_pose

    def _preprocess_obs(
        self, habitat_obs: habitat.core.simulator.Observations
    ) -> home_robot.core.interfaces.Observations:
        depth = self._preprocess_depth(habitat_obs["robot_head_depth"])
        (
            object_goal,
            start_recep_goal,
            end_recep_goal,
            goal_name,
        ) = self._preprocess_goal(habitat_obs, self.goal_type)

        if self.visualize:
            third_person_image = habitat_obs["robot_third_rgb"]
        else:
            third_person_image = None

        obs = home_robot.core.interfaces.Observations(
            rgb=habitat_obs["robot_head_rgb"],
            depth=depth,
            compass=habitat_obs["robot_start_compass"] - (np.pi / 2),
            gps=self._preprocess_xy(habitat_obs["robot_start_gps"]),
            task_observations={
                "object_goal": object_goal,
                "start_recep_goal": start_recep_goal,
                "end_recep_goal": end_recep_goal,
                "goal_name": goal_name,
                "object_embedding": habitat_obs["object_embedding"],
                "start_receptacle": habitat_obs["start_receptacle"],
                "goal_receptacle": habitat_obs["goal_receptacle"],
            },
            joint=habitat_obs["joint"],
            is_holding=habitat_obs["is_holding"],
            relative_resting_position=habitat_obs["relative_resting_position"],
            third_person_image=third_person_image,
            camera_pose=self.convert_pose_to_real_world_axis(
                np.asarray(habitat_obs["camera_pose"])
            ),
        )
        obs = self._preprocess_semantic(obs, habitat_obs)
        return obs

    def _preprocess_semantic(
        self, obs: home_robot.core.interfaces.Observations, habitat_obs
    ) -> home_robot.core.interfaces.Observations:
        if self.ground_truth_semantics:
            semantic = torch.from_numpy(
                habitat_obs["object_segmentation"].squeeze(-1).astype(np.int64)
            )
            recep_seg = torch.from_numpy(
                habitat_obs["receptacle_segmentation"].squeeze(-1).astype(np.int64)
            )

            recep_seg[recep_seg != 0] += 1
            semantic = semantic + recep_seg
            semantic[semantic == 0] = len(self._receptacle_sensor_name_to_id) + 1
            obs.semantic = semantic.numpy()
            obs.task_observations["recep_idx"] = 1
            obs.task_observations["semantic_max_val"] = (
                len(self._receptacle_sensor_name_to_id) + 1
            )
            # TODO Ground-truth semantic visualization
        else:
            obs = self.segmentation.predict(
                obs, depth_threshold=0.5, draw_instance_predictions=False
            )
            if type(self.semantic_category_mapping) == RearrangeDETICCategories:
                # First index is a dummy unused category
                obs.semantic[obs.semantic == 0] = (
                    self.semantic_category_mapping.num_sem_categories - 1
                )
                obs.task_observations["recep_idx"] = (
                    self.semantic_category_mapping.num_sem_obj_categories + 1
                )
                obs.task_observations["semantic_max_val"] = (
                    self.semantic_category_mapping.num_sem_categories - 1
                )
        obs.task_observations["semantic_frame"] = np.concatenate(
            [obs.rgb, obs.semantic[:, :, np.newaxis]], axis=2
        ).astype(np.uint8)
        return obs

    def _preprocess_depth(self, depth: np.array) -> np.array:
        rescaled_depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        rescaled_depth[depth == 0.0] = MIN_DEPTH_REPLACEMENT_VALUE
        rescaled_depth[depth == 1.0] = MAX_DEPTH_REPLACEMENT_VALUE
        return rescaled_depth[:, :, -1]

    def _preprocess_goal(
        self, obs: List[Observations], goal_type
    ) -> Tuple[Tensor, List[str]]:
        assert "object_category" in obs
        obj_goal_id, start_rec_goal_id, end_rec_goal_id, goal_name = (
            None,
            None,
            None,
            None,
        )
        # Check if small object category is included in goal specification
        obj_name = self._obj_id_to_name_mapping[obs["object_category"][0]]
        if goal_type in ["object", "object_on_recep", "ovmm"]:
            goal_name = obj_name
            obj_goal_id = 1  # semantic sensor returns binary mask for goal object
        if goal_type == "object_on_recep":
            # navigating to object on start receptacle (before grasping)
            goal_name = (
                obj_name
                + " on "
                + self._rec_id_to_name_mapping[obs["start_receptacle"][0]]
            )
            start_rec_goal_id = 2
        elif goal_type == "ovmm":
            # nav goal specification for ovmm task includes all three categories:
            start_receptacle = self._rec_id_to_name_mapping[obs["start_receptacle"][0]]
            goal_receptacle = self._rec_id_to_name_mapping[obs["goal_receptacle"][0]]
            goal_name = obj_name + " " + start_receptacle + " " + goal_receptacle
            if self.ground_truth_semantics:
                start_rec_goal_id = (
                    self._receptacle_sensor_name_to_id[start_receptacle] + 1
                )
                end_rec_goal_id = (
                    self._receptacle_sensor_name_to_id[goal_receptacle] + 1
                )
            else:
                # habitat goal ids (from obs) -> combined mapping (also used for detic predictions)
                name_to_id_mapping = {
                    v: k for k, v in self.obj_rec_combined_mapping.items()
                }
                obj_goal_id = name_to_id_mapping[obj_name]
                start_rec_goal_id = name_to_id_mapping[start_receptacle]
                end_rec_goal_id = name_to_id_mapping[goal_receptacle]

        elif goal_type == "recep":
            # navigating to end receptacle (before placing)
            goal_name = self._rec_id_to_name_mapping[obs["goal_receptacle"][0]]
            end_rec_goal_id = 3
        return obj_goal_id, start_rec_goal_id, end_rec_goal_id, goal_name

    def get_current_joint_pos(self, habitat_obs: Dict[str, Any]) -> np.array:
        """Returns the current absolute positions from habitat observations for the joints controlled by the action space"""
        complete_joint_pos = habitat_obs["joint"]
        curr_joint_pos = complete_joint_pos[
            self.joints_mask == 1
        ]  # The action space will have the same size as curr_joint_pos
        # If action controls the arm extension, get the final extension by summing over individiual joint positions
        if self.joints_mask[0] == 1:
            curr_joint_pos[SimJointActionIndex.ARM] = np.sum(
                complete_joint_pos[:4]
            )  # The first 4 values in sensor add up to give the complete extension
        return curr_joint_pos

    def _preprocess_action(
        self, action: Union[home_robot.core.interfaces.Action, Dict], habitat_obs
    ) -> int:
        """convert the ovmm agent's action outputs to continuous Habitat actions"""
        if type(action) in [ContinuousFullBodyAction, ContinuousNavigationAction]:
            grip_action = -1
            # Keep holding in case holding an object
            if habitat_obs["is_holding"][0] == 1:
                grip_action = 1
            waypoint_x, waypoint_y, turn = 0, 0, 0
            # Set waypoint correctly, if base waypoint is passed with the action
            if action.xyt is not None:
                if action.xyt[0] != 0:
                    waypoint_x = np.clip(action.xyt[0] / self.max_forward, -1, 1)
                elif action.xyt[1] != 0:
                    waypoint_y = np.clip(action.xyt[1] / self.max_forward, -1, 1)
                elif action.xyt[2] != 0:
                    turn = np.clip(action.xyt[2] / self.max_turn, -1, 1)
            arm_action = np.array([0] * self.joints_dof)
            # If action is of type ContinuousFullBodyAction, it would include waypoints for the joints
            if type(action) == ContinuousFullBodyAction:
                # We specify only one arm extension that rolls over to all the arm joints
                arm_action = np.concatenate([action.joints[0:1], action.joints[4:]])
            cont_action = np.concatenate(
                [
                    arm_action / self.max_joints_delta,
                    [grip_action] + [waypoint_x, waypoint_y, turn, -1],
                ]
            )
        elif type(action) == DiscreteNavigationAction:
            grip_action = -1
            if (
                habitat_obs["is_holding"][0] == 1
                and action != DiscreteNavigationAction.DESNAP_OBJECT
            ) or action == DiscreteNavigationAction.SNAP_OBJECT:
                grip_action = 1

            turn = 0
            forward = 0
            if action == DiscreteNavigationAction.TURN_RIGHT:
                turn = -self.discrete_turn_degrees
            elif action == DiscreteNavigationAction.TURN_LEFT:
                turn = self.discrete_turn_degrees
            elif action == DiscreteNavigationAction.MOVE_FORWARD:
                forward = self.discrete_forward

            arm_action = np.zeros(self.joints_dof)
            curr_joint_pos = self.get_current_joint_pos(habitat_obs)
            target_joint_pos = curr_joint_pos
            if action == DiscreteNavigationAction.MANIPULATION_MODE:
                # turn left by 90 degrees, positive for turn left
                turn = 90
                target_joint_pos = map_joint_q_state_to_action_space(STRETCH_PREGRASP_Q)
                arm_action = target_joint_pos - curr_joint_pos
            elif action == DiscreteNavigationAction.NAVIGATION_MODE:
                target_joint_pos = map_joint_q_state_to_action_space(
                    STRETCH_NAVIGATION_Q
                )
                arm_action = target_joint_pos - curr_joint_pos
            elif action == DiscreteNavigationAction.EXTEND_ARM:
                target_joint_pos = curr_joint_pos.copy()
                target_joint_pos[SimJointActionIndex.ARM] = STRETCH_ARM_EXTENSION
                target_joint_pos[SimJointActionIndex.LIFT] = STRETCH_ARM_LIFT
                arm_action = target_joint_pos - curr_joint_pos

            stop = float(action == DiscreteNavigationAction.STOP) * 2 - 1
            cont_action = np.concatenate(
                [
                    arm_action / self.max_joints_delta,
                    [
                        grip_action,
                        forward / self.max_forward,
                        0.0,
                        turn / self.max_turn_degrees,
                        stop,
                    ],
                ]
            )
        else:
            raise ValueError(
                "Action needs to be of one of the following types: DiscreteNavigationAction, ContinuousNavigationAction or ContinuousFullBodyAction"
            )
        return np.array(cont_action, dtype=np.float32)

    def _process_info(self, info: Dict[str, Any]) -> Any:
        if info and self.visualize:
            self.visualizer.visualize(**info)

    def apply_action(
        self,
        action: home_robot.core.interfaces.Action,
        info: Optional[Dict[str, Any]] = None,
    ):
        if info is not None:
            if type(action) == ContinuousNavigationAction:
                info["curr_action"] = str([round(a, 3) for a in action.xyt])
            if type(action) == DiscreteNavigationAction:
                info["curr_action"] = DiscreteNavigationAction(action).name
            self._process_info(info)
        habitat_action = self._preprocess_action(action, self._last_habitat_obs)
        habitat_obs, _, dones, infos = self.habitat_env.step(habitat_action)
        # copy the keys in info starting with the prefix "is_curr_skill" into infos
        for key in info:
            if key.startswith("is_curr_skill"):
                infos[key] = info[key]
        self._last_habitat_obs = habitat_obs
        self._last_obs = self._preprocess_obs(habitat_obs)
        return self._last_obs, dones, infos
