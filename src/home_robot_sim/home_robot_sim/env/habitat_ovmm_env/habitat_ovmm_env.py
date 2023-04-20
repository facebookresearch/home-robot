import os
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import habitat
import numpy as np
import torch
from habitat.core.environments import GymHabitatEnv
from habitat.core.simulator import Observations
from skimage import color
from torch import Tensor

import home_robot
from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot_sim.env.habitat_abstract_env import HabitatEnv
from home_robot_sim.env.habitat_objectnav_env.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
    RearrangeBasicCategories,
    RearrangeDETICCategories,
)
from home_robot_sim.env.habitat_objectnav_env.visualizer import Visualizer


class HabitatOpenVocabManipEnv(HabitatEnv):
    semantic_category_mapping: Union[RearrangeBasicCategories, RearrangeDETICCategories]

    def __init__(self, habitat_env: habitat.core.env.Env, config, dataset):
        super().__init__(habitat_env)
        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        self._dataset = dataset
        self.visualizer = Visualizer(config, dataset)
        self.goal_type = config.habitat.task.goal_type
        self.episodes_data_path = config.habitat.dataset.data_path
        self.video_dir = config.habitat_baselines.video_dir
        assert (
            "floorplanner" in self.episodes_data_path
            or "hm3d" in self.episodes_data_path
            or "mp3d" in self.episodes_data_path
        )

        if "floorplanner" in self.episodes_data_path:
            self._obj_name_to_id_mapping = self._dataset.obj_category_to_obj_category_id
            self._rec_name_to_id_mapping = (
                self._dataset.recep_category_to_recep_category_id
            )
            self._obj_id_to_name_mapping = {
                k: v for v, k in self._obj_name_to_id_mapping.items()
            }
            self._rec_id_to_name_mapping = {
                k: v for v, k in self._rec_name_to_id_mapping.items()
            }

            if self.ground_truth_semantics:
                self.semantic_category_mapping = RearrangeBasicCategories()
            else:
                # combining objs and recep IDs into one mapping
                self.obj_rec_combined_mapping = {}
                for i in range(
                    len(self._obj_id_to_name_mapping)
                    + len(self._rec_id_to_name_mapping)
                ):
                    if i < len(self._obj_id_to_name_mapping):
                        self.obj_rec_combined_mapping[
                            i + 1
                        ] = self._obj_id_to_name_mapping[i]
                    else:
                        self.obj_rec_combined_mapping[
                            i + 1
                        ] = self._rec_id_to_name_mapping[
                            i - len(self._obj_id_to_name_mapping)
                        ]
                self.semantic_category_mapping = RearrangeDETICCategories(
                    self.obj_rec_combined_mapping
                )

        if not self.ground_truth_semantics:
            from home_robot.perception.detection.detic.detic_perception import (
                DeticPerception,
            )

            # TODO Specify confidence threshold as a parameter
            self.segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary=",".join(
                    ["."] + list(self.obj_rec_combined_mapping.values()) + ["other"]
                ),
                sem_gpu_id=0,
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
            },
            joint=habitat_obs["joint"],
            is_holding=habitat_obs["is_holding"],
            relative_resting_position=habitat_obs["relative_resting_position"],
            third_person_image=habitat_obs["robot_third_rgb"],
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
            instance_id_to_category_id = (
                self.semantic_category_mapping.instance_id_to_category_id
            )
            semantic = torch.from_numpy(
                habitat_obs["object_segmentation"].squeeze(-1).astype(np.int64)
            )
            start_recep_seg = torch.from_numpy(
                habitat_obs["start_recep_segmentation"].squeeze(-1).astype(np.int64)
            )
            goal_recep_seg = torch.from_numpy(
                habitat_obs["goal_recep_segmentation"].squeeze(-1).astype(np.int64)
            )
            instance_id_to_category_id = (
                self.semantic_category_mapping.instance_id_to_category_id
            )
            # Assign semantic id of 1 for object_category, 2 for start_receptacle, 3 for goal_receptacle
            semantic = semantic + start_recep_seg * 2 + goal_recep_seg * 3
            semantic = torch.clip(semantic, 0, 3)
            # TODO: update semantic_category_mapping
            obs.semantic = instance_id_to_category_id[semantic]
            # TODO Ground-truth semantic visualization
        else:
            obs = self.segmentation.predict(obs, depth_threshold=0.5)
            if type(self.semantic_category_mapping) == RearrangeDETICCategories:
                # First index is a dummy unused category
                obs.semantic[obs.semantic == 0] = (
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
        if goal_type in ["object", "object_on_recep", "ovmm"]:
            goal_name = self._obj_id_to_name_mapping[obs["object_category"][0]]
            obj_goal_id = 1  # semantic sensor returns binary mask for goal object
        if goal_type == "object_on_recep":
            # navigating to object on start receptacle (before grasping)
            goal_name = (
                self._obj_id_to_name_mapping[obs["object_category"][0]]
                + " on "
                + self._rec_id_to_name_mapping[obs["start_receptacle"][0]]
            )
            start_rec_goal_id = 2
        elif goal_type == "ovmm":
            # nav goal specification for ovmm task includes all three categories:
            goal_name = (
                self._obj_id_to_name_mapping[obs["object_category"][0]]
                + " "
                + self._rec_id_to_name_mapping[obs["start_receptacle"][0]]
                + " "
                + self._rec_id_to_name_mapping[obs["goal_receptacle"][0]]
            )
            if self.ground_truth_semantics:
                start_rec_goal_id = 2
                end_rec_goal_id = 3
            else:
                # habitat goal ids (from obs) -> combined mapping (also used for detic predictions)
                obj_goal_id = (
                    obs["object_category"][0] + 1
                )  # detic predictions use mapping that starts from 1
                start_rec_goal_id = (
                    len(self._obj_id_to_name_mapping.keys())
                    + obs["start_receptacle"]
                    + 1
                )
                end_rec_goal_id = (
                    len(self._obj_id_to_name_mapping.keys())
                    + obs["goal_receptacle"]
                    + 1
                )

        elif goal_type == "recep":
            # navigating to end receptacle (before placing)
            goal_name = self._rec_id_to_name_mapping[obs["goal_receptacle"][0]]
            end_rec_goal_id = 3
        return obj_goal_id, start_rec_goal_id, end_rec_goal_id, goal_name

    def _preprocess_action(
        self, action: Union[home_robot.core.interfaces.Action, Dict], habitat_obs
    ) -> int:
        # convert planner output to continuous Habitat actions
        if isinstance(action, dict):
            grip_action = [-1]
            if "grip_action" in action:
                grip_action = action["grip_action"]
            base_vel = [0, 0]
            if "base_vel" in action:
                base_vel = action["base_vel"]
            arm_action = [0] * 7
            if "arm_action" in action:
                arm_action = action["arm_action"]
            rearrange_stop = [-1]
            if "rearrange_stop" in action:
                rearrange_stop = action["rearrange_stop"]
            cont_action = np.concatenate(
                [arm_action, grip_action, base_vel, [-1, -1, rearrange_stop[0], -1]]
            )
        else:
            grip_action = -1
            if (
                habitat_obs["is_holding"][0] == 1
                and action != DiscreteNavigationAction.DESNAP_OBJECT
            ) or action == DiscreteNavigationAction.SNAP_OBJECT:
                grip_action = 1

            waypoint = 0
            if action == DiscreteNavigationAction.TURN_RIGHT:
                waypoint = -1
            elif action in [
                DiscreteNavigationAction.TURN_LEFT,
                DiscreteNavigationAction.MOVE_FORWARD,
            ]:
                waypoint = 1

            face_arm = float(action == DiscreteNavigationAction.FACE_ARM) * 2 - 1
            stop = float(action == DiscreteNavigationAction.STOP) * 2 - 1
            reset_joints = (
                float(action == DiscreteNavigationAction.RESET_JOINTS) * 2 - 1
            )
            extend_arm = float(action == DiscreteNavigationAction.EXTEND_ARM) * 2 - 1
            arm_actions = [0] * 7
            cont_action = arm_actions + [
                grip_action,
                waypoint,
                (action == DiscreteNavigationAction.MOVE_FORWARD) * 2 - 1,
                extend_arm,
                face_arm,
                stop,
                reset_joints,
            ]
        return np.array(cont_action, dtype=np.float32)

    def _process_info(self, info: Dict[str, Any]) -> Any:
        if info:
            self.visualizer.visualize(**info)

    def apply_action(
        self,
        action: home_robot.core.interfaces.Action,
        info: Optional[Dict[str, Any]] = None,
    ):
        if info is not None:
            self._process_info(info)
        habitat_action = self._preprocess_action(action, self._last_habitat_obs)
        habitat_obs, _, dones, infos = self.habitat_env.step(habitat_action)
        self._last_habitat_obs = habitat_obs
        self._last_obs = self._preprocess_obs(habitat_obs)
        return self._last_obs, dones, infos
