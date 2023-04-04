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
    RearrangeCategories,
)
from home_robot_sim.env.habitat_objectnav_env.visualizer import Visualizer


class HabitatOpenVocabManipEnv(HabitatEnv):
    semantic_category_mapping: Union[RearrangeCategories]

    def __init__(self, habitat_env: habitat.core.env.Env, config, dataset):
        super().__init__(habitat_env)
        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        self.visualizer = Visualizer(config)
        self.goal_type = config.habitat.task.goal_type
        self.episodes_data_path = config.habitat.dataset.data_path
        self._dataset = dataset
        self.video_dir = config.habitat_baselines.video_dir
        assert (
            "floorplanner" in self.episodes_data_path
            or "hm3d" in self.episodes_data_path
            or "mp3d" in self.episodes_data_path
        )

        if "floorplanner" in self.episodes_data_path:
            self.semantic_category_mapping = RearrangeCategories()
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

        if not self.ground_truth_semantics:
            from home_robot.perception.detection.detic.detic_perception import (
                DeticPerception,
            )

            # TODO Specify confidence threshold as a parameter
            self.segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary=",".join(
                    list(self._obj_name_to_id_mapping.keys())
                    + list(self._rec_name_to_id_mapping.keys())
                ),
                sem_gpu_id=(-1 if config.NO_GPU else self.habitat_env.sim.gpu_device),
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

    def update_hab_pose(self, hab_pose):
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
            },
            third_person_image=habitat_obs["robot_third_rgb"],
            camera_pose=self.update_hab_pose(np.asarray(habitat_obs["camera_pose"])),
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
            obs.task_observations["semantic_frame"] = np.concatenate(
                [obs.rgb, obs.semantic[:, :, np.newaxis]], axis=2
            ).astype(np.uint8)
        else:
            obs = self.segmentation.predict(obs, depth_threshold=0.5)
            if type(self.semantic_category_mapping) == RearrangeCategories:
                # First index is a dummy unused category
                obs.semantic[obs.semantic == 0] = (
                    self.semantic_category_mapping.num_sem_categories - 1
                )
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

        if goal_type in ["object", "object_on_recep", "ovmm"]:
            goal_name = self._obj_id_to_name_mapping[obs["object_category"][0]]
            obj_goal_id = 1  # semantic sensor returns binary mask for goal object
        if goal_type == "object_on_recep":
            goal_name = (
                self._obj_id_to_name_mapping[obs["object_category"][0]]
                + " on "
                + self._rec_id_to_name_mapping[obs["start_receptacle"][0]]
            )
            start_rec_goal_id = 2
        elif goal_type == "ovmm":
            goal_name = (
                self._obj_id_to_name_mapping[obs["object_category"][0]]
                + " "
                + self._rec_id_to_name_mapping[obs["start_receptacle"][0]]
                + " "
                + self._rec_id_to_name_mapping[obs["goal_receptacle"][0]]
            )
            start_rec_goal_id = 2
            end_rec_goal_id = 3
        elif goal_type == "recep":
            goal_name = self._rec_id_to_name_mapping[obs["goal_receptacle"][0]]
            end_rec_goal_id = 3
        return obj_goal_id, start_rec_goal_id, end_rec_goal_id, goal_name

    def _preprocess_action(
        self, action: home_robot.core.interfaces.Action, habitat_obs
    ) -> int:
        # convert planner output to continuous Habitat actions
        grip_action = -1
        if (
            habitat_obs["is_holding"][0] == 1
            and action != DiscreteNavigationAction.DESNAP_OBJECT
        ) or action == DiscreteNavigationAction.SNAP_OBJECT:
            grip_action = 1

        turn = 0
        if action == DiscreteNavigationAction.TURN_RIGHT:
            turn = -1
        elif action == DiscreteNavigationAction.TURN_LEFT:
            turn = 1

        forward = float(action == DiscreteNavigationAction.MOVE_FORWARD)
        face_arm = float(action == DiscreteNavigationAction.FACE_ARM) * 2 - 1
        stop = float(action == DiscreteNavigationAction.STOP) * 2 - 1
        reset_joints = float(action == DiscreteNavigationAction.RESET_JOINTS) * 2 - 1
        extend_arm = float(action == DiscreteNavigationAction.EXTEND_ARM) * 2 - 1
        arm_actions = [0] * 7
        cont_action = arm_actions + [
            grip_action,
            forward,
            turn,
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
