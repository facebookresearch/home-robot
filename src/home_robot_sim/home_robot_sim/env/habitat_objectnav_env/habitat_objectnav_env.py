from typing import Any, Dict, List, Optional, Tuple, Union, cast

import habitat
import numpy as np
import torch
from habitat.core.simulator import Observations
from torch import Tensor

import home_robot
from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot_sim.env.habitat_abstract_env import HabitatEnv

from .constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
    FloorplannertoMukulIndoor,
    HM3DtoCOCOIndoor,
    RearrangeCategories,
    mukul_33categories_padded,
    rearrange_3categories_padded,
)
from .visualizer import Visualizer


class HabitatObjectNavEnv(HabitatEnv):
    semantic_category_mapping: Union[HM3DtoCOCOIndoor, FloorplannertoMukulIndoor]

    def __init__(self, habitat_env: habitat.core.env.Env, config):
        super().__init__(habitat_env)

        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        self.visualizer = Visualizer(config)
        self.goal_type = config.habitat.task.goal_type
        self.episodes_data_path = config.habitat.dataset.data_path
        assert (
            "floorplanner" in self.episodes_data_path
            or "hm3d" in self.episodes_data_path
            or "mp3d" in self.episodes_data_path
        )

        if "hm3d" in self.episodes_data_path:
            if config.AGENT.SEMANTIC_MAP.semantic_categories == "coco_indoor":
                self.semantic_category_mapping = HM3DtoCOCOIndoor()
            else:
                raise NotImplementedError
        elif (
            "floorplanner" in self.episodes_data_path
            and "CatNavToObjTask" in config.habitat.task.type
        ):
            self.semantic_category_mapping = RearrangeCategories()
            self._obj_name_to_id_mapping = {
                "action_figure": 0,
                "cup": 1,
                "dishtowel": 2,
                "hat": 3,
                "sponge": 4,
                "stuffed_toy": 5,
                "tape": 6,
                "vase": 7,
            }
            self._rec_name_to_id_mapping = {
                "armchair": 0,
                "armoire": 1,
                "bar_stool": 2,
                "coffee_table": 3,
                "desk": 4,
                "dining_table": 5,
                "kitchen_island": 6,
                "sofa": 7,
                "stool": 8,
            }
            self._obj_id_to_name_mapping = {
                k: v for v, k in self._obj_name_to_id_mapping.items()
            }
            self._rec_id_to_name_mapping = {
                k: v for v, k in self._rec_name_to_id_mapping.items()
            }
        elif "floorplanner" in self.episodes_data_path:
            if config.AGENT.SEMANTIC_MAP.semantic_categories == "mukul_indoor":
                self.semantic_category_mapping = FloorplannertoMukulIndoor()
            else:
                raise NotImplementedError
        elif "mp3d" in self.episodes_data_path:
            # TODO This is a hack to get unit tests running, we'll need to
            #  adapt this if we want to run ObjectNav on MP3D
            if config.AGENT.SEMANTIC_MAP.semantic_categories == "mukul_indoor":
                self.semantic_category_mapping = FloorplannertoMukulIndoor()
            else:
                raise NotImplementedError

        if not self.ground_truth_semantics:
            from home_robot.perception.detection.detic.detic_perception import (
                DeticPerception,
            )

            # TODO Specify confidence threshold as a parameter
            self.segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary=",".join(mukul_33categories_padded),
                sem_gpu_id=(-1 if config.NO_GPU else self.habitat_env.sim.gpu_device),
            )

    def reset(self):
        habitat_obs = self.habitat_env.reset()
        self.semantic_category_mapping.reset_instance_id_to_category_id(
            self.habitat_env
        )
        self._last_obs = self._preprocess_obs(habitat_obs)
        self.visualizer.reset()

    def _preprocess_obs(
        self, habitat_obs: habitat.core.simulator.Observations
    ) -> home_robot.core.interfaces.Observations:
        depth = self._preprocess_depth(habitat_obs["robot_head_depth"])
        object_goal, recep_goal, goal_name = self._preprocess_goal(
            habitat_obs, self.goal_type
        )

        obs = home_robot.core.interfaces.Observations(
            rgb=habitat_obs["robot_head_rgb"],
            depth=depth,
            compass=habitat_obs["robot_start_compass"],
            gps=self._preprocess_xy(habitat_obs["robot_start_gps"]),
            task_observations={
                "object_goal": object_goal,
                "recep_goal": recep_goal,
                "goal_name": goal_name,
            },
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
            obs.task_observations["semantic_frame"] = obs.rgb
        else:
            obs = self.segmentation.predict(obs, depth_threshold=0.5)
            if type(self.semantic_category_mapping) == FloorplannertoMukulIndoor:
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
        obj_goal_id, rec_goal_id, goal_name = None, None, None

        if goal_type in ["object", "object_on_recep"]:
            goal_name = self._obj_id_to_name_mapping[obs["object_category"][0]]
            obj_goal_id = 1  # semantic sensor returns binary mask for goal object
        if goal_type == "object_on_recep":
            goal_name = (
                self._obj_id_to_name_mapping[obs["object_category"][0]]
                + " on "
                + self._rec_id_to_name_mapping[obs["start_receptacle"][0]]
            )
            rec_goal_id = 2
        if goal_type == "recep":
            goal_name = self._rec_id_to_name_mapping[obs["goal_receptacle"][0]]
            rec_goal_id = 3
            obj_goal_id = None
        return obj_goal_id, rec_goal_id, goal_name

    def _preprocess_action(self, action: home_robot.core.interfaces.Action) -> int:
        # convert planner output to continuous Habitat actions
        action_map = {
            DiscreteNavigationAction.TURN_RIGHT: [0, 0, -1, -1],
            DiscreteNavigationAction.MOVE_FORWARD: [0, 1, 0, -1],
            DiscreteNavigationAction.TURN_LEFT: [0, 0, 1, -1],
            DiscreteNavigationAction.STOP: [0, 0, 0, 1],
        }
        cont_action = action_map[action]
        actions = {
            "action": ("arm_action", "base_velocity", "rearrange_stop"),
            "action_args": {
                "arm_action": np.array([cont_action[0]], dtype=np.float32),
                "base_vel": np.array(cont_action[1:3], dtype=np.float32),
                "rearrange_stop": np.array([cont_action[-1]], dtype=np.float32),
            },
        }
        return actions

    def _process_info(self, info: Dict[str, Any]) -> Any:
        if info:
            self.visualizer.visualize(**info)
