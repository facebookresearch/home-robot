from typing import Any, Dict, Optional, Tuple, Union, cast, List

import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions

import home_robot
from home_robot.utils.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
)
from home_robot_sim.env.habitat_abstract_env import HabitatEnv

from home_robot_sim.env.habitat_objectnav_env.constants import (
    FloorplannertoMukulIndoor,
    HM3DtoCOCOIndoor,
    HM3DtoHSSD28Indoor,
    LanguageNavCategories,
    coco_categories_mapping,
    hssd_28categories_padded,
    mukul_33categories_padded,
)
from home_robot_sim.env.habitat_objectnav_env.visualizer import Visualizer


class HabitatLanguageNavEnv(HabitatEnv):
    semantic_category_mapping: Union[HM3DtoCOCOIndoor, FloorplannertoMukulIndoor]

    def __init__(self, habitat_env: habitat.core.env.Env, config):
        super().__init__(habitat_env)

        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        self.visualizer = Visualizer(config)

        self.episodes_data_path = config.habitat.dataset.data_path
        assert (
            "floorplanner" in self.episodes_data_path
            or "hm3d" in self.episodes_data_path
            or "mp3d" in self.episodes_data_path
        )
        if "hm3d" in self.episodes_data_path:
            # if config.AGENT.SEMANTIC_MAP.semantic_categories == "coco_indoor":
            #     self.vocabulary = "coco"
            #     self.semantic_category_mapping = HM3DtoCOCOIndoor()
            # elif config.AGENT.SEMANTIC_MAP.semantic_categories == "hssd_28_cat":
            #     self.semantic_category_mapping = HM3DtoHSSD28Indoor()
            #     self.vocabulary = "custom"
            #     self.custom_vocabulary = hssd_28categories_padded 
            self.semantic_category_mapping = LanguageNavCategories()
            # else:
            #     raise NotImplementedError
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

        # if not self.ground_truth_semantics:
        #     from home_robot.perception.detection.detic.detic_perception import (
        #         DeticPerception,
        #     )

        #     # TODO Specify confidence threshold as a parameter
        #     if self.vocabulary == "custom":
        #         self.segmentation = DeticPerception(
        #             vocabulary=self.vocabulary,
        #             custom_vocabulary=",".join(self.custom_vocabulary),
        #             sem_gpu_id=(
        #                 -1 if config.NO_GPU else self.habitat_env.sim.gpu_device
        #             ),
        #         )
        #     else:
        #         self.segmentation = DeticPerception(
        #             vocabulary=self.vocabulary,
        #             sem_gpu_id=(
        #                 -1 if config.NO_GPU else self.habitat_env.sim.gpu_device
        #             ),
        #         )
        self.config = config

    def reset(self):
        habitat_obs = self.habitat_env.reset()
        target, landmarks = self._preprocess_goal(habitat_obs["languagegoal"])
        caption_tokens = (target, landmarks)

        if not self.ground_truth_semantics:
            vocabulary = [".", target, landmarks]
            self.init_perception_module(vocabulary=[".", target] + landmarks)

        self.semantic_category_mapping.reset_instance_id_to_category_id(
            self.habitat_env
        )
        self._last_obs = self._preprocess_obs(habitat_obs, caption_tokens)
        self.visualizer.reset()
        scene_id = self.habitat_env.current_episode.scene_id.split("/")[-1].split(".")[
            0
        ]
        self.visualizer.set_vis_dir(
            scene_id, self.habitat_env.current_episode.episode_id
        )

    def init_perception_module(self, vocabulary: Tuple[str, str]):
        from home_robot.perception.detection.detic.detic_perception import (
            DeticPerception,
        )
        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary=",".join(vocabulary),
            sem_gpu_id=(
                -1 if self.config.NO_GPU else self.habitat_env.sim.gpu_device
            ),
        )

        print("Initializing perception module with vocabulary:", vocabulary)

    def _preprocess_obs(
        self, habitat_obs: habitat.core.simulator.Observations, caption_tokens=None
    ) -> home_robot.core.interfaces.Observations:
        depth = self._preprocess_depth(habitat_obs["depth"])
        if caption_tokens is None:
            target, landmarks = self._preprocess_goal(habitat_obs["languagegoal"])
        else:
            target, landmarks = caption_tokens

        # TODO: extract all object/landmark tokens from instruction -> pass that as vocabulary to detic / for now just pass landmark and object
        obs = home_robot.core.interfaces.Observations(
            rgb=habitat_obs["rgb"],
            depth=depth,
            compass=habitat_obs["compass"],
            gps=self._preprocess_xy(habitat_obs["gps"]),
            task_observations={
                "target": target,
                "landmarks": landmarks,
                # "object_goal": goal_id,
                # "goal_name": goal_name,
                # "recep_goal": None,
            },
            camera_pose=None,
            third_person_image=None,
        )
        obs = self._preprocess_semantic(obs, habitat_obs["semantic"])
        return obs

    def _preprocess_semantic(
        self, obs: home_robot.core.interfaces.Observations, habitat_semantic: np.ndarray
    ) -> home_robot.core.interfaces.Observations:
        if self.ground_truth_semantics:
            raise NotImplementedError
            # if self.config.AGENT.SEMANTIC_MAP.semantic_categories == "hssd_28_cat":
            # instance_id_to_category_id = (
            #     self.semantic_category_mapping.instance_id_to_category_id
            # )
            # obs.semantic = instance_id_to_category_id[habitat_semantic[:, :, -1]]
            # # TODO Ground-truth semantic visualization
            # obs.task_observations["semantic_frame"] = obs.rgb
        else:
            obs = self.segmentation.predict(obs, depth_threshold=0.5)
            # if self.vocabulary == "coco":
            #     obs.semantic = np.vectorize(coco_categories_mapping.get)(obs.semantic)
            #     obs.semantic[obs.semantic == None] = (  # noqa: E711
            #         self.semantic_category_mapping.num_sem_categories - 1
            #     )
            #     obs.semantic = obs.semantic.astype(int)
            # if type(self.semantic_category_mapping) == FloorplannertoMukulIndoor:
            #     # First index is a dummy unused category
            #     obs.semantic[obs.semantic == 0] = (
            #         self.semantic_category_mapping.num_sem_categories - 1
            #     )
        obs.task_observations["semantic_frame"] = np.concatenate(
            [obs.rgb, obs.semantic[:, :, np.newaxis]], axis=2
        ).astype(np.uint8)
        return obs

    def _preprocess_depth(self, depth: np.array) -> np.array:
        rescaled_depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        rescaled_depth[depth == 0.0] = MIN_DEPTH_REPLACEMENT_VALUE
        rescaled_depth[depth == 1.0] = MAX_DEPTH_REPLACEMENT_VALUE
        return rescaled_depth[:, :, -1]

    def _preprocess_goal(self, goal: Dict) -> Tuple[str, str]:
        target = goal["category_name"]
        # caption = goal["caption"].replace('Instruction: ', '')
        landmarks = goal["landmarks"]

        if target in landmarks:
            landmarks.remove(target)

        target = "_".join(target.split())
        landmarks=["_".join(landmark.split()) for landmark in landmarks]
        # room = goal["room"]
        return target, landmarks

    def _preprocess_action(self, action: home_robot.core.interfaces.Action) -> int:
        discrete_action = cast(
            home_robot.core.interfaces.DiscreteNavigationAction, action
        )
        return HabitatSimActions[discrete_action.name.lower()]

    def _process_info(self, info: Dict[str, Any]) -> Any:
        if info:
            self.visualizer.visualize(**info)
