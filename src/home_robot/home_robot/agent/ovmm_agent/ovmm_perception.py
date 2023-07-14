# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
from typing import Dict, Tuple

from home_robot.core.interfaces import Observations
from home_robot.perception.constants import RearrangeDETICCategories
from home_robot.perception.detection.detic.detic_perception import DeticPerception


def read_category_map_file(
    category_map_file: str,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Reads a category map file in JSON and extracts mappings between category names and category IDs.
    These mappings are also present in the episodes file but are extracted to use in a stand-alone manner.
    Returns object and receptacle mappings.
    """
    with open(category_map_file) as f:
        category_map = json.load(f)

    obj_name_to_id_mapping = category_map["obj_category_to_obj_category_id"]
    rec_name_to_id_mapping = category_map["recep_category_to_recep_category_id"]
    obj_id_to_name_mapping = {k: v for v, k in obj_name_to_id_mapping.items()}
    rec_id_to_name_mapping = {k: v for v, k in rec_name_to_id_mapping.items()}

    return obj_id_to_name_mapping, rec_id_to_name_mapping


def build_vocab_from_category_map(
    obj_id_to_name_mapping: Dict[int, str], rec_id_to_name_mapping: Dict[int, str]
) -> RearrangeDETICCategories:
    """
    Build vocabulary from category maps that can be used for semantic sensor and visualizations.
    """
    obj_rec_combined_mapping = {}
    for i in range(len(obj_id_to_name_mapping) + len(rec_id_to_name_mapping)):
        if i < len(obj_id_to_name_mapping):
            obj_rec_combined_mapping[i + 1] = obj_id_to_name_mapping[i]
        else:
            obj_rec_combined_mapping[i + 1] = rec_id_to_name_mapping[
                i - len(obj_id_to_name_mapping)
            ]
    vocabulary = RearrangeDETICCategories(
        obj_rec_combined_mapping, len(obj_id_to_name_mapping)
    )
    return vocabulary


class OvmmPerception:
    """
    Wrapper around DETIC for use in OVMM Agent.
    It performs some preprocessing of observations necessary for OVMM skills.
    It also maintains a list of vocabularies to use in segmentation and can switch between them at runtime.
    """

    def __init__(self, config, gpu_device_id: int = 0, verbose: bool = False):
        self.config = config
        self._use_detic_viz = config.ENVIRONMENT.use_detic_viz
        self._vocabularies: Dict[int, RearrangeDETICCategories] = {}
        self._current_vocabulary: RearrangeDETICCategories = None
        self._current_vocabulary_id: int = None
        self.verbose = verbose
        # TODO Specify confidence threshold as a parameter
        self._segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary=".",
            sem_gpu_id=gpu_device_id,
            verbose=verbose,
        )

    @property
    def current_vocabulary_id(self) -> int:
        return self._current_vocabulary_id

    @property
    def current_vocabulary(self) -> RearrangeDETICCategories:
        return self._current_vocabulary

    def update_vocabulary_list(
        self, vocabulary: RearrangeDETICCategories, vocabulary_id: int
    ):
        """
        Update/insert a given vocabulary for the given ID.
        """
        self._vocabularies[vocabulary_id] = vocabulary

    def set_vocabulary(self, vocabulary_id: int):
        """
        Set given vocabulary ID to be the active vocabulary that the segmentation model uses.
        """
        vocabulary = self._vocabularies[vocabulary_id]
        self._segmentation.reset_vocab(
            ["."] + list(vocabulary.goal_id_to_goal_name.values()) + ["other"]
        )
        self.vocabulary_name_to_id = {
            name: id for id, name in vocabulary.goal_id_to_goal_name.items()
        }
        self._current_vocabulary = vocabulary
        self._current_vocabulary_id = vocabulary_id

    def _process_obs(self, obs: Observations):
        """
        Process observations. Add pointers to objects and other metadata in segmentation mask.
        """
        obs.semantic[obs.semantic == 0] = (
            self._current_vocabulary.num_sem_categories - 1
        )
        obs.task_observations["recep_idx"] = (
            self._current_vocabulary.num_sem_obj_categories + 1
        )
        obs.task_observations["semantic_max_val"] = (
            self._current_vocabulary.num_sem_categories - 1
        )
        if obs.task_observations["start_recep_name"] is not None:
            obs.task_observations["start_recep_goal"] = self.vocabulary_name_to_id[
                obs.task_observations["start_recep_name"]
            ]
        else:
            obs.task_observations["start_recep_name"] = None
        if obs.task_observations["place_recep_name"] is not None:
            obs.task_observations["end_recep_goal"] = self.vocabulary_name_to_id[
                obs.task_observations["place_recep_name"]
            ]
        else:
            obs.task_observations["end_recep_name"] = None
        if obs.task_observations["object_name"] is not None:
            obs.task_observations["object_goal"] = self.vocabulary_name_to_id[
                obs.task_observations["object_name"]
            ]
        else:
            obs.task_observations["object_goal"] = None

    def __call__(self, obs: Observations) -> Observations:
        return self.forward(obs)

    def forward(self, obs: Observations) -> Observations:
        """
        Run segmentation model and preprocess observations for OVMM skills
        """
        obs = self._segmentation.predict(
            obs, depth_threshold=0.5, draw_instance_predictions=self._use_detic_viz
        )
        self._process_obs(obs)
        return obs
