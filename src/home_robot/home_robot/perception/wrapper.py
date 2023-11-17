# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Any, Dict, Optional, Tuple

from home_robot.core.interfaces import Observations
from home_robot.perception.constants import RearrangeDETICCategories
from home_robot.utils.config import load_config


class OvmmPerception:
    """
    Wrapper around DETIC for use in OVMM Agent.
    It performs some preprocessing of observations necessary for OVMM skills.
    It also maintains a list of vocabularies to use in segmentation and can switch between them at runtime.
    """

    def __init__(
        self,
        config,
        gpu_device_id: int = 0,
        verbose: bool = False,
        module: str = "grounded_sam",
        module_kwargs: Dict[str, Any] = {},
    ):
        self.config = config
        self._use_detic_viz = config.ENVIRONMENT.use_detic_viz
        self._detection_module = getattr(config.AGENT, "detection_module", "detic")
        self._vocabularies: Dict[int, RearrangeDETICCategories] = {}
        self._current_vocabulary: RearrangeDETICCategories = None
        self._current_vocabulary_id: int = None
        self.verbose = verbose
        if self._detection_module == "detic":
            # Lazy import
            from home_robot.perception.detection.detic.detic_perception import (
                DeticPerception,
            )

            # TODO Specify confidence threshold as a parameter
            self._segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary=".",
                sem_gpu_id=gpu_device_id,
                verbose=verbose,
                **module_kwargs,
            )
        elif self._detection_module == "grounded_sam":
            from home_robot.perception.detection.grounded_sam.grounded_sam_perception import (
                GroundedSAMPerception,
            )

            self._segmentation = GroundedSAMPerception(
                custom_vocabulary=".",
                sem_gpu_id=gpu_device_id,
                verbose=verbose,
                **module_kwargs,
            )
        else:
            raise NotImplementedError

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
        self.segmenter_classes = (
            ["."] + list(vocabulary.goal_id_to_goal_name.values()) + ["other"]
        )
        self._segmentation.reset_vocab(self.segmenter_classes)

        self.vocabulary_name_to_id = {
            name: id for id, name in vocabulary.goal_id_to_goal_name.items()
        }
        self.vocabulary_id_to_name = vocabulary.goal_id_to_goal_name
        self.seg_id_to_name = dict(enumerate(self.segmenter_classes))
        self.name_to_seg_id = {v: k for k, v in self.seg_id_to_name.items()}

        self._current_vocabulary = vocabulary
        self._current_vocabulary_id = vocabulary_id

    def get_class_name_for_id(self, oid: int) -> str:
        """return name of a class from a detection"""
        return self._current_vocabulary.goal_id_to_goal_name[oid]

    def get_class_id_for_name(self, name: str) -> int:
        """return the id associated with a class"""
        return self._current_vocabulary.goal_name_to_goal_id[name]

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

    def predict(self, obs: Observations, depth_threshold: float = 0.5) -> Observations:
        """Run with no postprocessing. Updates observation to add semantics."""
        print(self.current_vocabulary.goal_id_to_goal_name.values())
        return self._segmentation.predict(
            obs,
            depth_threshold=depth_threshold,
            draw_instance_predictions=self._use_detic_viz,
        )

    def forward(self, obs: Observations, depth_threshold: float = 0.5) -> Observations:
        """
        Run segmentation model and preprocess observations for OVMM skills
        """
        obs = self.predict(obs, depth_threshold)
        self._process_obs(obs)
        return obs


def read_category_map_file(
    category_map_file: str,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Reads a category map file in JSON and extracts mappings between category names and category IDs.
    These mappings are also present in the episodes file but are extracted to use in a stand-alone manner.
    Returns object and receptacle mappings.
    """
    if os.environ["HOME_ROBOT_ROOT"]:
        category_map_file = os.path.join(
            os.environ["HOME_ROBOT_ROOT"], category_map_file
        )

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


def create_semantic_sensor(
    config=None,
    category_map_file: Optional[str] = None,
    device_id: int = 0,
    verbose: bool = True,
    module_kwargs: Dict[str, Any] = {},
    **kwargs,
):
    """Create segmentation sensor and load config. Returns config from file, as well as a OvmmPerception object that can be used to label scenes."""
    if verbose:
        print("- Loading configuration")
    if config is None:
        config = load_config(visualize=False, **kwargs)
    if category_map_file is None:
        category_map_file = config.ENVIRONMENT.category_map_file

    if verbose:
        print("- Create and load vocabulary and perception model")
    semantic_sensor = OvmmPerception(
        config=config,
        gpu_device_id=device_id,
        verbose=verbose,
        module="detic",
        module_kwargs=module_kwargs,
    )
    obj_name_to_id, rec_name_to_id = read_category_map_file(category_map_file)
    vocab = build_vocab_from_category_map(obj_name_to_id, rec_name_to_id)
    semantic_sensor.update_vocabulary_list(vocab, 0)
    semantic_sensor.set_vocabulary(0)
    return config, semantic_sensor
