from typing import Any, Dict, List, Optional, Tuple, Union, cast

import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions

import home_robot
from home_robot.perception.constants import (
    HM3DtoCOCOIndoor,
    LanguageNavCategories,
    all_hm3d_categories,
    coco_categories_mapping,
)
from home_robot.utils.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
)
from home_robot_sim.env.habitat_abstract_env import HabitatEnv
from home_robot_sim.env.habitat_goat_env.visualizer import Visualizer


class HabitatGoatEnv(HabitatEnv):
    semantic_category_mapping: Union[HM3DtoCOCOIndoor]

    def __init__(self, habitat_env: habitat.core.env.Env, config):
        super().__init__(habitat_env)

        self.min_depth = config.ENVIRONMENT.min_depth
        self.max_depth = config.ENVIRONMENT.max_depth
        self.ground_truth_semantics = config.GROUND_TRUTH_SEMANTICS
        self.visualizer = Visualizer(config)

        self.episodes_data_path = config.habitat.dataset.data_path

        assert "hm3d" in self.episodes_data_path, "only HM3D scenes supported for now."

        if "hm3d" in self.episodes_data_path:
            self.semantic_category_mapping = LanguageNavCategories()

        self.config = config
        self.current_episode = None

    def fetch_vocabulary(self, goals):
        # TODO: get open set vocabulary
        vocabulary = []
        for goal in goals:
            vocabulary.append(goal["target"])
            if "landmarks" in goal.keys():
                vocabulary += goal["landmarks"]
        return set(vocabulary)

    def reset(self):
        habitat_obs = self.habitat_env.reset()
        self.current_episode = self.habitat_env.current_episode
        self.active_task_idx = 0
        # goal_type, goal = self.update_and_fetch_goal()
        goals = habitat_obs["multigoal"]
        # open set vocabulary – all HM3D categories?
        vocabulary = self.fetch_vocabulary(goals)
        print("Vocabulary:", vocabulary)
        if not self.ground_truth_semantics:
            self.init_perception_module(vocabulary)

        self.semantic_category_mapping.reset_instance_id_to_category_id(
            self.habitat_env
        )
        self._last_obs = self._preprocess_obs(habitat_obs)
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
            sem_gpu_id=(-1 if self.config.NO_GPU else self.habitat_env.sim.gpu_device),
        )

        print("Initializing perception module with vocabulary:", vocabulary)

    def _preprocess_obs(
        self, habitat_obs: habitat.core.simulator.Observations
    ) -> home_robot.core.interfaces.Observations:
        depth = self._preprocess_depth(habitat_obs["depth"])
        goals, vocabulary = self._preprocess_goals(
            self.current_episode.tasks, habitat_obs
        )
        obs = home_robot.core.interfaces.Observations(
            rgb=habitat_obs["rgb"],
            depth=depth,
            compass=habitat_obs["compass"],
            gps=self._preprocess_xy(habitat_obs["gps"]),
            task_observations={
                "tasks": goals,
                "top_down_map": self.get_episode_metrics()["goat_top_down_map"],
            },
            camera_pose=None,
            third_person_image=None,
        )
        obs = self._preprocess_semantic(obs, habitat_obs["semantic"], vocabulary)
        return obs

    def _preprocess_semantic(
        self,
        obs: home_robot.core.interfaces.Observations,
        habitat_semantic: np.ndarray,
        vocabulary,
    ) -> home_robot.core.interfaces.Observations:
        if self.ground_truth_semantics:
            instance_id_to_category_id = (
                self.semantic_category_mapping.instance_id_to_category_id
            )
            obs.semantic = instance_id_to_category_id[habitat_semantic[:, :, -1]]
            obs.task_observations["instance_map"] = habitat_semantic[:, :, -1] + 1

            for idx_cat, obj_cat in enumerate(vocabulary):
                obj_cat = " ".join(obj_cat.split("_"))
                try:
                    if obj_cat in self.semantic_category_mapping.all_hm3d_categories:
                        idx = self.semantic_category_mapping.all_hm3d_categories.index(
                            obj_cat
                        )
                        obs.semantic[obs.semantic == idx] = -1 * (idx_cat + 1)
                except Exception as e:
                    print(e)
                    import pdb

                    pdb.set_trace()

            obs.semantic[obs.semantic >= 0] = 0
            obs.semantic = obs.semantic * -1
            # TODO Ground-truth semantic visualization
            obs.task_observations["semantic_frame"] = obs.rgb
        else:
            obs = self.segmentation.predict(obs, depth_threshold=0.5)

        obs.task_observations["semantic_frame"] = np.concatenate(
            [obs.rgb, obs.semantic[:, :, np.newaxis]], axis=2
        ).astype(np.uint8)
        return obs

    def _preprocess_depth(self, depth: np.array) -> np.array:
        rescaled_depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        rescaled_depth[depth == 0.0] = MIN_DEPTH_REPLACEMENT_VALUE
        rescaled_depth[depth == 1.0] = MAX_DEPTH_REPLACEMENT_VALUE
        return rescaled_depth[:, :, -1]

    def _preprocess_goals(self, tasks, habitat_obs):
        goals = []
        vocabulary = []
        for idx, task in enumerate(tasks):
            goal = {
                "type": task["task_type"],
            }

            if task["task_type"] == "objectnav":
                goal["target"] = task["object_category"]
            elif task["task_type"] == "languagenav":
                target = task["llm_response"]["target"]
                landmarks = task["llm_response"]["landmark"]
                if target in landmarks:
                    landmarks.remove(target)

                if "wall" in landmarks:
                    landmarks.remove("wall")  # unhelpful landmark

                target = "_".join(target.split())
                landmarks = ["_".join(landmark.split()) for landmark in landmarks]

                goal["target"] = target
                goal["landmarks"] = landmarks
                if "instructions" in task:
                    goal["instruction"] = task["instructions"][0]
            elif task["task_type"] == "imagenav":
                goal["target"] = task["object_category"]
                goal["image"] = habitat_obs["multigoal"][idx]["image"]

            if goal["target"] not in vocabulary:
                vocabulary.append(goal["target"])
            if "landmarks" in goal.keys():
                if goal["landmarks"] not in vocabulary:
                    vocabulary += goal["landmarks"]

            goal["semantic_id"] = vocabulary.index(goal["target"]) + 1
            goals.append(goal)
        return goals, vocabulary

    def _preprocess_action(self, action: home_robot.core.interfaces.Action) -> int:

        if type(action) == int:
            return action

        discrete_action = cast(
            home_robot.core.interfaces.DiscreteNavigationAction, action
        )
        return HabitatSimActions[discrete_action.name.lower()]

    def _process_info(self, info: Dict[str, Any]) -> Any:
        if info:
            if (
                self.habitat_env.current_episode.tasks[
                    self.habitat_env.task.current_task_idx
                ]["task_type"]
                != "imagenav"
            ):
                self.visualizer.visualize(**info)
