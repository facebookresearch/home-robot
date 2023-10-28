from typing import Any, Dict, List, Optional, Tuple, Union, cast

import habitat
import json
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
from home_robot.perception.detection.maskrcnn.maskrcnn_perception import (
    MaskRCNNPerception,
)

from home_robot.perception.constants import df as hm3d_mapping_df

all_ovon_categories_path = "/srv/flash1/rramrakhya3/fall_2023/goat/data/hm3d_meta/ovon_categories_final_split.json"
with open(all_ovon_categories_path, "r") as f:
    all_ovon_categories = json.load(f)

# all_ovon_categories = [y for x in all_ovon_categories.values() for y in x if type(y) == str]
all_ovon_categories = sorted(list(set(all_ovon_categories["val_seen"])))

all_ovon_categories = ["_".join(x.split(" ")) for x in all_ovon_categories]

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

        ovon_semantic_ids = []

        self.hm3d_mapping = {}

        for obj in self.habitat_env.sim.semantic_scene.objects:
            main_category = hm3d_mapping_df[hm3d_mapping_df['raw_category'] == obj.category.name()]
            
            # raw -> main category
            if len(main_category) == 0:
                continue
            else:
                main_category = main_category['category'].item()

            main_category = "_".join(main_category.split(" "))

            if main_category in all_ovon_categories:
                self.hm3d_mapping[int(obj.id.split('_')[-1])] = all_ovon_categories.index(main_category) + 1


        # for cat in hm3d_mapping_df['category'].tolist():
        #     if cat in all_ovon_categories:
        #         ovon_semantic_ids.append(all_ovon_categories.index(cat) + 1)
        #     else:
        #         ovon_semantic_ids.append(0)

        # hm3d_mapping_df['ovon_semantic_ids'] = ovon_semantic_ids

        # self.hm3d_mapping = hm3d_mapping_df.set_index('index')['ovon_semantic_ids'].to_dict()

        # self.segmentation = MaskRCNNPerception(
        #     sem_pred_prob_thr=0.9,
        #     sem_gpu_id=(-1 if self.config.NO_GPU else self.habitat_env.sim.gpu_device),
        # )

        if not self.ground_truth_semantics:
            self.init_perception_module()

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
        # vocabulary = self.fetch_vocabulary(goals)
        # print("Vocabulary:", vocabulary)
        if not self.ground_truth_semantics:
            self.init_perception_module()

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

    def init_perception_module(self, vocabulary=None):
        from home_robot.perception.detection.detic.detic_perception import (
            DeticPerception,
        )
        
        all_ovon_categories_path = "/srv/flash1/rramrakhya3/fall_2023/goat/data/hm3d_meta/ovon_categories_final_split.json"
        with open(all_ovon_categories_path, "r") as f:
            all_ovon_categories = json.load(f)

        # all_ovon_categories = [y for x in all_ovon_categories.values() for y in x if type(y) == str]
        all_ovon_categories = sorted(list(set(all_ovon_categories["val_seen"])))

        all_ovon_categories = ["_".join(x.split(" ")) for x in all_ovon_categories]

        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary="," + ",".join(all_ovon_categories),
            sem_gpu_id=(-1 if self.config.NO_GPU else self.habitat_env.sim.gpu_device),
        )

        # self.segmentation = MaskRCNNPerception(
        #     sem_pred_prob_thr=0.9,
        #     sem_gpu_id=(-1 if self.config.NO_GPU else self.habitat_env.sim.gpu_device),
        # )

        # from home_robot.perception.detection.grounded_sam.ram_perception import RAMPerception

        # self.segmentation = RAMPerception(
        #     custom_vocabulary=".",
        #     sem_gpu_id=(-1 if self.config.NO_GPU else self.habitat_env.sim.gpu_device),
        #     verbose=False,
        #     # **module_kwargs
        # )


    def _preprocess_obs(
        self, habitat_obs: habitat.core.simulator.Observations
    ) -> home_robot.core.interfaces.Observations:
        depth = self._preprocess_depth(habitat_obs["depth"])
        goals = self._preprocess_goals(
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
        obs = self._preprocess_semantic(obs, habitat_obs["semantic"])
        return obs

    def _preprocess_semantic(
        self,
        obs: home_robot.core.interfaces.Observations,
        habitat_semantic: np.ndarray,
        vocabulary=None,
    ) -> home_robot.core.interfaces.Observations:
        if self.ground_truth_semantics:

            obs.semantic = np.vectorize(lambda x: self.hm3d_mapping.get(x, 0))(habitat_semantic)[..., 0]
            obs.task_observations["instance_map"] = habitat_semantic[:, :, -1] + 1

            # import pdb;pdb.set_trace()
            # instance_id_to_category_id = (
            #     self.semantic_category_mapping.instance_id_to_category_id
            # )
            # obs.semantic = instance_id_to_category_id[habitat_semantic[:, :, -1]]
            # obs.task_observations["instance_map"] = habitat_semantic[:, :, -1] + 1

            # for idx_cat, obj_cat in enumerate(vocabulary):
            #     obj_cat = " ".join(obj_cat.split("_"))
            #     try:
            #         if obj_cat in self.semantic_category_mapping.all_hm3d_categories:
            #             idx = self.semantic_category_mapping.all_hm3d_categories.index(
            #                 obj_cat
            #             )
            #             obs.semantic[obs.semantic == idx] = -1 * (idx_cat + 1)
            #         else:

            #             all_categories = [x for x in self.semantic_category_mapping.all_hm3d_categories if type(x) == str]
            #             useful_categories = [x for x in all_categories if x in obj_cat or obj_cat in x]
            #             # print(obj_cat, useful_categories)
            #             for cat in useful_categories:
            #                 idx = self.semantic_category_mapping.all_hm3d_categories.index(cat)
            #                 obs.semantic[obs.semantic == idx] = -1 * (idx_cat + 1)
            #             # print("Object category not found:", obj_cat)
            #             # import pdb;pdb.set_trace()
            #     except Exception as e:
            #         print(e)
            #         import pdb

            #         pdb.set_trace()

            # obs.semantic[obs.semantic >= 0] = 0
            # obs.semantic = obs.semantic * -1
            # TODO Ground-truth semantic visualization
            obs.task_observations["semantic_frame"] = obs.rgb
        else:
            obs = self.segmentation.predict(obs)

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

        goals = habitat_obs['multigoal']

        for goal_v in goals:
            goal_v["semantic_id"] = all_ovon_categories.index("_".join(goal_v["category"].split(" "))) + 1
            if goal_v["image"] is not None:
                goal_v["type"] = "imagenav"
            elif goal_v["description"] :
                goal_v["type"] = "languagenav"
            else:
                goal_v["type"] = "objectnav"

        return goals

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
                ][1]
                != "image"
            ):
                info["top_down_map"] = self.get_observation().task_observations.get("top_down_map")
                self.visualizer.visualize(**info)
