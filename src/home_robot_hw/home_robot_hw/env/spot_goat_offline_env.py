import math
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

from home_robot.core.abstract_env import Env
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.perception.detection.maskrcnn.coco_categories import (
    coco_categories,
    coco_categories_color_palette,
)
from home_robot.perception.detection.maskrcnn.maskrcnn_perception import (
    MaskRCNNPerception,
)


class SpotGoatOfflineEnv(Env):
    def __init__(self, obs_dir: str):
        self.obs_dir = obs_dir

        self.sem_categories = list(coco_categories.keys())
        self.color_palette = coco_categories_color_palette
        self.num_sem_categories = len(coco_categories)
        self.segmentation = MaskRCNNPerception(
            sem_pred_prob_thr=0.9,
            sem_gpu_id=0,
        )

        self.goals = None
        self._episode_over = False

    def apply_action(
        self,
        action: Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ):
        pass

    def reset(self):
        super().reset()
        self.goals = None

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    def get_episode_metrics(self) -> Dict:
        pass

    def get_observation(self, t: int) -> Observations:
        with open(f"{self.obs_dir}/{t}.pkl", "rb") as f:
            obs = pickle.load(f)

        # Segment the image (can skip if already done)
        # obs = self.segmentation.predict(obs, depth_threshold=0.5)
        # obs.semantic[obs.semantic == 0] = self.num_sem_categories
        # obs.semantic = obs.semantic - 1

        # Instance mapping (can skip if already done)
        # obs.task_observations["instance_map"] += 1
        # obs.task_observations["instance_map"] = obs.task_observations[
        #     "instance_map"
        # ].astype(int)

        # Specify the goals
        obs.task_observations["tasks"] = self.goals

        return obs

    def set_goals(self, goals: List[Dict]):
        for goal in goals:
            assert goal["type"] in ["objectnav", "imagenav", "languagenav"]

            assert goal["target"] in self.sem_categories
            if goal["type"] == "languagenav":
                assert all(land in self.sem_categories for land in goal["landmarks"])

            goal["semantic_id"] = self.sem_categories.index(goal["target"])

        self.goals = goals
