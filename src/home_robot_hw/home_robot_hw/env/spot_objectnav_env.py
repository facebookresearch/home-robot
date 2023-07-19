from typing import Any, Dict, Optional

from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot_hw.env.spot_abstract_env import SpotEnv

CATEGORIES = [
    "chair",
    "couch",
    "potted_plant",
    "bed",
    "toilet",
    "tv",
    "dining_table",
    "oven",
    "sink",
    "refrigerator",
    "book",
    "person",  # clock
    "vase",
    "cup",
    "bottle",
]

CATEGORIES_COLOR_PALETTE = [
    0.9400000000000001,
    0.7818,
    0.66,  # chair
    0.9400000000000001,
    0.8868,
    0.66,  # couch
    0.8882000000000001,
    0.9400000000000001,
    0.66,  # potted plant
    0.7832000000000001,
    0.9400000000000001,
    0.66,  # bed
    0.6782000000000001,
    0.9400000000000001,
    0.66,  # toilet
    0.66,
    0.9400000000000001,
    0.7468000000000001,  # tv
    0.66,
    0.9400000000000001,
    0.8518000000000001,  # dining-table
    0.66,
    0.9232,
    0.9400000000000001,  # oven
    0.66,
    0.8182,
    0.9400000000000001,  # sink
    0.66,
    0.7132,
    0.9400000000000001,  # refrigerator
    0.7117999999999999,
    0.66,
    0.9400000000000001,  # book
    0.8168,
    0.66,
    0.9400000000000001,  # clock
    0.9218,
    0.66,
    0.9400000000000001,  # vase
    0.9400000000000001,
    0.66,
    0.8531999999999998,  # cup
    0.9400000000000001,
    0.66,
    0.748199999999999,  # bottle
]


class SpotObjectNavEnv(SpotEnv):
    def __init__(self, spot,position_control=False):
        super().__init__(spot)
        self.goal_options = CATEGORIES
        self.color_palette = CATEGORIES_COLOR_PALETTE
        self.position_control = position_control
        categories = [
            "other",
            *self.goal_options,
            "other",
        ]
        self.num_sem_categories = len(categories) - 1
        self.segmentation = DeticPerception(
            vocabulary="custom",
            custom_vocabulary=",".join(categories),
            sem_gpu_id=0,
        )
        self._episode_over = False

    def apply_action(
        self,
        action: Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ):
        if self.position_control:
            self.env.act_point(action,blocking=True)
        else:
            self.env.step(base_action=action)
            if action == DiscreteNavigationAction.STOP:
                self._episode_over = True

    def reset(self):
        super().reset()
        self.current_goal_name = None
        self.current_goal_id = None

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    def get_episode_metrics(self) -> Dict:
        pass

    def get_observation(self) -> Observations:
        obs = super().get_observation()

        # Segment the image
        obs = self.segmentation.predict(obs, depth_threshold=None)
        obs.semantic[obs.semantic == 0] = self.num_sem_categories
        obs.semantic = obs.semantic - 1

        # Instance mapping
        obs.task_observations["instance_map"] += 1
        obs.task_observations["instance_map"] = obs.task_observations[
            "instance_map"
        ].astype(int)

        # Specify the goal
        obs.task_observations.update(
            {
                "goal_id": self.current_goal_id,
                "goal_name": self.current_goal_name,
                "object_goal": self.current_goal_id,
                "recep_goal": self.current_goal_id,
            }
        )
        return obs

    def set_goal(self, goal: str):
        if goal in self.goal_options:
            self.current_goal_name = goal
            self.current_goal_id = self.goal_options.index(goal)
        else:
            raise NotImplementedError
