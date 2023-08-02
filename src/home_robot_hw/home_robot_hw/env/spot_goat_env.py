import math
from typing import Any, Dict, List, Optional
from bosdyn.client import math_helpers

import numpy as np

from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.perception.detection.maskrcnn.coco_categories import (
    coco_categories,
    coco_categories_color_palette,
)
from home_robot.perception.detection.maskrcnn.maskrcnn_perception import (
    MaskRCNNPerception,
)
from home_robot_hw.env.spot_abstract_env import SpotEnv


class SpotGoatEnv(SpotEnv):
    def __init__(self, spot, position_control=False):
        super().__init__(spot)
        self.spot = spot
        self.position_control = position_control

        self.sem_categories = list(coco_categories.keys())
        self.color_palette = coco_categories_color_palette
        self.num_sem_categories = len(coco_categories)
        self.segmentation = MaskRCNNPerception(
            sem_pred_prob_thr=0.8,
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
        if self.position_control:
            x, y, _ = action.xyt

            # angle from the origin to the STG
            angle_st_goal = math.atan2(x, y)
            dist = np.linalg.norm((x, y)) * 0.05
            xg = dist * np.cos(angle_st_goal + self.start_compass) + self.start_gps[0]
            yg = dist * np.sin(angle_st_goal + self.start_compass) + self.start_gps[1]

            # compute the angle from the current pose to the destination point
            # in robot global frame
            cx, cy, yaw = self.spot.get_xy_yaw()
            angle = math.atan2((yg - cy), (xg - cx)) % (2 * np.pi)
            rotation_speed = np.pi/8
            yaw_diff = math_helpers.angle_diff(angle,yaw)
            time=max(np.abs(yaw_diff)/rotation_speed,0.5)
            print(angle,yaw,yaw_diff,time)
            assert time > 0
            self.env.set_arm_yaw(yaw_diff,time=time)

            action = [xg, yg, angle]
            print("ObjectNavAgent point action", action)
            self.env.act_point(action, blocking=False)
        else:
            self.env.step(base_action=action)
            if action == DiscreteNavigationAction.STOP:
                self._episode_over = True

    def reset(self):
        super().reset()
        self.goals = None

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    def get_episode_metrics(self) -> Dict:
        pass

    def get_observation(self) -> Observations:
        obs = super().get_observation()

        # Segment the image
        obs = self.segmentation.predict(obs, depth_threshold=0.5)
        obs.semantic[obs.semantic == 0] = self.num_sem_categories
        obs.semantic = obs.semantic - 1

        # Instance mapping
        obs.task_observations["instance_map"] += 1
        obs.task_observations["instance_map"] = obs.task_observations[
            "instance_map"
        ].astype(int)

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
