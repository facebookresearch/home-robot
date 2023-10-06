import math
from typing import Any, Dict, Optional

import numpy as np
from bosdyn.client import math_helpers

from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.perception.detection.maskrcnn.coco_categories import (
    coco_categories,
    coco_categories_color_palette,
)
from home_robot.perception.detection.maskrcnn.maskrcnn_perception import (
    MaskRCNNPerception,
)
from home_robot_hw.env.spot_abstract_env import SpotEnv


class SpotObjectNavEnv(SpotEnv):
    def __init__(self, spot, position_control=False):
        super().__init__(spot)
        self.spot = spot
        self.position_control = position_control

        self.goal_options = list(coco_categories.keys())
        self.color_palette = coco_categories_color_palette
        self.num_sem_categories = len(coco_categories)
        self.segmentation = MaskRCNNPerception(
            sem_pred_prob_thr=0.9,
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
            rotation_speed = np.pi / 8
            yaw_diff = math_helpers.angle_diff(angle, yaw)
            time = max(np.abs(yaw_diff) / rotation_speed, 0.5)
            print(angle, yaw, yaw_diff, time)
            assert time > 0
            self.env.set_arm_yaw(yaw_diff, time=time)

            action = [xg, yg, angle]
            print("ObjectNavAgent point action", action)
            self.env.act_point(
                action,
                blocking=False,
                max_fwd_vel=0.5,
                max_ang_vel=np.pi / 10,
                max_hor_vel=0.4,
            )
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
        obs = self.segmentation.predict(obs, depth_threshold=0.5)
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
