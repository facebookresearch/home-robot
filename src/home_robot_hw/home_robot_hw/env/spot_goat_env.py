import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from bosdyn.client import math_helpers
from midas.model_loader import default_models, load_model
from midas.run import process

from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.perception.detection.maskrcnn.coco_categories import (
    coco_categories,
    coco_categories_color_palette,
)
from home_robot.perception.detection.maskrcnn.maskrcnn_perception import (
    MaskRCNNPerception,
)
from home_robot_hw.env.spot_abstract_env import SpotEnv


class Midas:
    def __init__(self, device):
        super().__init__()
        # midas params
        self.device = device
        self.model_type = "dpt_beit_large_512"
        self.optimize = False
        height = None
        square = False
        model_path = f"src/third_party/MiDaS/weights/{self.model_type}.pt"
        self.model, self.transform, self.net_w, self.net_h = load_model(
            device, model_path, self.model_type, self.optimize, height, square
        )

    # expects numpy rgb, [0,255]
    def depth_estimate(self, rgb, depth):
        image = self.transform({"image": (rgb / 255)})["image"]
        # compute
        with torch.no_grad():
            prediction = process(
                self.device,
                self.model,
                self.model_type,
                image,
                (self.net_w, self.net_h),
                rgb.shape[1::-1],
                self.optimize,
                False,
            )
        depth_valid = depth > 0

        # solve for MSE for the system of equations Ax = b where b is the observed depth and x is the predicted depth values
        x = np.stack(
            (prediction[depth_valid], np.ones_like(prediction[depth_valid])), axis=1
        ).T
        b = depth[depth_valid].T
        # 1 x 2 * 2 x n = 1 x n
        pinvx = np.linalg.pinv(x)
        A = b @ pinvx

        adjusted = prediction * A[0] + A[1]
        mse = ((A @ x - b) ** 2).mean()
        mean_error = np.abs(A @ x - b).mean()
        return adjusted, mse, mean_error


class SpotGoatEnv(SpotEnv):
    def __init__(
        self, spot, position_control: bool = False, estimated_depth_threshold: float = 5
    ):
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
        self.midas = Midas("cuda:0")
        self.estimated_depth_threshold = estimated_depth_threshold

    def patch_depth(self, obs):
        rgb, depth = obs.rgb, obs.depth
        monocular_estimate, mse, mean_error = self.midas.depth_estimate(rgb, depth)

        # clip at 0 if the linear transformation makes some points negative depth
        monocular_estimate[monocular_estimate < 0] = 0

        # threshold max distance to for estimated depth
        # monocular_estimate[monocular_estimate > self.estimated_depth_threshold] = 0

        try:
            # assign estimated depth where there are no values
            no_depth_mask = depth == 0

            # get a mask for the region of the image which has depth values (skip the blank borders)
            row, cols = np.where(~no_depth_mask)
            col_inds = np.indices(depth.shape)[1]
            depth_region = (col_inds >= cols.min()) & (col_inds <= cols.max())
            no_depth_mask = no_depth_mask & depth_region

            depth[no_depth_mask] = monocular_estimate[no_depth_mask]
            obs.depth = depth.copy()
            return obs
        except Exception as e:
            print(f"Initializing Midas depth completion failed: {e}")
            return obs

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
            rotation_speed = np.pi / 10
            yaw_diff = math_helpers.angle_diff(angle, yaw)
            time = max(np.abs(yaw_diff) / rotation_speed, 0.5)
            # print(angle,yaw,yaw_diff,time)
            assert time > 0
            self.env.set_arm_yaw(yaw_diff, time=time)

            action = [xg, yg, angle]
            # print("ObjectNavAgent point action", action)
            self.env.act_point(
                action,
                blocking=False,
                max_fwd_vel=0.25,
                max_ang_vel=np.pi / 10,
                max_hor_vel=0.15,
            )
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

        # fill in depth here before segmentation
        orig_depth = obs.depth.copy()
        obs = self.patch_depth(obs)

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
        obs.task_observations["orig_depth"] = orig_depth
        return obs

    def set_goals(self, goals: List[Dict]):
        for goal in goals:
            assert goal["type"] in ["objectnav", "imagenav", "languagenav"]

            assert goal["target"] in self.sem_categories
            if goal["type"] == "languagenav":
                assert all(land in self.sem_categories for land in goal["landmarks"])

            goal["semantic_id"] = self.sem_categories.index(goal["target"])

        self.goals = goals
