from typing import Any, Dict

import numpy as np
from slap_manipulation.policy.action_prediction_module import ActionPredictionModule
from slap_manipulation.policy.interaction_prediction_module import (
    InteractionPredictionModule,
)
from slap_manipulation.utils.pointcloud_preprocessing import (
    get_local_action_prediction_problem,
)

from home_robot.utils.point_cloud import show_point_cloud
from home_robot_hw.env.stretch_abstract_env import GRIPPER_IDX


class SLAPAgent(object):
    """
    Combined E2E agent which uses the SLAP architecture
    to predict actions given a language instruction
    """

    def __init__(self, cfg, device="cuda"):
        self._dry_run = cfg.SLAP.dry_run
        self.interaction_point = None
        self.cfg = cfg
        self.device = device
        self._curr_keyframe = -1
        self._last_action = None
        if not self._dry_run:
            # pass cfg parameters to IPM and APM
            self.interaction_prediction_module = InteractionPredictionModule()
            self.action_prediction_module = ActionPredictionModule(cfg)

    def load_models(self):
        self.interaction_prediction_module.load_state_dict(self.cfg.SLAP.ipm_path)
        self.interaction_prediction_module.to(self.device)
        self.action_prediction_module.load_state_dict(self.cfg.SLAP.apm_path)
        self.action_prediction_module.to(self.device)

    def get_proprio(self, time):
        if self._last_action is None:
            return np.array([2, 2, 2, 2, 2, 2, 2, -1])
        else:
            return np.concatenate((self._last_action, np.array([time])), axis=-1)

    def get_time(self, obs):
        norm_time = (
            2 * (self._curr_keyframe - 0) / obs.task_observations["num-keyframes"] - 1
        )
        return norm_time

    def to_torch(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Implement to_torch function in SLAP please")

    def to_numpy(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Implement to_numpy function in SLAP please")

    def to_device(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Implement to_device function in SLAP please")

    def create_interaction_prediction_input(
        self, obs, filter_depth=False, num_pts=8000, debug=False
    ) -> Dict[str, Any]:
        """method to convert obs into input expected by IPM
        takes raw data from stretch_manipulation_env, and language command from user.
        Converts it into input batch used in Interaction Prediction Module
        Return: obs_vector = ((rgb), xyz, proprio, lang)
            obs_vector[0]: tuple of features per point in PCD; each element is expected to be Nxfeat_dim
            obs_vector[1]: xyz coordinates of PCD points; N x 3
            obs_vector[2]: proprioceptive state of robot; 3-dim vector: [gripper-state, gripper-width, time] # probably do not need time for IPM training
            obs_vector[3]: language command; list of 1 string # should this be a list? only 1 string.
        """
        depth = obs.depth
        rgb = obs.rgb.astype(np.float64)
        xyz = obs.xyz.astype(np.float64)
        gripper = obs.joint[GRIPPER_IDX]

        # proprio looks different now
        # time depends on how many keyframes are there in the task
        time = self.get_time(obs)
        proprio = self.get_proprio(time)

        depth = depth.reshape(-1)
        rgb = rgb.reshape(-1, 3)
        # apply depth and z-filter for comparative distribution to training data
        if filter_depth:
            valid_depth = np.bitwise_and(depth > 0.1, depth < 1.0)
            rgb = rgb[valid_depth, :]
            xyz = xyz[valid_depth, :]
            z_mask = xyz[:, 2] > 0.7
            rgb = rgb[z_mask, :]
            xyz = xyz[z_mask, :]
        og_xyz = np.copy(xyz)
        og_rgb = np.copy(rgb)
        # get 8k points for tractable learning
        downsample_mask = np.arange(rgb.shape[0])
        np.random.shuffle(downsample_mask)
        if num_pts != -1:
            downsample_mask = downsample_mask[:num_pts]
        rgb = rgb[downsample_mask]
        xyz = xyz[downsample_mask]

        # mean-center the point cloud
        mean = xyz.mean(axis=0)
        xyz -= mean
        og_xyz -= mean

        if np.any(rgb > 1.0):
            rgb = rgb / 255.0
        if debug:
            print("create_action_prediction_input")
            show_point_cloud(xyz, rgb, orig=np.zeros(3))

        # input_vector = (rgb, xyz, proprio, lang, mean)
        input_data = {
            "rgb": rgb,
            "xyz,": xyz,
            "proprio": proprio,
            "lang": lang,
            "mean": mean,
            "og_xyz": og_xyz,
            "og_rgb": og_rgb,
        }
        return input_data

    def create_action_prediction_input(
        self,
        raw_data: Dict[str, Any],
        feat: np.ndarray,
        xyz: np.ndarray,
        p_i: np.ndarray,
        time: float = 0.0,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """takes p_i prediction, current gripper-state and open-loop input state
        converting into input batch for Action Prediction Module by cropping
        around predicted p_i: interaction_point"""
        cropped_feat, cropped_xyz, status = get_local_action_prediction_problem(
            self.cfg, feat, xyz, self.interaction_point
        )
        if np.any(cropped_feat > 1.0):
            cropped_feat = cropped_feat / 255.0
        if not status:
            raise RuntimeError(
                "Interaction Prediction Module predicted an interaction point with no tractable local problem around it"
            )
        proprio = self.get_proprio(raw_data, time=time)
        if debug:
            print("create_action_prediction_input")
            show_point_cloud(cropped_xyz, cropped_feat)
        input_data = {
            "cropped_feat": cropped_feat,
            "cropped_xyz": cropped_xyz,
            "proprio": proprio,
        }
        return input_data
        # return (cropped_feat, cropped_xyz, proprio)

    def predict(self, obs):
        info = {}
        action = None
        if self.interaction_point is None:
            if not self._dry_run:
                self.ipm_input = self.create_interaction_prediction_input(
                    obs, filter_depth=True
                )
                result = self.interaction_prediction_module.predict(**self.ipm_input)
                self.interaction_point = result[0]
            else:
                print("[SLAP] Predicting interaction point")
                self.interaction_point = np.random.rand(3)
            self._curr_keyframe = 0
        if self._dry_run:
            print(f"[SLAP] Predicting keyframe # {self._curr_keyframe}")
        else:
            apm_input = self.create_action_prediction_input(obs, **self.ipm_input)
            action = self.action_prediction_module.predict(**apm_input)
        self._curr_keyframe += 1
        return action, info

    def reset(self):
        self.interaction_point = None
        self._curr_keyframe = -1
        self._last_action = None
