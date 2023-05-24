from typing import Any, Dict

import numpy as np
from slap_manipulation.policy.action_prediction_module import ActionPredictionModule
from slap_manipulation.policy.interaction_prediction_module import (
    InteractionPredictionModule,
)
from slap_manipulation.utils.input_preprocessing import (
    get_local_action_prediction_problem,
)

from home_robot.utils.point_cloud import show_point_cloud
from home_robot_hw.env.stretch_abstract_env import GRIPPER_IDX


class SlapAgent(object):
    """
    Combined E2E agent which uses the SLAP architecture
    to predict actions given a language instruction
    """

    def __init__(self, cfg, device="cuda"):
        # pass cfg parameters to IPM and APM
        self.interaction_prediction_module = InteractionPredictionModule()
        self.action_prediction_module = ActionPredictionModule(cfg)
        self.interaction_point = None
        self.cfg = cfg
        self.device = device

    def load_models(self):
        self.interaction_prediction_module.load_state_dict(self.cfg.ipm_path)
        self.interaction_prediction_module.to(self.device)
        self.action_prediction_module.load_state_dict(self.cfg.apm_path)
        self.action_prediction_module.to(self.device)

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
        proprio = self.get_proprio(raw_data, time=-1.0)

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
        cfg,
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
            cfg, feat, xyz, self.interaction_point
        )
        if np.any(cropped_feat > 1.0):
            cropped_feat = cropped_feat / 255.0
        if not status:
            raise RuntimeError(
                "Interaction Prediction Module predicted an interaction point with no tractable local problem around it"
            )
        proprio = get_proprio(raw_data, time=time)
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
        if self.interaction_point is None:
            self.ipm_input = self.create_interaction_prediction_input(
                obs, filter_depth=True
            )
            result = self.interaction_prediction_module.predict(**self.ipm_input)
            self.interaction_point = result[0]
            return True
        else:
            apm_input = self.create_action_prediction_input(obs, **self.ipm_input)
            action = self.action_prediction_module.predict(**apm_input)
        return action

    def reset(self):
        self.interaction_point = None
