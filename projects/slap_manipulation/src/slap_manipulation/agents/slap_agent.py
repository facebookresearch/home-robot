from typing import Any, Dict

import numpy as np
import trimesh
from slap_manipulation.policy.action_prediction_module import ActionPredictionModule
from slap_manipulation.policy.interaction_prediction_module import (
    InteractionPredictionModule,
)
from slap_manipulation.utils.data_processing import (
    filter_and_remove_duplicate_points,
    voxelize_point_cloud,
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
        self._min_depth = self.cfg.SLAP.min_depth
        self._max_depth = self.cfg.SLAP.max_depth
        self._feat_dim = 1
        self._x_max = self.cfg.SLAP.x_max
        self._voxel_size_1 = self.cfg.SLAP.voxel_size_1
        self._voxel_size_2 = self.cfg.SLAP.voxel_size_2
        if not self._dry_run:
            # pass cfg parameters to IPM and APM
            self.interaction_prediction_module = InteractionPredictionModule()
            self.action_prediction_module = ActionPredictionModule(cfg.SLAP.APM)

    def load_models(self):
        self.interaction_prediction_module.load_state_dict(self.cfg.SLAP.IPM.path)
        self.interaction_prediction_module.to(self.device)
        self.action_prediction_module.load_state_dict(self.cfg.SLAP.APM.path)
        self.action_prediction_module.to(self.device)

    def get_proprio(self):
        if self._last_action is None:
            return np.array([2, 2, 2, 2, 2, 2, 2, -1])
        else:
            return np.concatenate((self._last_action), axis=-1)

    def get_time(self, time_as_float=False):
        if time_as_float:
            norm_time = 2 * (self._curr_keyframe - 0) / 5  # assuming max 6 waypoints
            return norm_time
        time_vector = np.zeros((1, 6))
        time_vector[0, self._curr_keyframe] = 1
        return time_vector

    def to_torch(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Implement to_torch function in SLAP please")

    def to_numpy(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Implement to_numpy function in SLAP please")

    def to_device(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Implement to_device function in SLAP please")

    def create_interaction_prediction_input_from_obs(
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
        rgb = obs.rgb.astype(np.float64) / 255.0
        xyz = obs.xyz.astype(np.float64)
        gripper = obs.joint[GRIPPER_IDX]
        camera_pose = obs.task_observations["base_camera_pose"]
        xyz = trimesh.transform_points(xyz.reshape(-1, 3), camera_pose)
        feat = obs.semantic

        # proprio looks different now
        proprio = self.get_proprio()

        depth = depth.reshape(-1)
        rgb = rgb.reshape(-1, 3)
        feat = feat.reshape(-1, self._feat_dim)

        # apply depth and z-filter for comparative distribution to training data
        if filter_depth:
            valid_depth = np.bitwise_and(
                depth > self._min_depth, depth < self._max_depth
            )
            rgb = rgb[valid_depth, :]
            xyz = xyz[valid_depth, :]
            feat = feat[valid_depth, :]
            x_mask = xyz[:, 0] < self._x_max
            rgb = rgb[x_mask]
            xyz = xyz[x_mask]
            feat = feat[x_mask]
            xyz, rgb, feat = xyz.reshape(-1, 3), rgb.reshape(-1, 3), feat.reshape(-1, 1)

        # voxelize at a granular voxel-size then choose X points
        xyz, rgb, feat = filter_and_remove_duplicate_points(
            xyz, rgb, feat, voxel_size=self._voxel_size_1, semantic_id=1
        )

        og_xyz = np.copy(xyz)
        og_rgb = np.copy(rgb)
        og_feat = np.copy(feat)

        # get 8k points for tractable learning
        downsample_mask = np.arange(rgb.shape[0])
        np.random.shuffle(downsample_mask)
        if num_pts != -1:
            downsample_mask = downsample_mask[:num_pts]
        rgb = rgb[downsample_mask]
        xyz = xyz[downsample_mask]
        feat = feat[downsample_mask]

        # mean-center the point cloud
        mean = xyz.mean(axis=0)
        xyz -= mean
        og_xyz -= mean

        if np.any(rgb > 1.0):
            rgb = rgb / 255.0
        if debug:
            print("create_action_prediction_input")
            show_point_cloud(xyz, rgb, orig=np.zeros(3))

        # voxelize rgb, xyz, and feat
        voxelized_xyz, voxelized_rgb, voxelized_feat = voxelize_point_cloud(
            xyz, rgb, feat=feat, voxel_size=self._voxel_size_2
        )

        # input_vector = (rgb, xyz, proprio, lang, mean)
        input_data = {
            "rgb": rgb,
            "xyz,": xyz,
            "feat": feat,
            "voxelized_rgb": voxelized_rgb,
            "voxelized_xyz": voxelized_xyz,
            "voxelized_feat": voxelized_feat,
            "proprio": proprio,
            "lang": obs.task_observations["task-name"],
            "mean": mean,
            "og_xyz": og_xyz,
            "og_rgb": og_rgb,
            "og_feat": og_feat,
        }
        return input_data

    def create_action_prediction_input_from_obs(
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
                self.ipm_input = self.create_interaction_prediction_input_from_obs(
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
            apm_input = self.create_action_prediction_input_from_obs(
                obs, **self.ipm_input
            )
            action = self.action_prediction_module.predict(**apm_input)
        self._curr_keyframe += 1
        return action, info

    def reset(self):
        self.interaction_point = None
        self._curr_keyframe = -1
        self._last_action = None
