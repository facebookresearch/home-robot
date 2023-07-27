# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import datetime
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import trimesh
import yaml
from slap_manipulation.policy.action_prediction_module import ActionPredictionModule
from slap_manipulation.policy.interaction_prediction_module import (
    InteractionPredictionModule,
)
from slap_manipulation.utils.pointcloud_preprocessing import (
    filter_and_remove_duplicate_points,
    get_local_action_prediction_problem,
    voxelize_point_cloud,
)

from home_robot.core.interfaces import Observations
from home_robot.utils.point_cloud import show_point_cloud
from home_robot_hw.env.stretch_abstract_env import GRIPPER_IDX


class SLAPAgent(object):
    """
    Combined E2E agent which uses the SLAP architecture
    to predict actions given a language description of a skill
    """

    def __init__(self, cfg, device: str = "cuda", task_id: int = -1):
        """Constructor for SLAPAgent, takes in configuration file and
        :task_id:. :task_id: is an int and maps to a manipulation skill.
        Each :task_id: represents a natural language action output generated
        by LLM. In the absence of this, one can use a dictionary to simulate it.
        Currently implemented by reading task, objects and num-actions described by
        configuration file :cfg:
        """
        self.task_id = task_id
        print("[SLAPAgent]: task_id = ", task_id)
        self._dry_run = cfg.SLAP.dry_run
        self.interaction_point = None
        self.cfg = cfg
        self.device = device
        self._curr_keyframe = -1
        self._last_action = None
        self._min_depth = self.cfg.SLAP.min_depth
        self._max_depth = self.cfg.SLAP.max_depth
        self._feat_dim = 1
        self._z_min = self.cfg.SLAP.z_min
        self._voxel_size_1 = self.cfg.SLAP.voxel_size_1
        self._voxel_size_2 = self.cfg.SLAP.voxel_size_2
        if cfg.SLAP.APM.skill_to_action_file is not None:
            self.skill_to_action = yaml.load(
                open(cfg.SLAP.APM.skill_to_action_file, "r"),
                Loader=yaml.FullLoader,
            )
        else:
            self.skill_to_action = None

    def get_goal_info(self) -> Dict[str, Any]:
        """returns goal information for the task from cfg
        information includes: task-name, object-list, num-actions
        """
        info = {}
        info["task-name"] = self.cfg.EVAL.task_name[self.task_id]
        info["object_list"] = self.cfg.EVAL.object_list[self.task_id]
        info["num-actions"] = self.cfg.EVAL.num_keypoints[self.task_id]
        return info

    def load_models(self):
        """loads weights for IPM and APM"""
        self.interaction_prediction_module = InteractionPredictionModule()
        self.action_prediction_module = ActionPredictionModule(self.cfg.SLAP.APM)
        self.interaction_prediction_module.load_weights(self.cfg.SLAP.IPM.path)
        self.interaction_prediction_module.to(self.device)
        self.action_prediction_module.load_weights(self.cfg.SLAP.APM.path)
        self.action_prediction_module.to(self.device)
        print("Loaded SLAP weights")

    def get_proprio(self):
        """initialzie proprio for LSTM"""
        return np.array([2, 2, 2, 2, 2, 2, 2, -1])

    def get_time(self, time_as_float: bool = False):
        """Returns the time element. Return 1x6 vector if time_as_float is False, else return float"""
        if time_as_float:
            norm_time = 2 * (self._curr_keyframe - 0) / 5  # assuming max 6 waypoints
            return norm_time
        time_vector = np.zeros((1, 6))
        time_vector[0, self._curr_keyframe] = 1
        return time_vector

    def to_torch(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """converts numpy arrays in input_dict to torch tensors"""
        for k, v in input_dict.items():
            if isinstance(v, np.ndarray):
                input_dict[k] = torch.from_numpy(v).float().to(self.device)
        return input_dict

    def to_numpy(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Implement to_numpy function in SLAP please")

    def to_device(
        self, input_dict: Dict[str, Any], device: str = "cuda"
    ) -> Dict[str, Any]:
        """converts tensors in input_dict to device"""
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                input_dict[k] = v.to(device)
        return input_dict

    def create_interaction_prediction_input_from_obs(
        self,
        obs: Observations,
        filter_depth: bool = False,
        num_pts: int = 8000,
        debug: bool = False,
        semantic_id: int = 1,
        zero_mean_norm: bool = False,
    ) -> Dict[str, Any]:
        """method to convert obs into input expected by IPM
        expects obs to have semantic-features, and ID of relevant
        object to be semantic_id in the mask
            :obs: Observations object from home-robot environment
            :filter_depth: if True, only keep points with depth in range [self._min_depth, self._max_depth]
            :num_pts: number of points to sample from original point-cloud
            :debug: if True, plot point-cloud and mask
            :semantic_id: ID of object to be segmented
            :zero_mean_norm: if True, normalize rgb to [-1, 1]
        """
        depth = obs.depth
        if zero_mean_norm:
            rgb = (obs.rgb.astype(np.float64) / 255.0) * 2.0 - 1.0
        else:
            rgb = obs.rgb.astype(np.float64) / 255.0
        xyz = obs.xyz.astype(np.float64)
        gripper = obs.joint[GRIPPER_IDX]
        feat = obs.semantic

        # only keep feat which is == semantic_id
        if feat is not None:
            # feat[feat >= (len(obs.task_observations["object_list"]) + 1)] = 0
            feat[feat != semantic_id] = 0

        # proprio looks different now
        proprio = self.get_proprio()

        depth = depth.reshape(-1)
        rgb = rgb.reshape(-1, 3)
        if feat is not None:
            feat = feat.reshape(-1, self._feat_dim)

        # apply depth and z-filter for comparative distribution to training data
        if filter_depth:
            valid_depth = np.bitwise_and(
                depth > self._min_depth, depth < self._max_depth
            )
            rgb = rgb[valid_depth, :]
            xyz = xyz[valid_depth, :]
            if feat is not None:
                feat = feat[valid_depth, :]
            z_mask = xyz[:, 2] > self._z_min
            rgb = rgb[z_mask]
            xyz = xyz[z_mask]
            if feat is not None:
                feat = feat[z_mask]
            xyz, rgb = xyz.reshape(-1, 3), rgb.reshape(-1, 3)
            if feat is not None:
                feat = feat.reshape(-1, 1)

        # voxelize at a granular voxel-size then choose X points
        xyz, rgb, feat = filter_and_remove_duplicate_points(
            xyz,
            rgb,
            feat,
            voxel_size=self._voxel_size_1,
        )

        og_xyz = np.copy(xyz)
        og_rgb = np.copy(rgb)
        og_feat = None
        if feat is not None:
            og_feat = np.copy(feat)

        # get 8k points for tractable learning
        downsample_mask = np.arange(rgb.shape[0])
        np.random.shuffle(downsample_mask)
        if num_pts != -1:
            downsample_mask = downsample_mask[:num_pts]
        rgb = rgb[downsample_mask]
        xyz = xyz[downsample_mask]
        if feat is not None:
            feat = feat[downsample_mask]

        # mean-center the point cloud
        xyz, og_xyz, mean = self.mean_center([xyz, og_xyz])

        if np.any(rgb > 1.0):
            raise RuntimeWarning(
                f"rgb values should be in range [0, 1] or [-1, 1]: got {rgb.min()} to {rgb.max()}"
            )
            rgb = rgb / 255.0
        if debug:
            print("create_interaction_prediction_input")
            show_point_cloud(xyz, rgb, orig=np.zeros(3))

        # voxelize rgb, xyz, and feat
        voxelized_xyz, voxelized_rgb, voxelized_feat = voxelize_point_cloud(
            xyz,
            rgb,
            feat=feat,
            voxel_size=self._voxel_size_2,
            debug_voxelization=debug,
        )

        input_data = {
            "rgb": rgb,
            "xyz": xyz,
            "feat": feat,
            "rgb_voxelized": voxelized_rgb,
            "xyz_voxelized": voxelized_xyz,
            "feat_voxelized": voxelized_feat,
            "proprio": proprio,
            "lang": obs.task_observations["task-name"],
            "mean": mean,
            "og_xyz": og_xyz,
            "og_rgb": og_rgb,
            "gripper-width": gripper,
            "og_feat": og_feat,
            "gripper-state": obs.task_observations["gripper-state"],
        }
        return self.to_torch(input_data)

    def mean_center(self, xyzs: List[np.ndarray]) -> List[np.ndarray]:
        """mean centers a list of point clouds based on the mean of 1st entry in the list"""
        mean = xyzs[0].mean(axis=0)
        return [xyz - mean for xyz in xyzs] + [mean]

    def create_action_prediction_input_from_obs(
        self,
        obs: Observations,
        ipm_data: Dict[str, Any],
        debug=False,
    ) -> Dict[str, Any]:
        """takes p_i prediction and open-loop input state converting into input
        batch for Action Prediction Module by cropping around predicted p_i:
        interaction_point"""
        xyz = ipm_data["og_xyz"]
        rgb = ipm_data["og_rgb"]
        if torch.any(rgb > 1.0):
            raise RuntimeWarning(
                f"rgb values should be in range [0, 1] or [-1, 1]: got {rgb.min()} to {rgb.max()}"
            )
        feat = ipm_data["og_feat"]
        combined_feat = torch.cat([rgb, feat], dim=-1)
        (cropped_feat, cropped_xyz, status,) = get_local_action_prediction_problem(
            self.cfg.SLAP.APM,
            combined_feat.detach().cpu().numpy(),
            xyz.detach().cpu().numpy(),
            self.interaction_point.detach().cpu().numpy(),
        )
        if not status:
            raise RuntimeError(
                "Interaction Prediction Module predicted an interaction point with no tractable local problem around it"
            )
        if debug:
            print("create_action_prediction_input")
            show_point_cloud(cropped_xyz, cropped_feat)
        all_cmd = self.skill_to_action[ipm_data["lang"]]
        input_data = self.to_torch(
            {
                "feat_crop": cropped_feat,
                "xyz_crop": cropped_xyz,
                "rgb_crop": None,
                "num-actions": obs.task_observations["num-actions"],
                "all_cmd": all_cmd,
            }
        )
        ipm_data.update(input_data)
        return ipm_data

    def predict(
        self,
        obs: Observations,
        ipm_only: bool = False,
        visualize: bool = False,
        save_logs: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Main method for SLAP, predicts interaction point and action given observation from GeneralLanguageEnv
        :ipm_only: if True, will only run interaction prediction module
        :visualize: if True, will visualize the interaction point and action prediction
        :save_logs: if True, will save logs for interaction point and action prediction
        """
        info = {}
        action = None
        if self.interaction_point is None:
            if not self._dry_run:
                self.ipm_input = self.create_interaction_prediction_input_from_obs(
                    obs, filter_depth=True, debug=False
                )
                result = self.interaction_prediction_module.predict(
                    self.ipm_input, debug=True
                )
                self.interaction_point = result[0]
                info["interaction_point"] = (
                    (self.interaction_point + self.ipm_input["mean"])
                    .detach()
                    .cpu()
                    .numpy()
                )
                scores = result[2]

                # get top 5% points from PCD
                (
                    top_xyz,
                    top_rgb,
                ) = self.interaction_prediction_module.get_top_attention(
                    self.ipm_input["xyz_voxelized"],
                    self.ipm_input["rgb_voxelized"],
                    scores,
                    threshold=10,
                    visualize=visualize,
                )
                if save_logs:
                    self.save_ipm_logs(info, obs, top_xyz, top_rgb)
            else:
                print("[SLAP] Predicting interaction point")
                self.interaction_point = np.random.rand(3)
        if self._dry_run:
            print(f"[SLAP] Predicting keyframe # {self._curr_keyframe}")
        elif not ipm_only:
            apm_input = self.create_action_prediction_input_from_obs(
                obs, self.ipm_input
            )
            action = self.action_prediction_module.predict(apm_input, debug=True)
            for i, act in enumerate(action):
                action[i] = act.detach().cpu().numpy()
            action = np.array(action)
            action[:, :3] += (
                (self.interaction_point + self.ipm_input["mean"])
                .detach()
                .cpu()
                .numpy()
                .reshape(1, 3)
            )
            if save_logs:
                self.save_apm_logs(obs, action)
            print(f"[SLAP] Predicted action: {action}")
        return action, info

    def save_ipm_logs(
        self,
        info: Dict[str, Any],
        obs: Observations,
        top_xyz: np.ndarray,
        top_rgb: np.ndarray,
    ):
        """saves logs for interaction point prediction; including xyz and mask for top 5% attention-scores
        :info: dictionary to store logs in
        :obs: observation from GeneralLanguageEnv
        :top_xyz: all xyz from PCD
        :top_rgb: all rgb from PCD with top 5% attention-scores colored red
        """
        dt = datetime.datetime.now().strftime("%d_%m_%H:%M:%S")
        info["top_xyz"] = top_xyz + self.ipm_input["mean"].detach().cpu().numpy()
        info["top_rgb"] = top_rgb
        filename = os.path.join(
            os.getcwd(),
            str(self.task_id)
            + f"_ipm_{dt}_"
            + obs.task_observations["task-name"]
            + "_".join(obs.task_observations["object_list"])
            + ".npz",
        )
        np.savez(
            filename,
            top_xyz=info["top_xyz"],
            top_rgb=info["top_rgb"],
            semantic_mask=obs.task_observations["semantic_frame"],
        )

    def save_apm_logs(self, obs: Observations, action: List[np.ndarray]):
        dt = datetime.datetime.now().strftime("%d_%m_%H:%M:%S")
        filename = os.path.join(
            os.getcwd(),
            str(self.task_id)
            + f"_apm_{dt}_"
            + obs.task_observations["task-name"]
            + "_".join(obs.task_observations["object_list"])
            + ".npz",
        )
        np.savez(filename, pred_action=action)

    def reset(self):
        self.interaction_point = None
        self._curr_keyframe = -1
        self._last_action = None
