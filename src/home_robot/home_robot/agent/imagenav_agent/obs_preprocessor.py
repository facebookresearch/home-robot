# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Tuple

import cv2
import numpy as np
import skimage.morphology
import torch
from numpy import ndarray
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm

import home_robot.utils.pose as pu
from home_robot.agent.goat_agent.utils.agent_utils import get_matches_against_memory
from home_robot.core.interfaces import Observations
from home_robot.perception.detection.detic.detic_mask import Detic

from .superglue import Matching


class ObsPreprocessor:
    """Preprocess raw home-Robot observations for consumption by an ImageNav agent."""

    def __init__(self, config: DictConfig, device: torch.device) -> None:
        self.device = device
        self.frame_height = config.ENVIRONMENT.frame_height
        self.frame_width = config.ENVIRONMENT.frame_width

        self.depth_filtering = config.AGENT.SEMANTIC_MAP.depth_filtering
        self.depth_filter_range_cm = config.AGENT.SEMANTIC_MAP.depth_filter_range_cm
        self.preprojection_kp_dilation = (
            config.AGENT.SEMANTIC_MAP.preprojection_kp_dilation
        )
        self.match_projection_threshold = (
            config.AGENT.SUPERGLUE.match_projection_threshold
        )

        self.store_all_categories_in_map = getattr(
            config.AGENT, "store_all_categories", False
        )
        self.record_instance_ids = getattr(
            config.AGENT.SEMANTIC_MAP, "record_instance_ids", False
        )

        #  initialize detection and localization modules
        self.matching = Matching(
            device=0,  # config.simulator_gpu_id
            config=config.AGENT.SUPERGLUE,
            default_vis_dir=f"{config.DUMP_LOCATION}/images/{config.EXP_NAME}",
            print_images=config.PRINT_IMAGES,
        )
        self.instance_seg = Detic(config.AGENT.DETIC)
        self.one_hot_encoding = torch.eye(
            config.AGENT.SEMANTIC_MAP.num_sem_categories, device=self.device
        )
        # init episode variables
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self.last_pose = None
        self.step = None

    def reset(self) -> None:
        """Reset for a new episode since pre-processing is temporally dependent."""
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self.last_pose = np.zeros(3)
        self.step = 0

    def reset_sub_episode(self) -> None:
        """Reset for a new sub-episode since pre-processing is temporally dependent."""
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self.step = 0

    def preprocess(
        self,
        obs: Observations,
        last_pose: Optional[ndarray] = None,
        instance_memory=None,
    ) -> Tuple[Tensor, Optional[Tensor], ndarray, ndarray]:
        """
        Preprocess observations of a single timestep batched across
        environments.

        Arguments:
            obs: list of observations of length num_environments

        Returns:
            obs_preprocessed: frame containing (RGB, depth, keypoint_loc) of
               shape (3 + 1 + 1, frame_height, frame_width)
            pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
               of shape (num_environments, 3)
            matches: keypoint correspondences from goal image to egocentric
               image of shape (1, n)
            confidence: keypoint correspondence confidence of shape (1, n)
            camera_pose: camera extrinsic pose of shape (num_environments, 4, 4)
        """
        if last_pose is not None:
            self.last_pose = last_pose
        if self.goal_image is None:
            img_goal = obs.task_observations["tasks"][self.current_task_idx]["image"]
            (
                self.goal_image,
                self.goal_image_keypoints,
            ) = self.matching.get_goal_image_keypoints(img_goal)
            self.goal_mask, _ = self.instance_seg.get_goal_mask(img_goal)

        pose_delta, self.last_pose = self._preprocess_pose_and_delta(obs)
        (
            obs_preprocessed,
            matches,
            confidence,
            all_matches,
            all_confidences,
        ) = self._preprocess_frame(obs, instance_memory)

        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)

        self.step += 1
        return (
            obs_preprocessed,
            self.goal_image,
            pose_delta,
            camera_pose,
            matches,
            confidence,
            all_matches,
            all_confidences,
        )

    def _preprocess_frame(
        self, obs: Observations, instance_memory
    ) -> Tuple[Tensor, ndarray, ndarray]:
        """Preprocess frame information in the observation."""

        def downscale(rgb: ndarray, depth: ndarray) -> Tuple[ndarray, ndarray]:
            """downscale RGB and depth frames to self.frame_{width,height}"""
            ds = rgb.shape[1] / self.frame_width
            if ds == 1:
                return rgb, depth
            dim = (self.frame_width, self.frame_height)
            rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, dim, interpolation=cv2.INTER_NEAREST)[:, :, None]
            return rgb, depth

        def preprocess_keypoint_localization(
            rgb: ndarray,
            goal_keypoints: torch.Tensor,
            rgb_keypoints: torch.Tensor,
            matches: ndarray,
            confidence: ndarray,
        ) -> ndarray:
            """
            Given keypoint correspondences, determine the egocentric pixel coordinates
            of matched keypoints that lie within a mask of the goal object.
            """
            goal_keypoints = goal_keypoints[0].cpu().to(dtype=int).numpy()
            rgb_keypoints = rgb_keypoints[0].cpu().to(dtype=int).numpy()
            confidence = confidence[0]
            matches = matches[0]

            # map the valid goal keypoints to ego keypoints
            is_in_mask = self.goal_mask[goal_keypoints[:, 1], goal_keypoints[:, 0]]
            has_high_confidence = confidence >= self.match_projection_threshold
            is_matching_kp = matches > -1
            valid = np.logical_and(is_in_mask, has_high_confidence, is_matching_kp)
            matched_rgb_keypoints = rgb_keypoints[matches[valid]]

            # set matched rgb keypoints as goal points
            kp_loc = np.zeros((*rgb.shape[:2], 1), dtype=rgb.dtype)
            kp_loc[matched_rgb_keypoints[:, 1], matched_rgb_keypoints[:, 0]] = 1

            if self.preprojection_kp_dilation > 0:
                disk = skimage.morphology.disk(self.preprojection_kp_dilation)
                kp_loc = np.expand_dims(cv2.dilate(kp_loc, disk, iterations=1), axis=2)

            return kp_loc

        depth = np.expand_dims(obs.depth, axis=2) * 100.0
        rgb, depth = downscale(obs.rgb, depth)

        # rgb = torch.from_numpy(obs.rgb).to(self.device)
        # depth = torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0

        (goal_keypoints, rgb_keypoints, matches, confidence) = self.matching(
            obs.rgb,
            goal_image=self.goal_image,
            goal_image_keypoints=self.goal_image_keypoints,
            step=self.step,
        )
        kp_loc = preprocess_keypoint_localization(
            obs.rgb, goal_keypoints, rgb_keypoints, matches, confidence
        )

        # if self.store_all_categories_in_map:
        semantic = obs.semantic
        # else:
        #     semantic = np.full_like(obs.semantic, 4)
        #     semantic[
        #         obs.semantic == current_goal_semantic_id
        #     ] = current_goal_semantic_id
        semantic = (
            self.one_hot_encoding[torch.from_numpy(semantic)][..., :-1].cpu().numpy()
        )

        obs_preprocessed = np.concatenate([rgb, depth, semantic, kp_loc], axis=2)
        # obs_preprocessed = obs_preprocessed.transpose(2, 0, 1)
        obs_preprocessed = torch.from_numpy(obs_preprocessed)
        obs_preprocessed = obs_preprocessed.to(device=self.device)
        # obs_preprocessed = obs_preprocessed.unsqueeze(0)

        # current_task = obs.task_observations["tasks"][self.current_task_idx]
        # current_goal_semantic_id = current_task["semantic_id"]

        # obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)

        if self.record_instance_ids:
            instances = obs.task_observations["instance_map"]
            # first create a mapping to 1, 2, ... num_instances
            instance_ids = np.unique(instances)
            # map instance id to index
            instance_id_to_idx = {
                instance_id: idx for idx, instance_id in enumerate(instance_ids)
            }
            # convert instance ids to indices, use vectorized lookup
            instances = torch.from_numpy(
                np.vectorize(instance_id_to_idx.get)(instances)
            ).to(self.device)
            # create a one-hot encoding
            instances = torch.eye(len(instance_ids), device=self.device)[instances]
            obs_preprocessed = torch.cat([obs_preprocessed, instances], dim=-1)

        obs_preprocessed = obs_preprocessed.unsqueeze(0).permute(0, 3, 1, 2)

        all_matches, all_confidences = [], []
        if self.record_instance_ids and self.step == 0:
            all_matches, all_confidences = get_matches_against_memory(
                instance_memory,
                self.matching,
                self.step,
                image_goal=self.goal_image,
                goal_image_keypoints=self.goal_image_keypoints,
            )

        return obs_preprocessed, matches, confidence, all_matches, all_confidences

    def _preprocess_pose_and_delta(self, obs: Observations) -> Tuple[Tensor, ndarray]:
        """merge GPS+compass. Compute the delta from the previous timestep."""
        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = (
            torch.tensor(pu.get_rel_pose_change(curr_pose, self.last_pose))
            .unsqueeze(0)
            .to(device=self.device)
        )
        return pose_delta, curr_pose
