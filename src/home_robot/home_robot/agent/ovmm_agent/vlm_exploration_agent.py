#!/usr/bin/env python
# -*- coding: utf-8 -*-
# quick fix for import
import home_robot.utils.pose as pu
from home_robot.manipulation import HeuristicPlacePolicy
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent
from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import clip
from typing import Any, Dict, Optional, Tuple
from enum import IntEnum, auto
import warnings
import json
import sys
sys.path.append(
    "src/home_robot/home_robot/perception/detection/minigpt4/MiniGPT-4/")


class Skill(IntEnum):
    NAV_TO_OBJ = auto()
    GAZE_AT_OBJ = auto()
    PICK = auto()
    NAV_TO_REC = auto()
    GAZE_AT_REC = auto()
    PLACE = auto()


def get_skill_as_one_hot_dict(curr_skill: Skill):
    skill_dict = {skill.name: 0 for skill in Skill}
    skill_dict[f"is_curr_skill_{Skill(curr_skill).name}"] = 1
    return skill_dict


class VLMExplorationAgent(OpenVocabManipAgent):

    def __init__(self, config, device_id: int = 0, obs_spaces=None, action_spaces=None):
        warnings.warn(
            "vlm agent is currently under development and not fully supported yet."
        )
        super().__init__(config, device_id=device_id)
        print ("Simple Exploration Agent (for VLM) created")

    def _preprocess_obs(self, obs: Observations):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map."""
        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm
        if self.store_all_categories_in_map:
            obj_goal_idx = obs.task_observations["object_goal"]
            if "start_recep_goal" in obs.task_observations:
                start_recep_idx = obs.task_observations["start_recep_goal"]
            if "end_recep_goal" in obs.task_observations:
                end_recep_idx = obs.task_observations["end_recep_goal"]
        else:
            obj_goal_idx, start_recep_idx, end_recep_idx = 1, 2, 3
        semantic = np.full_like(obs.semantic, 4)
        semantic = self.one_hot_encoding[torch.from_numpy(
            semantic).to(self.device)]

        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)
        if self.record_instance_ids:
            instances = obs.task_observations["instance_map"]
            # first create a mapping to 1, 2, ... num_instances
            instance_ids = np.unique(instances)
            # map instance id to index
            instance_id_to_idx = {
                instance_id: idx for idx, instance_id in enumerate(instance_ids)
            }
            # convert instance ids to indices, use vectorized lookup
            instances = torch.from_numpy(np.vectorize(
                instance_id_to_idx.get)(instances)).to(self.device)
            # create a one-hot encoding
            instances = torch.eye(len(instance_ids), device=self.device)[
                instances]
            obs_preprocessed = torch.cat([obs_preprocessed, instances], dim=-1)

        if self.evaluate_instance_tracking:
            gt_instance_ids = (
                torch.from_numpy(obs.task_observations["gt_instance_ids"])
                .to(self.device)
                .long()
            )
            gt_instance_ids = self.one_hot_instance_encoding[gt_instance_ids]
            obs_preprocessed = torch.cat(
                [obs_preprocessed, gt_instance_ids], dim=-1)

        obs_preprocessed = obs_preprocessed.unsqueeze(0).permute(0, 3, 1, 2)

        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_poses[0])
        ).unsqueeze(0)
        self.last_poses[0] = curr_pose
        object_goal_category = None
        end_recep_goal_category = None
        if (
            "object_goal" in obs.task_observations
            and obs.task_observations["object_goal"] is not None
        ):
            if self.verbose:
                print("object goal =", obs.task_observations["object_goal"])
            object_goal_category = torch.tensor(obj_goal_idx).unsqueeze(0)
        start_recep_goal_category = None
        if (
            "start_recep_goal" in obs.task_observations
            and obs.task_observations["start_recep_goal"] is not None
        ):
            if self.verbose:
                print("start_recep goal =",
                      obs.task_observations["start_recep_goal"])
            start_recep_goal_category = torch.tensor(
                start_recep_idx).unsqueeze(0)
        if (
            "end_recep_goal" in obs.task_observations
            and obs.task_observations["end_recep_goal"] is not None
        ):
            if self.verbose:
                print("end_recep goal =",
                      obs.task_observations["end_recep_goal"])
            end_recep_goal_category = torch.tensor(end_recep_idx).unsqueeze(0)
        goal_name = [obs.task_observations["goal_name"]]

        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)
        return (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            start_recep_goal_category,
            end_recep_goal_category,
            goal_name,
            camera_pose,
        )
