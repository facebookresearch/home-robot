#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import IntEnum, auto
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
import clip
import json
from PIL import Image
import matplotlib.pyplot as plt
from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.manipulation import HeuristicPlacePolicy


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


class LLMAgent(OpenVocabManipAgent):
    """Simple object nav agent based on a 2D semantic map."""

    def __init__(self,
                 config,
                 device_id: int = 0,
                 obs_spaces=None,
                 action_spaces=None):
        super().__init__(config, device_id=device_id)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        # self.goal_obs = {}
        self.clip_threshold = 0.8
        self.num_object_pixel_threshold = 500
        
        self.object_of_interest = None
        self.object_names = None
        
        self.memory = {}

    def show_image(self, rgb):
        """Simple helper function to save images"""
        im = Image.fromarray(rgb)
        im.save("datadump/test.png")

    def _update_memory(self, obs):
        text = clip.tokenize(self.object_names).to(self.device)
        # curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])

        # record GT label of if the object has been found
        if obs.task_observations['object_goal'] in obs.semantic:
            self.memory[self.timesteps[0]]['is_found'] = True

        for object_id in self.object_of_interest:
            if object_id not in obs.semantic: continue

            if self._get_num_goal_pixels(
                    obs, object_id) < self.num_object_pixel_threshold:
                continue

            image_arr = self._segment_goal(obs, object_id)
            image = self.preprocess(
                Image.fromarray(image_arr)).unsqueeze(0).to(self.device)

            with torch.no_grad():

                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)
                logits_per_image, logits_per_text = self.model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                if np.max(probs) > self.clip_threshold:
                # if True:
                    self.show_image(image_arr)
                    print(self.object_names[np.argmax(probs)])
                    print("Label probs:", probs)

                    self.show_image(image_arr)
                    self.memory[self.timesteps[0]]['clip_features'].append(
                        image_features.squeeze(0).tolist())

            # self.memory.append(image_features.squeeze(0))

    def _segment_goal(self, obs, object_id):
        segmented_rgb = np.zeros(obs.rgb.shape, dtype=np.uint8)
        min_h = obs.rgb.shape[0]
        min_w = obs.rgb.shape[1]
        max_h = 0
        max_w = 0

        for i, ele in np.ndenumerate(obs.semantic):
            if ele == object_id:
                segmented_rgb[i] = obs.rgb[i]
                if i[0] < min_h: min_h = i[0]
                if i[1] < min_w: min_w = i[1]
                if i[0] > max_h: max_h = i[0]
                if i[1] > max_w: max_w = i[1]

        cropped = np.zeros((max_h - min_h, max_w - min_w, obs.rgb.shape[2]),
                           dtype=np.uint8)
        for h in range(cropped.shape[0]):
            for w in range(cropped.shape[1]):
                cropped[h, w] = segmented_rgb[h + min_h, w + min_w]
        return cropped


#      def _is_object_perceived(self, obs):
#      # now using GT segments
#      # object_id = obs.task_observations['object_goal']
#      for object_id in self.object_of_interest:
#          if object_id not in self.goal_obs: self.goal_obs[object_id] = None
#          if object_id in obs.semantic:
#              num_goal_pixels = self._get_num_goal_pixels(obs, object_id)
#              if num_goal_pixels > self._get_num_goal_pixels(
#                      self.goal_obs[object_id], object_id):
#                  self.goal_obs[object_id] = obs
#          elif self.goal_obs[object_id]:
#              return True
#      return False
#

    def _get_num_goal_pixels(self, obs, object_id):
        count = 0
        if not obs:
            return count
        for i, ele in np.ndenumerate(obs.semantic):
            if ele == object_id:
                count += 1
        return count

    def act(self, obs: Observations
            ) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """State machine"""
        if self.object_of_interest is None:
            self.object_of_interest = np.array([
                obs.task_observations['object_goal'],
                obs.task_observations['start_recep_goal'],
                obs.task_observations['end_recep_goal']
            ])
        if self.object_names is None:
            self.object_names = obs.task_observations["goal_name"].split(' ')

        # if self._is_object_perceived(obs):
        self.memory[self.timesteps[0]] = {
            "clip_features": [],
            "is_found": False
        }
        if any(ele in obs.semantic for ele in self.object_of_interest):
            print(f'Update robot memory at timestep {self.timesteps[0]}')
            self._update_memory(obs)
            with open("datadump/memory_data.json", 'w') as json_file:
                json.dump(self.memory, json_file)
        return super().act(obs)
