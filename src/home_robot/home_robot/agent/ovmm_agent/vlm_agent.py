#!/usr/bin/env python
# -*- coding: utf-8 -*-
# quick fix for import
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


class VLMAgent(OpenVocabManipAgent):

    def __init__(self, config, device_id: int = 0, obs_spaces=None, action_spaces=None):
        warnings.warn(
            "vlm agent is currently under development and not fully supported yet."
        )
        super().__init__(config, device_id=device_id)
        from minigpt4_example import Predictor
        self.vlm = Predictor()

    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """State machine"""
        print("\n")
        self.vlm.predict(Image.fromarray(obs.rgb))
        return super().act(obs)
