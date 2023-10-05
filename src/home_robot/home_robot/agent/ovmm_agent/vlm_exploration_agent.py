#!/usr/bin/env python
# -*- coding: utf-8 -*-
# quick fix for import
import json
import os
import shutil
import sys
import warnings
from enum import IntEnum, auto
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations

sys.path.append("src/home_robot/home_robot/perception/detection/minigpt4/MiniGPT-4/")


class Skill(IntEnum):
    NAV_TO_OBJ = auto()
    GAZE_AT_OBJ = auto()
    PICK = auto()
    NAV_TO_REC = auto()
    GAZE_AT_REC = auto()
    PLACE = auto()
    EXPLORE = auto()
    NAV_TO_INSTANCE = auto()
    FALL_WAIT = auto()


class VLMExplorationAgent(OpenVocabManipAgent):
    def __init__(
        self, config, device_id: int = 0, obs_spaces=None, action_spaces=None, args=None
    ):
        warnings.warn(
            "VLM (MiniGPT-4) agent is currently under development and not fully supported yet."
        )
        super().__init__(config, device_id=device_id)
        print("VLM Exploration Agent created")
        self.vlm_freq = 10
        self.high_level_plan = None
        self.max_context_length = 20
        self.planning_times = 1
        self.remaining_actions = None
        self.args = args
        # print_image currently breaks
        self.obs_path = "data_obs/"
        shutil.rmtree(self.obs_path, ignore_errors=True)
        os.makedirs(self.obs_path, exist_ok=True)

    def _explore(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        nav_to_obj_type = self.config.AGENT.SKILLS.NAV_TO_OBJ.type
        if self.skip_skills.nav_to_obj:
            terminate = True
        elif nav_to_obj_type == "heuristic":
            if self.verbose:
                print("[OVMM AGENT] step heuristic nav policy")
            action, info, terminate = self._heuristic_nav(obs, info)
        elif nav_to_obj_type == "rl":
            action, info, terminate = self.nav_to_obj_agent.act(obs, info)
        else:
            raise ValueError(
                f"Got unexpected value for NAV_TO_OBJ.type: {nav_to_obj_type}"
            )
        new_state = None
        if terminate:
            action = None
            new_state = True
        return action, info, new_state

    def reset_vectorized(self, episodes=None):
        """Initialize agent state."""
        super().reset_vectorized()
        self.states = torch.tensor([Skill.EXPLORE] * self.num_environments)
        self.remaining_actions = None
        self.high_level_plan = None
        self.world_representation = None

    def act(
        self, obs: Observations
    ) -> Tuple[DiscreteNavigationAction, Dict[str, Any], Observations]:
        """State machine"""

        if self.timesteps[0] == 0:
            self._init_episode(obs)

        if self.config.GROUND_TRUTH_SEMANTICS == 0:
            obs = self.semantic_sensor(obs)
        else:
            obs.task_observations["semantic_frame"] = None
        info = self._get_info(obs)

        # Image.fromarray(obs.rgb, "RGB").save(self.obs_path +
        #                                      str(self.timesteps[0])+'.png')

        self.timesteps[0] += 1

        is_finished = False
        action = None

        while action is None:
            if self.states[0] == Skill.EXPLORE:
                obs.task_observations["instance_id"] = 100000000000
                action, info, new_state = self._explore(obs, info)
            else:
                raise ValueError

        # update the curr skill to the new skill whose action will be executed
        info["curr_skill"] = Skill(self.states[0].item()).name
        if self.verbose:
            print(
                f'Executing skill {info["curr_skill"]} at timestep {self.timesteps[0]}'
            )
        if self.args.max_step == self.timesteps[0]:
            print("Max agent step reached -- wrapping up...")
            is_finished = True

        return action, info, obs, is_finished
