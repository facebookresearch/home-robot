# TODO: WIP
# from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent
from enum import IntEnum
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.nn import DataParallel

import home_robot.utils.pose as pu
from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.agent.objectnav_agent.objectnav_agent_module import ObjectNavAgentModule
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.navigation_planner.discrete_planner import DiscretePlanner


class Skill(IntEnum):
    NAV_TO_OBJ = 0
    PICK = 1
    NAV_TO_REC = 2
    PLACE = 3


class OpenVocabManipAgent(ObjectNavAgent):
    """Simple object nav agent based on a 2D semantic map"""

    def __init__(self, config, device_id: int = 0, obs_spaces=None, action_spaces=None):
        super().__init__(config, device_id=device_id)
        self.states = None
        self.place_start_step = None
        self.pick_start_step = None
        # self.ppo_agent = PPOAgent(config, obs_spaces=obs_spaces, action_spaces=action_spaces)

    def reset_vectorized(self):
        """Initialize agent state."""
        super().reset_vectorized()
        # self.ppo_agent.reset_vectorized()
        self.states = torch.tensor([Skill.NAV_TO_OBJ] * self.num_environments)
        self.place_start_step = torch.tensor([0] * self.num_environments)
        self.pick_start_step = torch.tensor([0] * self.num_environments)

    def get_nav_to_recep(self):
        return (self.states == Skill.NAV_TO_REC).float().to(device=self.device)

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.states[e] = Skill.NAV_TO_OBJ
        self.place_start_step[e] = 0
        self.pick_start_step[e] = 0
        super().reset_vectorized_for_env(e)

    def act(
        self, habitat_obs, obs: Observations
    ) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """State machine"""
        action, info = super().act(obs)
        # snap the object
        # rotate 90 degrees so that arm faces the receptacle
        # extend camera
        if self.states[0] == Skill.PLACE:
            place_step = self.timesteps[0] - self.place_start_step[0]
            num_turns = np.round(90 / 10)
            fall_steps = 20
            # first orient
            if place_step <= num_turns:
                action = DiscreteNavigationAction.TURN_LEFT
            elif place_step == num_turns + 1:
                action = DiscreteNavigationAction.FACE_ARM
            elif place_step == num_turns + 2:
                action = DiscreteNavigationAction.EXTEND_ARM
            elif place_step == num_turns + 3:
                action = DiscreteNavigationAction.DESNAP_OBJECT
            elif place_step <= num_turns + 3 + fall_steps:
                # allow the object to come to rest
                action = DiscreteNavigationAction.EMPTY_ACTION
            elif place_step == num_turns + fall_steps + 4:
                action = DiscreteNavigationAction.STOP
        elif self.states[0] == Skill.PICK:
            pick_step = self.timesteps[0] - self.pick_start_step[0]
            num_turns = np.round(90 / 10)
            # first orient
            if pick_step <= num_turns:
                action = DiscreteNavigationAction.TURN_LEFT
            elif pick_step == num_turns + 1:
                action = DiscreteNavigationAction.FACE_ARM
            elif pick_step == num_turns + 2:
                action = DiscreteNavigationAction.SNAP_OBJECT
            elif pick_step == num_turns + 3:
                action = DiscreteNavigationAction.RESET_JOINTS
                self.timesteps_before_goal_update[0] = 0
                self.states[0] = Skill.NAV_TO_REC
        else:
            # unsnap to drop the object
            if action == DiscreteNavigationAction.STOP:
                if self.states[0] == Skill.NAV_TO_OBJ:
                    action = DiscreteNavigationAction.RESET_JOINTS
                    self.states[0] = Skill.PICK
                    self.pick_start_step[0] = self.timesteps[0]
                elif self.states[0] == Skill.NAV_TO_REC:
                    action = DiscreteNavigationAction.RESET_JOINTS
                    self.states[0] = Skill.PLACE
                    self.place_start_step[0] = self.timesteps[0]
        return action, info
