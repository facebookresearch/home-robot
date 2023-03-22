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


class Skill:
    NAV_TO_OBJ = 0
    GAZE = 1
    NAV_TO_REC = 2
    PLACE = 3


class OpenVocabManipAgent(ObjectNavAgent):
    """Simple object nav agent based on a 2D semantic map"""

    def __init__(self, config, device_id: int = 0):
        super().__init__(config, device_id=device_id)
        self.states = None

    def reset_vectorized(self):
        """Initialize agent state."""
        super().reset_vectorized()
        self.states = torch.tensor([Skill.NAV_TO_OBJ] * self.num_environments)

    def get_nav_to_recep(self):
        return (self.states == Skill.NAV_TO_REC).float().to(device=self.device)

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.states[e] = Skill.NAV_TO_OBJ
        super().reset_vectorized_for_env(e)

    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """State machine"""
        action, info = super().act(obs)
        if action == DiscreteNavigationAction.STOP:
            if self.states[0] == Skill.NAV_TO_OBJ:
                action = DiscreteNavigationAction.EMPTY_ACTION
                self.states[0] = Skill.NAV_TO_REC
        return action, info
