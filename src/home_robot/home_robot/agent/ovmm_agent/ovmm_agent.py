from enum import IntEnum
from typing import Any, Dict, Tuple

import numpy as np
import torch

from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations


class Skill(IntEnum):
    NAV_TO_OBJ = 0
    ORIENT_OBJ = 1
    PICK = 2
    NAV_TO_REC = 3
    PLACE = 4


class OpenVocabManipAgent(ObjectNavAgent):
    """Simple object nav agent based on a 2D semantic map"""

    def __init__(self, config, device_id: int = 0, obs_spaces=None, action_spaces=None):
        super().__init__(config, device_id=device_id)
        self.states = None
        self.place_start_step = None
        self.orient_start_step = None
        self.pick_done = None
        self.place_done = None
        self.gaze_agent = None
        if config.AGENT.SKILLS.PICK.type == "gaze":
            self.gaze_agent = PPOAgent(
                config,
                config.AGENT.SKILLS.PICK,
                device_id=device_id,
                obs_spaces=obs_spaces,
                action_spaces=action_spaces,
            )
        self.skip_nav_to_obj = config.AGENT.skip_nav_to_obj
        self.skip_nav_to_rec = config.AGENT.skip_nav_to_rec
        self.skip_place = config.AGENT.skip_place
        self.skip_pick = config.AGENT.skip_pick
        self.skip_orient_obj = config.AGENT.skip_orient_obj
        self.config = config

    def _get_vis_inputs(self, obs: Observations) -> Dict[str, torch.Tensor]:
        """Get inputs for visual skill."""
        return {
            "semantic_frame": obs.task_observations["semantic_frame"],
            "goal_name": obs.task_observations["goal_name"],
            "third_person_image": obs.third_person_image,
            "timestep": self.timesteps[0],
        }

    def reset_vectorized(self):
        """Initialize agent state."""
        super().reset_vectorized()
        if self.gaze_agent is not None:
            self.gaze_agent.reset_vectorized()
        self.states = torch.tensor([Skill.NAV_TO_OBJ] * self.num_environments)
        self.place_start_step = torch.tensor([0] * self.num_environments)
        self.orient_start_step = torch.tensor([0] * self.num_environments)
        self.pick_done = torch.tensor([0] * self.num_environments)
        self.place_done = torch.tensor([0] * self.num_environments)

    def get_nav_to_recep(self):
        return (self.states == Skill.NAV_TO_REC).float().to(device=self.device)

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.states[e] = Skill.NAV_TO_OBJ
        self.place_start_step[e] = 0
        self.orient_start_step[e] = 0
        self.pick_done[e] = 0
        self.place_done[e] = 0
        super().reset_vectorized_for_env(e)
        if self.gaze_agent is not None:
            self.gaze_agent.reset_vectorized_for_env(e)

    def _switch_to_next_skill(self, e: int):
        """Switch to the next skill."""
        skill = self.states[e]
        if skill == Skill.NAV_TO_OBJ:
            self.states[e] = Skill.ORIENT_OBJ
            self.orient_start_step[e] = self.timesteps[e]
        elif skill == Skill.ORIENT_OBJ:
            self.states[e] = Skill.PICK
        elif skill == Skill.PICK:
            self.states[e] = Skill.NAV_TO_REC
        elif skill == Skill.NAV_TO_REC:
            self.place_start_step[e] = self.timesteps[e]
            self.states[e] = Skill.PLACE
        elif skill == Skill.PLACE:
            self.place_done[0] = 1

    def _hardcoded_place(self):
        place_step = self.timesteps[0] - self.place_start_step[0]
        turn_angle = self.config.ENVIRONMENT.turn_angle
        forward_steps = 0
        fall_steps = 20
        num_turns = np.round(90 / turn_angle)
        forward_and_turn_steps = forward_steps + num_turns
        if place_step <= forward_steps:
            # for experimentation (TODO: Remove. ideally nav should drop us close)
            action = DiscreteNavigationAction.MOVE_FORWARD
        elif place_step <= forward_and_turn_steps:
            # first orient
            action = DiscreteNavigationAction.TURN_LEFT
        elif place_step == forward_and_turn_steps + 1:
            action = DiscreteNavigationAction.FACE_ARM
        elif place_step == forward_and_turn_steps + 2:
            action = DiscreteNavigationAction.EXTEND_ARM
        elif place_step == forward_and_turn_steps + 3:
            # desnap to drop the object
            action = DiscreteNavigationAction.DESNAP_OBJECT
        elif place_step <= forward_and_turn_steps + 3 + fall_steps:
            # allow the object to come to rest
            action = DiscreteNavigationAction.EMPTY_ACTION
        elif place_step == forward_and_turn_steps + fall_steps + 4:
            action = DiscreteNavigationAction.STOP
        return action

    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """State machine"""
        # TODO: from config
        vis_inputs = self._get_vis_inputs(obs)
        turn_angle = self.config.ENVIRONMENT.turn_angle

        self.timesteps[0] += 1
        if self.states[0] == Skill.NAV_TO_OBJ:
            if self.skip_nav_to_obj:
                self._switch_to_next_skill(e=0)
            elif self.config.AGENT.SKILLS.NAV_TO_OBJ.type == "modular":
                action, info = super().act(obs)
                self.timesteps[0] -= 1  # objectnav agent increments timestep
                if action == DiscreteNavigationAction.STOP:
                    action = DiscreteNavigationAction.RESET_JOINTS
                    self._switch_to_next_skill(e=0)
                return action, info
            else:
                raise NotImplementedError
        if self.states[0] == Skill.ORIENT_OBJ:
            num_turns = np.round(90 / turn_angle)
            orient_step = self.timesteps[0] - self.orient_start_step[0]
            if self.skip_orient_obj:
                self._switch_to_next_skill(e=0)
            elif orient_step <= num_turns:
                return DiscreteNavigationAction.TURN_LEFT, vis_inputs
            elif orient_step == num_turns + 1:
                self._switch_to_next_skill(e=0)
                return DiscreteNavigationAction.FACE_ARM, vis_inputs
        if self.states[0] == Skill.PICK:
            if self.skip_pick:
                self._switch_to_next_skill(e=0)
            elif self.pick_done[0]:
                self._switch_to_next_skill(e=0)
                self.timesteps_before_goal_update[0] = 0
                self.pick_done[0] = 0
                return DiscreteNavigationAction.RESET_JOINTS, vis_inputs
            elif self.config.AGENT.SKILLS.PICK.type == "gaze":
                action, term = self.gaze_agent.act(obs)
                if term:
                    # the object is at the center of the frame, try to grasp the object
                    action["grip_action"] = [1]
                    self.pick_done[0] = 1
                return action, vis_inputs
            elif self.config.AGENT.SKILLS.PICK.type == "hardcoded":
                self.pick_done[0] = 1
                return DiscreteNavigationAction.SNAP_OBJECT, vis_inputs
            else:
                raise NotImplementedError
        if self.states[0] == Skill.NAV_TO_REC:
            if self.skip_nav_to_rec:
                self._switch_to_next_skill(e=0)
            elif self.config.AGENT.SKILLS.NAV_TO_REC.type == "modular":
                action, info = super().act(obs)
                self.timesteps[0] -= 1  # objectnav agent increments timestep
                if action == DiscreteNavigationAction.STOP:
                    action = DiscreteNavigationAction.RESET_JOINTS
                    self._switch_to_next_skill(e=0)
                return action, info
            else:
                raise NotImplementedError
        if self.states[0] == Skill.PLACE:
            if self.skip_place:
                return DiscreteNavigationAction.STOP, vis_inputs
            elif self.config.AGENT.SKILLS.PLACE.type == "hardcoded":
                action = self._hardcoded_place()
                return action, vis_inputs
            else:
                raise NotImplementedError
        return DiscreteNavigationAction.STOP, vis_inputs
