import os
from enum import IntEnum
from typing import Any, Dict, Tuple

import numpy as np
import torch

from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.pick_policy.heuristic_pick_policy import HeuristicPickPolicy
from home_robot.place_policy.heuristic_place_policy import HeuristicPlacePolicy


class Skill(IntEnum):
    NAV_TO_OBJ = 0
    ORIENT_OBJ = 1
    GAZE_OBJ = 2
    PICK_OBJ = 3
    NAV_TO_REC = 4
    PLACE = 5


def get_skill_as_one_hot_dict(curr_skill: Skill):
    skill_dict = {skill.name: 0 for skill in Skill}
    skill_dict[f"is_curr_skill_{Skill(curr_skill).name}"] = 1
    return skill_dict


class OpenVocabManipAgent(ObjectNavAgent):
    """Simple object nav agent based on a 2D semantic map"""

    def __init__(self, config, device_id: int = 0, obs_spaces=None, action_spaces=None):
        super().__init__(config, device_id=device_id)
        self.states = None
        self.place_start_step = None
        self.orient_start_step = None
        self.is_gaze_done = None
        self.place_done = None
        self.gaze_agent = None
        self.nav_to_obj_agent = None
        self.pick_policy = None
        self.place_policy = None
        if config.AGENT.SKILLS.PLACE.type == "heuristic":
            self.place_policy = HeuristicPlacePolicy(config, self.device)
        # if config.AGENT.SKILLS.PICK_OBJ.type == "heuristic":
        #     self.pick_policy = HeuristicPickPolicy(config, self.device)
        if config.AGENT.SKILLS.GAZE_OBJ.type == "gaze":
            self.gaze_agent = PPOAgent(
                config,
                config.AGENT.SKILLS.GAZE_OBJ,
                device_id=device_id,
                obs_spaces=obs_spaces,
                action_spaces=action_spaces,
            )
        if config.AGENT.SKILLS.NAV_TO_OBJ.type == "rl":
            self.nav_to_obj_agent = PPOAgent(
                config,
                config.AGENT.SKILLS.NAV_TO_OBJ,
                device_id=device_id,
                obs_spaces=obs_spaces,
                action_spaces=action_spaces,
            )
        self.skip_nav_to_obj = config.AGENT.skip_nav_to_obj
        self.skip_nav_to_rec = config.AGENT.skip_nav_to_rec
        self.skip_place = config.AGENT.skip_place
        self.skip_pick = config.AGENT.skip_pick
        self.skip_orient_obj = config.AGENT.skip_orient_obj
        self.skip_gaze = config.AGENT.skip_gaze
        self.config = config

    def _get_info(self, obs: Observations) -> Dict[str, torch.Tensor]:
        """Get inputs for visual skill."""
        info = {
            "semantic_frame": obs.task_observations["semantic_frame"],
            "goal_name": obs.task_observations["goal_name"],
            "third_person_image": obs.third_person_image,
            "timestep": self.timesteps[0],
            "curr_skill": Skill(self.states[0].item()).name,
            "skill_done": "",  # Set if skill gets done
        }
        # only the current skill has corresponding value as 1
        info = {**info, **get_skill_as_one_hot_dict(self.states[0].item())}
        return info

    def reset_vectorized(self, episodes):
        """Initialize agent state."""
        super().reset_vectorized()
        self.planner.set_vis_dir(
            episodes[0].scene_id.split("/")[-1].split(".")[0], episodes[0].episode_id
        )
        if self.gaze_agent is not None:
            self.gaze_agent.reset_vectorized()
        if self.nav_to_obj_agent is not None:
            self.nav_to_obj_agent.reset_vectorized()
        self.states = torch.tensor([Skill.NAV_TO_OBJ] * self.num_environments)
        self.place_start_step = torch.tensor([0] * self.num_environments)
        self.orient_start_step = torch.tensor([0] * self.num_environments)
        self.is_gaze_done = torch.tensor([0] * self.num_environments)
        self.place_done = torch.tensor([0] * self.num_environments)

    def get_nav_to_recep(self):
        return (self.states == Skill.NAV_TO_REC).float().to(device=self.device)

    def reset_vectorized_for_env(self, e: int, episode):
        """Initialize agent state for a specific environment."""
        self.states[e] = Skill.NAV_TO_OBJ
        self.place_start_step[e] = 0
        self.orient_start_step[e] = 0
        self.is_gaze_done[e] = 0
        self.place_done[e] = 0
        self.place_policy = HeuristicPlacePolicy(self.config, self.device)
        super().reset_vectorized_for_env(e)
        self.planner.set_vis_dir(
            episode.scene_id.split("/")[-1].split(".")[0], episode.episode_id
        )
        if self.gaze_agent is not None:
            self.gaze_agent.reset_vectorized_for_env(e)
        if self.nav_to_obj_agent is not None:
            self.nav_to_obj_agent.reset_vectorized_for_env(e)

    def _switch_to_next_skill(
        self, e: int, info: Dict[str, Any], start_in_same_step: bool = False
    ):
        """Switch to the next skill for environment `e`.

        This function transitions to the next skill for the specified environment `e`.
        `start_in_same_step` indicates whether the next skill is started in the same timestep (eg. when the previous skill was skipped).
        """
        skill = self.states[e]
        print(f"Skipping {skill} in timestep {self.timesteps[e]}")
        info["skill_done"] = Skill(skill.item()).name
        if skill == Skill.NAV_TO_OBJ:
            self.states[e] = Skill.GAZE_OBJ
        elif skill == Skill.GAZE_OBJ:
            self.states[e] = Skill.ORIENT_OBJ
        elif skill == Skill.ORIENT_OBJ:
            self.states[e] = Skill.PICK_OBJ
        elif skill == Skill.PICK_OBJ:
            self.timesteps_before_goal_update[0] = 0
            self.states[e] = Skill.NAV_TO_REC
        elif skill == Skill.NAV_TO_REC:
            self.place_start_step[e] = self.timesteps[e] + 1
            self.states[e] = Skill.PLACE
            if start_in_same_step:
                self.place_start_step[e] -= 1
        elif skill == Skill.PLACE:
            self.place_done[0] = 1
        return info

    def _heuristic_nav(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any]:
        action, planner_info = super().act(obs)
        info = {**planner_info, **info}
        self.timesteps[0] -= 1  # objectnav agent increments timestep
        info["timestep"] = self.timesteps[0]
        if action == DiscreteNavigationAction.STOP:
            action = DiscreteNavigationAction.NAVIGATION_MODE
            info = self._switch_to_next_skill(e=0, info=info)
        return action, info

    def _heuristic_pick(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any]:
        action, info = self.pick_policy(obs, info)
        if action == DiscreteNavigationAction.STOP:
            action = DiscreteNavigationAction.NAVIGATION_MODE
            info = self._switch_to_next_skill(e=0, info=info)
        return action, info

    def _rl_nav_to_obj(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any]:
        """
        Gets the next action to execute from the RL-based nav-to-object policy
        """
        action, term = self.nav_to_obj_agent.act(obs)
        if term:
            action = DiscreteNavigationAction.NAVIGATION_MODE
            self._switch_to_next_skill(e=0, info=info)
        return action, info

    def _hardcoded_place(self):
        """Hardcoded place skill execution
        Orients the agent's arm and camera towards the recetacle, extends arm and releases the object"""
        place_step = self.timesteps[0] - self.place_start_step[0]
        turn_angle = self.config.ENVIRONMENT.turn_angle
        forward_steps = 0
        fall_steps = 20
        num_turns = np.round(90 / turn_angle)
        forward_and_turn_steps = forward_steps + num_turns
        if place_step < forward_steps:
            # for experimentation (TODO: Remove. ideally nav should drop us close)
            action = DiscreteNavigationAction.MOVE_FORWARD
        elif place_step < forward_and_turn_steps:
            # first orient
            action = DiscreteNavigationAction.TURN_LEFT
        elif place_step == forward_and_turn_steps:
            action = DiscreteNavigationAction.MANIPULATION_MODE
        elif place_step == forward_and_turn_steps + 1:
            action = DiscreteNavigationAction.EXTEND_ARM
        elif place_step == forward_and_turn_steps + 2:
            # desnap to drop the object
            action = DiscreteNavigationAction.DESNAP_OBJECT
        elif place_step <= forward_and_turn_steps + 2 + fall_steps:
            # allow the object to come to rest
            action = DiscreteNavigationAction.EMPTY_ACTION
        elif place_step == forward_and_turn_steps + fall_steps + 3:
            action = DiscreteNavigationAction.STOP
        else:
            raise ValueError(
                f"Something is wrong. Episode should have ended. Place step: {place_step}, Timestep: {self.timesteps[0]}"
            )
        return action

    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """State machine"""
        info = self._get_info(obs)

        self.timesteps[0] += 1
        print(f'Executing skill {info["curr_skill"]} at timestep {self.timesteps[0]}')
        # Since heuristic nav is not properly vectorized, this agent currently only supports 1 env
        # _switch_to_next_skill is thus always invoked with e=0
        if self.states[0] == Skill.NAV_TO_OBJ:
            nav_to_obj_type = self.config.AGENT.SKILLS.NAV_TO_OBJ.type
            if self.skip_nav_to_obj:
                info = self._switch_to_next_skill(
                    e=0, info=info, start_in_same_step=True
                )
            elif nav_to_obj_type == "heuristic":
                return self._heuristic_nav(obs, info)
            elif nav_to_obj_type == "rl":
                return self._rl_nav_to_obj(obs, info)
            else:
                raise ValueError(
                    f"Got unexpected value for NAV_TO_OBJ.type: {nav_to_obj_type}"
                )
        if self.states[0] == Skill.GAZE_OBJ:
            if self.skip_gaze:
                info = self._switch_to_next_skill(
                    e=0, info=info, start_in_same_step=True
                )
            elif self.is_gaze_done[0]:
                info = self._switch_to_next_skill(e=0, info=info)
                self.is_gaze_done[0] = 0
                return DiscreteNavigationAction.NAVIGATION_MODE, info
            elif self.config.AGENT.SKILLS.GAZE_OBJ.type == "rl":
                action, term = self.gaze_agent.act(obs)
                if term:
                    action = (
                        {}
                    )  # TODO: update after simultaneous gripping/motion is supported
                    action["grip_action"] = [1]  # grasp the object when gaze is done
                    self.is_gaze_done[0] = 1
                return action, info
            else:
                raise NotImplementedError
        if self.states[0] == Skill.ORIENT_OBJ:
            info = self._switch_to_next_skill(e=0, info=info)
            return DiscreteNavigationAction.MANIPULATION_MODE, info
        if self.states[0] == Skill.PICK_OBJ:
            if self.config.AGENT.SKILLS.PICK_OBJ.type == "oracle":
                return DiscreteNavigationAction.SNAP_OBJECT, info
            elif self.config.AGENT.SKILLS.PICK_OBJ.type == "heuristic":
                raise NotImplementedError
            #     return self._heuristic_pick(obs, info)
            else:
                raise NotImplementedError
        if self.states[0] == Skill.NAV_TO_REC:
            if self.skip_nav_to_rec:
                info = self._switch_to_next_skill(
                    e=0, info=info, start_in_same_step=True
                )
            elif self.config.AGENT.SKILLS.NAV_TO_REC.type == "heuristic":
                return self._heuristic_nav(obs, info)
            else:
                raise NotImplementedError
        if self.states[0] == Skill.PLACE:
            if self.skip_place:
                info["skill_done"] = "PLACE"
                return DiscreteNavigationAction.STOP, info
            elif self.config.AGENT.SKILLS.PLACE.type == "hardcoded":
                action = self._hardcoded_place()
                return action, info
            elif self.config.AGENT.SKILLS.PLACE.type == "heuristic_debug":
                action, info = self.place_policy(obs, info)
                return action, info
            else:
                raise NotImplementedError
        return DiscreteNavigationAction.STOP, info
