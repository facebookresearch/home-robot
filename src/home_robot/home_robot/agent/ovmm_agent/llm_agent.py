#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import IntEnum, auto
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
import clip
from PIL import Image
import matplotlib.pyplot as plt

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


class OpenVocabManipAgent(ObjectNavAgent):
    """Simple object nav agent based on a 2D semantic map."""

    def __init__(self,
                 config,
                 device_id: int = 0,
                 obs_spaces=None,
                 action_spaces=None):
        super().__init__(config, device_id=device_id)
        self.states = None
        self.place_start_step = None
        self.pick_start_step = None
        self.is_gaze_done = None
        self.place_done = None
        self.gaze_agent = None
        self.nav_to_obj_agent = None
        self.pick_policy = None
        self.place_policy = None
        self.skip_skills = config.AGENT.skip_skills
        if config.AGENT.SKILLS.PLACE.type == "heuristic_debug":
            self.place_policy = HeuristicPlacePolicy(config, self.device)
        skip_both_gaze = self.skip_skills.gaze_at_obj and self.skip_skills.gaze_at_rec
        if config.AGENT.SKILLS.GAZE_OBJ.type == "rl" and not skip_both_gaze:
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
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        # self.goal_obs = {}
        self.clip_threshold = 0.8
        self.num_object_pixel_threshold = 500
        self.object_of_interest = np.array([1, 2,
                                            3])  # predefine a list of objects that we want to segment
        self.goal_object = 1
        self.memory = {}

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
            episodes[0].scene_id.split("/")[-1].split(".")[0],
            episodes[0].episode_id)
        if self.gaze_agent is not None:
            self.gaze_agent.reset_vectorized()
        if self.nav_to_obj_agent is not None:
            self.nav_to_obj_agent.reset_vectorized()
        self.states = torch.tensor([Skill.NAV_TO_OBJ] * self.num_environments)
        self.pick_start_step = torch.tensor([0] * self.num_environments)
        self.place_start_step = torch.tensor([0] * self.num_environments)
        self.is_gaze_done = torch.tensor([0] * self.num_environments)
        self.place_done = torch.tensor([0] * self.num_environments)

    def get_nav_to_recep(self):
        return (self.states == Skill.NAV_TO_REC).float().to(device=self.device)

    def reset_vectorized_for_env(self, e: int, episode):
        """Initialize agent state for a specific environment."""
        self.states[e] = Skill.NAV_TO_OBJ
        self.place_start_step[e] = 0
        self.pick_start_step[e] = 0
        self.is_gaze_done[e] = 0
        self.place_done[e] = 0
        self.place_policy = HeuristicPlacePolicy(self.config, self.device)
        super().reset_vectorized_for_env(e)
        self.planner.set_vis_dir(
            episode.scene_id.split("/")[-1].split(".")[0], episode.episode_id)
        if self.gaze_agent is not None:
            self.gaze_agent.reset_vectorized_for_env(e)
        if self.nav_to_obj_agent is not None:
            self.nav_to_obj_agent.reset_vectorized_for_env(e)

    def _switch_to_next_skill(
            self, e: int, next_skill: Skill,
            info: Dict[str, Any]) -> DiscreteNavigationAction:
        """Switch to the next skill for environment `e`.

        This function transitions to the next skill for the specified environment `e`.
        Initial setup for each skill is done here, and each skill can return a single
        action to take when starting (meant to switch between navigation and manipulation modes)
        """

        info["skill_done"] = Skill(self.states[e].item()).name
        action = None
        if next_skill == Skill.NAV_TO_OBJ:
            # action = DiscreteNavigationAction.NAVIGATION_MODE
            pass
        elif next_skill == Skill.GAZE_AT_OBJ:
            pass
        elif next_skill == Skill.PICK:
            self.pick_start_step[e] = self.timesteps[e]
        elif next_skill == Skill.NAV_TO_REC:
            self.timesteps_before_goal_update[e] = 0
            if not self.skip_skills.nav_to_rec:
                action = DiscreteNavigationAction.NAVIGATION_MODE
        elif next_skill == Skill.GAZE_AT_REC:
            # We reuse gaze agent between pick and place
            if self.gaze_agent is not None:
                self.gaze_agent.reset_vectorized_for_env(e)
        elif next_skill == Skill.PLACE:
            self.place_start_step[e] = self.timesteps[e]
        self.states[e] = next_skill
        return action

    def _heuristic_nav(
            self, obs: Observations,
            info: Dict[str, Any]) -> Tuple[DiscreteNavigationAction, Any]:
        action, planner_info = super().act(obs)
        info = {**planner_info, **info}
        self.timesteps[0] -= 1  # objectnav agent increments timestep
        info["timestep"] = self.timesteps[0]
        return action, info

    def _heuristic_pick(
            self, obs: Observations,
            info: Dict[str, Any]) -> Tuple[DiscreteNavigationAction, Any]:
        """Heuristic pick skill execution"""
        raise NotImplementedError  # WIP
        action, info = self.pick_policy(obs, info)
        if action == DiscreteNavigationAction.STOP:
            action = DiscreteNavigationAction.NAVIGATION_MODE
            info = self._switch_to_next_skill(e=0, info=info)
        return action, info

    def _hardcoded_place(self):
        """Hardcoded place skill execution
        Orients the agent's arm and camera towards the recetacle, extends arm and releases the object"""
        place_step = self.timesteps[0] - self.place_start_step[0]
        forward_steps = 0
        fall_steps = 20
        if place_step < forward_steps:
            # for experimentation (TODO: Remove. ideally nav should drop us close)
            action = DiscreteNavigationAction.MOVE_FORWARD
        elif place_step == forward_steps:
            action = DiscreteNavigationAction.MANIPULATION_MODE
        elif place_step == forward_steps + 1:
            action = DiscreteNavigationAction.EXTEND_ARM
        elif place_step == forward_steps + 2:
            # desnap to drop the object
            action = DiscreteNavigationAction.DESNAP_OBJECT
        elif place_step <= forward_steps + 2 + fall_steps:
            # allow the object to come to rest
            action = DiscreteNavigationAction.EMPTY_ACTION
        elif place_step == forward_steps + fall_steps + 3:
            action = DiscreteNavigationAction.STOP
        else:
            raise ValueError(
                f"Something is wrong. Episode should have ended. Place step: {place_step}, Timestep: {self.timesteps[0]}"
            )
        return action

    """
    The following methods each correspond to a skill/state this agent can execute.
    They take sensor observations as input and return the action to take and
    the state to transition to. Either the action has a value and the new state doesn't,
    or the action has no value and the new state does. The latter case indicates
    a state transition.
    """

    def _nav_to_obj(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        nav_to_obj_type = self.config.AGENT.SKILLS.NAV_TO_OBJ.type
        if self.skip_skills.nav_to_obj:
            action = DiscreteNavigationAction.STOP
        elif nav_to_obj_type == "heuristic":
            action, info = self._heuristic_nav(obs, info)
        elif nav_to_obj_type == "rl":
            action, term = self.nav_to_obj_agent.act(obs)
        else:
            raise ValueError(
                f"Got unexpected value for NAV_TO_OBJ.type: {nav_to_obj_type}")
        new_state = None
        if action == DiscreteNavigationAction.STOP:
            action = None
            new_state = Skill.GAZE_AT_OBJ
        return action, info, new_state

    def _gaze_at_obj(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        if self.skip_skills.gaze_at_obj:
            term = True
        else:
            action, term = self.gaze_agent.act(obs)
        new_state = None
        if term:
            # action = (
            #     {}
            # )  # TODO: update after simultaneous gripping/motion is supported
            # action["grip_action"] = [1]  # grasp the object when gaze is done
            # self.is_pick_done[0] = 1
            action = None
            new_state = Skill.PICK
        return action, info, new_state

    def _pick(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        if self.skip_skills.pick:
            action = None
        elif self.config.AGENT.SKILLS.PICK.type == "oracle":
            pick_step = self.timesteps[0] - self.pick_start_step[0]
            if pick_step == 0:
                action = DiscreteNavigationAction.SNAP_OBJECT
            elif pick_step == 1:
                action = None
            else:
                raise ValueError(
                    "Still in oracle pick. Should've transitioned to next skill."
                )
        new_state = None
        if action is None:
            new_state = Skill.NAV_TO_REC
        return action, info, new_state

    def _nav_to_rec(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        nav_to_rec_type = self.config.AGENT.SKILLS.NAV_TO_REC.type
        if self.skip_skills.nav_to_rec:
            action = DiscreteNavigationAction.STOP
        elif nav_to_rec_type == "heuristic":
            action, info = self._heuristic_nav(obs, info)
        elif nav_to_rec_type == "rl":
            raise NotImplementedError
        else:
            raise ValueError(
                f"Got unexpected value for NAV_TO_REC.type: {nav_to_rec_type}")
        new_state = None
        if action == DiscreteNavigationAction.STOP:
            action = None
            new_state = Skill.GAZE_AT_REC
        return action, info, new_state

    def _gaze_at_rec(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        if self.skip_skills.gaze_at_rec:
            term = True
        else:
            action, term = self.gaze_agent.act(obs)
        new_state = None
        if term:
            action = None
            new_state = Skill.PLACE
        return action, info, new_state

    def _place(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        place_type = self.config.AGENT.SKILLS.PLACE.type
        if self.skip_skills.place:
            action = DiscreteNavigationAction.STOP
        elif place_type == "hardcoded":
            action = self._hardcoded_place()
        elif place_type == "heuristic_debug":
            action, info = self.place_policy(obs, info)
        else:
            raise ValueError(
                f"Got unexpected value for PLACE.type: {place_type}")
        return action, info, None

    def show_image(self, rgb):
        """Simple helper function to save images"""
        im = Image.fromarray(rgb)
        im.save("test.png")

    def _update_memory(self, obs):

        object_names = obs.task_observations["goal_name"].split(' ')

        text = clip.tokenize(object_names).to(self.device)
        # curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])

        # record GT label of if the object has been found
        if self.goal_object in obs.semantic:
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
                    self.show_image(image_arr)
                    print(object_names[np.argmax(probs)])
                    print("Label probs:", probs)
                    
                    self.show_image(image_arr) 
                    self.memory[self.timesteps[0]]['clip_features'].append(image_features.squeeze(0).tolist())
                     
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
        
        info = self._get_info(obs)
        self.timesteps[0] += 1

        print(
            f'Executing skill {info["curr_skill"]} at timestep {self.timesteps[0]}'
        )

        action = None
        while action is None:
            if self.states[0] == Skill.NAV_TO_OBJ:
                action, info, new_state = self._nav_to_obj(obs, info)
            elif self.states[0] == Skill.GAZE_AT_OBJ:
                action, info, new_state = self._gaze_at_obj(obs, info)
            elif self.states[0] == Skill.PICK:
                action, info, new_state = self._pick(obs, info)
            elif self.states[0] == Skill.NAV_TO_REC:
                action, info, new_state = self._nav_to_rec(obs, info)
            elif self.states[0] == Skill.GAZE_AT_REC:
                action, info, new_state = self._gaze_at_rec(obs, info)
            elif self.states[0] == Skill.PLACE:
                action, info, new_state = self._place(obs, info)
            else:
                raise ValueError

            # Since heuristic nav is not properly vectorized, this agent currently only supports 1 env
            # _switch_to_next_skill is thus invoked with e=0
            if new_state:
                assert (
                    action is None
                ), f"action must be None when switching states, found {action} instead"
                action = self._switch_to_next_skill(0, new_state, info)

        # if self._is_object_perceived(obs):
        self.memory[self.timesteps[0]] = {"clip_features": [], "is_found": False}
        if any(ele in obs.semantic for ele in self.object_of_interest):
            print(f'Update robot memory at timestep {self.timesteps[0]}')
            self._update_memory(obs)

        return action, info
