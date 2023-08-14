#!/usr/bin/env python
# -*- coding: utf-8 -*-
# quick fix for import
import json
import sys
import warnings
from enum import IntEnum, auto
from typing import Any, Dict, Optional, Tuple

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import home_robot.utils.pose as pu
from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent
from home_robot.agent.ovmm_agent.vlm_exploration_agent import VLMExplorationAgent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.manipulation import HeuristicPlacePolicy

sys.path.append("src/home_robot/home_robot/perception/detection/minigpt4/MiniGPT-4/")


class Skill(IntEnum):
    NAV_TO_OBJ = auto()
    GAZE_AT_OBJ = auto()
    PICK = auto()
    NAV_TO_REC = auto()
    GAZE_AT_REC = auto()
    PLACE = auto()
    EXPLORE = auto()


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
        # if config.GROUND_TRUTH_SEMANTICS == 0 or self.store_all_categories_in_map:
        #     raise NotImplementedError
        from minigpt4_example import Predictor

        # self.vlm = Predictor()
        print("VLM Agent created")

        self.vlm_freq = 1

        self.explore_agent = VLMExplorationAgent(config=config)
        self.explore_agent.reset()
        print(
            "Also created a simple exploration agent for frontier exploration behaviors"
        )

    def switch_high_level_action(self):
        self.states[0] == Skill.EXPLORE
        return

    def _explore(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        # nav_to_obj_type = self.config.AGENT.SKILLS.NAV_TO_OBJ.type
        # if self.skip_skills.nav_to_obj:
        #     terminate = True
        # elif nav_to_obj_type == "heuristic":
        #     if self.verbose:
        #         print("[OVMM AGENT] step heuristic nav policy")
        #     action, info, terminate = self._heuristic_nav(obs, info)
        # elif nav_to_obj_type == "rl":
        #     action, info, terminate = self.nav_to_obj_agent.act(obs, info)
        # else:
        #     raise ValueError(
        #         f"Got unexpected value for NAV_TO_OBJ.type: {nav_to_obj_type}"
        #     )
        action, info, obs = self.explore_agent.act(obs)
        new_state = None
        # if terminate:
        #     action = None
        # new_state = Skill.GAZE_AT_OBJ
        return action, info, new_state

    def reset_vectorized(self, episodes=None):
        """Initialize agent state."""
        super().reset_vectorized()
        self.states = torch.tensor([Skill.EXPLORE] * self.num_environments)

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

        self.timesteps[0] += 1

        if self.timesteps[0] % self.vlm_freq == 0:

            self.switch_high_level_action()
            # import pdb; pdb.set_trace()

        action = None
        while action is None:
            if self.states[0] == Skill.EXPLORE:
                action, info, new_state = self._explore(obs, info)
            if self.states[0] == Skill.NAV_TO_INSTANCE:
                obs.task_observations["instance_id"] = 1  # TODO: get this from model
                action, info, new_state = self._nav_to_obj(obs, info)
            elif self.states[0] == Skill.GAZE_AT_OBJ:
                action, info, new_state = self._gaze_at_obj(obs, info)
            elif self.states[0] == Skill.PICK:
                pick_instance_id = 1  # TODO
                category_id = self.instance_memory.instance_views[0][
                    pick_instance_id
                ].category_id
                obs.task_observations["object_goal"] = category_id
                action, info, new_state = self._pick(obs, info)
            # elif self.states[0] == Skill.NAV_TO_REC:
            #     action, info, new_state = self._nav_to_rec(obs, info)
            # elif self.states[0] == Skill.GAZE_AT_REC:
            #     action, info, new_state = self._gaze_at_rec(obs, info)
            elif self.states[0] == Skill.PLACE:
                action, info, new_state = self._place(obs, info)
            elif self.states[0] == Skill.FALL_WAIT:
                action, info, new_state = self._fall_wait(obs, info)
            else:
                raise ValueError

            # Since heuristic nav is not properly vectorized, this agent currently only supports 1 env
            # _switch_to_next_skill is thus invoked with e=0
            if new_state:
                # mark the current skill as done
                info["skill_done"] = info["curr_skill"]
                assert (
                    action is None
                ), f"action must be None when switching states, found {action} instead"
                action = self._switch_to_next_skill(0, new_state, info)
        # update the curr skill to the new skill whose action will be executed
        info["curr_skill"] = Skill(self.states[0].item()).name
        if self.verbose:
            print(
                f'Executing skill {info["curr_skill"]} at timestep {self.timesteps[0]}'
            )
        return action, info, obs

    # def get_perceived_objects(self, obs):
    #     conv = self.vlm.init_conv(Image.fromarray(obs.rgb))
    #     prompt = " Please only answer Yes or No."
    #     object_answer = True if "Yes" in self.vlm.ask("Is there " +
    #                                                   obs.task_observations["object_name"] + " in the image?" + prompt, conv) else False
    #     s_recep_answer = True if "Yes" in self.vlm.ask("Is there " +
    #                                                    obs.task_observations["start_recep_name"] + " in the image?" + prompt, conv) else False
    #     g_recep_answer = True if "Yes" in self.vlm.ask("Is there " +
    #                                                    obs.task_observations["place_recep_name"] + " in the image?" + prompt, conv) else False
    #     return object_answer, s_recep_answer, g_recep_answer

    # def _preprocess_obs(self, obs: Observations):
    #     """Take a home-robot observation, preprocess it to put it into the correct format for the
    #     semantic map."""
    #     rgb = torch.from_numpy(obs.rgb).to(self.device)
    #     depth = (
    #         torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
    #     )  # m to cm
    #     if self.store_all_categories_in_map:
    #     #     semantic = obs.semantic
    #         obj_goal_idx = obs.task_observations["object_goal"]
    #         if "start_recep_goal" in obs.task_observations:
    #             start_recep_idx = obs.task_observations["start_recep_goal"]
    #         if "end_recep_goal" in obs.task_observations:
    #             end_recep_idx = obs.task_observations["end_recep_goal"]
    #     else:
    #     #     semantic = np.full_like(obs.semantic, 4)
    #         obj_goal_idx, start_recep_idx, end_recep_idx = 1, 2, 3
    #     #     goal, start, end = self.get_perceived_objects(obs)
    #     #     if goal:
    #     #         semantic[
    #     #             obs.semantic == obs.task_observations["object_goal"]
    #     #         ] = obj_goal_idx
    #     #     if "start_recep_goal" in obs.task_observations and start:
    #     #         semantic[
    #     #             obs.semantic == obs.task_observations["start_recep_goal"]
    #     #         ] = start_recep_idx
    #     #     if "end_recep_goal" in obs.task_observations and end:
    #     #         semantic[
    #     #             obs.semantic == obs.task_observations["end_recep_goal"]
    #     #         ] = end_recep_idx
    #     semantic = np.full_like(obs.semantic, 4)
    #     semantic = self.one_hot_encoding[torch.from_numpy(
    #         semantic).to(self.device)]

    #     obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1)
    #     if self.record_instance_ids:
    #         instances = obs.task_observations["instance_map"]
    #         # first create a mapping to 1, 2, ... num_instances
    #         instance_ids = np.unique(instances)
    #         # map instance id to index
    #         instance_id_to_idx = {
    #             instance_id: idx for idx, instance_id in enumerate(instance_ids)
    #         }
    #         # convert instance ids to indices, use vectorized lookup
    #         instances = torch.from_numpy(np.vectorize(
    #             instance_id_to_idx.get)(instances)).to(self.device)
    #         # create a one-hot encoding
    #         instances = torch.eye(len(instance_ids), device=self.device)[
    #             instances]
    #         obs_preprocessed = torch.cat([obs_preprocessed, instances], dim=-1)

    #     if self.evaluate_instance_tracking:
    #         gt_instance_ids = (
    #             torch.from_numpy(obs.task_observations["gt_instance_ids"])
    #             .to(self.device)
    #             .long()
    #         )
    #         gt_instance_ids = self.one_hot_instance_encoding[gt_instance_ids]
    #         obs_preprocessed = torch.cat(
    #             [obs_preprocessed, gt_instance_ids], dim=-1)

    #     obs_preprocessed = obs_preprocessed.unsqueeze(0).permute(0, 3, 1, 2)

    #     curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
    #     pose_delta = torch.tensor(
    #         pu.get_rel_pose_change(curr_pose, self.last_poses[0])
    #     ).unsqueeze(0)
    #     self.last_poses[0] = curr_pose
    #     object_goal_category = None
    #     end_recep_goal_category = None
    #     if (
    #         "object_goal" in obs.task_observations
    #         and obs.task_observations["object_goal"] is not None
    #     ):
    #         if self.verbose:
    #             print("object goal =", obs.task_observations["object_goal"])
    #         object_goal_category = torch.tensor(obj_goal_idx).unsqueeze(0)
    #     start_recep_goal_category = None
    #     if (
    #         "start_recep_goal" in obs.task_observations
    #         and obs.task_observations["start_recep_goal"] is not None
    #     ):
    #         if self.verbose:
    #             print("start_recep goal =",
    #                   obs.task_observations["start_recep_goal"])
    #         start_recep_goal_category = torch.tensor(
    #             start_recep_idx).unsqueeze(0)
    #     if (
    #         "end_recep_goal" in obs.task_observations
    #         and obs.task_observations["end_recep_goal"] is not None
    #     ):
    #         if self.verbose:
    #             print("end_recep goal =",
    #                   obs.task_observations["end_recep_goal"])
    #         end_recep_goal_category = torch.tensor(end_recep_idx).unsqueeze(0)
    #     goal_name = [obs.task_observations["goal_name"]]

    #     camera_pose = obs.camera_pose
    #     if camera_pose is not None:
    #         camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)
    #     return (
    #         obs_preprocessed,
    #         pose_delta,
    #         object_goal_category,
    #         start_recep_goal_category,
    #         end_recep_goal_category,
    #         goal_name,
    #         camera_pose,
    #     )
