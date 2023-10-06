#!/usr/bin/env python
# -*- coding: utf-8 -*-
# quick fix for import
import json
import os
import pdb
import random
import shutil
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
    NAV_TO_INSTANCE = auto()
    FALL_WAIT = auto()


class VLMAgent(OpenVocabManipAgent):
    def __init__(
        self, config, device_id: int = 0, obs_spaces=None, action_spaces=None, args=None
    ):
        warnings.warn(
            "VLM (MiniGPT-4) agent is currently under development and not fully supported yet."
        )
        super().__init__(config, device_id=device_id)
        # if config.GROUND_TRUTH_SEMANTICS == 0 or self.store_all_categories_in_map:
        #     raise NotImplementedError
        from minigpt4_example import Predictor

        if args and hasattr(args, "task"):
            print("Reset the agent task to " + args.task)
            self.set_task(args.task)
        self.vlm = Predictor(args)
        print("VLM Agent created")
        self.vlm_freq = 10
        self.high_level_plan = None
        self.max_context_length = 20
        self.planning_times = 1
        self.remaining_actions = None

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

    def _nav_to_obj(
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

    def _gaze_at_obj(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        gaze_step = self.timesteps[0] - self.gaze_at_obj_start_step[0]
        if self.skip_skills.gaze_at_obj:
            terminate = True
        elif gaze_step == 0:
            return DiscreteNavigationAction.POST_NAV_MODE, info, None
        else:
            action, info, terminate = self.gaze_agent.act(obs, info)
        new_state = None

        if terminate:
            # action = (
            #     {}
            # )  # TODO: update after simultaneous gripping/motion is supported
            # action["grip_action"] = [1]  # grasp the object when gaze is done
            # self.is_pick_done[0] = 1
            action = None
            new_state = True
        return action, info, new_state

    def set_task(self, task):
        self.task = task

    def switch_high_level_action(self):
        self.states = torch.tensor([Skill.EXPLORE] * self.num_environments)
        if self.timesteps[0] >= 20:
            self.states = torch.tensor([Skill.NAV_TO_INSTANCE] * self.num_environments)
        return

    def reset_vectorized(self, episodes=None):
        """Initialize agent state."""
        super().reset_vectorized()
        self.states = torch.tensor([Skill.EXPLORE] * self.num_environments)
        self.remaining_actions = None
        self.high_level_plan = None
        self.world_representation = None

    def ask_vlm_for_plan(self, world_representation):
        sample = self.vlm.prepare_sample(self.task, world_representation)
        plan = self.vlm.evaluate(sample)
        return plan

    def get_obj_centric_world_representation(self, external_instance_memory=None):
        if external_instance_memory:
            self.instance_memory = external_instance_memory
        crops = []
        for global_id, instance in self.instance_memory.instances[0].items():
            instance_crops = instance.instance_views
            crops.append((global_id, random.sample(instance_crops, 1)[0].cropped_image))
        # TODO: the model currenly can only handle 20 crops
        if len(crops) > self.max_context_length:
            print(
                "\nWarning: this version of minigpt4 can only handle limited size of crops -- sampling a subset of crops from the instance memory..."
            )
            crops = random.sample(crops, self.max_context_length)
        import shutil

        debug_path = "crops_for_planning/"
        shutil.rmtree(debug_path, ignore_errors=True)
        os.mkdir(debug_path)
        ret = []
        for id, crop in enumerate(crops):
            Image.fromarray(crop[1], "RGB").save(debug_path + str(id) + ".png")
            ret.append(str(id) + ".png")
        return ret

    def _pick(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        """Handle picking policies, either in sim or on the real robot."""
        action = None
        if self.skip_skills.pick:
            action = None
        elif self.config.AGENT.SKILLS.PICK.type == "oracle":
            pick_step = self.timesteps[0] - self.pick_start_step[0]
            if pick_step == 0:
                action = DiscreteNavigationAction.MANIPULATION_MODE
            elif pick_step == 1:
                action = DiscreteNavigationAction.SNAP_OBJECT
            elif pick_step == 2:
                action = None
            else:
                raise ValueError(
                    "Still in oracle pick. Should've transitioned to next skill."
                )
        elif self.config.AGENT.SKILLS.PICK.type == "heuristic":
            action, info = self.pick_policy(obs, info)
        elif self.config.AGENT.SKILLS.PICK.type == "hw":
            # use the hardware pick skill
            pick_step = self.timesteps[0] - self.pick_start_step[0]
            if pick_step == 0:
                action = DiscreteNavigationAction.MANIPULATION_MODE
            elif pick_step < self.max_pick_attempts:
                # If we have not seen an object mask to try to grasp...
                if not obs.task_observations["prev_grasp_success"]:
                    action = DiscreteNavigationAction.PICK_OBJECT
                else:
                    action = None
        else:
            raise NotImplementedError(
                f"pick type not supported: {self.config.AGENT.SKILLS.PICK.type}"
            )
        new_state = None
        if action in [None, DiscreteNavigationAction.STOP]:
            new_state = True
            action = None
        return action, info, new_state

    def _switch_to_next_skill(
        self, e: int, next_skill, info: Dict[str, Any]
    ) -> DiscreteNavigationAction:
        """Switch to the next skill for environment `e`.

        This function transitions to the next skill for the specified environment `e`.
        Initial setup for each skill is done here, and each skill can return a single
        action to take when starting (meant to switch between navigation and manipulation modes)
        """

        # action = None
        if type(next_skill) == bool:
            self.remaining_actions.pop(0)
            if len(self.remaining_actions) == 0:
                return True
            next_skill = self.from_pred_to_skill(self.remaining_actions[0])
        # if next_skill == Skill.NAV_TO_INSTANCE:
        #     action = DiscreteNavigationAction.NAVIGATION_MODE
        #     pass
        if next_skill == Skill.GAZE_AT_OBJ:
            self.gaze_at_obj_start_step[e] = self.timesteps[e]
        elif next_skill == Skill.PICK:
            self.pick_start_step[e] = self.timesteps[e]
        # elif next_skill == Skill.PLACE:
        #     self.place_start_step[e] = self.timesteps[e]
        # elif next_skill == Skill.FALL_WAIT:
        #     self.fall_wait_start_step[e] = self.timesteps[e]
        # import pdb
        # pdb.set_trace()
        self.states[e] = next_skill

    def from_pred_to_skill(self, pred):
        if "goto" in pred:
            return Skill.NAV_TO_INSTANCE
        if "pickup" in pred:
            return Skill.PICK
        raise NotImplementedError

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

        Image.fromarray(obs.rgb, "RGB").save(
            self.obs_path + str(self.timesteps[0]) + ".png"
        )

        self.timesteps[0] += 1
        if not self.high_level_plan:
            if self.timesteps[0] % self.vlm_freq == 0 and self.timesteps[0] != 0:
                for _ in range(self.planning_times):
                    self.world_representation = (
                        self.get_obj_centric_world_representation()
                    )  # a list of images
                    self.high_level_plan = self.ask_vlm_for_plan(
                        self.world_representation
                    )
                    # self.high_level_plan = "goto(crop_2); pickup(crop_2); goto(crop_11); goto(crop_4); goto(crop_6)"
                    if self.high_level_plan:
                        print("plan found by VLMs!!!!!!!!")
                        print(self.high_level_plan)
                        self.remaining_actions = self.high_level_plan.split("; ")
                        dry_run = input(
                            "type y if you want to switch to the next task, otherwise will execute the plan: "
                        )
                        if "y" in dry_run:
                            return None, info, obs, True
                        break

        if self.remaining_actions and len(self.remaining_actions) > 0:
            current_high_level_action = self.remaining_actions[0]
            # hard code the first action to be nav_to_instance
            if self.states[0] == Skill.EXPLORE:
                self.states = torch.tensor(
                    [Skill.NAV_TO_INSTANCE] * self.num_environments
                )

        is_finished = False
        action = None

        while action is None:
            if self.states[0] == Skill.EXPLORE:
                obs.task_observations["instance_id"] = 100000000000
                action, info, new_state = self._explore(obs, info)
            elif self.states[0] == Skill.NAV_TO_INSTANCE:
                current_high_level_action = self.remaining_actions[0]
                nav_instance_id = int(
                    self.world_representation[
                        int(
                            current_high_level_action.split("(")[1]
                            .split(")")[0]
                            .split(", ")[0]
                            .split("_")[1]
                        )
                    ].split(".")[0]
                )
                obs.task_observations["instance_id"] = nav_instance_id
                print("Navigating to instance of category: " + str(nav_instance_id))
                action, info, new_state = self._nav_to_obj(obs, info)
            elif self.states[0] == Skill.GAZE_AT_OBJ:
                current_high_level_action = self.remaining_actions[0]
                pick_instance_id = int(
                    self.world_representation[
                        int(
                            current_high_level_action.split("(")[1]
                            .split(")")[0]
                            .split(", ")[0]
                            .split("_")[1]
                        )
                    ].split(".")[0]
                )
                category_id = self.instance_memory.instance_views[0][
                    pick_instance_id
                ].category_id
                obs.task_observations["object_goal"] = category_id
                action, info, new_state = self._gaze_at_obj(obs, info)
            elif self.states[0] == Skill.PICK:
                current_high_level_action = self.remaining_actions[0]
                pick_instance_id = int(
                    self.world_representation[
                        int(
                            current_high_level_action.split("(")[1]
                            .split(")")[0]
                            .split(", ")[0]
                            .split("_")[1]
                        )
                    ].split(".")[0]
                )
                category_id = self.instance_memory.instance_views[0][
                    pick_instance_id
                ].category_id
                obs.task_observations["object_goal"] = category_id
                # import pdb; pdb.set_trace()
                action, info, new_state = self._pick(obs, info)
            # elif self.states[0] == Skill.PLACE:
            #     action, info, new_state = self._place(obs, info)
            # elif self.states[0] == Skill.FALL_WAIT:
            #     action, info, new_state = self._fall_wait(obs, info)
            else:
                raise ValueError

            # Since heuristic nav is not properly vectorized, this agent currently only supports 1 env
            # _switch_to_next_skill is thus invoked with e=0
            if new_state:
                print("Done with current skill...")
                # mark the current skill as done
                info["skill_done"] = info["curr_skill"]
                assert (
                    action is None
                ), f"action must be None when switching states, found {action} instead"
                # if len(self.remaining_actions) == 0:
                #     is_finished = True
                # else:
                self._switch_to_next_skill(0, new_state, info)
        # update the curr skill to the new skill whose action will be executed
        info["curr_skill"] = Skill(self.states[0].item()).name
        if self.verbose:
            print(
                f'Executing skill {info["curr_skill"]} at timestep {self.timesteps[0]}'
            )
        return action, info, obs, is_finished
