# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from datetime import datetime
from enum import IntEnum, auto
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.manipulation import HeuristicPickPolicy, HeuristicPlacePolicy
from home_robot.perception.constants import RearrangeBasicCategories
from home_robot.perception.wrapper import (
    OvmmPerception,
    build_vocab_from_category_map,
    read_category_map_file,
)


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


class SemanticVocab(IntEnum):
    FULL = auto()
    SIMPLE = auto()
    ALL = auto()


def get_skill_as_one_hot_dict(curr_skill: Skill):
    skill_dict = {f"is_curr_skill_{skill.name}": 0 for skill in Skill}
    skill_dict[f"is_curr_skill_{Skill(curr_skill).name}"] = 1
    return skill_dict


class OpenVocabManipAgent(ObjectNavAgent):
    """Simple object nav agent based on a 2D semantic map."""

    def __init__(self, config, device_id: int = 0):
        super().__init__(config, device_id=device_id)
        self.states = None
        self.place_start_step = None
        self.pick_start_step = None
        self.gaze_at_obj_start_step = None
        self.fall_wait_start_step = None
        self.is_gaze_done = None
        self.place_done = None
        self.gaze_agent = None
        self.nav_to_obj_agent = None
        self.nav_to_rec_agent = None
        self.pick_agent = None
        self.place_agent = None
        self.pick_policy = None
        self.place_policy = None
        self.semantic_sensor = None

        if config.GROUND_TRUTH_SEMANTICS == 1 and not self.store_all_categories_in_map:
            # currently we get ground truth semantics of only the target object category and all scene receptacles from the simulator
            raise NotImplementedError

        self.skip_skills = config.AGENT.skip_skills
        self.max_pick_attempts = 10
        if config.GROUND_TRUTH_SEMANTICS == 0:
            self.semantic_sensor = OvmmPerception(config, device_id, self.verbose)
            self.obj_name_to_id, self.rec_name_to_id = read_category_map_file(
                config.ENVIRONMENT.category_map_file
            )
        if config.AGENT.SKILLS.PICK.type == "heuristic" and not self.skip_skills.pick:
            self.pick_policy = HeuristicPickPolicy(
                config, self.device, verbose=self.verbose
            )
        if config.AGENT.SKILLS.PLACE.type == "heuristic" and not self.skip_skills.place:
            self.place_policy = HeuristicPlacePolicy(
                config, self.device, verbose=self.verbose
            )
        elif config.AGENT.SKILLS.PLACE.type == "rl" and not self.skip_skills.place:
            from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent

            self.place_agent = PPOAgent(
                config,
                config.AGENT.SKILLS.PLACE,
                device_id=device_id,
            )
        skip_both_gaze = self.skip_skills.gaze_at_obj and self.skip_skills.gaze_at_rec
        if config.AGENT.SKILLS.GAZE_OBJ.type == "rl" and not skip_both_gaze:
            from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent

            self.gaze_agent = PPOAgent(
                config,
                config.AGENT.SKILLS.GAZE_OBJ,
                device_id=device_id,
            )
        if (
            config.AGENT.SKILLS.NAV_TO_OBJ.type == "rl"
            and not self.skip_skills.nav_to_obj
        ):
            from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent

            self.nav_to_obj_agent = PPOAgent(
                config,
                config.AGENT.SKILLS.NAV_TO_OBJ,
                device_id=device_id,
            )
        if (
            config.AGENT.SKILLS.NAV_TO_REC.type == "rl"
            and not self.skip_skills.nav_to_rec
        ):
            from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent

            self.nav_to_rec_agent = PPOAgent(
                config,
                config.AGENT.SKILLS.NAV_TO_REC,
                device_id=device_id,
            )
        self._fall_wait_steps = getattr(config.AGENT, "fall_wait_steps", 0)
        self.config = config

    def _get_info(self, obs: Observations) -> Dict[str, torch.Tensor]:
        """Get inputs for visual skill."""
        use_detic_viz = self.config.ENVIRONMENT.use_detic_viz

        if self.config.GROUND_TRUTH_SEMANTICS == 1 or use_detic_viz:
            semantic_category_mapping = None  # Visualizer handles mapping
        elif self.semantic_sensor.current_vocabulary_id == SemanticVocab.SIMPLE:
            semantic_category_mapping = RearrangeBasicCategories()
        else:
            semantic_category_mapping = self.semantic_sensor.current_vocabulary

        if use_detic_viz:
            semantic_frame = obs.task_observations["semantic_frame"]
        else:
            semantic_frame = np.concatenate(
                [obs.rgb, obs.semantic[:, :, np.newaxis]], axis=2
            ).astype(np.uint8)

        info = {
            "semantic_frame": semantic_frame,
            "semantic_category_mapping": semantic_category_mapping,
            "goal_name": obs.task_observations["goal_name"],
            "third_person_image": obs.third_person_image,
            "timestep": self.timesteps[0],
            "curr_skill": Skill(self.states[0].item()).name,
            "skill_done": "",  # Set if skill gets done
        }
        # only the current skill has corresponding value as 1
        info = {**info, **get_skill_as_one_hot_dict(self.states[0].item())}
        return info

    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()

    def reset_vectorized(self):
        """Initialize agent state."""
        super().reset_vectorized()

        if self.gaze_agent is not None:
            self.gaze_agent.reset_vectorized()
        if self.nav_to_obj_agent is not None:
            self.nav_to_obj_agent.reset_vectorized()
        if self.place_agent is not None:
            self.place_agent.reset_vectorized()
        if self.nav_to_rec_agent is not None:
            self.nav_to_rec_agent.reset_vectorized()
        self.states = torch.tensor([Skill.NAV_TO_OBJ] * self.num_environments)
        self.pick_start_step = torch.tensor([0] * self.num_environments)
        self.gaze_at_obj_start_step = torch.tensor([0] * self.num_environments)
        self.place_start_step = torch.tensor([0] * self.num_environments)
        self.gaze_at_obj_start_step = torch.tensor([0] * self.num_environments)
        self.fall_wait_start_step = torch.tensor([0] * self.num_environments)
        self.is_gaze_done = torch.tensor([0] * self.num_environments)
        self.place_done = torch.tensor([0] * self.num_environments)
        if self.place_policy is not None:
            self.place_policy.reset()
        if self.pick_policy is not None:
            self.pick_policy.reset()

    def get_nav_to_recep(self):
        return (self.states == Skill.NAV_TO_REC).float().to(device=self.device)

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.states[e] = Skill.NAV_TO_OBJ
        self.place_start_step[e] = 0
        self.pick_start_step[e] = 0
        self.gaze_at_obj_start_step[e] = 0
        self.fall_wait_start_step[e] = 0
        self.is_gaze_done[e] = 0
        self.place_done[e] = 0
        if self.place_policy is not None:
            self.place_policy.reset()
        if self.pick_policy is not None:
            self.pick_policy.reset()
        super().reset_vectorized_for_env(e)
        if self.gaze_agent is not None:
            self.gaze_agent.reset_vectorized_for_env(e)
        if self.nav_to_obj_agent is not None:
            self.nav_to_obj_agent.reset_vectorized_for_env(e)
        if self.place_agent is not None:
            self.place_agent.reset_vectorized_for_env(e)
        if self.nav_to_rec_agent is not None:
            self.nav_to_rec_agent.reset_vectorized_for_env(e)

    def _init_episode(self, obs: Observations):
        """
        This method is called at the first timestep of every episode before any action is taken.
        """
        if self.verbose:
            print("Initializing episode...")
        if self.config.GROUND_TRUTH_SEMANTICS == 0:
            self._update_semantic_vocabs(obs)
            if self.store_all_categories_in_map:
                self._set_semantic_vocab(SemanticVocab.ALL, force_set=True)
            elif (
                self.config.AGENT.SKILLS.NAV_TO_OBJ.type == "rl"
                and not self.skip_skills.nav_to_obj
            ):
                self._set_semantic_vocab(SemanticVocab.FULL, force_set=True)
            else:
                self._set_semantic_vocab(SemanticVocab.SIMPLE, force_set=True)

    def _switch_to_next_skill(
        self, e: int, next_skill: Skill, info: Dict[str, Any]
    ) -> DiscreteNavigationAction:
        """Switch to the next skill for environment `e`.

        This function transitions to the next skill for the specified environment `e`.
        Initial setup for each skill is done here, and each skill can return a single
        action to take when starting (meant to switch between navigation and manipulation modes)
        """

        action = None
        if next_skill == Skill.NAV_TO_OBJ:
            # action = DiscreteNavigationAction.NAVIGATION_MODE
            pass
        elif next_skill == Skill.GAZE_AT_OBJ:
            if not self.store_all_categories_in_map:
                self._set_semantic_vocab(SemanticVocab.SIMPLE, force_set=False)
            self.gaze_at_obj_start_step[e] = self.timesteps[e]
        elif next_skill == Skill.PICK:
            self.pick_start_step[e] = self.timesteps[e]
        elif next_skill == Skill.NAV_TO_REC:
            self.timesteps_before_goal_update[e] = 0
            if not self.skip_skills.nav_to_rec:
                action = DiscreteNavigationAction.NAVIGATION_MODE
                if (
                    self.config.AGENT.SKILLS.NAV_TO_OBJ.type == "rl"
                    and not self.store_all_categories_in_map
                ):
                    self._set_semantic_vocab(SemanticVocab.FULL, force_set=False)
        elif next_skill == Skill.GAZE_AT_REC:
            if not self.store_all_categories_in_map:
                self._set_semantic_vocab(SemanticVocab.SIMPLE, force_set=False)
            # We reuse gaze agent between pick and place
            if self.gaze_agent is not None:
                self.gaze_agent.reset_vectorized_for_env(e)
        elif next_skill == Skill.PLACE:
            self.place_start_step[e] = self.timesteps[e]
        elif next_skill == Skill.FALL_WAIT:
            self.fall_wait_start_step[e] = self.timesteps[e]
        self.states[e] = next_skill
        return action

    def _update_semantic_vocabs(
        self, obs: Observations, update_full_vocabulary: bool = True
    ):
        """
        Sets vocabularies for semantic sensor at the start of episode.
        Optional-
        :update_full_vocabulary: if False, only updates simple vocabulary
        True by default
        """
        obj_id_to_name = {
            0: obs.task_observations["object_name"],
        }
        simple_rec_id_to_name = {
            0: obs.task_observations["start_recep_name"],
            1: obs.task_observations["place_recep_name"],
        }

        # Simple vocabulary contains only object and necessary receptacles
        simple_vocab = build_vocab_from_category_map(
            obj_id_to_name, simple_rec_id_to_name
        )
        self.semantic_sensor.update_vocabulary_list(simple_vocab, SemanticVocab.SIMPLE)

        if update_full_vocabulary:
            # Full vocabulary contains the object and all receptacles
            full_vocab = build_vocab_from_category_map(
                obj_id_to_name, self.rec_name_to_id
            )
            self.semantic_sensor.update_vocabulary_list(full_vocab, SemanticVocab.FULL)

        # All vocabulary contains all objects and all receptacles
        all_vocab = build_vocab_from_category_map(
            self.obj_name_to_id, self.rec_name_to_id
        )
        self.semantic_sensor.update_vocabulary_list(all_vocab, SemanticVocab.ALL)

    def _set_semantic_vocab(self, vocab_id: SemanticVocab, force_set: bool):
        """
        Set active vocabulary for semantic sensor to use to the given ID.
        """
        # import pdb; pdb.set_trace()
        if self.config.GROUND_TRUTH_SEMANTICS == 0 and (
            force_set or self.semantic_sensor.current_vocabulary_id != vocab_id
        ):
            self.semantic_sensor.set_vocabulary(vocab_id)

    def _heuristic_nav(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any]:
        action, planner_info = super().act(obs)
        # info overwrites planner_info entries for keys with same name
        info = {**planner_info, **info}
        self.timesteps[0] -= 1  # objectnav agent increments timestep
        info["timestep"] = self.timesteps[0]
        if action == DiscreteNavigationAction.STOP:
            terminate = True
        else:
            terminate = False
        return action, info, terminate

    def _heuristic_pick(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any]:
        """Heuristic pick skill execution"""
        raise NotImplementedError  # WIP
        action, info = self.pick_agent(obs, info)
        if action == DiscreteNavigationAction.STOP:
            action = DiscreteNavigationAction.NAVIGATION_MODE
            info = self._switch_to_next_skill(e=0, info=info)
        return action, info

    def _hardcoded_place(self):
        """Hardcoded place skill execution
        Orients the agent's arm and camera towards the recetacle, extends arm and releases the object
        """
        place_step = self.timesteps[0] - self.place_start_step[0]
        forward_steps = 0
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
        elif place_step <= forward_steps + 3:
            # allow the object to come to rest
            action = DiscreteNavigationAction.STOP
        else:
            raise ValueError(
                f"Something is wrong. Episode should have ended. Place step: {place_step}, Timestep: {self.timesteps[0]}"
            )
        return action

    def _rl_place(self, obs: Observations, info: Dict[str, Any]):
        place_step = self.timesteps[0] - self.place_start_step[0]
        if place_step == 0:
            action = DiscreteNavigationAction.POST_NAV_MODE
        elif self.place_done[0] == 1:
            action = DiscreteNavigationAction.STOP
            self.place_done[0] = 0
        else:
            action, info, terminate = self.place_agent.act(obs, info)
            if terminate:
                action = DiscreteNavigationAction.DESNAP_OBJECT
                self.place_done[0] = 1
        return action, info

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
            new_state = Skill.GAZE_AT_OBJ
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
            new_state = Skill.PICK
        return action, info, new_state

    def _pick(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        """Handle picking policies, either in sim or on the real robot."""
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
            new_state = Skill.NAV_TO_REC
            action = None
        return action, info, new_state

    def _nav_to_rec(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        nav_to_rec_type = self.config.AGENT.SKILLS.NAV_TO_REC.type
        if self.skip_skills.nav_to_rec:
            terminate = True
        elif nav_to_rec_type == "heuristic":
            action, info, terminate = self._heuristic_nav(obs, info)
        elif nav_to_rec_type == "rl":
            action, info, terminate = self.nav_to_rec_agent.act(obs, info)
        else:
            raise ValueError(
                f"Got unexpected value for NAV_TO_REC.type: {nav_to_rec_type}"
            )
        new_state = None
        if terminate:
            action = None
            new_state = Skill.GAZE_AT_REC
        return action, info, new_state

    def _gaze_at_rec(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        if self.skip_skills.gaze_at_rec:
            terminate = True
        else:
            action, info, terminate = self.gaze_agent.act(obs, info)
        new_state = None
        if terminate:
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
        elif place_type == "heuristic":
            action, info = self.place_policy(obs, info)
        elif place_type == "rl":
            action, info = self._rl_place(obs, info)
        else:
            raise ValueError(f"Got unexpected value for PLACE.type: {place_type}")
        new_state = None
        if action == DiscreteNavigationAction.STOP:
            action = None
            new_state = Skill.FALL_WAIT
        return action, info, new_state

    def _fall_wait(
        self, obs: Observations, info: Dict[str, Any]
    ) -> Tuple[DiscreteNavigationAction, Any, Optional[Skill]]:
        if self.timesteps[0] - self.fall_wait_start_step[0] < self._fall_wait_steps:
            action = DiscreteNavigationAction.EMPTY_ACTION
        else:
            action = DiscreteNavigationAction.STOP
        return action, info, None

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
