# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum
from typing import Any, Dict, List, Tuple

import torch

from home_robot.agent.objectnav_agent import ObjectNavAgent
from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent, SemanticVocab
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
from home_robot.manipulation import HeuristicPlacePolicy
from home_robot.perception.wrapper import (
    OvmmPerception,
    build_vocab_from_category_map,
    read_category_map_file,
)


class SimpleTaskState(Enum):
    """Track task state."""

    NOT_STARTED = 0
    FIND_OBJECT = 1
    ORIENT_OBJ = 2
    GAZE_OBJECT = 3
    PICK_OBJECT = 4
    ORIENT_NAV = 5
    FIND_GOAL = 6
    ORIENT_PLACE = 7
    PLACE_OBJECT = 8
    DONE = 9


class PickAndPlaceAgent(OpenVocabManipAgent):
    """Create a simple version of a pick and place agent which uses a 2D semantic map to find
    objects and try to grasp them."""

    # For debugging
    # Force the robot to jump right to an attempt to pick objects

    def __init__(
        self,
        config,
        device_id: int = 0,
        skip_find_object=False,
        skip_place=False,
        skip_orient=False,
        skip_pick=False,
        skip_gaze=False,
        test_place=False,
        skip_orient_place=True,
        min_distance_goal_cm: float = 50.0,
        continuous_angle_tolerance: float = 30.0,
    ):
        """Create the component object nav agent as a PickAndPlaceAgent object.

        Args:
            config: A configuration object containing various parameters.
            device_id (int, optional): The ID of the device to use. Defaults to 0.
            skip_find_object (bool, optional): Whether to skip the exploration and navigation step. Useful for debugging. Defaults to False.
            skip_place (bool, optional): Whether to skip the object-placement step. Useful for debugging. Defaults to False.
            skip_orient (bool, optional): Whether to skip orientating towards the objects. Useful for debugging. Defaults to False.
            skip_pick (bool, optional): Whether to skip the object-pickup step. Useful for debugging. Defaults to False.
            skip_gaze (bool, optional): Whether to skip the gaze step. Useful for debugging. Defaults to False.
            test_place (bool, optional): go directly to finding and placing
            skip_orient_place (bool, optional): skip orienting in manipulation mode before placing
        """

        # Flags used for skipping through state machine when debugging
        self.device = device_id
        self.skip_find_object = skip_find_object
        self.skip_place = skip_place
        self.skip_orient = skip_orient
        self.skip_gaze = skip_gaze
        self.skip_pick = skip_pick
        self.test_place = test_place
        self.skip_orient_place = skip_orient_place
        self.config = config
        self.timestep = 0
        self.semantic_sensor = OvmmPerception(config, device_id)
        self.obj_name_to_id, self.rec_name_to_id = read_category_map_file(
            config.ENVIRONMENT.category_map_file
        )

        # Create place policy
        if not self.skip_place:
            self.place_policy = HeuristicPlacePolicy(self.config, self.device)

        # Agent for object nav
        self.object_nav_agent = ObjectNavAgent(
            config,
            device_id,
            min_goal_distance_cm=min_distance_goal_cm,
            continuous_angle_tolerance=continuous_angle_tolerance,
        )
        if not self.skip_gaze and hasattr(self.config.AGENT.SKILLS, "GAZE"):
            from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent

            self.gaze_agent = PPOAgent(
                config,
                config.AGENT.SKILLS.GAZE,
                device_id=device_id,
            )
        else:
            self.gaze_agent = None

        self.reset()

    def _get_vis_inputs(self, obs: Observations) -> Dict[str, torch.Tensor]:
        return {
            "semantic_frame": obs.task_observations["semantic_frame"],
            "goal_name": obs.task_observations["goal_name"],
            "third_person_image": obs.third_person_image,
            "found_goal": False,
        }

    def reset(self):
        """Clear internal task state and reset component agents."""
        self.state = SimpleTaskState.FIND_OBJECT
        if self.test_place:
            # TODO: remove debugging code
            # If we want to find the goal first...
            # self.state = SimpleTaskState.FIND_GOAL
            # If we just want to place...
            self.state = SimpleTaskState.PLACE_OBJECT
        self.object_nav_agent.reset()
        if self.gaze_agent is not None:
            self.gaze_agent.reset()
        self.timestep = 0

    def _preprocess_obs_for_find(self, obs: Observations) -> Observations:
        task_info = obs.task_observations
        # Recep goal is unused by our object nav policies
        obs.task_observations["recep_goal"] = None
        obs.task_observations["start_recep_goal"] = task_info["start_recep_goal"]
        obs.task_observations["object_goal"] = task_info["object_goal"]
        obs.task_observations["goal_name"] = task_info["object_name"]
        return obs

    def _preprocess_obs_for_place(
        self, obs: Observations, info: Dict
    ) -> Tuple[Observations, Dict]:
        """Process information we need for the place skills."""
        task_info = obs.task_observations
        # Receptacle goal used for placement
        obs.task_observations["end_recep_goal"] = task_info["end_recep_goal"]
        # Start receptacle goal unused
        obs.task_observations["start_recep_goal"] = None
        # Object goal unused - we already presumably have it in our hands
        obs.task_observations["object_goal"] = None
        obs.task_observations["goal_name"] = task_info["place_recep_name"]
        info["goal_name"] = obs.task_observations["goal_name"]
        return obs, info

    def _get_info(self, obs: Observations) -> Dict:
        """Get inputs for visual skill."""
        info = {
            "semantic_frame": obs.task_observations["semantic_frame"],
            "goal_name": obs.task_observations["goal_name"],
            "curr_skill": str(self.state),
            "skill_done": "",  # Set if skill gets done
            "timestep": self.timestep,
        }
        return info

    def act(self, obs: Observations) -> Tuple[Action, Dict[str, Any]]:
        """
        Act end-to-end. Checks the current internal task state; will call the appropriate agent.

        Arguments:
            obs: home_robot observation object containing sensor measurements.

        Returns:
            action: home_robot action
            info: additional information (e.g., for debugging, visualization)
        """
        if self.timestep == 0:
            self._update_semantic_vocabs(obs)
            self._set_semantic_vocab(SemanticVocab.SIMPLE, force_set=True)
        obs = self.semantic_sensor(obs)

        info = self._get_info(obs)
        self.timestep += 1  # Update step counter for visualizations
        action = DiscreteNavigationAction.STOP
        action_info = None
        # Look for the goal object.
        if self.state == SimpleTaskState.FIND_OBJECT:
            if self.skip_find_object:
                # transition to the next state
                action = DiscreteNavigationAction.STOP
                self.state = SimpleTaskState.ORIENT_OBJ
            else:
                obs = self._preprocess_obs_for_find(obs)
                action, action_info = self.object_nav_agent.act(obs)
                if action == DiscreteNavigationAction.STOP:
                    self.state = SimpleTaskState.ORIENT_OBJ
        # If we have found the object, then try to pick it up.
        if self.state == SimpleTaskState.ORIENT_OBJ:
            self.state = SimpleTaskState.GAZE_OBJECT
            if not self.skip_orient:
                # orient to face the object
                return (
                    DiscreteNavigationAction.MANIPULATION_MODE,
                    action_info,
                    obs,
                )
        if self.state == SimpleTaskState.GAZE_OBJECT:
            if self.skip_gaze or self.gaze_agent is None:
                self.state = SimpleTaskState.PICK_OBJECT
            else:
                # act using the policy predictions until termination condition is hit
                action, does_want_terminate = self.gaze_agent.act(obs)
                if does_want_terminate:
                    self.state = SimpleTaskState.PICK_OBJECT
                return action, None, obs
        if self.state == SimpleTaskState.PICK_OBJECT:
            # Try to grab the object
            if obs.task_observations["prev_grasp_success"]:
                print("[Agent] Attempted a grasp. Moving on...")
                if not self.skip_place:
                    self.state = SimpleTaskState.ORIENT_NAV
                else:
                    print("[Agent] staying in grasp state to continue testing.")
            return DiscreteNavigationAction.PICK_OBJECT, action_info, obs
        if self.state == SimpleTaskState.ORIENT_NAV:
            self.state = SimpleTaskState.FIND_GOAL
            return DiscreteNavigationAction.NAVIGATION_MODE, action_info, obs
        if self.state == SimpleTaskState.FIND_GOAL:
            # Find the goal location
            obs, info = self._preprocess_obs_for_place(obs, info)
            action, action_info = self.object_nav_agent.act(obs)
            if action == DiscreteNavigationAction.STOP:
                self.state = SimpleTaskState.PLACE_OBJECT
        if self.state == SimpleTaskState.ORIENT_PLACE:
            # TODO: this is not currently used
            self.state = SimpleTaskState.PLACE_OBJECT
            if not self.skip_orient_place:
                # orient to face the object
                return (
                    DiscreteNavigationAction.MANIPULATION_MODE,
                    action_info,
                    obs,
                )
        if self.state == SimpleTaskState.PLACE_OBJECT:
            # place the object somewhere - hopefully in front of the agent.
            obs, info = self._preprocess_obs_for_place(obs, info)
            # action, action_info = self.place_agent.act(obs)
            action, action_info = self.place_policy.forward(obs, info)
            if action == DiscreteNavigationAction.STOP:
                self.state = SimpleTaskState.DONE
        if self.state == SimpleTaskState.DONE:
            # We're done - just stop execution entirely.
            action = DiscreteNavigationAction.STOP
            action_info = info

        # If we did not find anything else to do, just stop
        return action, action_info, obs
