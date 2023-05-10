from enum import Enum
from typing import Any, Dict, List, Tuple

import torch

from home_robot.agent.objectnav_agent import ObjectNavAgent
from home_robot.agent.ovmm_agent.ppo_agent import PPOAgent
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations


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


class PickAndPlaceAgent(Agent):
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
        """

        # Flags used for skipping through state machine when debugging
        self.skip_find_object = skip_find_object
        self.skip_place = skip_place
        self.skip_orient = skip_orient
        self.skip_gaze = skip_gaze
        self.skip_pick = skip_pick
        self.config = config
        # Agent for object nav
        self.object_nav_agent = ObjectNavAgent(config, device_id)
        if not self.skip_gaze and hasattr(self.config.AGENT.SKILLS, "GAZE"):
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
        self.object_nav_agent.reset()
        if self.gaze_agent is not None:
            self.gaze_agent.reset()

    def _preprocess_obs_for_find(self, obs: Observations) -> Observations:
        task_info = obs.task_observations
        # Recep goal is unused by our object nav policies
        obs.task_observations["recep_goal"] = None
        obs.task_observations["start_recep_goal"] = task_info["start_recep_id"]
        obs.task_observations["object_goal"] = task_info["object_id"]
        obs.task_observations["goal_name"] = task_info["object_name"]
        return obs

    def _preprocess_obs_for_place(self, obs: Observations) -> Observations:
        task_info = obs.task_observations
        # Receptacle goal used for placement
        obs.task_observations["end_recep_goal"] = task_info["place_recep_id"]
        # Start receptacle goal unused
        obs.task_observations["start_recep_goal"] = None
        # Object goal unused - we already presumably have it in our hands
        obs.task_observations["object_goal"] = None
        obs.task_observations["goal_name"] = task_info["place_recep_name"]
        return obs

    def act(self, obs: Observations) -> Tuple[Action, Dict[str, Any]]:
        """
        Act end-to-end. Checks the current internal task state; will call the appropriate agent.

        Arguments:
            obs: home_robot observation object containing sensor measurements.

        Returns:
            action: home_robot action
            info: additional information (e.g., for debugging, visualization)
        """

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
                return DiscreteNavigationAction.MANIPULATION_MODE, action_info
        if self.state == SimpleTaskState.GAZE_OBJECT:
            if self.skip_gaze or self.gaze_agent is None:
                self.state = SimpleTaskState.PICK_OBJECT
            else:
                # act using the policy predictions until termination condition is hit
                action, does_want_terminate = self.gaze_agent.act(obs)
                if does_want_terminate:
                    self.state = SimpleTaskState.PICK_OBJECT
                return action, None
        if self.state == SimpleTaskState.PICK_OBJECT:
            # Try to grab the object
            if not self.skip_place:
                self.state = SimpleTaskState.ORIENT_NAV
            return DiscreteNavigationAction.PICK_OBJECT, action_info
        if self.state == SimpleTaskState.ORIENT_NAV:
            self.state = SimpleTaskState.FIND_GOAL
            return DiscreteNavigationAction.NAVIGATION_MODE, action_info
        elif self.state == SimpleTaskState.FIND_GOAL:
            # Find the goal location
            obs = self._preprocess_obs_for_place(obs)
            action, action_info = self.object_nav_agent.act(obs)
        # If we did not find anything else to do, just stop
        return action, action_info
