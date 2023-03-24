from enum import Enum
from typing import Any, Dict, List, Tuple

from home_robot.agent.objectnav_agent import ObjectNavAgent
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations


class SimpleTaskState(Enum):
    """Track task state."""

    NOT_STARTED = 0
    FIND_OBJECT = 1
    ORIENT_OBJ = 2
    PICK_OBJECT = 3
    ORIENT_NAV = 4
    FIND_GOAL = 5
    ORIENT_PLACE = 6
    PLACE_OBJECT = 7


class PickAndPlaceAgent(Agent):
    """Create a simple version of a pick and place agent which uses a 2D semantic map to find
    objects and try to grasp them."""

    # For debugging
    # Force the robot to jump right to an attempt to pick objects

    def __init__(
        self, config, device_id: int = 0, skip_find_object=False, skip_place=False
    ):
        """Create the component object nav agent"""

        # Flags used for skipping through state machine when debugging
        self.skip_find_object = skip_find_object
        self.skip_place = skip_place

        # Agent for object nav
        self.object_nav_agent = ObjectNavAgent(config, device_id)
        self.reset()

    def reset(self):
        """Clear internal task state and reset component agents."""
        self.state = SimpleTaskState.FIND_OBJECT
        self.object_nav_agent.reset()

    def _preprocess_obs_for_find(self, obs: Observations) -> Observations:
        task_info = obs.task_observations
        obs.task_observations["recep_goal"] = task_info["start_recep_id"]
        obs.task_observations["object_goal"] = task_info["object_id"]
        obs.task_observations["goal_name"] = task_info["object_name"]
        return obs

    def _preprocess_obs_for_place(self, obs: Observations) -> Observations:
        task_info = obs.task_observations
        obs.task_observations["recep_goal"] = task_info["place_recep_id"]
        obs.task_observations["object_goal"] = None
        obs.task_observations["goal_name"] = task_info["place_recep_name"]
        return obs

    def act(self, obs: Observations) -> Tuple[Action, Dict[str, Any]]:
        """
        Act end-to-end. Checks the current internal task state; will call the appropriate agent.

        Arguments:
            obs: home_robot observation

        Returns:
            action: home_robot action
            info: additional information (e.g., for debugging, visualization)
        """
        action = DiscreteNavigationAction.STOP
        action_info = None
        # Look for the goal object.
        if self.state == SimpleTaskState.FIND_OBJECT:
            obs = self._preprocess_obs_for_find(obs)
            action, action_info = self.object_nav_agent.act(obs)
            if self.skip_find_object:
                print("-> Actually predicted:", action)
                action = DiscreteNavigationAction.STOP
                self.state = SimpleTaskState.PICK_OBJECT
            elif action == DiscreteNavigationAction.STOP:
                self.state = SimpleTaskState.ORIENT_OBJ
        # If we have found the object, then try to pick it up.
        if self.state == SimpleTaskState.ORIENT_OBJ:
            # Try to grab the object.
            # If we grasped the object, then we should increment our state again
            self.state = SimpleTaskState.PICK_OBJECT
            return DiscreteNavigationAction.MANIPULATION_MODE, action_info
        if self.state == SimpleTaskState.PICK_OBJECT:
            # Try to grab the object
            if not self.skip_place:
                self.state = SimpleTaskState.ORIENT_NAV
            return DiscreteNavigationAction.PICK_OBJECT, action_info
        elif self.state == SimpleTaskState.ORIENT_NAV:
            return DiscreteNavigationAction.NAVIGATION_MODE, action_info
        elif self.state == SimpleTaskState.FIND_GOAL:
            # Find the goal location
            obs = self._preprocess_obs_for_place(obs)
            action, action_info = self.object_nav_agent.act(obs)
        # If we did not find anything else to do, just stop
        return action, action_info
