from enum import Enum
from typing import Any, Dict, List, Tuple

from home_robot.agent.objectnav_agent import ObjectNavAgent
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations


class SimpleTaskState(Enum):
    """Track task state."""

    NOT_STARTED = 0
    FIND_OBJECT = 1
    PICK_OBJECT = 2
    FIND_GOAL = 3
    PLACE_OBJECT = 4


class PickAndPlaceAgent(Agent):
    """Create a simple version of a pick and place agent which uses a 2D semantic map to find
    objects and try to grasp them."""

    def __init__(self, config, device_id: int = 0):
        """Create the component object nav agent"""
        self.object_nav_agent = ObjectNavAgent(config, device_id)
        self.reset()

    def reset(self):
        """Clear internal task state and reset component agents."""
        self.state = SimpleTaskState.FIND_OBJECT
        self.object_nav_agent.reset()

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
            action, action_info = self.object_nav_agent.act(obs)
            if action == DiscreteNavigationAction.STOP:
                self.state = SimpleTaskState.PICK_OBJECT
        # If we have found the object, then try to pick it up.
        if self.state == SimpleTaskState.PICK_OBJECT:
            # Try to grab the object.
            # If we grasped the object, then we should increment our state again
            return DiscreteNavigationAction.PICK_OBJECT, action_info
        # If we did not find anything else to do, just stop
        return action, action_info
