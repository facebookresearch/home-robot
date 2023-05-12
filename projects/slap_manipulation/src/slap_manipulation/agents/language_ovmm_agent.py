from typing import List

from home_robot.agent.hierarchical.pick_and_place_agent import (
    PickAndPlaceAgent,
    SimpleTaskState,
)
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations


class LangAgent(PickAndPlaceAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.steps = []
        self.state = SimpleTaskState.NOT_STARTED
        self.current_step = ""

    def get_steps(self, task: str) -> List[str]:
        """takes in a task string and returns a list of steps to complete the task"""
        raise NotImplementedError

    def get_actions(self, current_step: str) -> List[str]:
        """takes in current step in natural language and returns list of actions required to complete it"""
        raise NotImplementedError

    def get_ovmm_action_for_current_step(self):
        action = []
        info = {}
        if "locate" in self.current_step or "goto" in self.current_step:
            # figure out how to pass a navigation command with info
            info["skill_language"] = self.current_step
        if "pick" in self.current_step:
            # figure out how to  pass a pick_object action with info
            info["action_list"] = self.get_actions(self.current_step)
        return action, info

    def is_done(self):
        if len(self.steps) == 0 and self.state != SimpleTaskState.NOT_STARTED:
            return True
        else:
            return False

    def act(self, obs: Observations, task: str):
        if self.state == SimpleTaskState.NOT_STARTED and len(self.steps) == 0:
            self.steps = self.get_steps(task)
        self.current_step = self.steps.pop(0)
        action, info = self.get_ovmm_action_for_current_step()
        return action, info
