from typing import Any, Dict, List, Tuple

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

    def is_busy(self) -> bool:
        return self.state != SimpleTaskState.IDLE

    def get_steps(self, task: str) -> List[str]:
        """takes in a task string and returns a list of steps to complete the task"""
        raise NotImplementedError

    def get_actions(self, obs: Observations) -> Tuple[Action, Dict[str, Any]]:
        """takes in current step in natural language and returns list of actions required to complete it"""
        info = {}
        if self.state == SimpleTaskState.FIND_OBJECT:
            info["complex_action"] = True
            info["action_length"] = 3
            info["skill_language"] = self.current_step
            if self.skip_find_object:
                # transition to the next state
                action = DiscreteNavigationAction.STOP
                self.state = SimpleTaskState.ORIENT_OBJ
            else:
                obs = self._preprocess_obs_for_find(obs)
                action, info = self.object_nav_agent.act(obs)
                if action == DiscreteNavigationAction.STOP:
                    self.state = SimpleTaskState.ORIENT_OBJ
        if self.state == SimpleTaskState.ORIENT_OBJ:
            self.state = SimpleTaskState.GAZE_OBJECT
            if not self.skip_orient:
                # orient to face the object
                return DiscreteNavigationAction.MANIPULATION_MODE, info
        # If we have found the object, then try to pick it up.
        if self.state == SimpleTaskState.GAZE_OBJECT:
            if self.skip_gaze or self.gaze_agent is None:
                self.state = SimpleTaskState.IDLE
            else:
                # act using the policy predictions until termination condition is hit
                action, does_want_terminate = self.gaze_agent.act(obs)
                if does_want_terminate:
                    self.state = SimpleTaskState.IDLE
        if self.state == SimpleTaskState.PICK_OBJECT:
            pass
        if self.state == SimpleTaskState.PLACE_OBJECT:
            pass
        # calling SLAP will be yet another hard-coded if-else-then here
        # return value here would be a ContinuousManipulationMode (or equivalent)
        # and info["action_list"] = List[(xyz, quat, g)]
        return action, info

    def get_ovmm_action_for_current_step(self, obs: Observations):
        action = []
        info = {}
        if "locate" in self.current_step:
            if self.state == SimpleTaskState.IDLE:
                self.state = SimpleTaskState.FIND_OBJECT
            action, info = self.get_actions(obs)
        if "pick" in self.current_step:
            if self.state == SimpleTaskState.IDLE:
                self.state = SimpleTaskState.PICK_OBJECT
            action, info = self.get_actions(obs)
        if "place" in self.current_step:
            if self.state == SimpleTaskState.IDLE:
                self.state = SimpleTaskState.PLACE_OBJECT
            action, info = self.get_actions(obs)
        # for SLAP version this will be where we infer using SLAP open-loop
        return action, info

    def is_done(self):
        if len(self.steps) == 0 and self.state != SimpleTaskState.NOT_STARTED:
            return True
        else:
            return False

    def act(self, obs: Observations, task: str) -> Tuple[Action, Dict[str, Any]]:
        if self.state == SimpleTaskState.NOT_STARTED and len(self.steps) == 0:
            self.steps = self.get_steps(task)
        self.current_step = self.steps.pop(0)
        action, info = self.get_ovmm_action_for_current_step(obs)
        return action, info
