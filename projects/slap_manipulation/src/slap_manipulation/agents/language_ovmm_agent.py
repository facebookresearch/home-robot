from home_robot.agent.hierarchical.pick_and_place_agent import (
    PickAndPlaceAgent,
    SimpleTaskState,
)
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations


class LangAgent(PickAndPlaceAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.state = SimpleTaskState.NOT_STARTED

    def get_ovmm_action_for_current_step(self):
        action = None
        info = {}
        if "locate" in self.current_step or "goto" in self.current_step:
            # figure out how to pass a navigation command with info
            return action, info
        if "pick" in self.current_step:
            # figure out how to  pass a pick_object action with info
            return action, info

    def is_done(self):
        if len(self.steps == 0 and self.state != SimpleTaskState.NOT_STARTED):
            return True
        else:
            return False

    def act(self, obs: Observations, task: str):
        if self.state == SimpleTaskState.NOT_STARTED and len(self.steps) == 0:
            self.steps = self.get_steps(task)
        self.current_step = self.steps.pop(0)
        action, info = self.get_ovmm_action_for_current_step()
        return action, info
