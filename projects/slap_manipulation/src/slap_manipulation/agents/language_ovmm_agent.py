# %%
from typing import Any, Dict, List, Tuple
from glob import glob
import pandas as pd
from home_robot.agent.hierarchical.pick_and_place_agent import (
    PickAndPlaceAgent,
    SimpleTaskState,
)
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations
# %%
def get_task_plans_from_gt(index, datafile='../../../datasets/BringXFromYSurfaceToHuman.json', root='../datasets/'):
    """Reads the dataset files and return a list of task plans"""
    if datafile == 'all':
        files = glob(root+'*.json')
        dflist = []
        for file in files:
            dflist.append(pd.read_json(file))    
        df = pd.concat(dflist)
    else:
        df = pd.read_json(datafile)
    assert index < len(df), f"Index {index} is out of range"
    steps_list = df.iloc[index]['steps']
    # steps_df = pd.DataFrame.from_records(steps_list)
    code  = get_codelist(steps_list)
    return code
    
def get_codelist(steps_list):
    codelist = []
    for step in steps_list:
        if step['verb'] =='goto':
            pass
        codelist += [f"self.{step['verb']}('{step['noun']}', motion_profile={step['adverb']}, obs=obs)"]
    return codelist
# %%    
class LangAgent(PickAndPlaceAgent):
    def __init__(self, cfg, debug=True, **kwargs):
        super().__init__(cfg, **kwargs)
        # read the yaml 
        # get velocities 
        # get accelerations
        
        self.steps = []
        self.state = SimpleTaskState.NOT_STARTED
        self.mode = "navigation"  # TODO: turn into an enum
        self.current_step = ""
        # for testing
        self.testing = True
        self.debug = debug
        self.dry_run = False
        self.task_plans = get_task_plans_from_gt() 
        # self.task_defs = {
        #     0: "place the apple on the table",
        #     1: "place the banana on the table",
        #     2: "find my bottle",
        # }
        # {
        #     0: [
        #         "locate('apple')",
        #         "pick_up('apple')",
        #         "place('table')",
        #     ],
        #     1: [
        #         "locate('banana')",
        #         "pick_up('banana')",
        #         "place('table')",
        #     ],
        #     2: ["self.locate('bottle', obs)"],
        #     3: [
        #         "self.locate('bottle', obs)",
        #         "self.locate('can', obs)",
        #     ],
        # }

    # ---override methods---
    def reset(self):
        """Clear internal task state and reset component agents."""
        self.state = SimpleTaskState.NOT_STARTED
        if self.test_place:
            self.state = SimpleTaskState.FIND_GOAL
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

    # --unique methods--
    def skill_is_done(self) -> bool:
        return self.state == SimpleTaskState.IDLE

    def get_steps(self, task: str):
        """takes in a task string and returns a list of steps to complete the task"""
        if self.testing:
            # task is expected to be an int as a str
            self.steps = self.task_plans[int(task)]
        else:
            raise NotImplementedError(
                "Getting plans outside of test tasks is not implemented yet"
            )

    def goto(self, object_name, obs):
        if self.debug:
            print("In locate skill")
        info = {}
        if self.skip_find_object:
            # transition to the next state
            action = DiscreteNavigationAction.STOP
            self.state = SimpleTaskState.IDLE
        else:
            if self.mode != "navigation":
                print(f"Changing from {self.mode=} to ")
                self.mode = "navigation"
                print(f"{self.mode=}")
                self.state = SimpleTaskState.FIND_GOAL
                return DiscreteNavigationAction.NAVIGATION_MODE, {}
            else:
                print("Already in Nav Mode")
                self.state = SimpleTaskState.FIND_GOAL
                # obs.task_observations["goal_name"] = object_name
                obs = self._preprocess_obs_for_find(obs)
                breakpoint()
                action, info = self.object_nav_agent.act(obs)
                if action == DiscreteNavigationAction.STOP or self.dry_run:
                    self.state = SimpleTaskState.IDLE
        return action, info

    def pick_up(self, object_name, obs):
        if self.debug:
            print("In pick_up skill")
        # return following if agent currently not in manip mode
        if self.mode == "navigation":
            print("Change the mode of the robot to manipulation mode")
            self.mode = "manipulation"
            return DiscreteNavigationAction.MANIPULATION_MODE, {}
        else:
            print("DRYRUN: Run SLAP on: pick-up", object_name, obs)
            return DiscreteNavigationAction.PICK_OBJECT, {}

    def place(self, object_name, obs):
        if self.debug:
            print("In place skill")
        if self.mode == "navigation":
            print("Change the mode of the robot to manipulation mode")
            self.mode = "manipulation"
            return DiscreteNavigationAction.MANIPULATION_MODE, {}
        else:
            print("DRYRUN: Run SLAP on: place-on", object_name, obs)
            return DiscreteNavigationAction.PLACE_OBJECT, {}

    def task_is_done(self):
        if len(self.steps) == 0 and self.state == SimpleTaskState.IDLE:
            return True
        else:
            return False

    def act(self, obs: Observations, task: str) -> Tuple[Action, Dict[str, Any]]:
        if self.state == SimpleTaskState.NOT_STARTED and len(self.steps) == 0:
            self.get_steps(task)
        print(f"{self.steps=}")
        if (
            self.state == SimpleTaskState.IDLE
            or self.state == SimpleTaskState.NOT_STARTED
        ):
            print(self.state)
            self.current_step = self.steps.pop(0)
        print(f"evaling: {self.current_step=}")
        action, info = eval(self.current_step)
        return action, info
