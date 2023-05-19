import logging
from enum import Enum
from glob import glob
from typing import Any, Dict, List, Tuple

import pandas as pd

from home_robot.agent.ovmm_agent.pick_and_place_agent import (
    PickAndPlaceAgent,
    SimpleTaskState,
)
from home_robot.core.interfaces import Action, DiscreteNavigationAction, Observations


class GeneralTaskState(Enum):
    NOT_STARTED = 0
    PREPPING = 1
    DOING_TASK = 2
    IDLE = 3
    STOP = 4


def get_task_plans_from_gt(
    index, datafile="./datasets/BringXFromYSurfaceToHuman.json", root="./datasets/"
):
    """Reads the dataset files and return a list of task plans"""
    if datafile == "all":
        files = glob(root + "*.json")
        dflist = []
        for file in files:
            dflist.append(pd.read_json(file))
        df = pd.concat(dflist)
    else:
        df = pd.read_json(datafile)
    assert index < len(df), f"Index {index} is out of range"
    steps_list = df.iloc[index]["steps"]
    # steps_df = pd.DataFrame.from_records(steps_list)
    code = get_codelist(steps_list)
    return code


def get_codelist(steps_list):
    codelist = []
    for step in steps_list:
        codelist += [
            f"self.{step['verb']}('{step['noun']}', motion_profile={step['adverb']}, obs=obs)"
        ]
    return codelist


class LangAgent(PickAndPlaceAgent):
    def __init__(self, cfg, debug=True, **kwargs):
        super().__init__(cfg, **kwargs)
        self.steps = []
        self.state = GeneralTaskState.NOT_STARTED
        self.mode = "navigation"  # TODO: turn into an enum
        self.current_step = ""
        # for testing
        self.testing = True
        self.debug = debug
        self.dry_run = False
        if not self.debug:
            self.task_plans = get_task_plans_from_gt
        else:
            self.task_defs = {
                0: "place the apple on the table",
                1: "place the banana on the table",
                2: "find my bottle",
            }
            self.task_plans = {
                0: [
                    "self.locate('mug', obs)",
                    "self.pick_up('mug', obs)",
                    "self.locate('table', obs)",
                    "self.place('table', obs)",
                ],
                1: [
                    "locate('banana')",
                    "pick_up('banana')",
                    "place('table')",
                ],
                2: ["self.locate('bottle', obs)"],
                3: [
                    "self.locate('bottle', obs)",
                    "self.locate('can', obs)",
                ],
            }

    # ---override methods---
    def reset(self):
        """Clear internal task state and reset component agents."""
        self.state = GeneralTaskState.NOT_STARTED
        self.object_nav_agent.reset()
        if self.gaze_agent is not None:
            self.gaze_agent.reset()

    def _preprocess_obs(self, obs: Observations, object_name: str) -> Observations:
        # we do not differentiate b/w obejcts or receptacles
        # everything is a semantic goal to be found
        # start_recep_goal and "end_recep_goal" are always None
        obs.task_observations["end_recep_goal"] = None
        obs.task_observations["start_recep_goal"] = None
        obs.task_observations["object_goal"] = 1
        obs.task_observations["goal_name"] = object_name
        return obs

    # --unique methods--
    def skill_is_done(self) -> bool:
        return self.state == GeneralTaskState.IDLE

    def task_is_done(self) -> bool:
        return len(self.steps) == 0 and self.state == GeneralTaskState.IDLE

    def is_busy(self) -> bool:
        return (
            self.state == GeneralTaskState.PREPPING
            or self.state == GeneralTaskState.DOING_TASK
        )

    def get_steps(self, task: str):
        """takes in a task string and returns a list of steps to complete the task"""
        if self.testing:
            # task is expected to be an int as a str
            if self.debug:
                self.steps = self.task_plans[int(task)]
            else:
                self.steps = self.task_plans(int(task))
        else:
            raise NotImplementedError(
                "Getting plans outside of test tasks is not implemented yet"
            )

    def locate(self, object_name, obs):
        if self.debug:
            print("[LangAgent]: In locate skill")
        info = {}
        if self.skip_find_object:
            # transition to the next state
            action = DiscreteNavigationAction.STOP
            self.state = GeneralTaskState.IDLE
        else:
            if not self.is_busy():
                print("[LangAgent]: Changing mode, setting goals")
                self.mode = "navigation"
                print(f"[LangAgent]: {self.mode=}")
                self.state = GeneralTaskState.PREPPING
                info["not_viz"] = True
                info["object_name"] = object_name
                return DiscreteNavigationAction.NAVIGATION_MODE, info
            else:
                self.state = GeneralTaskState.DOING_TASK
                obs = self._preprocess_obs(obs, object_name)
                action, info = self.object_nav_agent.act(obs)
                if action == DiscreteNavigationAction.STOP or self.dry_run:
                    self.state = GeneralTaskState.IDLE
        return action, info

    def pick_up(self, object_name, obs):
        info = {}
        if self.debug:
            print("[LangAgent]: In pick_up skill")
        # return following if agent currently not in manip mode
        if (
            self.state == GeneralTaskState.IDLE
            or self.state == GeneralTaskState.NOT_STARTED
        ):
            print(
                "[LangAgent]: Change the mode of the robot to manipulation mode; set goals"
            )
            self.mode = "manipulation"
            # TODO: can check if new obejct_name is same as last; if yes, then don't change
            info["not_viz"] = True
            info["object_name"] = object_name
            return DiscreteNavigationAction.MANIPULATION_MODE, info
        else:
            print("[LangAgent]: DRYRUN: Run SLAP on: pick-up", object_name, obs)
            return DiscreteNavigationAction.PICK_OBJECT, None

    def place(self, object_name, obs):
        if self.debug:
            print("[LangAgent]: In place skill")
        if self.mode == "navigation":
            print("[LangAgent]: Change the mode of the robot to manipulation mode")
            self.mode = "manipulation"
            return DiscreteNavigationAction.MANIPULATION_MODE, {}
        else:
            print("[LangAgent]: DRYRUN: Run SLAP on: place-on", object_name, obs)
            return DiscreteNavigationAction.PLACE_OBJECT, {}

    def act(self, obs: Observations, task: str) -> Tuple[Action, Dict[str, Any]]:
        if self.state == GeneralTaskState.NOT_STARTED and len(self.steps) == 0:
            self.get_steps(task)
        if not self.is_busy():
            print(f"[LangAgent]: {self.state=}")
            self.current_step = self.steps.pop(0)
        print(f"[LangAgent]: evaling: {self.current_step=}")
        action, info = eval(self.current_step)
        return action, info
