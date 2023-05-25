import logging
from enum import Enum
from glob import glob
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from slap_manipulation.agents.slap_agent import SLAPAgent

from home_robot.agent.ovmm_agent.pick_and_place_agent import PickAndPlaceAgent
from home_robot.core.interfaces import (
    Action,
    ContinuousEndEffectorAction,
    DiscreteNavigationAction,
    GeneralTaskState,
    Observations,
)


def get_task_plans_from_oracle(
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


class GeneralLanguageAgent(PickAndPlaceAgent):
    def __init__(self, cfg, debug=True, **kwargs):
        super().__init__(cfg, **kwargs)
        self.steps = []
        self.state = GeneralTaskState.NOT_STARTED
        self.mode = "navigation"  # TODO: turn into an enum
        self.current_step = ""
        self.cfg = cfg
        # for testing
        self.testing = True
        self.debug = debug
        self.dry_run = self.cfg.AGENT.dry_run
        self.slap_model = SLAPAgent(cfg)
        self.num_actions_done = 0
        self._language = yaml.load(
            open(self.cfg.AGENT.language_file, "r"), Loader=yaml.FullLoader
        )
        self._task_information = yaml.load(
            open(self.cfg.AGENT.task_information_file, "r"), Loader=yaml.FullLoader
        )  # read from a YAML
        if not self.debug:
            self.task_plans = get_task_plans_from_oracle
        else:
            self.task_defs = {
                0: "place the apple on the table",
                1: "place the banana on the table",
                2: "find my bottle",
            }
            self.task_plans = {
                0: [
                    "self.goto(['chair', 'bottle'], obs=obs)",
                    "self.pick_up(['bottle'], obs=obs)",
                    "self.goto(['table'], obs=obs)",
                    "self.place(['table'], obs=obs)",
                ],
                1: [
                    "goto('banana')",
                    "pick_up('banana')",
                    "place('table')",
                ],
                2: ["self.goto('bottle', obs)"],
                3: [
                    "self.goto('bottle', obs)",
                    "self.goto('can', obs)",
                ],
                4: ["self.open(['drawer'], obs)"],
            }

    # ---override methods---
    def reset(self):
        """Clear internal task state and reset component agents."""
        self.state = GeneralTaskState.NOT_STARTED
        self.object_nav_agent.reset()
        if self.gaze_agent is not None:
            self.gaze_agent.reset()

    def soft_reset(self):
        self.num_actions_done = 0

    def _preprocess_obs(
        self, obs: Observations, object_list: List[str]
    ) -> Observations:
        # we do not differentiate b/w obejcts or receptacles
        # everything is a semantic goal to be found
        # start_recep_goal and "end_recep_goal" are always None
        obs.task_observations["end_recep_goal"] = None
        obs.task_observations["end_recep_name"] = None
        if len(object_list) > 1:
            obs.task_observations["start_recep_goal"] = 1
            obs.task_observations["object_goal"] = 2
            obs.task_observations["start_recep_name"] = object_list[0]
            obs.task_observations["goal_name"] = object_list[1]
        else:
            obs.task_observations["object_goal"] = 1
            obs.task_observations["goal_name"] = object_list[0]
            obs.task_observations["start_recep_goal"] = None
            obs.task_observations["start_recep_name"] = None
        return obs

    def _preprocess_obs_for_place(
        self, obs: Observations, object_list: List[str]
    ) -> Observations:
        # we do not differentiate b/w obejcts or receptacles
        # everything is a semantic goal to be found
        # start_recep_goal and "end_recep_goal" are always None
        obs.task_observations["end_recep_goal"] = 1
        obs.task_observations["end_recep_name"] = None
        obs.task_observations["object_goal"] = None
        obs.task_observations["goal_name"] = object_list[0]
        obs.task_observations["start_recep_goal"] = None
        obs.task_observations["start_recep_name"] = None
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

    def goto(self, object_list, obs: Observations):
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
                info["object_list"] = object_list
                return DiscreteNavigationAction.NAVIGATION_MODE, info
            else:
                self.state = GeneralTaskState.DOING_TASK
                obs = self._preprocess_obs(obs, object_list)
                action, info["viz"] = self.object_nav_agent.act(obs)
                if action == DiscreteNavigationAction.STOP or self.dry_run:
                    self.state = GeneralTaskState.IDLE
        return action, info

    def pick_up(self, object_list, obs):
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
            info["object_list"] = object_list
            self.state = GeneralTaskState.PREPPING
            return DiscreteNavigationAction.MANIPULATION_MODE, info
        else:
            print("[LangAgent]: Picking up with heuristic", object_list, obs)
            self.state = GeneralTaskState.IDLE
            return DiscreteNavigationAction.PICK_OBJECT, None

    def place(self, object_list, obs):
        info = {}
        if self.debug:
            print("[LangAgent]: In place skill")
        if not self.is_busy():
            self.mode = "manipulation"
            info["not_viz"] = True
            info["object_list"] = object_list
            self.state = GeneralTaskState.PREPPING
            return DiscreteNavigationAction.MANIPULATION_MODE, info
        else:
            print("[LangAgent]: DRYRUN: Run SLAP on: place-on", object_list, obs)
            self.state = GeneralTaskState.DOING_TASK
            # place the object somewhere - hopefully in front of the agent.
            obs = self._preprocess_obs_for_place(obs, object_list)
            action, action_info = self.place_policy.forward(obs, info)
            if action == DiscreteNavigationAction.STOP:
                self.state = GeneralTaskState.IDLE
            return action, action_info

    def open(self, object_list: List[str], obs: Observations):
        info = {}
        action = None
        language = self._language["open"][object_list[0]]
        num_actions = self._task_information[language]
        if not self.is_busy():
            print("[LangAgent]: Changing mode, setting goals")
            info["not_viz"] = True
            info["object_list"] = object_list
            self.state = GeneralTaskState.PREPPING
            return DiscreteNavigationAction.MANIPULATION_MODE, info
        else:
            if self.dry_run:
                print("[LangAgent] Call APM given p_i")
                print("[LangAgent] Call IPM to generate p_i")
                # maybe obs can take care of interaction_point too?
                action = ContinuousEndEffectorAction(
                    pos=np.random.rand(3), ori=np.random.rand(4), g=1.0
                )
                return action, None
            else:
                self.interaction_point = self.slap_model.predict(obs)
                action = ContinuousEndEffectorAction(
                    self.interaction_point[:3],
                    self.interaction_point[3:7],
                    self.interaction_point[-1],
                )
                self.state = GeneralTaskState.DOING_TASK
                return action, info

    def act(self, obs: Observations, task: str) -> Tuple[Action, Dict[str, Any]]:
        if self.state == GeneralTaskState.NOT_STARTED and len(self.steps) == 0:
            self.get_steps(task)
        if not self.is_busy():
            print(f"[LangAgent]: {self.state=}")
            self.current_step = self.steps.pop(0)
        print(f"[LangAgent]: evaling: {self.current_step=}")
        action, info = eval(self.current_step)
        return action, info
