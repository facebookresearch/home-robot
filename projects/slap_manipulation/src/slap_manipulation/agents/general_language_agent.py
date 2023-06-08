import json
import re
from enum import Enum
from glob import glob
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import trimesh.transformations as tra
import yaml
from slap_manipulation.agents.slap_agent import SLAPAgent

from home_robot.agent.ovmm_agent.pick_and_place_agent import PickAndPlaceAgent
from home_robot.core.interfaces import (
    Action,
    ContinuousEndEffectorAction,
    ContinuousNavigationAction,
    DiscreteNavigationAction,
    GeneralTaskState,
    Observations,
)
from home_robot.utils.geometry import (
    sophus2xyt,
    xyt2sophus,
    xyt_base_to_global,
    xyt_global_to_base,
)
from home_robot.utils.point_cloud import show_point_cloud
from home_robot_hw.ros.utils import matrix_to_pose_msg


def evaluate_expression(expression, dummy_value) -> List[str]:
    # if expression.startswith('[') and expression.endswith(']'):
    if type(expression) == list:
        return expression
    try:
        result = eval(expression)
    except SyntaxError:
        print(f"SyntaxError: {expression}")
        result = dummy_value
    return result


def separate_into_codelist(string: str) -> List[str]:
    # Remove leading/trailing whitespace and split the string by ') '
    function_calls = string.split(")")
    # Remove the ')' character from each function call
    new_function_calls = []
    for call in function_calls:
        call += ")"  # add back the ')'
        call = call.strip(" \n\t!;")
        call = call.replace("\n\t", "")
        call = call.replace("!", "")
        new_function_calls.append(call)
    return new_function_calls


def get_taskplan_for_robot(steps: List[str]) -> List[dict]:
    """
    Test this function whenever prompt is changed.
    """
    # Define the pattern using regex
    pattern = r"(\w+)\((.*?)?(,\s*([\w']+)=\'([^\)]+)\')?\)"
    steps_table = []
    for step in steps:
        # Match the pattern and extract values
        matches = re.match(pattern, step)
        if matches is None:
            continue
        # Extract verb, object, and speed values
        verb = matches.group(1)
        noun = evaluate_expression(matches.group(2), ["dummy"])
        # speed_key = matches.group(4)
        adverb = matches.group(5)
        if adverb is not None:
            adverb = adverb.upper()
        # print(verb, noun, adverb)
        steps_table.append({"verb": verb, "noun": noun, "adverb": adverb})
    return steps_table


def get_task_plans_from_llm(
    index, json_path="./llm_eval/icl_llama-7b.json", input_string="prediction"
):
    """Reads the dataset files and return a list of task plans"""
    with open(json_path, "r") as f:
        data_dict = json.load(f)
        df = pd.json_normalize(data_dict["data"])
    steps_string = df.iloc[index][input_string]
    code_list = separate_into_codelist(steps_string)
    steps_list = get_taskplan_for_robot(code_list)
    code = get_codelist(steps_list)
    return code


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
        codelist += [f"self.{step['verb']}({step['noun']}, obs=obs)"]
    return codelist


class GeneralLanguageAgent(PickAndPlaceAgent):
    def __init__(self, cfg, debug=True, task_id: int = -1, **kwargs):
        super().__init__(cfg, **kwargs)
        # Visualizations
        self.task_id = task_id
        print(f"[GeneralLanguageAgent]: {self.task_id=}")
        self.steps = []
        self.state = GeneralTaskState.NOT_STARTED
        self.mode = "navigation"  # TODO: turn into an enum
        self.current_step = ""
        self.cfg = cfg
        # for testing
        self.testing = False
        self.gt_run = cfg.CORLAGENT.gt_run
        self.debug = debug
        self.dry_run = self.cfg.AGENT.dry_run
        self.slap_model = SLAPAgent(cfg, task_id=self.task_id)
        if not self.cfg.SLAP.dry_run:
            self.slap_model.load_models()
        self.num_actions_done = 0
        self._language = yaml.load(
            open(self.cfg.AGENT.language_file, "r"), Loader=yaml.FullLoader
        )
        self._task_information = yaml.load(
            open(self.cfg.AGENT.task_information_file, "r"), Loader=yaml.FullLoader
        )  # read from a YAML
        if not self.debug and not self.gt_run:
            self.task_plans = get_task_plans_from_llm  # get_task_plans_from_oracle
        elif self.gt_run:
            self.task_plans = {
                0: [
                    "self.goto(['bottle'], obs)",
                    "self.take_bottle(['bottle'], obs)",
                    "self.goto(['person'], obs)",
                    "self.handover(['person'], obs)",
                ],
                1: [
                    "self.pick_up(['bottle'], obs)",
                    "self.goto(['counter'], obs)",
                    "self.place(['counter'], obs)",
                    "self.goto(['drawer', 'drawer handle'], obs)",
                    "self.open_object(['drawer handle'], obs)",
                    "self.goto(['bottle'], obs)",
                    "self.take_bottle(['bottle'], obs)",
                    "self.goto(['drawer handle'], obs)",
                    "self.place(['drawer handle'], obs)",
                    "self.goto(['drawer handle'], obs)",
                    "self.close_object(['drawer handle'], obs)",
                ],
                2: [
                    "self.goto(['drawer', 'drawer handle'], obs)",
                    "self.open_object(['drawer handle',], obs)",
                    "self.goto(['drawer handle', 'lemon'], obs)",
                    "self.pick_up(['lemon'], obs)",
                    "self.goto(['table'], obs)",
                    "self.place(['table'], obs)",
                ],
                3: [
                    # "self.goto(['drawer', 'drawer handle'], obs)",
                    # "self.open_object(['drawer handle',], obs)",
                    # "self.goto(['drawer handle', 'headphones'], obs)",
                    # "self.pick_up(['headphones'], obs)",
                    "self.goto(['person'], obs)",
                    "self.handover(['person'], obs)",
                ],
                4: [
                    # "self.goto(['cup'], obs)",
                    "self.pick_up(['cup'], obs)",
                    "self.goto(['bowl'], obs)",
                    "self.pour_into_bowl(['bowl'], obs)",
                    "self.goto(['basket'], obs)",
                    "self.place(['basket'], obs)",
                ],
            }
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
                4: [
                    "self.open_object(['drawer'], obs)",
                    "self.open_object(['cabinet'], obs)",
                ],
                5: [
                    "self.goto(['drawer handle'], obs)",
                    "self.open_object(['drawer handle',], obs)",
                ],
                6: [
                    "self.open_object(['drawer', 'drawer handle'], obs)",
                ],
                7: [
                    "self.goto(['table', 'cup'], obs)",
                    "self.pick_up(['cup'], obs)",
                    "self.goto(['person'], obs)",
                    "self.handover(['person'], obs)",
                ],
                8: [
                    "self.goto(['table', 'cup'], obs)",
                    "self.pick_up(['cup'], obs)",
                    "self.goto(['counter'], obs)",
                    "self.handover(['counter'], obs)",
                ],
                9: [
                    "self.goto(['person'], obs)",
                    "self.handover(['person'], obs)",
                ],
                10: [
                    "self.goto(['drawer', 'drawer handle'], obs)",
                    "self.open_object(['drawer handle',], obs)",
                    "self.goto(['lemon'], obs)",
                    "self.pick_up(['lemon'], obs)",
                ],
                11: [
                    "self.goto(['lemon'], obs)",
                    "self.pick_up(['lemon'], obs)",
                ],
                12: [
                    "self.goto(['bottle'], obs)",
                    "self.take_bottle(['bottle'], obs)",
                ],
                13: [
                    "self.goto(['bowl'], obs)",
                    "self.pour_into_bowl(['bowl'], obs)",
                ],
                14: [
                    "self.goto(['drawer handles'], obs)",
                    "self.close_object(['drawer'], obs)",
                ],
            }

    # ---override methods---
    def reset(self):
        """Clear internal task state and reset component agents."""
        self.state = GeneralTaskState.NOT_STARTED
        self.object_nav_agent.reset()
        if self.gaze_agent is not None:
            self.gaze_agent.reset()

    def soft_reset(self):
        print("[GeneralLanguageAgent] ObjectNav reset")
        self.state = GeneralTaskState.IDLE
        self.num_actions_done = 0
        self.slap_model.reset()
        self.object_nav_agent.reset()

    def _preprocess_obs(
        self, obs: Observations, object_list: List[str]
    ) -> Observations:
        # we do not differentiate b/w obejcts or receptacles
        # everything is a semantic goal to be found
        if len(object_list) > 1:
            obs.task_observations["start_recep_goal"] = 1
            obs.task_observations["object_goal"] = 2
            obs.task_observations["start_recep_name"] = object_list[0]
            obs.task_observations["goal_name"] = object_list[1]
            obs.task_observations["end_recep_goal"] = None
            obs.task_observations["end_recep_name"] = None
        else:
            obs.task_observations["end_recep_goal"] = 1
            obs.task_observations["end_recep_name"] = object_list[0]
            obs.task_observations["start_recep_goal"] = None
            obs.task_observations["start_recep_name"] = None
            obs.task_observations["object_goal"] = None
            obs.task_observations["goal_name"] = object_list[0]
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
        if self.testing or self.debug or self.gt_run:
            self.steps = self.task_plans[int(task)]
        else:
            raise NotImplementedError(
                "Getting plans outside of test tasks is not implemented yet"
            )

    def goto(self, object_list: List[str], obs: Observations):
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
                if "place" in self.steps[0] or "pick_up" in self.steps[0]:
                    print(
                        f"[GeneralLanguageAgent] Preparing for next task: {self.steps[0]} by decreasing rad + min_dist"
                    )
                    self.object_nav_agent.planner.min_goal_distance_cm = 50
                    self.object_nav_agent.planner.goal_dilation_selem_radius = 7
                else:
                    print(
                        f"[GeneralLanguageAgent] Preparing for next task: {self.steps[0]} by increasing rad + min_dist"
                    )
                    self.object_nav_agent.planner.min_goal_distance_cm = 80
                    self.object_nav_agent.planner.goal_dilation_selem_radius = 25
                action, info["viz"] = self.object_nav_agent.act(obs)
                if action == DiscreteNavigationAction.STOP or self.dry_run:
                    self.soft_reset()
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
            # TODO: can check if new obejct_name is same as last;
            # if yes, then don't change (saves a lot of time!!)
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

    def open_object(self, object_list: List[str], obs: Observations):
        language = "open-object-drawer"
        num_actions = self._task_information[language]
        return self.call_slap(language, num_actions, obs, object_list)

    def close_object(self, object_list: List[str], obs: Observations):
        language = "close-object-drawer"
        num_actions = self._task_information[language]
        return self.call_slap(language, num_actions, obs, object_list)

    def handover(self, object_list: List[str], obs: Observations):
        language = "handover-to-person"
        num_actions = self._task_information[language]
        return self.call_slap(language, num_actions, obs, object_list)

    def take_bottle(self, object_list: List[str], obs: Observations):
        language = "take-bottle"
        num_actions = self._task_information[language]
        return self.call_slap(language, num_actions, obs, object_list)

    def pour_into_bowl(self, object_list: List[str], obs: Observations):
        language = "pour-into-bowl"
        num_actions = self._task_information[language]
        return self.call_slap(language, num_actions, obs, object_list)

    def call_slap(self, language: str, num_actions: int, obs, object_list: List[str]):
        info = {}
        action = None
        obs.task_observations["task-name"] = language
        obs.task_observations["num-actions"] = num_actions
        obs.task_observations["object_list"] = object_list
        if not self.is_busy() or self.state == GeneralTaskState.PREPPING:
            if self.state == GeneralTaskState.PREPPING:
                self.state = GeneralTaskState.DOING_TASK
                info["object_list"] = object_list
                print(f"[AGENT] {object_list=}")
                return DiscreteNavigationAction.MANIPULATION_MODE, info
            print("[LangAgent]: Changing mode, setting goals")
            self.state = GeneralTaskState.PREPPING
            # rotate the obs before sending it in
            camera_pose = obs.task_observations["base_camera_pose"]
            xyz = tra.transform_points(obs.xyz.reshape(-1, 3), camera_pose)

            # PCD comes from nav mode, where robot base rot is diff
            # from that we trained on, so add another rotation
            rot_matrix = tra.euler_matrix(0, 0, -np.pi / 2)
            obs.xyz = tra.transform_points(xyz, rot_matrix)

            result, info = self.slap_model.predict(obs)

            # rotate back the predicted point
            rot_matrix = tra.euler_matrix(0, 0, np.pi / 2)
            info["interaction_point"] = tra.transform_points(
                info["interaction_point"].reshape(-1, 3), rot_matrix
            ).reshape(-1)

            # top_xyz = info["top_xyz"]
            # top_rgb = info["top_rgb"]
            # from numpy.linalg import eig
            #
            # # Step 1: Compute the centroid
            # centroid = np.mean(top_xyz, axis=0)
            # # Step 2: project points to xy
            # xy_points = top_xyz[:, :2]
            # # Step 3: Compute the covariance matrix of the projected points
            # differences = xy_points - centroid[:2]
            # covariance_matrix = np.dot(differences.T, differences) / len(xy_points)
            # # Step 4: Compute eigenvalues and eigenvectors of the projected points
            # eigenvalues, eigenvectors = eig(covariance_matrix)
            # # Step 5: find the z-axis, perpendicular to PCD's axis
            # perpendicular_orientation = np.arctan2(
            #     eigenvectors[1, 0], eigenvectors[0, 0]
            # )
            # # Step 6: Project a point along the perpendicular to the z-axis
            # # this depends on the task
            # distance = 0.75  # Distance in meters (50 cm)
            # projection_vector = np.array(
            #     [
            #         np.cos(perpendicular_orientation),
            #         np.sin(perpendicular_orientation),
            #         0,
            #     ]
            # )
            # projected_point = info["interaction_point"] + distance * projection_vector
            # projected_point[2] = perpendicular_orientation + np.deg2rad(180)
            if "open-object" in language:
                info["global_offset_vector"] = np.array([0, 1, 0])
                info["global_orientation"] = np.deg2rad(-90)
                info["offset_distance"] = 0.83
            if "close-object" in language:
                info["global_offset_vector"] = np.array([0, 1, 0])
                info["global_orientation"] = np.deg2rad(-90)
                info["offset_distance"] = 0.5
            if "handover" in language:
                info["global_offset_vector"] = np.array([-1, 0, 0])
                info["global_orientation"] = np.deg2rad(0)
                info["offset_distance"] = 0.95
            if "take-bottle" == language:
                info["global_offset_vector"] = np.array([0, 1, 0])
                info["global_orientation"] = np.deg2rad(-90)
                info["offset_distance"] = 0.65
            if "pour-into-bowl" == language:
                info["global_offset_vector"] = np.array([0, 1, 0])
                info["global_orientation"] = np.deg2rad(-90)
                info["offset_distance"] = 0.65
            projected_point = np.copy(info["interaction_point"])
            projected_point[2] = 0
            info["SLAP"] = True
            action = ContinuousNavigationAction(projected_point)
            self.slap_model.reset()
            return action, info
        else:
            # rotate the obs before sending it in
            camera_pose = obs.task_observations["base_camera_pose"]
            obs.xyz = tra.transform_points(obs.xyz.reshape(-1, 3), camera_pose)
            result, info = self.slap_model.predict(obs)
            if result is not None:
                action = ContinuousEndEffectorAction(
                    result[:, :3], result[:, 3:7], np.expand_dims(result[:, 7], -1)
                )
            else:
                action = ContinuousEndEffectorAction(
                    np.random.rand(1, 3), np.random.rand(1, 4), np.random.rand(1, 1)
                )
            self.soft_reset()
            self.state = GeneralTaskState.IDLE
            return action, info

    def act(self, obs: Observations, task: str) -> Tuple[Action, Dict[str, Any]]:
        # while True:
        #     breakpoint()
        if self.state == GeneralTaskState.NOT_STARTED and len(self.steps) == 0:
            self.get_steps(task)
        if not self.is_busy():
            print(f"[LangAgent]: {self.state=}")
            self.current_step = self.steps.pop(0)
        print(f"[LangAgent]: evaling: {self.current_step=}")
        action, info = eval(self.current_step)
        return action, info
