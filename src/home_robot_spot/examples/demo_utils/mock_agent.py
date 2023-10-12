# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import math
import os
import pickle
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

import home_robot_spot.nav_client as nc
import matplotlib.pyplot as plt
import numpy as np
import open3d
import torch
from atomicwrites import atomic_write
from home_robot.agent.ovmm_agent import (
    OvmmPerception,
    build_vocab_from_category_map,
    read_category_map_file,
)
from home_robot.mapping.voxel import SparseVoxelMap  # Aggregate 3d information
from home_robot.mapping.voxel import (  # Sample positions in free space for our robot to move to
    SparseVoxelMapNavigationSpace,
)
from home_robot.motion import ConfigurationSpace, Planner, PlanResult
from home_robot.motion.rrt_connect import RRTConnect
from home_robot.motion.shortcut import Shortcut
from home_robot.motion.spot import (  # Just saves the Spot robot footprint for kinematic planning
    SimpleSpotKinematics,
)
from home_robot.perception.encoders import ClipEncoder
from home_robot.utils.config import Config, get_config, load_config
from home_robot.utils.demo_chat import DemoChat
from home_robot.utils.geometry import xyt_global_to_base
from home_robot.utils.point_cloud import numpy_to_pcd
from home_robot.utils.visualization import get_x_and_y_from_path
from home_robot_spot import SpotClient, VoxelMapSubscriber
from home_robot_spot.grasp_env import GraspController
from loguru import logger
from PIL import Image


class MockSpotDemoAgent:
    def __init__(
        self,
        parameters: Dict[str, Any],
        spot_config: Config,
        dock: Optional[int] = None,
        path: str = None,
    ):
        self.voxel_map = SparseVoxelMap()

    def say(self, msg: str):
        """Provide input either on the command line or via chat client"""
        if self.chat is not None:
            self.chat.output(msg)
        else:
            print(msg)

    def ask(self, msg: str) -> str:
        """Receive input from the user either via the command line or something else"""
        if self.chat is not None:
            return self.chat.input(msg)
        else:
            return input(msg)

    def confirm_plan(self, plan: str):
        print(f"Received plan: {plan}")
        if "confirm_plan" not in self.parameters or self.parameters["confirm_plan"]:
            execute = self.ask("Do you want to execute (replan otherwise)? (y/n): ")
            return execute[0].lower() == "y"
        else:
            if plan[:7] == "explore":
                print("Currently we do not explore! Explore more to start with!")
                return False
            return True

    def run(self):
        # Should load parameters from the yaml file
        self.voxel_map.read_from_pickle(input_path)
        world_representation = get_obj_centric_world_representation(
            voxel_map.get_instances(), args.context_length
        )
        # task is the prompt, save it
        data["prompt"] = self.get_language_task()
        output = stub.stream_act_on_observations(
            ProtoConverter.wrap_obs_iterator(
                episode_id=random.randint(1, 1000000),
                obs=world_representation,
                goal=data["prompt"],
            )
        )
        if confirm_plan(plan):
            # now it is hacky to get two instance ids TODO: make it more general for all actions
            # get pick instance id
            current_high_level_action = plan.split("; ")[0]
            pick_instance_id = int(
                world_representation.object_images[
                    int(
                        current_high_level_action.split("(")[1]
                        .split(")")[0]
                        .split(", ")[0]
                        .split("_")[1]
                    )
                ].crop_id
            )
            if len(plan.split(": ")) > 2:
                # get place instance id
                current_high_level_action = plan.split("; ")[2]
                place_instance_id = int(
                    world_representation.object_images[
                        int(
                            current_high_level_action.split("(")[1]
                            .split(")")[0]
                            .split(", ")[0]
                            .split("_")[1]
                        )
                    ].crop_id
                )
                print("place_instance_id", place_instance_id)


class MethodCall:
    def __init__(self, method_name, args, kwargs, response):
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
        self.response = response


class AttributeAccess:
    def __init__(self, attribute_name, value):
        self.attribute_name = attribute_name
        self.value = value


class MockWrapper:
    def __init__(self, obj):
        self.obj = obj
        self.calls = []

    def save_to_file(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.calls, f)

    def load_from_file(self, filename):
        with open(filename, "rb") as f:
            self.calls = pickle.load(f)

    def replay(self):
        for call in self.calls:
            if isinstance(call, MethodCall):
                func = getattr(self.obj, call.method_name)
                func(*call.args, **call.kwargs)
            elif isinstance(call, AttributeAccess):
                setattr(self.obj, call.attribute_name, call.value)

    def __getattr__(self, name):
        attr = getattr(self.obj, name)
        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                self.calls.append(MethodCall(name, args, kwargs, result))
                return result

            return wrapper
        else:
            self.calls.append(AttributeAccess(name, attr))
            return attr
