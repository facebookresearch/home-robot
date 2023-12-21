# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import sys
import time
import timeit
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import open3d
import rospy
import torch

# Mapping and perception
from home_robot.core.robot import GraspClient, RobotClient
from home_robot.perception import OvmmPerception, create_semantic_sensor

# Import planning tools for exploration
from home_robot.perception.encoders import ClipEncoder

# Other tools
from home_robot.utils.config import get_config, load_config

# Chat and UI tools
from home_robot.utils.point_cloud import numpy_to_pcd, show_point_cloud
from home_robot.utils.rpc import get_vlm_rpc_stub
from home_robot.utils.visualization import get_x_and_y_from_path

# Hardware inteface
from home_robot_hw.remote import StretchClient
from home_robot_hw.ros.grasp_helper import GraspClient as RosGraspClient
from home_robot_hw.ros.visualizer import Visualizer


class Funmap(GraspClient):
    def __init__(self, robot_client: RobotClient, semantic_sensor: OvmmPerception):
        self.robot = robot_client
        self.semantic_sensor = semantic_sensor

    def try_grasping(self, object_goal: str):
        self.run_grasping(object_goal)

    def run_grasping(self, to_grasp="cup", to_place="chair"):
        """Start running grasping code here"""
        robot, semantic_sensor = self.robot, self.semantic_sensor
        robot.switch_to_manipulation_mode()
        robot.move_to_demo_pregrasp_posture()
        rospy.sleep(2)

        def within(x, y):
            return (
                x >= 0
                and x < obs.semantic.shape[0]
                and y >= 0
                and y < obs.semantic.shape[1]
            )

        if to_grasp is not None:
            ### GRASPING ROUTINE
            # Get observations from the robot
            obs = robot.get_observation()
            # Predict masks
            obs = semantic_sensor.predict(obs)

            print(f"Try to grasp {to_grasp}:")
            to_grasp_oid = None
            for oid in np.unique(obs.semantic):
                if oid == 0:
                    continue
                cid, classname = semantic_sensor.current_vocabulary.map_goal_id(oid)
                print(f"- {oid} {cid} = {classname}")
                if classname == to_grasp:
                    to_grasp_oid = oid

            x, y = np.mean(np.where(obs.semantic == to_grasp_oid), axis=1)
            if not within(x, y):
                print("WARN: to_grasp object not within valid semantic map bounds")
                return
            x = int(x)
            y = int(y)

            c_x, c_y, c_z = obs.xyz[x, y]
            c_pt = np.array([c_x, c_y, c_z, 1.0])
            m_pt = obs.camera_pose @ c_pt
            m_x, m_y, m_z, _ = m_pt

            print(f"- Execute grasp at {m_x=}, {m_y=}, {m_z=}.")
            robot._ros_client.trigger_grasp(m_x, m_y, m_z)
            robot.switch_to_manipulation_mode()
            robot.move_to_demo_pregrasp_posture()
            print(" - Done grasping!")

        if to_place is not None:
            ### PLACEMENT ROUTINE
            # Get observations from the robot
            obs = robot.get_observation()
            # Predict masks
            obs = semantic_sensor.predict(obs)

            to_place_oid = None
            for oid in np.unique(obs.semantic):
                if oid == 0:
                    continue
                cid, classname = semantic_sensor.current_vocabulary.map_goal_id(oid)
                print(f"- {oid} {cid} = {classname}")
                if classname == to_place:
                    to_place_oid = oid

            x, y = np.mean(np.where(obs.semantic == to_place_oid), axis=1)
            if not within(x, y):
                print("WARN: to_place object not within valid semantic map bounds")
                return
            x = int(x)
            y = int(y)

            c_x, c_y, c_z = obs.xyz[x, y]
            c_pt = np.array([c_x, c_y, c_z, 1.0])
            m_pt = obs.camera_pose @ c_pt
            m_x, m_y, m_z, _ = m_pt

            print(f"- Execute place at {m_x=}, {m_y=}, {m_z=}.")
            robot._ros_client.trigger_placement(m_x, m_y, m_z)
            robot.switch_to_manipulation_mode()
            robot.move_to_demo_pregrasp_posture()
            rospy.sleep(2)
            print(" - Done placing!")
