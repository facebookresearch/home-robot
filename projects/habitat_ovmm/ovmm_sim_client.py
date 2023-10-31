# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from home_robot.core.interfaces import (
    ContinuousNavigationAction,
    DiscreteNavigationAction,
    Observations,
)
from home_robot.core.robot import ControlMode, RobotClient
from home_robot.motion.stretch import HelloStretchKinematics
from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import (
    HabitatOpenVocabManipEnv,
)


class OvmmSimClient(RobotClient):
    """Defines the ovmm simulation robot as a RobotClient child
    class so the sim can be used with the cortex demo code"""

    def __init__(
        self,
        sim_env: HabitatOpenVocabManipEnv,
        is_stretch_robot: bool,
    ):
        super().__init__()

        self.env = sim_env
        self.obs = None
        self.done = None
        self.hab_info = None

        if is_stretch_robot:
            self._robot_model = HelloStretchKinematics(
                urdf_path="",
                ik_type="pinocchio",
                visualize=False,
                grasp_frame=None,
                ee_link_name=None,
                manip_mode_controlled_joints=None,
            )

    def navigate_to(
        self, xyt: ContinuousNavigationAction, relative=False, blocking=False
    ):
        """Move to xyt in global coordinates or relative coordinates."""
        assert relative == True
        self.obs, self.done, self.hab_info = self.env.apply_action(xyt)

    def reset(self):
        """Reset everything in the robot's internal state"""
        self.obs = self.env.reset()
        self.done = False

    def switch_to_navigation_mode(self) -> bool:
        """Apply sim navigation mode action and set internal state"""
        self.env.apply_action(DiscreteNavigationAction.NAVIGATION_MODE)
        self._base_control_mode = ControlMode.NAVIGATION

        return True

    def switch_to_manipulation_mode(self) -> bool:
        """Apply sim manipulation mode action and set internal state"""
        self.env.apply_action(DiscreteNavigationAction.MANIPULATION_MODE)
        self._base_control_mode = ControlMode.MANIPULATION

        return True

    def get_robot_model(self) -> RobotModel:
        """return a model of the robot for planning"""
        return self._robot_model

    def get_observation(self):
        """Return obs from last apply action"""
        return self.obs

    def move_to_nav_posture(self):
        """No applicable action in sim"""
        self.env.apply_action(DiscreteNavigationAction.EMPTY_ACTION)

    def move_to_manip_posture(self):
        """No applicable action in sim"""
        self.env.apply_action(DiscreteNavigationAction.EMPTY_ACTION)

    def get_base_pose(self):
        """xyt position of robot"""
        return [self.obs.gps[0], self.obs.gps[1], self.obs.compass]

    def apply_action(self, action):
        self.obs, self.done, self.hab_info = self.env.apply_action(action)


class SimGraspPlanner(GraspClient):
    """Interface to simulation grasping"""

    def __init__(
        self,
        robot_client: RobotClient,
    ):
        self.robot_client = robot_client

    def set_robot_client(self, robot_client: RobotClient):
        """Update the robot client this grasping client uses"""
        self.robot_client = robot_client

    def try_grasping(self, object_goal: Optional[str] = None) -> bool:
        """Grasp the object by snapping object in sim"""
        self.robot_client.apply_action(DiscreteNavigationAction.SNAP_OBJECT)
        return True

    def try_placing(self, object_goal: Optional[str] = None) -> bool:
        """Place the object by de-snapping object in sim"""
        self.robot_client.apply_action(DiscreteNavigationAction.DESNAP_OBJECT)
        return True
