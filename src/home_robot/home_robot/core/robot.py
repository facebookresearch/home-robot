# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch

from home_robot.core.interfaces import ContinuousNavigationAction
from home_robot.motion.robot import RobotModel


class ControlMode(Enum):
    IDLE = 0
    VELOCITY = 1
    NAVIGATION = 2
    MANIPULATION = 3


class RobotClient(ABC):
    """Connection to a robot."""

    def __init__(self):
        # Init control mode
        self._base_control_mode = ControlMode.IDLE

    @abstractmethod
    def navigate_to(
        self, xyt: ContinuousNavigationAction, relative=False, blocking=False
    ):
        """Move to xyt in global coordinates or relative coordinates."""
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """Reset everything in the robot's internal state"""
        raise NotImplementedError()

    @abstractmethod
    def switch_to_navigation_mode(self):
        raise NotImplementedError()

    def in_manipulation_mode(self) -> bool:
        """is the robot ready to grasp"""
        return self._base_control_mode == ControlMode.MANIPULATION

    def in_navigation_mode(self) -> bool:
        """Is the robot to move around"""
        return self._base_control_mode == ControlMode.NAVIGATION

    @abstractmethod
    def get_robot_model() -> RobotModel:
        """return a model of the robot for planning"""
        raise NotImplementedError()


class GraspClient(ABC):
    """Connection to grasping."""

    def set_robot_client(self, robot_client: RobotClient):
        """Update the robot client this grasping client uses"""
        self.robot_client = robot_client

    def try_grasping(self, object_goal: Optional[str] = None) -> bool:
        raise NotImplementedError
