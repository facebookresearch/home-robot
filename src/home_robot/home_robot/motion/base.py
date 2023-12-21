# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from home_robot.motion.space import ConfigurationSpace

"""
This just defines the standard interface for a motion planner
"""


class PlanResult(object):
    """Stores motion plan. Can be extended."""

    def __init__(
        self, success, trajectory: Optional[List] = None, reason: Optional[str] = None
    ):
        self.success = success
        self.trajectory = trajectory
        self.reason = reason

    def get_success(self):
        """Was the trajectory planning successful?"""
        return self.success

    def get_trajectory(self, *args, **kwargs) -> Optional[List]:
        """Return the trajectory"""
        return self.trajectory


class Planner(ABC):
    """planner base class"""

    def __init__(self, space: ConfigurationSpace, validate_fn: Callable):
        self.space = space
        self.validate = validate_fn

    @abstractmethod
    def plan(self, start, goal) -> PlanResult:
        """returns a trajectory"""
        raise NotImplementedError
