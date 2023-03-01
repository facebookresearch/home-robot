# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from home_robot.motion.robot import Robot
from home_robot.motion.space import Space

"""
This just defines the standard interface for a motion planner
"""


class Planner(object):
    """planner base class"""

    # def __init__(self, space: Space, validate_fn):
    def __init__(self, robot: Robot):
        self.robot = robot
        # self.Space = space
        # self.validate = validate_fn

    def plan(self, q0, qg):
        """returns a trajectory"""
        raise NotImplementedError
