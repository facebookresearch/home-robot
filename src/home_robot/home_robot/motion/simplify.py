# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from random import random
from typing import Callable, List

import numpy as np

from home_robot.motion.base import Planner, PlanResult
from home_robot.motion.rrt import TreeNode


class Simplify(Planner):
    """Define RRT planning problem and parameters. Holds two different trees and tries to connect them with some probabability."""

    def __init__(
        self,
        planner: Planner,
        min_step: float = 0.1,
        max_step: float = 1.0,
        min_angle: float = np.deg2rad(5),
    ):
        self.min_step = min_step
        self.max_step = max_step
        self.min_angle = min_angle
