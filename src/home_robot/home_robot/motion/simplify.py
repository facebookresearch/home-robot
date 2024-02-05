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
        self.planner = planner
        self.reset()

    def reset(self):
        self.nodes = None

    def plan(self, start, goal, verbose: bool = False, **kwargs) -> PlanResult:
        """Do plan simplification"""
        self.planner.reset()
        if verbose:
            print("Call internal planner")
        res = self.planner.plan(start, goal, verbose=verbose, **kwargs)
        self.nodes = self.planner.nodes
        if not res.success or len(res.trajectory) < 4:
            # Planning failed so nothing to do here
            return res

        prev_node = None
        idx0 = 0
        prev_theta = 0
        for i, node in enumerate(self.nodes):
            # loop over nodes
            if prev_node is None:
                prev_node = node
                prev_theta = node.state[-1]
                idx0 = i
                continue
            else:
                theta_dist = node.state[-1] - prev_theta
                print(idx0)
                print(theta_dist)

        return res  # PlanResult(True, new_trajectory)
