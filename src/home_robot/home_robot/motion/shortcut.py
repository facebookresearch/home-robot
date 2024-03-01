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


class Shortcut(Planner):
    """Define RRT planning problem and parameters. Holds two different trees and tries to connect them with some probabability."""

    def __init__(
        self,
        planner: Planner,
        shortcut_iter: int = 100,
    ):
        self.planner = planner
        super(Shortcut, self).__init__(self.planner.space, self.planner.validate)
        self.shortcut_iter = shortcut_iter
        self.reset()

    def reset(self):
        self.nodes = None

    def plan(self, start, goal, verbose: bool = False, **kwargs) -> PlanResult:
        """Do shortcutting"""
        self.planner.reset()
        if verbose:
            print("Call internal planner")
        res = self.planner.plan(start, goal, verbose=verbose, **kwargs)
        self.nodes = self.planner.nodes
        if not res.success or len(res.trajectory) < 4:
            # Planning failed so nothing to do here
            return res
        # Now try to shorten things
        for i in range(self.shortcut_iter):
            # Sample two indices
            idx0 = np.random.randint(len(res.trajectory) - 3)
            idx1 = np.random.randint(idx0 + 1, len(res.trajectory))
            node_a = res.trajectory[idx0]
            node_b = res.trajectory[idx1]
            # Extend between them
            previous_node = node_a
            for qi in self.space.extend(node_a.state, node_b.state):
                if np.all(qi == node_b.state):
                    node_b.parent = previous_node
                    break
                if not self.validate(qi):
                    break
                else:
                    self.nodes.append(TreeNode(qi, parent=previous_node))
                    previous_node = self.nodes[-1]
        new_trajectory = res.trajectory[-1].backup()
        return PlanResult(True, new_trajectory, planner=self)
