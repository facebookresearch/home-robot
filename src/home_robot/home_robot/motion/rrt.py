# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from home_robot.motion.base import Planner
from home_robot.motion.space import Space


def RRT(object):
    """Define RRT planning problem and parameters"""

    def __init__(self, space: Space, validate_fn, max_iter=1000):
        super(RRT, self).__init__(space, validate_fn)

    def plan(self, q0, qg):
        """plan from start to goal. creates a new tree"""
        raise NotImplementedError
