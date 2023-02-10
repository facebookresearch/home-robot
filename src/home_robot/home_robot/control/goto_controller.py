#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import List, Optional

import numpy as np
import sophus as sp

from home_robot.utils.geometry import xyt_global_to_base, sophus2xyt, xyt2sophus

from .feedback.velocity_controllers import DDVelocityControlNoplan


log = logging.getLogger(__name__)


class GotoVelocityController:
    """
    Self-contained controller module for moving a diff drive robot to a target goal.
    Target goal is update-able at any given instant.
    """

    def __init__(
        self,
        hz: float,
        odom_only_feedback: bool = True,
    ):
        self.hz = hz
        self.odom_only = odom_only_feedback

        # Control module
        self.control = DDVelocityControlNoplan(hz)

        # Initialize
        self.xyt_loc = np.zeros(3)
        self.xyt_loc_odom = np.zeros(3)
        self.xyt_goal: Optional[np.ndarray] = None

        self.active = False
        self.track_yaw = True

    def update_pose_feedback(self, pose):
        self.xyt_loc = sophus2xyt(pose)

    def update_goal(self, pose: sp.SE3):
        """
        if self.odom_only:
            # Project absolute goal from current odometry reading
            pose_delta = xyt2sophus(self.xyt_loc_odom).inverse() * pose
            pose_goal = xyt2sophus(self.xyt_loc_odom) * pose_delta
        else:
            # Assign absolute goal directly
            pose_goal = pose
        """

        pose_goal = pose
        self.xyt_goal = sophus2xyt(pose_goal)

    def set_yaw_tracking(self, value: bool):
        self.track_yaw = value

    def _compute_error_pose(self):
        """
        Updates error based on robot localization
        """
        xyt_err = xyt_global_to_base(self.xyt_goal, self.xyt_loc)
        if not self.track_yaw:
            xyt_err[2] = 0.0
        else:
            xyt_err[2] = (xyt_err[2] + np.pi) % (2 * np.pi) - np.pi

        return xyt_err

    def step(self):
        # Get state estimation
        xyt_err = self._compute_error_pose()

        # Compute control
        v_cmd, w_cmd = self.control(xyt_err)

        return v_cmd, w_cmd
