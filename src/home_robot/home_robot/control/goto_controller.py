#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import List, Optional

import numpy as np

from home_robot.utils.config import get_control_config

from .feedback.velocity_controllers import DDVelocityControlNoplan


log = logging.getLogger(__name__)


def xyt_global_to_base(xyt_world2target, xyt_world2base):
    x_diff = xyt_world2target[0] - xyt_world2base[0]
    y_diff = xyt_world2target[1] - xyt_world2base[1]
    theta_diff = xyt_world2target[2] - xyt_world2base[2]
    theta_base = xyt_world2base[2]

    xyt_base2target = np.zeros(3)
    xyt_base2target[0] = x_diff * np.cos(theta_base) + y_diff * np.sin(theta_base)
    xyt_base2target[1] = x_diff * -np.sin(theta_base) + y_diff * np.cos(theta_base)
    xyt_base2target[2] = theta_diff

    return xyt_base2target


class GotoVelocityController:
    """
    Self-contained controller module for moving a diff drive robot to a target goal.
    Target goal is update-able at any given instant.
    """

    def __init__(
        self,
        cfg_name: Optional[str] = None,
    ):
        if cfg_name is None:
            cfg_name = "noplan_velocity_sim"
        cfg = get_control_config(cfg_name)

        # Control module
        self.control = DDVelocityControlNoplan(cfg)

        # Initialize
        self.xyt_loc = np.zeros(3)
        self.xyt_goal: Optional[np.ndarray] = None

        self.active = False
        self.track_yaw = True

    def update_pose_feedback(self, xyt_current: np.ndarray):
        self.xyt_loc = xyt_current

    def update_goal(self, xyt_goal: np.ndarray):
        self.xyt_goal = xyt_goal

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

    def compute_control(self):
        # Get state estimation
        xyt_err = self._compute_error_pose()

        # Compute control
        v_cmd, w_cmd = self.control(xyt_err)

        return v_cmd, w_cmd
