#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from home_robot.utils.config import get_control_config

from .feedback.velocity_controllers import DDVelocityControlNoplan

log = logging.getLogger(__name__)

DEFAULT_CFG_NAME = "noplan_velocity_sim"


def xyt_global_to_base(xyt_world2target, xyt_world2base):
    """Transforms SE2 coordinates from global frame to local frame

    This function was created to temporarily remove dependency on sophuspy from the controller.
    TODO: Unify geometry utils across repository

    Args:
        xyt_world2target: SE2 transformation from world to target
        xyt_world2base: SE2 transformation from world to base

    Returns:
        SE2 transformation from base to target
    """
    x_diff = xyt_world2target[0] - xyt_world2base[0]
    y_diff = xyt_world2target[1] - xyt_world2base[1]
    theta_diff = xyt_world2target[2] - xyt_world2base[2]
    base_cos = np.cos(xyt_world2base[2])
    base_sin = np.sin(xyt_world2base[2])

    xyt_base2target = np.zeros(3)
    xyt_base2target[0] = x_diff * base_cos + y_diff * base_sin
    xyt_base2target[1] = x_diff * -base_sin + y_diff * base_cos
    xyt_base2target[2] = theta_diff

    return xyt_base2target


def xyt_base_to_global(xyt_base2target, xyt_world2base):
    """Transforms SE2 coordinates from local frame to global frame

    This function was created to temporarily remove dependency on sophuspy from the controller.
    TODO: Unify geometry utils across repository

    Args:
        xyt_base2target: SE2 transformation from base to target
        xyt_world2base: SE2 transformation from world to base

    Returns:
        SE2 transformation from world to target
    """
    base_cos = np.cos(xyt_world2base[2])
    base_sin = np.sin(xyt_world2base[2])
    x_base2target_global = xyt_base2target[0] * base_cos - xyt_base2target[1] * base_sin
    y_base2target_global = xyt_base2target[0] * base_sin + xyt_base2target[1] * base_cos

    xyt_world2target = np.zeros(3)
    xyt_world2target[0] = xyt_world2base[0] + x_base2target_global
    xyt_world2target[1] = xyt_world2base[1] + y_base2target_global
    xyt_world2target[2] = xyt_world2base[2] + xyt_base2target[2]

    return xyt_world2target


class GotoVelocityController:
    """
    Self-contained controller module for moving a diff drive robot to a target goal.
    Target goal is update-able at any given instant.
    """

    def __init__(
        self,
        cfg: Optional["DictConfig"] = None,
    ):
        if cfg is None:
            cfg = get_control_config(DEFAULT_CFG_NAME)

        # Control module
        self.control = DDVelocityControlNoplan(cfg)

        # Initialize
        self.xyt_loc = np.zeros(3)
        self.xyt_goal: Optional[np.ndarray] = None

        self.active = False
        self.track_yaw = True

    def update_pose_feedback(self, xyt_current: np.ndarray):
        self.xyt_loc = xyt_current

    def update_goal(self, xyt_goal: np.ndarray, relative: bool = False):
        if relative:
            self.xyt_goal = xyt_base_to_global(xyt_goal, self.xyt_loc)
        else:
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
