# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np


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
