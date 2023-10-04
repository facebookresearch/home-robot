# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

PI2 = 2 * np.pi


def angle_difference(angle1: float, angle2: float):
    """angle difference"""
    angle1 = angle1 % PI2
    angle2 = angle2 % PI2
    return np.abs(angle1 - angle2)


def interpolate_angles(start_angle, end_angle, step_size: float = 0.1):
    start_angle = start_angle % PI2
    end_angle = end_angle % PI2
    interpolated_angle = start_angle + (step_size * np.sign(end_angle - start_angle))
    return interpolated_angle % PI2
