# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

PI2 = 2 * np.pi


def angle_difference(angle1: float, angle2: float) -> float:
    """Calculate the smallest difference between two angles in radians."""
    angle1 = angle1 % PI2
    angle2 = angle2 % PI2
    diff = np.abs(angle1 - angle2)
    return min(diff, PI2 - diff)


def interpolate_angles(start_angle: float, end_angle: float, step_size: float = 0.1) -> float:
    """Interpolate between two angles in radians with a given step size."""
    start_angle = start_angle % PI2
    end_angle = end_angle % PI2
    direction = np.sign((end_angle - start_angle + PI2) % PI2 - np.pi)
    interpolated_angle = start_angle + (step_size * direction)
    return interpolated_angle % PI2
