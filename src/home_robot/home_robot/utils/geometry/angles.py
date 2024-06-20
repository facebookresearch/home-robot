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


def interpolate_angles(
    start_angle: float, end_angle: float, step_size: float = 0.1
) -> float:
    """Interpolate between two angles in radians with a given step size."""
    start_angle = start_angle % PI2
    end_angle = end_angle % PI2
    diff1 = (end_angle - start_angle) % PI2
    diff2 = (start_angle - end_angle) % PI2
    if diff1 <= diff2:
        direction = 1
        delta = diff1
    else:
        direction = -1
        delta = diff2
    step = min(delta, step_size) * direction
    interpolated_angle = start_angle + step
    return interpolated_angle % PI2


if __name__ == "__main__":
    print(interpolate_angles(4.628, 4.28))
