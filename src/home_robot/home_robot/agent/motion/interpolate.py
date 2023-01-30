# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import scipy
from scipy.interpolate import CubicSpline


def interpolate(t, q):
    """using time and q generate a trajectory that we hope a robot could follow"""
