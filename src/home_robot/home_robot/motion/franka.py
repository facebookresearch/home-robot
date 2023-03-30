# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np


class FrankaPanda(object):
    """contains information about the franka robot"""

    def __init__(self):
        # Distance from ee frame
        grasp_offset = np.eye(4)
        grasp_offset[2, 3] = 0.22
        self.grasp_offset = grasp_offset
        self.max_grasp = 0.08

    def apply_grasp_offset(self, ee_pose):
        return ee_pose @ self.grasp_offset
