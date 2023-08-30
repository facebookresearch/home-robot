# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.motion.space import XYT


class SparseVoxelMapNavigationSpace(XYT):
    """subclass for sampling XYT states from explored space"""

    def __init__(self, voxel_map: SparseVoxelMap):
        self.map = voxel_map

    def sample(self):
        """Sample any position that corresponds to an "explored" location. Goals are valid if they are within a reasonable distance of explored locations. Paths through free space are ok and don't collide."""
        # Extract 2d map from this - hopefully it is already cached
        obstacles, explored = self.map.get_2d_map()

        # Sample any point which is explored and not an obstacle

        # Sample a random orientation

        raise NotImplementedError()
