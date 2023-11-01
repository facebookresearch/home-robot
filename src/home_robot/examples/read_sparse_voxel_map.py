# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path

import click
import open3d
import rospy
import torch

from home_robot.mapping import SparseVoxelMap, SparseVoxelMapNavigationSpace


@click.option(
    "--input-path",
    type=click.Path(),
    default="output.pkl",
    help="Input path with default value 'output.npy'",
)
def main(
    input_path,
    voxel_size: float = 0.01,
    show_maps: bool = True,
):
    """Simple script to load a voxel map"""
    input_path = Path(input_path)
    print("Loading:", input_path)
    voxel_map = SparseVoxelMap(resolution=voxel_size)
    voxel_map.read_from_pickle(input_path)
    voxel_map.show(instances=True)
    voxel_map.get_2d_map(debug=show_maps)


if __name__ == "__main__":
    """run the test script."""
    main()
