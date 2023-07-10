# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
# function pick_points(pcd) taken from: examples/python/visualization/interactive_visualization.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Script to label interaction points for SLAP dataset for tasks
needing explicit supervision, like pour-into-bowl where gripper does not explicitly touch bowl.

Script supports following modes:
        1. Read: Shows 0th image of each episode and associated labeled point cloud
        2. Write: Shows 0th image and queries if user wants to label an interaction point"""

import glob
import os
from typing import List

import click
import h5py
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

import home_robot.utils.data_tools.image as image
from home_robot.utils.point_cloud import numpy_to_pcd, show_pcd


def pick_points(pcd: o3d.geometry.PointCloud) -> List[int]:
    """Helper file to pick points from point cloud using Open3D's visualizer"""
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


@click.command()
@click.option("--data-dir", type=str, default="~/data/dataset.h5")
@click.option("--template", type=str, default="*/*.h5")
@click.option(
    "--mode",
    type=click.Choice(["read", "write"], case_sensitive=True),
    default="read",
)
def main(data_dir: str, template: str, mode: str):
    files = glob.glob(os.path.join(data_dir, template))
    if mode == "read":
        print("Nothing will be written to H5s, this is to show labeled points")
    for file in files:
        # get object category to look for given task
        if mode == "read":
            h5file = h5py.File(file, "r")
        else:
            h5file = h5py.File(file, "a")
        for g_name in h5file.keys():
            rgb = image.img_from_bytes(h5file[g_name]["head_rgb/0"][()])
            xyz = h5file[g_name]["head_xyz"][()][0]
            print(f"Showing {g_name=} from {file=}...")
            if mode == "write":
                res = input("Do you wish to label this trial? (y/n): ")
                if res == "y" or res == "Y":
                    pcd = numpy_to_pcd(xyz, rgb / 255.0)
                    points = pick_points(pcd)
                    print(f"Picked point is: {points}")
                    h5file[g_name]["interaction_point_index"] = points
            input("Press enter to continue...")


if __name__ == "__main__":
    main()
