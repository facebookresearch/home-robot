# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Script to label episode as success/failure (key: demo_status) for cases which
   were mislabeled by data-collector. Shows: 
       point-cloud, rgb-image, number-of-keyframes and current demo_status label
"""

import glob
import os
from typing import List

import click
import h5py
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

import home_robot.utils.data_tools.image as image
from home_robot.utils.point_cloud import numpy_to_pcd, show_point_cloud


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
            print(file, g_name)
            print("Showing initial point-cloud...")
            show_point_cloud(xyz, rgb / 255.0)
            print(f"Number of keyframes: {h5file[g_name]['head_xyz'][()].shape[0]}")
            print(f"Current demo-status: {h5file[g_name]['demo_status'][()]}")
            if mode == "write":
                override_status = input("Enter y if you want to override demo-status")
                if override_status == "y":
                    new_status = int(
                        input("Enter the new status. 0 for fail, 1 for success: ")
                    )
                    h5file[g_name]["demo_status"][...] = new_status


if __name__ == "__main__":
    main()
