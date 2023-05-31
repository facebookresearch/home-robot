import glob
import os
from typing import List

import click
import h5py
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

import home_robot.utils.data_tools.image as image
from home_robot.utils.point_cloud import numpy_to_pcd


def pick_points(pcd):
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
    type=click.Choice(["read", "test", "write", "visualize"], case_sensitive=True),
    default="read",
)
def main(data_dir, template, mode):
    # depth_factor = 10000
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
            # depth = (
            #     image.img_from_bytes(h5file[g_name]["head_depth/0"][()])
            #     / depth_factor
            # )
            pcd = numpy_to_pcd(xyz, rgb / 255.0)
            points = pick_points(pcd)
            print(f"Picked point is: {points}")
            if mode == "write":
                h5file[g_name]["interaction_point_index"] = points
            input("Press enter to continue...")


if __name__ == "__main__":
    main()
