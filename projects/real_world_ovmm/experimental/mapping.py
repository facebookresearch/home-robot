# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This file is a test script for instance mapping on the robot.
# - create an environment and an agent
# - explore slightly
# - update agent using some components like FastMobileSlam
# - use the instance map and update it
# - save some images
# - use them for training

import click
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import rospy
from fastsam import FastSAM, FastSAMPrompt

import home_robot
import home_robot_hw
from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.utils.point_cloud import create_visualization_geometries, numpy_to_pcd
from home_robot_hw.remote.api import StretchClient
from home_robot_hw.utils.config import load_config


def load_sam_model(path_to_weights):
    model = FastSAM(path_to_weights)
    return model


def try_sam(image, model, debug=True):
    DEVICE = "cpu"
    everything_results = model(
        image,
        device=DEVICE,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )
    prompt_process = FastSAMPrompt(image, everything_results, device=DEVICE)

    # everything prompt
    ann = prompt_process.everything_prompt()
    if debug:
        prompt_process.plot(
            annotations=ann,
            output_path="./output.jpg",
        )
    return everything_results


def show_map(
    xyz,
    rgb,
    robot_model: np.ndarray,
    robot_pose: np.ndarray,
    orig: np.ndarray = None,
    R: np.ndarray = None,
    save: str = None,
    grasps: list = None,
    size: float = 0.1,
) -> None:

    pcd = numpy_to_pcd(xyz, rgb)
    geoms = create_visualization_geometries(pcd=pcd, orig=orig, size=size)
    geoms += [robot_model]
    o3d.visualization.draw_geometries(geoms)


@click.command()
@click.option("--reset-nav", default=False, is_flag=True)
@click.option("--sam-weights", default="./weights/FastSAM-x.pt")
@click.option(
    "--debug",
    default=False,
    is_flag=True,
    help="Add pauses for debugging manipulation behavior.",
)
def main(
    debug=False,
    reset_nav=False,
    sam_weights=None,
    voxel_size=0.1,
):
    print("- Creating robot client")
    robot = StretchClient()
    print("- Loading SAM weights")
    sam = load_sam_model(sam_weights)
    print("- Creating voxel map with resolution =", voxel_size, "m")
    voxel_map = SparseVoxelMap(resolution=voxel_size)

    # Placeholder robot mesh
    print("- create robot mesh geometry for Stretch")
    # robot_mesh_data = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    # robot_base = o3d.geometry.TriangleMesh.create_box(width=0.33, height=0.15, depth=0.36)
    robot_base = o3d.geometry.TriangleMesh.create_box(
        width=0.33, height=0.36, depth=0.15
    )

    t = 0
    while not rospy.is_shutdown():
        t += 1
        print("STEP =", t)

        # Get information from the head camera
        rgb, depth, xyz = robot.head.get_images(
            compute_xyz=True,
        )
        # Get the camera pose and make sure this works properly
        camera_pose = robot.head.get_pose(rotated=False)
        robot_pose = robot.nav.get_base_pose()

        # run a segmenter
        res = try_sam(rgb, sam, debug=True)

        # Update the voxel map
        voxel_map.add(camera_pose, xyz, rgb)
        pc_xyz, pc_rgb = voxel_map.get_data()
        show_map(pc_xyz, pc_rgb / 255, robot_base, robot_pose, orig=np.zeros(3))
        breakpoint()


if __name__ == "__main__":
    main()
