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
import rospy
from fastsam import FastSAM, FastSAMPrompt

import home_robot
import home_robot_hw
from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.utils.point_cloud import show_point_cloud
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


@click.command()
@click.option("--test-pick", default=False, is_flag=True)
@click.option("--test-gaze", default=False, is_flag=True)
@click.option("--test-place", default=False, is_flag=True)
@click.option("--skip-gaze", default=True, is_flag=True)
@click.option("--reset-nav", default=False, is_flag=True)
@click.option("--dry-run", default=False, is_flag=True)
@click.option("--pick-object", default="cup")
@click.option("--start-recep", default="table")
@click.option("--goal-recep", default="chair")
@click.option("--sam-weights", default="./weights/FastSAM-x.pt")
@click.option(
    "--cat-map-file", default="projects/real_world_ovmm/configs/example_cat_map.json"
)
@click.option("--max-num-steps", default=200)
@click.option("--visualize-maps", default=False, is_flag=True)
@click.option("--visualize-grasping", default=False, is_flag=True)
@click.option(
    "--debug",
    default=False,
    is_flag=True,
    help="Add pauses for debugging manipulation behavior.",
)
def main(
    test_pick=False,
    reset_nav=False,
    pick_object="cup",
    start_recep="table",
    goal_recep="chair",
    dry_run=False,
    visualize_maps=False,
    visualize_grasping=False,
    test_place=False,
    cat_map_file=None,
    max_num_steps=200,
    sam_weights=None,
    **kwargs,
):
    print("- Creating robot client")
    robot = StretchClient()
    print("- Loading SAM weights")
    sam = load_sam_model(sam_weights)
    print("- Creating voxel map")
    voxel_map = SparseVoxelMap(resolution=0.01)

    t = 0
    while not rospy.is_shutdown():
        t += 1
        print("STEP =", t)

        # Get information from the head camera
        rgb, depth, xyz = robot.head.get_images(
            compute_xyz=True,
        )
        # Get the camera pose and make sure this works properly
        camera_pose = self.robot.head.get_pose(rotated=False)

        # run a segmenter
        res = try_sam(obs.rgb, sam, debug=True)

        # Update the voxel map
        voxel_map.add(camera_pose, xyz, rgb)
        pc_xyz, pc_rgb = voxel_map.get_data()
        show_point_cloud(pc_xyz, pc_rgb / 255, orig=np.zeros(3))
        breakpoint()


if __name__ == "__main__":
    main()
