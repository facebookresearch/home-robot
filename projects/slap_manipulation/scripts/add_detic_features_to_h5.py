# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Stand-alone script to add Detic features to pre-recorded images
# Assumes data is stored in H5s with the following structure:
# H5-file-|
#         |-episode0-|
#         |          |-head_rgb/<frame_number> (RGB Image)
#         |          |-head_depth/<frame_number> (DEPTH Image)
#         |          |-head_xyz/<frame_number> (XYZ point-cloud)
#         |        ++|-head_semantic_frame
#         |        ++|-head_semantic_mask
# As outlined above two new keys are added to each episode which store the
# semantic frame (visualizable BBox output) and mask for the head camera


import glob
import os
from typing import List

import click
import h5py
import numpy as np
from matplotlib import pyplot as plt

import home_robot.utils.data_tools.image as image
from home_robot.core.interfaces import Observations
from home_robot.perception.detection.detic.detic_perception import DeticPerception

MY_CATEGORIES = [
    "cup",
    "bottle",
    "drawer",
    "basket",
    "bowl",
    "computer",
    "mug",
]
TASK_TO_OBJECT_MAP = {
    "open-object-drawer": ["drawer handle", "drawer"],
    "close-object-drawer": ["drawer handle", "drawer"],
    "pour-into-bowl": ["bowl"],
    "pour-into-sink": ["sink"],
    "handover-to-person": ["person"],
    "take-bottle": ["bottle"],
    "sweep-table-with-brush": ["sponge", "squeegee", "brush"],
}


def sandwich(obj_list: List[str]) -> List[str]:
    """Returns a new list with :obj_list: sandwiched between two "other"
    string tokens"""
    return ["other"] + obj_list + ["other"]


@click.command()
@click.option("--data-dir", type=str, default="~/data/")
@click.option("--template", type=str, default="*/*.h5")
@click.option(
    "--mode",
    type=click.Choice(["read", "write", "visualize"], case_sensitive=True),
    default="read",
)
def main(data_dir, template, mode):
    segmentation = DeticPerception(
        vocabulary="custom",
        custom_vocabulary=",".join(MY_CATEGORIES),
        sem_gpu_id=0,
    )
    depth_factor = 10000
    files = glob.glob(os.path.join(data_dir, template))
    # prev_object_for_task = None
    print("Add detic features")
    for file in files:
        # get object category to look for given task
        if mode == "read":
            h5file = h5py.File(file, "r")
        else:
            h5file = h5py.File(file, "a")
        task_name = h5file[list(h5file.keys())[0]]["task_name"][()].decode("utf-8")
        print(f"Processing {task_name} in {file}")
        if mode in ["read", "write"]:
            object_for_task = TASK_TO_OBJECT_MAP[task_name]
            segmentation.reset_vocab(sandwich(object_for_task))
        for g_name in h5file.keys():
            if mode == "read":
                rgb = image.img_from_bytes(h5file[g_name]["head_rgb/0"][()])
                depth = (
                    image.img_from_bytes(h5file[g_name]["head_depth/0"][()])
                    / depth_factor
                )
                detic_obs = Observations(
                    rgb=rgb,
                    depth=None,
                    xyz=h5file[g_name]["head_xyz"][()][0],
                    gps=np.zeros(2),  # TODO Replace
                    compass=np.zeros(1),  # TODO Replace
                    task_observations={},
                )
                result = segmentation.predict(detic_obs)
                plt.imsave(
                    f"outputs/debug/{task_name}_{g_name}.png",
                    result.task_observations["semantic_frame"],
                )
                plt.show()
            elif mode == "visualize" and "head_semantic_frame" in h5file[g_name].keys():
                plt.imsave(
                    f"outputs/debug/visualize/{task_name}_{g_name}.png",
                    h5file[g_name]["head_semantic_frame"][()],
                )
                plt.show()
            elif mode == "write":
                if "head_semantic_frame" not in h5file[g_name].keys():
                    rgb = image.img_from_bytes(h5file[g_name]["head_rgb/0"][()])
                    depth = (
                        image.img_from_bytes(h5file[g_name]["head_depth/0"][()])
                        / depth_factor
                    )
                    detic_obs = Observations(
                        rgb=rgb,
                        depth=None,
                        xyz=h5file[g_name]["head_xyz"][()][0],
                        gps=np.zeros(2),  # TODO Replace
                        compass=np.zeros(1),  # TODO Replace
                        task_observations={},
                    )
                    result = segmentation.predict(detic_obs)
                    h5file[g_name]["head_semantic_frame"] = result.task_observations[
                        "semantic_frame"
                    ]
                    h5file[g_name]["head_semantic_mask"] = result.semantic
                    plt.imshow(result.task_observations["semantic_frame"])
                    plt.show()
                else:
                    print(f"Already processed {g_name} in {file}")
        h5file.close()


if __name__ == "__main__":
    main()
