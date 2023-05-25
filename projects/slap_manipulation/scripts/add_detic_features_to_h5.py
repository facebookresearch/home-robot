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

MY_CATEGORIES = ["cup", "bottle", "drawer", "basket", "bowl", "computer", "mug"]
TASK_TO_OBJECT_MAP = {
    "open-top-drawer": ["drawer handle"],
    "close-top-drawer": ["drawer handle"],
}


def sandwich(obj_list: List[str]):
    return ["other"] + obj_list + ["other"]


@click.command()
@click.option("--data-dir", type=str, default="~/data/dataset.h5")
@click.option("--template", type=str, default="*/*.h5")
@click.option(
    "--mode", type=click.Choice(["read", "write"], case_sensitive=True), default="read"
)
def main(data_dir, template, mode):
    segmentation = DeticPerception(
        vocabulary="custom",
        custom_vocabulary=",".join(MY_CATEGORIES),
        sem_gpu_id=0,
    )
    depth_factor = 10000
    files = glob.glob(os.path.join(data_dir, template))
    for file in files:
        # get object category to look for given task
        if mode == "read":
            h5file = h5py.File(file, "r")
        else:
            h5file = h5py.File(file, "a")
        if mode == "write":
            task_name = h5file[list(h5file.keys())[0]]["task_name"][()].decode("utf-8")
            object_for_task = TASK_TO_OBJECT_MAP[task_name]
        # segmentation.reset_vocab(sandwich(object_for_task))
        for g_name in h5file.keys():
            if mode == "read":
                plt.imshow(h5file[g_name]["head_semantic_frame"][()])
                plt.show()
            else:
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
