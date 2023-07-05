# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to test Detic's detection on a single, live image. Orient head-camera
towards the test-scene and fire the detector.
Takes in optional categorical input for any user-defined object.
"""

import click
import numpy as np
import rospy
from matplotlib import pyplot as plt
from PIL import Image

from home_robot.core.interfaces import Observations
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.utils.data_tools.loader import DatasetBase
from home_robot.utils.data_tools.writer import DataWriter
from home_robot_hw.remote.api import StretchClient

MY_CATEGORIES = ["cup", "bottle", "drawer", "basket", "bowl", "computer", "mug"]


@click.command()
@click.option(
    "--category",
    multiple=True,
    default=MY_CATEGORIES,
    help="Categories to detect in given image, takes multiple values",
)
def main(category):
    robot = StretchClient()
    segmentation = DeticPerception(
        vocabulary="custom",
        custom_vocabulary=",".join(category),
        sem_gpu_id=0,
    )
    rgb, depth, xyz = robot.head.get_images(
        compute_xyz=True,
    )
    detic_obs = Observations(
        rgb=rgb,
        depth=depth,
        xyz=xyz,
        gps=np.zeros(2),  # TODO Replace
        compass=np.zeros(1),  # TODO Replace
        task_observations={},
    )
    result = segmentation.predict(detic_obs)
    plt.imshow(result.task_observations["semantic_frame"])
    plt.show()


if __name__ == "__main__":
    main()
