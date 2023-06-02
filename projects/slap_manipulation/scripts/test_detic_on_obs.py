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
@click.option("--category", multiple=True, default=MY_CATEGORIES)
def main(category):
    robot = StretchClient()
    segmentation = DeticPerception(
        vocabulary="custom",
        custom_vocabulary=",".join(category),
        sem_gpu_id=0,
    )
    # image = Image.open("./desk.jpg")
    # image_np = np.array(image)
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
