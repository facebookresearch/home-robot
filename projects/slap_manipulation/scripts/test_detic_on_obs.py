import click
import numpy as np
import rospy
from matplotlib import pyplot as plt
from PIL import Image

from home_robot.core.interfaces import Observations
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot.utils.data_tools.loader import DatasetBase
from home_robot.utils.data_tools.writer import DataWriter
from home_robot_hw.env.stretch_manipulation_env import StretchManipulationEnv

MY_CATEGORIES = ["cup", "bottle", "drawer", "basket", "bowl", "computer", "mug"]


@click.command()
@click.option("--category", multiple=True, default=MY_CATEGORIES)
def main(category):
    rospy.init_node("test_detic")
    robot = StretchManipulationEnv(init_cameras=True)
    segmentation = DeticPerception(
        vocabulary="custom",
        custom_vocabulary=",".join(category),
        sem_gpu_id=0,
    )
    # image = Image.open("./desk.jpg")
    # image_np = np.array(image)
    obs = robot.get_observation()
    breakpoint()
    detic_obs = Observations(
        rgb=obs["rgb"],
        depth=obs["depth"],
        xyz=obs["xyz"],
        gps=np.zeros(2),  # TODO Replace
        compass=np.zeros(1),  # TODO Replace
        task_observations={},
    )
    result = segmentation.predict(detic_obs)
    plt.imshow(result.task_observations["semantic_frame"])
    plt.show()


if __name__ == "__main__":
    main()
