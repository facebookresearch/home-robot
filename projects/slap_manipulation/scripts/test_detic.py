import numpy as np
import rospy
from matplotlib import pyplot as plt
from PIL import Image
from slap_manipulation.env.stretch_manipulation_env import StretchManipulationEnv

from home_robot.core.interfaces import Observations
from home_robot.perception.detection.detic.detic_perception import DeticPerception

REAL_WORLD_CATEGORIES = ["cup", "bottle", "drawer", "basket", "bowl", "computer", "mug"]

if __name__ == "__main__":
    rospy.init_node("test_detic")
    robot = StretchManipulationEnv(init_cameras=True)
    segmentation = DeticPerception(
        vocabulary="custom",
        custom_vocabulary=",".join(REAL_WORLD_CATEGORIES),
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
