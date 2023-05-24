import numpy as np
import rospy
from matplotlib import pyplot as plt
from PIL import Image
from slap_manipulation.env.language_planner_env import GeneralLanguageEnv

from home_robot_hw.env.stretch_pick_and_place_env import load_config

MY_CATEGORIES = ["cup", "bottle", "drawer", "mug"]

if __name__ == "__main__":
    rospy.init_node("test_detic")
    config = load_config(
        visualize=True,
        config_path="projects/slap_manipulation/configs/language_agent.yaml",
    )
    env = GeneralLanguageEnv(
        config=config,
        test_grasping=False,
        dry_run=True,
        segmentation_method="detic",
    )
    env.reset()
    info = {
        "object_list": MY_CATEGORIES,
    }
    env.set_goal(info)
    env.robot.move_to_manip_posture()
    # image = Image.open("./desk.jpg")
    # image_np = np.array(image)
    obs = env.get_observation()
    plt.imshow(obs.task_observations["semantic_frame"])
    plt.show()
