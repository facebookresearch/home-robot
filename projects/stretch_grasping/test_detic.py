import numpy as np
import torch
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

from home_robot.utils.config import get_config
from home_robot.perception.detection.detic.detic_perception import DeticPerception
from home_robot_hw.remote import StretchClient
from home_robot_hw.env.stretch_pick_and_place_env import StretchPickandPlaceEnv
from home_robot.motion.stretch import STRETCH_PREGRASP_Q

import rospy
import ros_numpy
from sensor_msgs.msg import Image


GOAL_OPTIONS = [
    "other",
    "chair",
    "cup",
    "table",
    "other",
]


if __name__ == '__main__':
    robot = StretchClient()
    detic = DeticPerception(
        vocabulary="custom",
        custom_vocabulary=",".join(GOAL_OPTIONS),
        sem_gpu_id=0,
    )

    # Move robot to manip
    robot.switch_to_manipulation_mode()
    robot.head.look_at_ee(blocking=False)
    robot.manip.goto_joint_positions(
        robot.manip._extract_joint_pos(STRETCH_PREGRASP_Q)
    )
    
    # Create a configuration file
    config_path = "projects/stretch_grasping/configs/agent/floorplanner_eval.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 0
    config.EXP_NAME = "debug"
    config.freeze()

    env = StretchPickandPlaceEnv(
        config=config, test_grasping=False, dry_run=True,
        ros_grasping=False,
    )
    env.reset(GOAL_OPTIONS[1], GOAL_OPTIONS[2], GOAL_OPTIONS[3])

    publisher = rospy.Publisher('detic', Image, queue_size=1)

    blocking = False

    try:
        while not rospy.is_shutdown():
            pan, tilt = robot.head.get_pan_tilt()

            if blocking:
                pt_input = input(f"Current pan tilt: {pan}, {tilt}. Set pan tilt: ")
                if pt_input[0] == 'q':
                    print("Quitting")
                    break
                try:
                    if pt_input:
                        pt_list = [float(v) for v in pt_input.split(" ")]
                        pt_list[0] = np.clip(pt_list[0], -3.9, 1.5)
                        pt_list[1] = np.clip(pt_list[1], -1.53, 0.79)
                        robot.head.set_pan_tilt(pan=pt_list[0], tilt=pt_list[1])
                except Exception:
                    print("Warning: invalid input")

            rgb, dpt = robot.head.get_images()
            pred = detic.predictor(rgb)

            # Now also check the environment code
            obs = env.get_observation()

            visualizer = Visualizer(
                rgb, detic.metadata, instance_mode=ColorMode.IMAGE
            )
            # Get a pretty rgb image
            vis_img = visualizer.draw_instance_predictions(
                predictions=pred["instances"].to(torch.device("cpu"))
            ).get_image()

            if blocking:
                plt.subplot(141)
                plt.title('Detic Output')
                plt.imshow(vis_img)
                plt.axis('off')

                plt.subplot(142)
                plt.title('Semantic')
                plt.axis('off')
                plt.imshow(obs.semantic)

                plt.subplot(143)
                plt.title('Instance Map')
                plt.imshow(obs.task_observations['instance_map'])
                plt.axis('off')

                plt.subplot(144)
                plt.title('Goal Mask')
                plt.imshow(obs.task_observations['goal_mask'])
                plt.axis('off')

                plt.show()
            else:
                publisher.publish(ros_numpy.msgify(Image, vis_img, encoding='rgb8'))

    except KeyboardInterrupt:
        print("Terminated by user.")