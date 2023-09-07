from home_robot.utils.config import get_config
from home_robot_hw.env.spot_goat_env import SpotGoatEnv
from spot_wrapper.spot import Spot, SpotCamIds
from home_robot.agent.goat_agent.goat_agent import GoatAgent
import numpy as np
import cv2
import time
from spot_rl.models import OwlVit
from spot_wrapper.spot import image_response_to_cv2 as imcv2

config_path = "projects/spot/configs/config.yaml"
config, config_str = get_config(config_path)
config.defrost()

V = OwlVit([["an image of a tajin bottle"]], 0.05, False)
spot = Spot("RealNavEnv")

gaze_arm_joint_angles = np.deg2rad(config.GAZE_ARM_JOINT_ANGLES)
place_arm_joint_angles = np.deg2rad(config.PLACE_ARM_JOINT_ANGLES)

with spot.get_lease(hijack=True):

    spot.power_on()
    # breakpoint()
    try:
        spot.undock()
    except:
        spot.blocking_stand()

    time.sleep(1)

    spot.set_arm_joint_positions(gaze_arm_joint_angles, travel_time=1.0)
    spot.open_gripper()
    time.sleep(1)
    # spot.block_until_arm_arrives(gaze_arm_joint_angles, timeout_sec=1.5)

    while True:

        print("Resetting environment...")
        image_responses = spot.get_image_responses([SpotCamIds.HAND_COLOR])
        hand_image_response = image_responses[0]  # only expecting one image
        img = imcv2(hand_image_response)
        # Make image_response cv2 (H,W,C) 

        # run_inference with owlvit
        owlvit_response = V.run_inference(img)

        if owlvit_response != []:

            # get bounding box from owlvit_response
            print(owlvit_response)
            bbox = owlvit_response[0][2]

            # center owlvit
            #x
            center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
            cv2.circle(img, (int(center[0]), int(center[1])), 20, (0, 0, 255), -1)
            # draw bounding box
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
            
            # show
            cv2.imwrite("test_tajin.png", img)
            cv2.imshow("img", img)
            # cv2.waitKey(0)
            # time.sleep(5)

            # Grasp
            print("Attempting to grasp...")
            print("Center: ", center)
            sucess = spot.grasp_point_in_image(hand_image_response, 
                                               pixel_xy=center, 
                                               timeout=10, 
                                               top_down_grasp=True)
            
            
            print("Grasp sucessful: ", sucess)
            if sucess:
                # Set the arm to carry position
                spot.set_arm_joint_positions(place_arm_joint_angles, travel_time=1.0)

# receives spot (no env)
# assumes its already standing and in front of object
# uses any object detection model to get bounding box and center of it
# gets the center, attemps the grasp
# Uses success, if its successful we set the ee to carry position
# If fails we can retry k=3 times
'''
class GraspingControler():
    def __init__(self, spot):
        self.spot = spot

'''