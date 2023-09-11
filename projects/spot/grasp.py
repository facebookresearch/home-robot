from home_robot.utils.config import get_config
from spot_wrapper.spot import Spot, SpotCamIds
import numpy as np
import cv2
import time
from spot_rl.models import OwlVit
from spot_wrapper.spot import image_response_to_cv2 as imcv2

class GraspController():
    def __init__(self, config=get_config("projects/spot/configs/config.yaml"), spot=None, objects=[['ball', 'lion']], confidence=0.05, show_img=False, top_grasp=False, hor_grasp=False):
        self.spot = spot
        self.labels = [f'an image of {y}' for x in objects for y in x]
        self.confidence = confidence
        self.show_img = show_img
        self.top_grasp = top_grasp
        self.hor_grasp = hor_grasp
        self.owl = OwlVit(self.labels, self.confidence, self.show_img)
        self.look = np.deg2rad(config.GAZE_ARM_JOINT_ANGLES)
        self.stow = np.deg2rad(config.PLACE_ARM_JOINT_ANGLES)
        
    def reset_to_look(self):
        self.spot.set_arm_joint_positions(self.look, travel_time=1.0)
        time.sleep(1) 
    def reset_to_stow(self):   
        self.spot.set_arm_joint_positions(self.stow, travel_time=1.0)
        time.sleep(1)  
    def find_obj(self, img) -> np.ndarray:
        self.reset_to_look()
        if isinstance(img, np.ndarray) or isinstance(img, list):
            if isinstance(img, list):
                img = np.asarray(img)
                print(f" > Converted img from list -> {type(img)}")
            coords = self.owl.run_inference(img)
            if len(coords)>0:
                print(f" > Result -- {coords}")
                bounding_box = coords[0][2]
                center = np.array([(bounding_box[0] + bounding_box[2])/2, (bounding_box[1] + bounding_box[3])/2])
                cv2.circle(img, (int(center[0]), int(center[1])), 20, (0, 0, 255), -1)
                cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 255, 0), 3)
                if self.show_img:
                    cv2.imshow("img", img)
                else:
                    filename = f"{coords[0][0].replace(' ', '_')}.jpg"
                    cv2.imwrite(filename, img)
                    print(f" > Saved {filename}")
                return center
        else:
            raise TypeError(f"img is of type {type(img)}, expected is numpy array") 
        
    def grasp(self, image_responses, timeout=10, count=3):
        # TODO receive label (optionally) here and update label from owlvit ()

        #image_responses = spot.get_image_responses([SpotCamIds.HAND_COLOR])
        hand_image_response = image_responses[0]  # only expecting one image
        img = imcv2(hand_image_response)
        k = 0
        while True:
            pixels = self.find_obj(img=img)
            print(f" > Grasping object at {pixels}")
            success = self.spot.grasp_point_in_image(hand_image_response, 
                                                pixel_xy=pixels, 
                                                timeout=timeout,
                                                top_down_grasp=self.top_grasp, 
                                                horizontal_grasp=self.hor_grasp
                                                )
            if success:
                self.reset_to_stow()
                time.sleep(1)
                break
            else:
                k = k + 1
                print(f" > Could not find object from the labels, tries left: {count - k}")
            if k == count:
                print("> Ending trial as target trials reached")
                retry = input("Would you like to retry? y/n, or enter 'c' to enter a new label and retry: ")
                if retry == "y":
                    #@JAY add a look around script and then replace with gaze
                    k = 0
                    continue
                if retry == "c":
                    new_label = input("Enter new label: ")
                    self.update_label(new_label)
                    k = 0
                    count = 0
                else:
                    break
        print('Sucess')
        time.sleep(1)
    def update_label(self, new_label):
        self.labels.append(new_label)
        self.owl = OwlVit(self.labels, self.confidence, self.show_img)


if __name__ == "__main__":
    config_path = "projects/spot/configs/config.yaml"
    config, config_str = get_config(config_path)
    config.defrost()
    spot = Spot("RealNavEnv")
    gaze = GraspController(config=config, spot=spot, objects=[["bottle of water"]], confidence=0.05, show_img=True, top_grasp=False, hor_grasp=True)
    with spot.get_lease(hijack=True):
        spot.power_on()
        try:
            spot.undock()
        except:
            spot.blocking_stand()
        time.sleep(1)
        #spot.set_arm_joint_positions(gaze_arm_joint_angles, travel_time=1.0)
        spot.open_gripper()
        time.sleep(1)
        print("Resetting environment...")
        image_responses = spot.get_image_responses([SpotCamIds.HAND_COLOR])
        gaze.grasp(image_responses=image_responses)
