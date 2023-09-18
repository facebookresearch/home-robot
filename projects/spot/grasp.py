import time

import cv2
import numpy as np
from spot_rl.models import OwlVit
from spot_wrapper.spot import Spot, SpotCamIds
from spot_wrapper.spot import image_response_to_cv2 as imcv2

from home_robot.utils.config import get_config


class GraspController:
    def __init__(
        self,
        config=get_config("projects/spot/configs/config.yaml"),
        spot=None,
        objects=[["ball", "lion"]],
        confidence=0.05,
        show_img=False,
        top_grasp=False,
        hor_grasp=False,
    ):
        self.spot = spot
        self.labels = [f"an image of {y}" for x in objects for y in x]
        self.confidence = confidence
        self.show_img = show_img
        self.top_grasp = top_grasp
        self.hor_grasp = hor_grasp
        self.detector = OwlVit(self.labels, self.confidence, self.show_img)
        self.look = np.deg2rad(config.SPOT.GAZE_ARM_JOINT_ANGLES)
        self.stow = np.deg2rad(config.SPOT.PLACE_ARM_JOINT_ANGLES)
        self.pick_location = []

    def reset_to_look(self):
        """
        Reset the robotic arm to a predefined 'look' position.

        This method sets the joint positions of the
        robotic arm to a predefined 'look' configuration.
        The 'travel_time' parameter controls the speed of
        the movement, and a delay of 1 second is added
        to ensure stability after the movement.

        Args:
            None

        Returns:
            None
        """
        self.spot.set_arm_joint_positions(self.look, travel_time=1.0)
        time.sleep(1)

    def reset_to_stow(self):
        """
        Reset the robotic arm to a predefined 'stow' position.

        This method sets the joint positions of the robotic arm
        to a predefined 'stow' configuration.
        The 'travel_time' parameter controls the speed of the movement,
        and a delay of 1 second is added
        to ensure stability after the movement.

        Args:
            None

        Returns:
            None
        """
        self.spot.set_arm_joint_positions(self.stow, travel_time=1.0)
        time.sleep(1)

    def find_obj(self, img) -> np.ndarray:
        """
        Detect and locate an object in an image.

        This method resets the robotic arm to a predefined
        'look' position and then attempts to detect and locate
        an object within the given image. It draws a bounding box and a
        center point on the detected object,
        and optionally displays the annotated image.

        Args:
            img (numpy.ndarray or list): The image in which to detect the
            object. It can be either a numpy.ndarray
                or a list. If it's a list, it will be converted to
                a numpy array.

        Returns:
            np.ndarray: A numpy array representing the center coordinates of the detected object.

        Raises:
            NotImplementedError: If the object cannot be found in the image.
            TypeError: If the input image is not of type numpy.ndarray or list.
        """
        self.reset_to_look()
        if isinstance(img, np.ndarray) or isinstance(img, list):
            if isinstance(img, list):
                img = np.asarray(img)
                print(f" > Converted img from list -> {type(img)}")
            coords = self.detector.run_inference(img)
            if len(coords) > 0:
                print(f" > Result -- {coords}")
                bounding_box = coords[0][2]
                center = np.array(
                    [
                        (bounding_box[0] + bounding_box[2]) / 2,
                        (bounding_box[1] + bounding_box[3]) / 2,
                    ]
                )
                cv2.circle(img, (int(center[0]), int(center[1])), 20, (0, 0, 255), -1)
                cv2.rectangle(
                    img,
                    (bounding_box[0], bounding_box[1]),
                    (bounding_box[2], bounding_box[3]),
                    (0, 255, 0),
                    3,
                )
                if self.show_img:
                    cv2.imshow("img", img)
                else:
                    filename = f"{coords[0][0].replace(' ', '_')}.jpg"
                    cv2.imwrite(filename, img)
                    print(f" > Saved {filename}")
                return center
            else:
                return None
        else:
            raise TypeError(f"img is of type {type(img)}, expected is numpy array")

    def grasp(self, hand_image_response, pixels, timeout=10, count=3):
        # hand_image_response = image_responses_local[0]  # only expecting one image
        # img = imcv2(hand_image_response)
        k = 0
        while True:
            # pixels = self.find_obj(img=img)
            if pixels is not None:
                print(f" > Grasping object at {pixels}")
                success = self.spot.grasp_point_in_image(
                    hand_image_response,
                    pixel_xy=pixels,
                    timeout=timeout,
                    top_down_grasp=self.top_grasp,
                    horizontal_grasp=self.hor_grasp,
                )
                if success:
                    print(" > Sucess")
                    self.pick_location = self.spot.get_arm_joint_positions(
                        as_array=True
                    )
                    self.reset_to_stow()
                    time.sleep(1)
                    return success
                k = k + 1
                print(
                    f" > Could not find object from the labels, tries left: {count - k}"
                )
                if k >= count:
                    print(" > Ending trial as target trials reached")
                    return success
                    break
            else:
                return None
                # new_look = self.look
                # new_look[0] -=np.pi/4
                # self.spot.set_arm_joint_positions(new_look, travel_time=1)
                # pixels = self.find_obj(img=imcv2(self.spot.get_image_responses([SpotCamIds.HAND_COLOR])[0]))
        time.sleep(1)

    def update_label(self, new_label: str):
        """
        Update the labels associated with an image and configure an OwlVit detector.

        This method appends a new label, formatted as "an image of {new_label}",
        to the list of labels.
        It also updates the OwlVit detector with the modified list of labels,
        confidence settings, and
        whether to display images.

        Args:
            new_label (str): Classification of the object to be detected

        Returns:
            None
        """
        self.labels.append(f"an image of {new_label}")
        self.detector = OwlVit(self.labels, self.confidence, self.show_img)

    def get_pick_location(self):
        return self.pick_location


if __name__ == "__main__":
    CONFIG_PATH = "projects/spot/configs/config.yaml"
    config, config_str = get_config(CONFIG_PATH)
    config.defrost()
    spot = Spot("RealNavEnv")
    gaze = GraspController(
        config=config,
        spot=spot,
        objects=[["penguin plush"]],
        confidence=0.05,
        show_img=False,
        top_grasp=False,
        hor_grasp=True,
    )
    with spot.get_lease(hijack=True):
        spot.power_on()
        try:
            spot.undock()
        except:
            spot.blocking_stand()
        time.sleep(1)
        # spot.set_arm_joint_positions(gaze_arm_joint_angles, travel_time=1.0)
        spot.open_gripper()
        time.sleep(1)
        print("Resetting environment...")
        image_responses = spot.get_image_responses([SpotCamIds.HAND_COLOR])
        pixel = gaze.find_obj(img=imcv2(image_responses[0]))
        gaze.grasp(hand_image_response=image_responses[0], pixels=pixel)
        time.sleep(1)
        pick = gaze.get_pick_location()
        spot.set_arm_joint_positions(pick, travel_time=1)
        time.sleep(1)
        spot.open_gripper()
        time.sleep(2)
