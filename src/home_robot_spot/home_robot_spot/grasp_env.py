# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger
from spot_rl.models import OwlVit
from spot_wrapper.spot import Spot, SpotCamIds
from spot_wrapper.spot import image_response_to_cv2 as imcv2

from home_robot.utils.config import get_config


class GraspController:
    def __init__(
        self,
        config=None,
        spot: Optional[Spot] = None,
        objects: Optional[List[List[str]]] = [["ball", "lion"]],
        confidence=0.05,
        show_img=False,
        top_grasp=False,
        hor_grasp=False,
    ):
        self.spot = spot
        self.confidence = confidence
        self.show_img = show_img
        self.top_grasp = top_grasp
        self.hor_grasp = hor_grasp
        if objects is not None:
            self.set_objects(objects)
        self.look = np.deg2rad(config.SPOT.GAZE_ARM_JOINT_ANGLES)
        self.stow = np.deg2rad(config.SPOT.PLACE_ARM_JOINT_ANGLES)
        self.pick_location: List[float] = []

    def set_objects(self, objects: List[List[str]]):
        """set the objects"""
        self.labels = [f"an image of {y}" for x in objects for y in x]
        self.detector = OwlVit(self.labels, self.confidence, self.show_img)

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

    def spot_is_disappointed(self):
        """
        Perform a disappointed arm motion with Spot's arm.

        This method moves Spot's arm back and forth three times to create
        a disappointed motion.

        Returns:
            None
        """
        # Define the angles for disappointed motion
        disappointed_angles = [-np.pi / 8, np.pi / 8]
        self.reset_to_look()
        for _ in range(3):
            for angle in disappointed_angles:
                self.look[3] = angle
                self.spot.set_arm_joint_positions(self.look, travel_time=1)
                time.sleep(1)

        # Reset the arm to its original position
        self.look[3] = 0
        self.spot.set_arm_joint_positions(self.look, travel_time=1)
        time.sleep(0.5)

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
                logger.info(f" > Converted img from list -> {type(img)}")
            coords = self.detector.run_inference(img)
            if len(coords) > 0:
                logger.info(f" > Result -- {coords}")
                bounding_box = coords[0][2]
                confidence = coords[0][1]
                center = np.array(
                    [
                        (bounding_box[0] + bounding_box[2]) / 2,
                        (bounding_box[1] + bounding_box[3]) / 2,
                    ]
                )
                cv2.circle(img, (int(center[0]), int(center[1])), 10, (0, 0, 255), -1)
                cv2.rectangle(
                    img,
                    (bounding_box[0], bounding_box[1]),
                    (bounding_box[2], bounding_box[3]),
                    (0, 255, 0),
                    3,
                )
                if self.show_img:
                    cv2.imshow("img", img)

                filename = f"{coords[0][0].replace(' ', '_')}.jpg"
                cv2.imwrite(filename, img)
                logger.info(f" > Saved {filename}")
                return center, confidence
            else:
                return None, None
        else:
            raise TypeError(f"img is of type {type(img)}, expected is numpy array")

    def sweep(self, finish_sweep_before_deciding=True):
        """
        Perform a sweeping motion while looking for an object.

        This method moves the robot's arm through a series of predefined angles,
        capturing images at each position and searching for an object in the images.

        Returns:
            tuple or None: If an object is found, returns a tuple (x, y) representing
            the pixel coordinates of the object. If no object is found, returns None.
        """
        new_look = self.look
        sweep_angles = [0] + [
            -np.pi / 4 + i * np.pi / 8 for i in range(5)
        ]  # Compute sweep angles

        matches = []
        for angle in sweep_angles:
            new_look[0] = angle
            logger.info(f" > Moving to a new position at angle {angle}")
            self.spot.set_arm_joint_positions(new_look, travel_time=1)
            time.sleep(1.0)
            responses = self.spot.get_image_responses([SpotCamIds.HAND_COLOR])
            logger.info(" > Looking for the object")
            pixel, confidence = self.find_obj(img=imcv2(responses[0]))
            if pixel is not None:
                matches.append([pixel, responses[0], confidence, new_look])
                logger.info(
                    f" > Object found at {pixel} with spot coords: {self.spot.get_arm_proprioception()}"
                )

                # Return first match
                if not finish_sweep_before_deciding:
                    return responses[0], pixel, self.look
        self.reset_to_look()

        # No matches
        if len(matches) == 0:
            return None, None, None

        # Return best match
        best_match = max(matches, key=lambda match: match[2])
        self.spot.set_arm_joint_positions(best_match[3], travel_time=1)
        return best_match[1], best_match[0], best_match[3]

    def grasp(self, hand_image_response, pixels, timeout=10, count=3):
        """
        Attempt to grasp an object using the robot's hand.

        Parameters:
            - hand_image_response (object): The image response containing the object to grasp.
            - pixels (tuple or None): The pixel coordinates (x, y) of the object in the image.
                                    If set to None, the function will return None.
            - timeout (int, optional): Maximum time (in seconds) to wait for the grasp to succeed.
                                    Defaults to 10 seconds.
            - count (int, optional): Maximum number of grasp attempts before giving up.
                                    Defaults to 3 attempts.

        Returns:
            - success (bool or None): True if the grasp was successful, False if not, None if no pixels provided.

        Note:
            This function attempts to grasp an object located at the specified pixel coordinates in the image.
            It uses the 'spot.grasp_point_in_image' method to perform the grasp operation. If successful, it sets
            the 'pick_location' attribute and then resets the robot's arm to a stow position. The function
            allows for multiple attempts (up to 'count' times) to grasp the object within the specified 'timeout'.
            If 'pixels' is None, the function returns None.

        Example Usage:
            success = robot.grasp(image_response, (320, 240))
            if success:
                print("Grasp successful!")
            else:
                print("Grasp failed.")
        """
        k = 0
        while True:
            if pixels is not None:
                logger.info(f" > Grasping object at {pixels}")
                self.reset_to_look()
                grasp_look = self.look
                grasp_look[-2] = np.deg2rad(90)
                self.spot.set_arm_joint_positions(grasp_look, travel_time=1.0)
                time.sleep(1)
                success = self.spot.grasp_point_in_image(
                    hand_image_response,
                    pixel_xy=pixels,
                    timeout=timeout,
                    top_down_grasp=self.top_grasp,
                    horizontal_grasp=self.hor_grasp,
                )
                if success:
                    logger.info(" > Sucess")
                    self.pick_location = self.spot.get_arm_joint_positions(
                        as_array=True
                    )[-2:]
                    self.reset_to_stow()
                    time.sleep(1)
                    return success
                k = k + 1
                logger.info(
                    f" > Could not find object from the labels, tries left: {count - k}"
                )
                if k >= count:
                    logger.info(" > Ending trial as target trials reached")
                    return success
            else:
                return None

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
        """
        Get the pick location for an item.

        Returns:
            The pick location as a string.

        This method returns the pick location for the item, which is a string representing
        the location where the item can be picked in a warehouse or similar environment.
        """
        if self.pick_location is not None:
            return self.pick_location
        return None

    def gaze_and_grasp(self, finish_sweep_before_deciding=True):
        hand_image_response, pixels, arm_pos = self.sweep(finish_sweep_before_deciding)
        if arm_pos is not None:
            self.spot.set_arm_joint_positions(arm_pos, travel_time=1.5)
        if pixels is not None:
            logger.info(
                f" > Object found at {pixels} with spot coords: {self.spot.get_arm_joint_positions(as_array=True)}"
            )
            success = self.grasp(hand_image_response=hand_image_response, pixels=pixels)
            return success
        else:
            logger.info(" > No object found after sweep...BBBBOOOOOOOOOOOOOOOOO :((")
            self.spot_is_disappointed()
            return None


if __name__ == "__main__":
    CONFIG_PATH = "projects/spot/configs/config.yaml"
    config, config_str = get_config(CONFIG_PATH)
    config.defrost()
    spot = Spot("RealNavEnv")
    gaze = GraspController(
        config=config,
        spot=spot,
        objects=[["soft toy"]],
        confidence=0.1,
        show_img=True,
        top_grasp=False,
        hor_grasp=True,
    )
    with spot.get_lease(hijack=True):
        spot.power_on()
        spot.blocking_stand()
        time.sleep(1)
        # spot.set_arm_joint_positions(gaze_arm_joint_angles, travel_time=1.0)
        spot.open_gripper()
        time.sleep(1)
        logger.info("Resetting environment...")
        success = gaze.gaze_and_grasp()
