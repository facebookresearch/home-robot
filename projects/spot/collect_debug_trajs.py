# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from home_robot.agent.goat_agent.goat_agent import GoatAgent
# from home_robot.mapping.voxel import SparseVoxelMapfrom home_robot.utils.config import get_config
# from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
# import pytorch3d

import argparse
import dataclasses
import pickle as pkl
import sys
import time
import timeit
from pathlib import Path
from typing import Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import transforms3d as t3d
import trimesh
import trimesh.transformations as tra
from bosdyn.api import image_pb2

# Some tools from importing data loader
from natsort import natsorted
from spot_rl.models import OwlVit
from spot_wrapper.spot import Spot, SpotCamIds, build_image_request
from spot_wrapper.spot import image_response_to_cv2 as imcv2
from tqdm import tqdm

from home_robot.utils.config import get_config
from home_robot.utils.image import Camera as PinholeCamera
from home_robot.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates
from home_robot_hw.env.spot_goat_env import SpotGoatEnv

RGB_FORMAT = image_pb2.Image.PixelFormat.PIXEL_FORMAT_RGB_U8
DEPTH_FORMAT = image_pb2.Image.PixelFormat.PIXEL_FORMAT_DEPTH_U16
BODY_THERSH = 10000 / 255
HAND_THERSH = 6000 / 255
RGB_THRESH = 1
"""
back_depth
back_depth_in_visual_frame
back_fisheye_image
frontleft_depth
frontleft_depth_in_visual_frame
frontleft_fisheye_image
frontright_depth
frontright_depth_in_visual_frame
frontright_fisheye_image
hand_color_image
hand_color_in_hand_depth_frame
hand_depth
hand_depth_in_hand_color_frame
hand_image
left_depth
left_depth_in_visual_frame
left_fisheye_image
right_depth
right_depth_in_visual_frame
right_fisheye_image
"""
GOAT_SOURCES = [
    # ("hand_depth_in_hand",DEPTH_FORMAT,HAND_THERSH,None),
    ("hand_depth", DEPTH_FORMAT, HAND_THERSH, None),
    ("hand_color_image", RGB_FORMAT, RGB_THRESH, None),
    ("hand_depth_in_hand_color_frame", DEPTH_FORMAT, HAND_THERSH, None),
    ("back_depth", DEPTH_FORMAT, BODY_THERSH, None),
    # ("back_fisheye_image",RGB_FORMAT,RGB_THRESH,None),
    ("frontleft_depth", DEPTH_FORMAT, BODY_THERSH, cv2.ROTATE_90_CLOCKWISE),
    # ("frontleft_fisheye_image",RGB_FORMAT,RGB_THRESH,cv2.ROTATE_90_CLOCKWISE),
    ("frontright_depth", DEPTH_FORMAT, BODY_THERSH, cv2.ROTATE_90_CLOCKWISE),
    # ("frontright_fisheye_image",RGB_FORMAT,RGB_THRESH,cv2.ROTATE_90_CLOCKWISE),
    ("left_depth", DEPTH_FORMAT, BODY_THERSH, None),
    # ("left_fisheye_image",RGB_FORMAT,RGB_THRESH,None),
    ("right_depth", DEPTH_FORMAT, BODY_THERSH, cv2.ROTATE_180),
    # ("right_fisheye_image",RGB_FORMAT,RGB_THRESH,cv2.ROTATE_180),
]


def flatten(l):
    return [item for sublist in l for item in sublist]


def create_grid_ndc(
    height_pixels, width_pixels, stacked=False, flatten=False, **kwargs
):
    """
    Returns rays that cast pixels in image.
    Order is consisteny with omnidata dataloader
    Args:
        width_pixels: int
        height_pixels: int
        **kwargs: Device, dtype, etc.
    """
    # create grid of uv-values
    u_min, u_max = -1.0, 1.0
    v_min, v_max = -1.0, 1.0
    half_du = 0.5 * (u_max - u_min) / width_pixels
    half_dv = 0.5 * (v_max - v_min) / height_pixels
    yy, xx = torch.meshgrid(
        torch.linspace(v_max - half_dv, v_min + half_dv, height_pixels, **kwargs),
        torch.linspace(u_max - half_du, u_min + half_du, width_pixels, **kwargs),
        indexing="ij",
    )
    if flatten:
        xx, yy = xx.flatten(), yy.flatten()
    if stacked:
        return torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    return xx, yy


config_path = "projects/spot/configs/config.yaml"
config, config_str = get_config(config_path)
# config.defrost()
spot = Spot("RealNavEnv")
gaze_arm_joint_angles = np.deg2rad(config.GAZE_ARM_JOINT_ANGLES)

print(gaze_arm_joint_angles)
print(config)

reqs = [
    build_image_request(name, quality_percent=75, pixel_format=format, resize_ratio=1)
    for name, format, _, _ in GOAT_SOURCES
]


def get_images():
    responses = {}
    for req in reqs:
        try:
            response = spot.image_client.get_image([req])[0]
            responses[req.image_source_name] = response
        except Exception as e:
            print(f"Req failed: {req}: {e=}")
            continue
    # images = [imcv2(x) for x in responses]
    # images = [np.clip(im/dat[2],0,255).astype(np.uint8) for im,dat in zip(images,GOAT_SOURCES)]
    return responses


data = {}
with spot.get_lease(hijack=True):
    spot.power_on()
    try:
        spot.undock()
    except:
        spot.blocking_stand()
    time.sleep(1)
    spot.set_arm_joint_positions(gaze_arm_joint_angles, travel_time=1.0)
    spot.open_gripper()
    time.sleep(1)

    print("Resetting environment...")
    # image_responses = spot.get_image_responses([SpotCamIds.HAND_COLOR])
    data["0"] = get_images()

    print("Move forward a little bit")
    cmd = spot.set_base_position(x_pos=1, y_pos=0, yaw=0, end_time=100)
    time.sleep(2.0)
    data["1"] = get_images()

    print("Move back to start")
    spot.set_base_position(x_pos=0, y_pos=0, yaw=0, end_time=100)
    time.sleep(2.0)
    data["2"] = get_images()

    print("Move arm")
    arm_test_q = np.deg2rad([45, -160, 100, 0, 90, 0])
    spot.set_arm_joint_positions(arm_test_q, travel_time=1.0)
    time.sleep(2.0)
    data["3"] = get_images()

    print("Move arm to start")
    spot.set_arm_joint_positions(gaze_arm_joint_angles, travel_time=1.0)
    time.sleep(2.0)
    data["4"] = get_images()

    print("Rotate")
    spot.set_base_position(x_pos=0, y_pos=0, yaw=0.79, end_time=100)
    time.sleep(2.0)
    data["5"] = get_images()

    print("Rotate to start")
    spot.set_base_position(x_pos=0, y_pos=0, yaw=0, end_time=100)
    time.sleep(2.0)
    data["6"] = get_images()

with open("data.pkl", "wb") as f:
    pkl.dump(data, f)

"""
   def set_base_position(
        self,
        x_pos,
        y_pos,
        yaw,
        end_time,
        relative=False,
        max_fwd_vel=2,
        max_hor_vel=2,
        max_ang_vel=np.pi / 2,
        disable_obstacle_avoidance=False,
        blocking=False,
    ):
"""
