# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# To get this to work:
#  pip install torch-scatter einops lpip

import os
import timeit

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from home_robot.utils.point_cloud import show_point_cloud
from home_robot_hw.remote.api import StretchClient

txt_path = "~/Downloads/RealEstate10K/train/0000cc6d8b108390.txt"
txt_path = os.path.expanduser(txt_path)
video_path = "~/Downloads/RealEstate10K/0000cc6d8b108390.mp4"
video_path = os.path.expanduser(video_path)

# Create a VideoCapture object to read the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_list = []

zerodepth_model = torch.hub.load(
    "TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True
)

with open(txt_path, "r") as file:
    # Read all lines from the file and store them in a list
    lines = file.readlines()

debug = False
Ks = []
poses = []
height = 720
width = 1280
new_height = int(height / 4)
new_width = int(width / 4)
for line in lines[1:]:
    line = line.strip()
    words = line.split()
    timestamp = int(words[0])
    floats = [float(word) for word in words[1:]]
    K = np.eye(3)
    K[0, 0] = floats[0] * new_width
    K[1, 1] = floats[1] * new_height
    K[0, 2] = floats[2] * new_width
    K[1, 2] = floats[3] * new_height
    pose = floats[6:]
    assert len(pose) == 12
    pose = np.array(pose).reshape(3, 4)
    if debug:
        print(f"{K=}")
        print(f"{pose=}")
    Ks.append(K)
    poses.append(pose)

from home_robot.mapping.voxel.voxel import SparseVoxelMap
from home_robot.utils.image import Camera

camera = Camera.from_K(K, width=new_width, height=new_height)
svm = SparseVoxelMap(use_instance_memory=False, min_depth=-999, max_depth=999)

show = False
debug = False
skip_fade_in = 100
frame_count = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame_count += 1
    if frame_count <= skip_fade_in:
        continue

    assert frame.shape[0] == height, f"{height=} {frame.shape=}"
    assert frame.shape[1] == width, f"{width=} {frame.shape=}"

    # Use the resize function to downsample the image
    downsized_image = cv2.resize(frame, (new_width, new_height))

    # Convert the frame to RGB (OpenCV loads images in BGR format by default)
    rgb = cv2.cvtColor(downsized_image, cv2.COLOR_BGR2RGB)

    # Append the RGB frame to the list
    frame_list.append(rgb)

    if debug:
        plt.imshow(rgb)
        plt.axis("off")
        plt.title(str(frame_count))
        plt.show()

    # intrinsics = torch.zeros((1, 3, 3))
    K = Ks[frame_count - 1]
    pose = poses[frame_count - 1]

    orig_rgb = rgb / 255.0
    device = torch.device("cuda")
    zerodepth_model = zerodepth_model.to(device)
    rgb = torch.FloatTensor(orig_rgb[None]).permute(0, 3, 1, 2).to(device)
    intrinsics = torch.FloatTensor(K[None]).to(device)
    print("Predicting depth...")
    t0 = timeit.default_timer()
    pred_depth = zerodepth_model(rgb, intrinsics)[0, 0].detach().cpu().numpy()
    t1 = timeit.default_timer()
    print("...done. Took", t1 - t0, "seconds.")

    if show:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(orig_rgb)
        plt.subplot(1, 2, 2)
        plt.imshow(pred_depth)
        plt.show()

    pred_xyz = camera.depth_to_xyz(pred_depth)
    if show:
        print("Predicted xyz and showing...")
        show_point_cloud(pred_xyz, orig_rgb, orig=np.zeros(3))
    camera_pose = np.eye(4)
    camera_pose[:3, :] = pose
    camera_pose = torch.from_numpy(camera_pose).float()
    svm.add(
        camera_pose,
        orig_rgb.reshape(-1, 3) * 255,
        xyz=torch.from_numpy(pred_xyz).reshape(-1, 3).float(),
        camera_K=K,
        xyz_frame="camera",
    )
    svm.show()

# Release the video capture object
cap.release()
