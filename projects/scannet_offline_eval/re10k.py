# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# To get this to work:
#  pip install torch-scatter einops lpip

import glob
import os
import timeit
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from home_robot.utils.point_cloud import show_point_cloud
from home_robot_hw.remote.api import StretchClient


def convert_timestamp(timestamp: int) -> str:
    timestamp = int(timestamp / 1000)
    str_hour = str(int(timestamp / 3600000)).zfill(2)
    str_min = str(int(int(timestamp % 3600000) / 60000)).zfill(2)
    str_sec = str(int(int(int(timestamp % 3600000) % 60000) / 1000)).zfill(2)
    str_mill = str(int(int(int(timestamp % 3600000) % 60000) % 1000)).zfill(3)
    return str_hour + ":" + str_min + ":" + str_sec + "." + str_mill


def extract_frame(video_path: Path, timestamp: int, output_path: Path):
    timestamp = convert_timestamp(timestamp)
    command = f"ffmpeg -ss {timestamp} -i {video_path} -frames:v 1 -f image2 {output_path} -hide_banner -loglevel error -y"
    os.system(command)


def main():
    txt_path = "~/Downloads/RealEstate10K/066c35b1abc706be.txt"
    txt_path = os.path.expanduser(txt_path)
    video_path = "~/Downloads/RealEstate10K/-aldZQifF2U.mp4"
    video_path = os.path.expanduser(video_path)
    frames_path = "~/Downloads/RealEstate10K/-aldZQifF2U"
    frames_path = os.path.expanduser(frames_path)
    try:
        os.mkdir(frames_path)
    except FileExistsError:
        pass

    # Create a VideoCapture object to read the video file
    # cap = cv2.VideoCapture(video_path)

    # if not cap.isOpened():
    #    print("Error: Could not open video file.")
    #    exit()

    frame_list = []

    # zerodepth_model = torch.hub.load(
    #    "TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True
    # )
    torch.hub.help(
        "intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True
    )  # Triggers fresh download of MiDaS repo
    repo = "isl-org/ZoeDepth"
    model_zoe_k = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
    zerodepth_model = model_zoe_k
    # model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
    # zerodepth_model = model_zoe_n

    with open(txt_path, "r") as file:
        # Read all lines from the file and store them in a list
        lines = file.readlines()

    debug = False
    Ks = []
    poses = []
    height = 1080
    width = 1920
    new_height = int(height / 8)
    new_width = int(width / 8)
    for i, line in enumerate(lines[1:]):
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
        output_path = Path(frames_path) / (str(i).zfill(5) + ".png")
        extract_frame(Path(video_path), timestamp, output_path)

    frames = glob.glob(frames_path + "/*.png")
    assert len(frames) == len(Ks)
    print("All frames:", frames)
    from home_robot.mapping.voxel.voxel import SparseVoxelMap
    from home_robot.utils.image import Camera

    camera = Camera.from_K(K, width=new_width, height=new_height)
    svm = SparseVoxelMap(use_instance_memory=False, min_depth=-999, max_depth=999)

    show = False
    debug = False
    skip_fade_in = 0
    for i, frame_file in enumerate(tqdm(frames, ncols=50)):
        # ret, frame = cap.read()
        frame = np.array(Image.open(frame_file))

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
            plt.title(str(i + 1))
            plt.show()

        # intrinsics = torch.zeros((1, 3, 3))
        K = Ks[i]
        pose = poses[i]

        orig_rgb = rgb / 255.0
        device = torch.device("cuda")
        zerodepth_model = zerodepth_model.to(device)
        rgb = torch.FloatTensor(orig_rgb[None]).permute(0, 3, 1, 2).to(device)
        intrinsics = torch.FloatTensor(K[None]).to(device)
        # print("Predicting depth...")
        t0 = timeit.default_timer()
        # print(rgb.shape)
        # pred_depth = zerodepth_model.infer(rgb, intrinsics)[0, 0].detach().cpu().numpy()
        # pred_depth = zerodepth_model(rgb, intrinsics)[0, 0].detach().cpu().numpy()
        pred_depth = zerodepth_model.infer(rgb)[0, 0].detach().cpu().numpy()
        t1 = timeit.default_timer()
        # print("...done. Took", t1 - t0, "seconds.")

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
    # cap.release()


if __name__ == "__main__":
    main()
