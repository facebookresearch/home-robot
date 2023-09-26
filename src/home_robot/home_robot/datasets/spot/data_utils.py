# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import dataclasses
import datetime
import pickle as pkl
import sys
import timeit
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import bosdyn.client.frame_helpers as frame_helpers
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import seaborn as sns
import torch
import torchvision
from bosdyn.api import image_pb2
from google.protobuf.timestamp_pb2 import Timestamp
from natsort import natsorted
from pytorch3d.io import IO
from pytorch3d.structures import Pointclouds, join_pointclouds_as_scene
from pytorch3d.vis.plotly_vis import AxisArgs
from tqdm import tqdm

from home_robot.core.interfaces import Observations
from home_robot.utils.bboxes_3d import (
    BBoxes3D,
    join_boxes_as_batch,
    join_boxes_as_scene,
)
from home_robot.utils.bboxes_3d_plotly import (
    create_triad_pointclouds,
    plot_scene_with_bboxes,
)
from home_robot.utils.image import Camera as PinholeCamera
from home_robot.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates


def torch_image_from_spot_image_response(
    image_response: image_pb2.ImageResponse, reorient=False
):
    is_depth = (
        image_response.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
    )
    if is_depth:
        dtype = np.uint16
    else:
        dtype = np.uint8
    # img = np.fromstring(image_response.shot.image.data, dtype=dtype)
    img = np.frombuffer(image_response.shot.image.data, dtype=dtype)

    if image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(
            image_response.shot.image.rows, image_response.shot.image.cols
        )
    else:
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED) / 255.0  # -1

    if reorient:
        img = np.rot90(img, k=3)

    if is_depth:
        img = img * 1.0 / image_response.source.depth_scale
    return torch.from_numpy(img).float()


def get_camera_from_spot_image_response(response: image_pb2.ImageResponse):
    # parent_tform_child	SE3Pose		Transform representing the pose of the child frame in the parent's frame.
    # https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html#bosdyn-api-FrameTreeSnapshot
    # Build camera model
    tform_snap = response.shot.transforms_snapshot
    body_tform_camera = frame_helpers.get_a_tform_b(
        tform_snap, frame_helpers.BODY_FRAME_NAME, response.shot.frame_name_image_sensor
    ).to_matrix()
    body_tform_vision = frame_helpers.get_a_tform_b(
        tform_snap, frame_helpers.BODY_FRAME_NAME, frame_helpers.VISION_FRAME_NAME
    ).to_matrix()
    cam_to_world = torch.from_numpy(body_tform_vision).inverse() @ torch.from_numpy(
        body_tform_camera
    )
    height = response.shot.image.rows
    width = response.shot.image.cols
    camera = PinholeCamera(
        pos=cam_to_world[:3, 3],
        orn=cam_to_world[:3, :3],
        height=height,
        width=width,
        fx=response.source.pinhole.intrinsics.focal_length.x,
        fy=response.source.pinhole.intrinsics.focal_length.y,
        px=response.source.pinhole.intrinsics.principal_point.x,
        py=response.source.pinhole.intrinsics.principal_point.y,
        near_val=None,
        far_val=None,
        pose_matrix=None,
        proj_matrix=None,
        view_matrix=None,
        fov=None,
    )

    # # Later on we can try converting between conventions. Here's a non-working example
    # from home_robot.utils.camera_view_conventions import convert_camera_view_coordinate_system, CameraViewCoordSystem
    # SPOT = {'up': '-Y', 'right': '+X', 'forward': '+Z'}
    # M = convert_camera_view_coordinate_system(M, source_convention=SPOT, output_convention=CameraViewCoordSystem.OPENGL.value).transformedRT
    # cams = cameras_from_opencv_projection(
    #     R=M[:, :3, :3], # .permute([0,2,1])
    #     tvec=M[:, :3, 3],
    #     # tvec = torch.stack(tvecs).float(),
    #     camera_matrix=torch.stack(intrinsics).float(),
    #     image_size=torch.stack(image_sizes).float()
    # )
    # # cams = PerspectiveCameras(
    # #     device="cpu",
    # #     R=M[:, :3, :3].permute([0,2,1]),
    # #     T=M[:, :3, 3],
    # #     K=torch.stack(intrinsics).float())
    return camera


def get_base_gps_compass_from_spot_image_response(response: image_pb2.ImageResponse):
    raise NotImplementedError


def build_obs_from_spot_image_responses(
    depth_response: image_pb2.ImageResponse,
    color_response: Optional[image_pb2.ImageResponse],
):
    # Get camera parameters
    camera: PinholeCamera = get_camera_from_spot_image_response(depth_response)
    cam_to_world = torch.eye(4)
    cam_to_world[:3, :3] = camera.orn
    cam_to_world[:3, 3] = camera.pos

    # get images
    color = None
    if color_response is not None:
        color = torch_image_from_spot_image_response(color_response)
    depth = torch_image_from_spot_image_response(depth_response)

    # Other metadata
    timestamp = depth_response.shot.acquisition_time.ToDatetime()

    # Build obs
    return Observations(
        gps=None,
        compass=None,
        rgb=color,
        depth=depth,
        camera_pose=cam_to_world,
        camera_K=camera.K[:3, :3],
        task_observations={"timestamp": timestamp},
    )


class SpotPickleDataset:
    def __init__(self, dir: Union[Path, str]):
        self.scenes = [
            "data_09_05_02_obs",
            "with_body_poses_obs",
            "trajectory_chair_table_obs",
            "data_09_05_01_obs",
        ]

        self.root_dir = Path(dir)

    def __getitem__(self, idx: int):
        input_path = self.root_dir / self.scenes[idx]
        pkl_files = input_path.glob("*.pkl")
        sorted_pkl_files = natsorted(pkl_files, key=lambda f: int(f.stem))
        results = []
        for i, pkl_file in enumerate(
            tqdm(sorted_pkl_files, desc="Loading SPOT pickles")
        ):
            # print("-", i, pkl_file)
            with open(pkl_file, "rb") as f:
                obs = pkl.load(f)
                results.append(obs)
        return results


# # Other trajectories use different formats
# with open(DATA_DIR / 'debug_data_depth.pkl', 'rb') as f:
#     traj = pkl.load(f)
#     traj_list = [None for _ in traj]
#     for k, v in traj.items():
#         traj_list[int(k)] = v
#     assert len(traj_list) == len(traj)
#     traj = traj_list

# with open(DATA_DIR / 'debug_data.pkl', 'rb') as f:
#     traj = pkl.load(f)

###############################
# This can all go in __main__
###############################
if __name__ == "__main__":
    depth_key = "frontleft_depth_in_visual_frame"
    rgb_key = "frontleft_fisheye_image"
    rgb_video_file_name = "traj.mp4"
    use_sparse_voxel_map_agent = False
    OUTPUT_PLY_FILE = "traj.ply"
    DATA_DIR = Path(
        "/private/home/ssax/home-robot/src/home_robot/home_robot/datasets/spot/data"
    )

    def is_interactive():
        import __main__ as main

        return not hasattr(main, "__file__")

    def in_notebook():
        try:
            from IPython import get_ipython

            if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
                return False
        except ImportError:
            return False
        except AttributeError:
            return False
        return True

    # LOAD DATA
    ds = SpotPickleDataset(DATA_DIR)
    traj = ds[1]

    if in_notebook():
        i, j = 7, 16

        rgb_video_file_name = "traj.mp4"

        im1 = traj[i].raw_obs["cameras"][rgb_key]["image"]
        im2 = traj[j].raw_obs["cameras"][rgb_key]["image"]
        plt.imshow(im1)
        plt.show()
        plt.imshow(im2)
        plt.show()

        im1 = traj[i].raw_obs["cameras"][depth_key]["image"]
        im2 = traj[j].raw_obs["cameras"][depth_key]["image"]
        plt.imshow(im1)
        plt.show()
        plt.imshow(im2)
        plt.show()

    if rgb_video_file_name is not None:
        torchvision.io.write_video(
            rgb_video_file_name,
            torch.tensor(
                np.array([obs.raw_obs["cameras"][rgb_key]["image"] for obs in traj])
            ),
            fps=5,
        )

    colors = []
    xyz = []
    depths = []
    Rs, tvecs, intrinsics = [], [], []

    # CREATE OBSERVATIONS TO FEED TO AGENT
    for spot_responses in traj:
        depth_response: image_pb2.ImageResponse = spot_responses.raw_obs["cameras"][
            "frontleft_depth_in_visual_frame"
        ]["raw_response"]
        color_response: image_pb2.ImageResponse = spot_responses.raw_obs["cameras"][
            "frontleft_fisheye_image"
        ]["raw_response"]

        # # With saving all ImageResponses and sane naming we could do:
        # response_key = 'frontleft_depth_in_visual_frame'
        # depth_response: image_pb2.ImageResponse = obs.raw_obs[response_key]
        # color_response = None
        # if response.shot.frame_name_image_sensor != response_key
        #     color_response: image_pb2.ImageResponse = obs.raw_obs[response_key]

        obs = build_obs_from_spot_image_responses(depth_response, color_response)

        # unproject, to visualize pointcloud, but this could build a map
        depth = obs.depth
        color = obs.rgb
        cam_to_world = obs.camera_pose
        K = obs.camera_K
        keep_mask = (0.4 < depth) & (depth < 4.0)
        full_world_xyz = unproject_masked_depth_to_xyz_coordinates(  # Batchable!
            depth=depth.unsqueeze(0).unsqueeze(1),
            pose=cam_to_world.unsqueeze(0),
            # pose=torch.eye(4).unsqueeze(0),
            mask=~keep_mask.unsqueeze(0).unsqueeze(1),
            inv_intrinsics=torch.linalg.inv(torch.tensor(K[:3, :3])).unsqueeze(0),
        )

        # print(full_world_xyz.shape, ((0.4 < depth) & (depth < 4.0)).sum())

        xyz.append(full_world_xyz.view(-1, 3))
        # print(color[keep_mask].shape)
        colors.append(color[keep_mask])
        depths.append(obs.depth)
        Rs.append(cam_to_world[:3, :3])
        tvecs.append(cam_to_world[:3, 3])
        intrinsics.append(torch.from_numpy(K[:3, :3]))

        # plt.imshow(depth)
        # plt.show()
        # print(color.shape, depth.shape, color.max())
        # alpha = 0.3
        # plt.imshow(color * alpha + color * keep_mask.unsqueeze(-1) * (1 - alpha))
        # plt.show()

    pointclouds = Pointclouds(points=xyz, features=colors)
    triads = create_triad_pointclouds(
        torch.stack(Rs, dim=0), torch.stack(tvecs, dim=0), n_points=20, scale=0.2
    )

    if in_notebook():
        fig = plot_scene_with_bboxes(
            plots={
                "Scene": {
                    "Camera Triads": triads,
                    "points": pointclouds
                    # "Cameras": cams,
                }
            },
            xaxis={"backgroundcolor": "rgb(230, 200, 200)"},
            yaxis={"backgroundcolor": "rgb(200, 230, 200)"},
            zaxis={"backgroundcolor": "rgb(200, 200, 230)"},
            axis_args=AxisArgs(showgrid=True),
            pointcloud_marker_size=3,
            pointcloud_max_points=20_000,
            height=1000,
            # width=1000,
        )
    pcd = join_pointclouds_as_scene([pointclouds, triads])
    IO().save_pointcloud(pcd, OUTPUT_PLY_FILE)

    if in_notebook():
        fig.show()
