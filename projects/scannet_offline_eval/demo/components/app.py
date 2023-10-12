# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import pickle as pkl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import dash
import dash_bootstrap_components as dbc
import numpy as np
import torch

# from dash_extensions.websockets import SocketPool, run_server
from dash_extensions.enrich import BlockingCallbackTransform, DashProxy
from loguru import logger
from pytorch3d.vis.plotly_vis import get_camera_wireframe
from torch_geometric.nn.pool.voxel_grid import voxel_grid

from home_robot.core.interfaces import Observations
from home_robot.mapping.voxel.voxel import SparseVoxelMap
from home_robot.utils.point_cloud_torch import get_bounds

from .directory_watcher import DirectoryWatcher, get_most_recent_viz_directory


@dataclass
class AppConfig:
    pointcloud_update_freq_ms: int = 2000
    video_feed_update_freq_ms: int = 1000

    directory_watcher_update_freq_ms: int = 600
    directory_watch_path: Optional[str] = get_most_recent_viz_directory()
    pointcloud_voxel_size: float = 0.035
    convert_rgb_to_bgr: bool = True
    ignore_box_classes: List[int] = field(default_factory=list)
    target_box_width: int = 10

    pcl_max_height: float = 1.8
    pcl_min_height: float = 0.1

    camera_initial_distance: float = 3.5


RGB_TO_BGR = [2, 1, 0]
app_config = AppConfig()

if app_config.directory_watch_path is None:
    raise ValueError("No directory found with published observations.")


app = DashProxy(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder="../assets",
    prevent_initial_callbacks=True,
    transforms=[BlockingCallbackTransform(timeout=10)],
)

server = app.server


#########################################
# Observation consumer
#########################################
class SparseVoxelMapDirectoryWatcher:
    def __init__(
        self,
        watch_dir: Path,
        fps: int = 1,
        convert_rgb_to_bgr=True,
        sparse_voxel_map_kwargs: Dict = {},
    ):
        self.svm = SparseVoxelMap(**sparse_voxel_map_kwargs)
        # self.svm.step_and_return_update = step_and_return_update
        self.obs_watcher = DirectoryWatcher(
            watch_dir,
            on_new_obs_callback=self.add_obs,
            rate_limit=fps,
        )
        self.points = []
        self.rgb = []
        self.bounds = []
        self.box_bounds = []
        self.box_names = []
        self.rgb_jpeg = None
        self.cam_coords = dict(x=[], y=[], z=[])
        self.convert_rgb_to_bgr = convert_rgb_to_bgr
        self.watch_dir = watch_dir
        self._vocab = self.load_vocab()

    def load_vocab(self):
        vocab_file = os.path.join(self.watch_dir, "vocab_dict.pkl")
        if not os.path.exists(vocab_file):
            return None
        with open(vocab_file, "rb") as f:
            self._vocab = pkl.load(f)
        return self._vocab

    @torch.no_grad()
    def add_obs(self, obs) -> bool:
        if obs is None:
            return True

        if not self._vocab:
            self.load_vocab()

        if obs["limited_obs"]:
            obs["rgb"] = torch.from_numpy(obs["obs"].rgb)
            obs["depth"] = obs["obs"].depth
            obs["camera_pose"] = torch.from_numpy(obs["obs"].camera_pose).float()

        # print(obs['obs'].rgb)
        # # Assuming obs["rgb"] is your BGR image in tensor form
        # bgr_image = obs["rgb"].cpu().numpy()

        # # Convert the BGR image to RGB
        # rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # # Encode the RGB image to JPEG format
        # self.rgb_jpeg = cv2.imencode(".jpg", rgb_image.astype(np.uint8))[1].tobytes()
        rgb_image = obs["rgb"]
        rgb_ten = rgb_image[..., RGB_TO_BGR] if self.convert_rgb_to_bgr else obs["rgb"]
        # rgb_ten = (torch.flip(obs["obstacles"], dims=(0, 1)) > 0) * 255
        # rgb_ten = rgb_ten[256:-256, 256:-256]

        self.rgb_jpeg = cv2.imencode(".jpg", (rgb_ten.cpu().numpy()).astype(np.uint8))[
            1
        ].tobytes()  # * 255

        pose = obs["camera_pose"].cpu().detach()
        R = pose[:3, :3]
        t = pose[:3, -1]
        cam_points = get_camera_wireframe(0.2)
        # Convert p3d (opengl) to opencv
        cam_points[:, 1] *= -1
        # cam_points[:, 2] *= -1
        cam_points_world = cam_points @ R.T + t.unsqueeze(0)  # (cam_points @ R) # + t)
        x, y, z = [v.cpu().numpy().tolist() for v in cam_points_world.unbind(1)]
        self.cam_coords = {"x": x, "y": y, "z": z}
        if obs["limited_obs"]:
            return True

        old_points = self.svm.voxel_pcd._points
        old_rgb = self.svm.voxel_pcd._rgb
        if obs["rgb"].max() > 1.0:  # added nomalization
            obs["rgb"] = obs["rgb"] / 255.0
        svm_watcher.svm.add(**obs)

        # Only send _new_ points to the frontend
        keep = (app_config.pcl_min_height < self.svm.voxel_pcd._points[:, 2]) & (
            self.svm.voxel_pcd._points[:, 2] < app_config.pcl_max_height
        )
        self.svm.voxel_pcd._points = self.svm.voxel_pcd._points[keep]
        self.svm.voxel_pcd._rgb = self.svm.voxel_pcd._rgb[keep]
        self.svm.voxel_pcd._weights = self.svm.voxel_pcd._weights[keep]

        new_points = self.svm.voxel_pcd._points
        new_bounds = get_bounds(new_points).cpu()
        new_rgb = self.svm.voxel_pcd._rgb
        total_points = len(new_points)
        if old_points is not None:
            # Add new points
            voxel_size = self.svm.voxel_pcd.voxel_size
            mins, maxes = get_bounds(new_points).unbind(dim=-1)
            voxel_idx_old = voxel_grid(
                pos=old_points, size=voxel_size, start=mins, end=maxes
            )
            voxel_idx_new = voxel_grid(
                pos=new_points, size=voxel_size, start=mins, end=maxes
            )
            novel_idxs = torch.isin(voxel_idx_new, voxel_idx_old, invert=True)

            new_rgb = new_rgb[novel_idxs]

            new_points = new_points[novel_idxs]

            logger.debug(
                f"At obs {len(self.points)} PTC now has {total_points} points ({len(old_points)} old, {len(new_points)} new, sending {float(len(new_points)) / total_points * 100:0.2f}%)"
            )

        self.points.append(new_points.cpu())
        self.rgb.append(new_rgb.cpu())
        self.bounds.append(new_bounds)

        # Record bounding box update
        if "box_bounds" in obs:
            self.box_bounds.append(obs["box_bounds"].cpu())
        else:
            logger.warning("No box bounds in obs")

        if "box_names" in obs:
            self.box_names.append(obs["box_names"].cpu())
        else:
            logger.warning("No box names in obs")

        # breakpoint()
        logger.debug(f"Added obs {len(self.points)} and {len(self.rgb)} rgbs.")
        return True

    def get_points_since(self):
        pass

    def begin(self):
        self.obs_watcher.start()

    def cancel(self):
        self.obs_watcher.stop()

    def unpause(self):
        self.obs_watcher.unpause()

    def pause(self):
        self.obs_watcher.pause()


svm_watcher = SparseVoxelMapDirectoryWatcher(
    watch_dir=app_config.directory_watch_path,
    fps=1.0 / (app_config.directory_watcher_update_freq_ms / 1000.0),
    sparse_voxel_map_kwargs=dict(
        resolution=app_config.pointcloud_voxel_size,
        use_instance_memory=False,
        min_depth=0.5,
        max_depth=3.5,
        obs_min_height=0.1,  # Originally .1, floor appears noisy in the 3d map of freemont so we're being super conservative
        obs_max_height=1.8,  # Originally 1.8, spot is shorter than stretch tho
        obs_min_density=12,  # Originally 10, making it bigger because theres a bunch on noise
    ),
    convert_rgb_to_bgr=app_config.convert_rgb_to_bgr,
)
