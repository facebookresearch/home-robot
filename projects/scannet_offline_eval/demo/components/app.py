import logging
from pathlib import Path
from typing import Dict

import dash
import dash_bootstrap_components as dbc
import torch
from loguru import logger
from torch_geometric.nn.pool.voxel_grid import voxel_grid

from home_robot.core.interfaces import Observations
from home_robot.mapping.voxel.voxel import SparseVoxelMap
from home_robot.utils.point_cloud_torch import get_bounds

from .directory_watcher import DirectoryWatcher

app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder="../assets"
)
server = app.server


@torch.no_grad()
def step_and_return_update(self, obs):
    old_points = self.svm.voxel_map.voxel_pcd._points
    old_rgb = self.svm.voxel_map.voxel_pcd._rgb
    self.svm.step(obs)
    new_points = self.svm.voxel_map.voxel_pcd._points
    new_rgb = self.svm.voxel_map.voxel_pcd._rgb

    mins, maxes = get_bounds(new_points).unbind(dim=-1)
    voxel_idx_old = voxel_grid(pos=old_points, size=voxel_size, start=mins, end=maxes)
    voxel_idx_new = voxel_grid(pos=new_points, size=voxel_size, start=mins, end=maxes)
    novel_idxs = torch.isin(voxel_idx_old, voxel_idx_new, invert=False)
    new_rgb = new_rgb[novel_idxs]
    new_points = new_points[novel_idxs]


class SparseVoxelMapDirectoryWatcher:
    def __init__(
        self,
        watch_dir: Path,
        fps: int = 1,
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

    @torch.no_grad()
    def add_obs(self, obs):
        old_points = self.svm.voxel_pcd._points
        old_rgb = self.svm.voxel_pcd._rgb
        self.svm.add(**obs)

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

            logger.warning(
                f"{len(old_points)} + {len(new_points)} of {total_points} points ({float(len(new_points)) / total_points * 100:0.2f}%)"
            )

        self.points.append(new_points.cpu())
        self.rgb.append(new_rgb.cpu())
        self.bounds.append(new_bounds)

        # Record bounding box update
        self.box_bounds.append(obs["box_bounds"].cpu())
        self.box_names.append(obs["box_names"].cpu())

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
    watch_dir="publishers/published_trajectory/obs",
    fps=1,
    sparse_voxel_map_kwargs=dict(
        resolution=0.05,
    ),
)
