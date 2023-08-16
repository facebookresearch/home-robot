# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pickle
from typing import Tuple

import numpy as np
import open3d as o3d
import trimesh

from home_robot.utils.point_cloud import numpy_to_pcd, pcd_to_numpy, show_point_cloud


def combine_point_clouds(
    pc_xyz: np.ndarray,
    pc_rgb: np.ndarray,
    xyz: np.ndarray,
    rgb: np.ndarray,
    sparse_voxel_size: float = 0.01,
) -> np.ndarray:
    """Tool to combine point clouds without duplicates. Concatenate, voxelize, and then return
    the finished results."""
    if pc_rgb is None:
        pc_rgb, pc_xyz = rgb, xyz
    else:
        pc_rgb = np.concatenate([pc_rgb, rgb], axis=0)
        pc_xyz = np.concatenate([pc_xyz, xyz], axis=0)
    return numpy_to_pcd(pc_xyz, pc_rgb).voxel_down_sample(voxel_size=sparse_voxel_size)


DEFAULT_GRID_SIZE = [512, 512]
GRID_CHANNELS = 3


class SparseVoxelMap(object):
    """Create a voxel map object which captures 3d information."""

    def __init__(
        self,
        resolution=0.01,
        feature_dim=3,
        grid_size: Tuple[int, int] = None,
        grid_resolution: float = 0.05,
        obs_min_height=0.1,
        obs_max_height=1.8,
        obs_min_density=5,
    ):
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.xyz = None
        self.feats = None
        self.observations = []
        self.obs_min_height = obs_min_height
        self.obs_max_height = obs_max_height
        self.obs_min_density = obs_min_density
        self.grid_resolution = grid_resolution
        if grid_size is not None:
            self.grid_size = [grid_size[0], grid_size[1]]
        else:
            self.grid_size = DEFAULT_GRID_SIZE
        # Track the center of the grid - (0, 0) in our coordinate system
        # We then just need to update everything when we want to track obstacles
        self.grid_origin = np.array(self.grid_size + [0]) // 2

    def add(self, camera_pose: np.ndarray, xyz: np.ndarray, feats: np.ndarray, **info):
        """Add this to our history of observations. Also update the current running map."""
        assert xyz.shape[-1] == 3
        assert feats.shape[-1] == self.feature_dim
        assert xyz.shape[0] == feats.shape[0]

        if len(xyz.shape) > 2:
            xyz = xyz.reshape(-1, 3)
            feats = feats.reshape(-1, 3)

        self.observations.append((camera_pose, xyz, feats, info))
        world_xyz = trimesh.transform_points(xyz, camera_pose)

        # Combine point clouds by adding in the current view to the previous ones and
        # voxelizing.
        self._pcd = combine_point_clouds(
            self.xyz,
            self.feats,
            world_xyz,
            feats,
            sparse_voxel_size=self.resolution,
        )
        self.xyz, self.feats = pcd_to_numpy(self._pcd)

    def get_data(self, in_place: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return the current point cloud and features; optionally copying."""
        if in_place or self.xyz is None:
            return self.xyz, self.feats
        else:
            return self.xyz.copy(), self.feats.copy()

    def write_to_pickle(self, filename: str):
        """Write out to a pickle file. This is a rough, quick-and-easy output for debugging, not intended to replace the scalable data writer in data_tools for bigger efforts."""
        data = {}
        data["poses"] = []
        data["xyz"] = []
        data["feats"] = []
        for camera_pose, xyz, feats, info in self.observations:
            # add it to pickle
            data["poses"].append(camera_pose)
            data["xyz"].append(xyz)
            data["feats"].append(feats)
            for k, v in info.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)
        data["world_xyx"] = self.xyz
        data["world_feats"] = self.feats
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def recompute_map(self):
        """Recompute the entire map from scratch instead of doing incremental updates"""
        self.xyz, self.feats = None, None
        for camera_pose, xyz, feats, _ in self.observations:
            world_xyz = trimesh.transform_points(xyz, camera_pose)
            if self.xyz is None:
                pc_xyz, pc_rgb = world_xyz, feats
            else:
                pc_rgb = np.concatenate([self.feats, feats], axis=0)
                pc_xyz = np.concatenate([self.xyz, world_xyz], axis=0)
        self._pcd = numpy_to_pcd(pc_xyz, pc_rgb).voxel_down_sample(
            voxel_size=self.resolution
        )
        self.xyz, self.feats = pcd_to_numpy(self._pcd)

    def get_2d_map(
        self, debug: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get 2d map with explored area and frontiers."""

        # Convert metric measurements to discrete
        # Gets the xyz correctly - for now everyhting is assumed to be within correct distance of origin
        xyz = ((self.xyz / self.grid_resolution) + self.grid_origin).astype(np.uint32)

        # Crop to robot height
        min_height = self.obs_min_height / self.grid_resolution
        max_height = self.obs_max_height / self.grid_resolution
        # NOTE: keep this if we only care about obstacles
        # voxels = np.zeros(self.grid_size + [int(max_height - min_height)])
        # But we might want to track floor pixels as well
        voxels = np.zeros(self.grid_size + [int(max_height)])
        obs_mask = np.bitwise_and(xyz[:, -1] > min_height, xyz[:, -1] < max_height)
        x_coords = xyz[obs_mask, 0]
        y_coords = xyz[obs_mask, 1]
        z_coords = (xyz[obs_mask, 2] - min_height).astype(np.uint32)
        voxels[x_coords, y_coords, z_coords] = 1

        # Compute the obstacle voxel grid based on what we've seen
        obstacle_voxels = voxels[:, :, min_height:]
        obstacles_soft = np.sum(obstacle_voxels, axis=-1)
        obstacles = obstacles_soft > self.obs_min_density

        # Explored area = only floor mass
        floor_voxels = voxels[:, :, :min_height]
        explored = np.sum(floor_voxels, axis=-1)

        # Frontier consists of floor voxels adjacent to empty voxels
        # TODO

        if debug:
            import matplotlib.pyplot as plt

            plt.subplot(1, 2, 1)
            plt.imshow(obstacles_soft)
            plt.subplot(1, 2, 2)
            plt.imshow(explored)

        # Add places where there are obstacles above a certain height and density
        # Add frontiers where there are ground points and no obstacles
        return obstacles, explored

    def get_kd_tree(self):
        return o3d.geometry.KDTreeFlann(self._pcd)

    def reset(self) -> None:
        """Clear out the entire voxel map."""
        self.xyz = None
        self.feats = None
        self.observations = []

    def show(self):
        """Display the aggregated point cloud."""

        # Create a combined point cloud
        # Do the other stuff we need
        pc_xyz, pc_rgb = self.voxel_map.get_data()
        show_point_cloud(pc_xyz, pc_rgb / 255, orig=np.zeros(3))
