# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pickle
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as open3d
import torch
import trimesh
from scipy.ndimage import distance_transform_edt

from home_robot.core.interfaces import Observations
from home_robot.mapping.semantic.instance_tracking_modules import (
    InstanceMemory,
    InstanceView,
)
from home_robot.motion.space import XYT
from home_robot.utils.point_cloud import (
    create_visualization_geometries,
    numpy_to_pcd,
    pcd_to_numpy,
    show_point_cloud,
)

# TODO: add K
Frame = namedtuple(
    "Frame", ["camera_pose", "xyz", "rgb", "feats", "depth", "base_pose", "obs", "info"]
)


DEFAULT_GRID_SIZE = [512, 512]
GRID_CHANNELS = 3


def combine_point_clouds(
    pc_xyz: np.ndarray,
    pc_rgb: np.ndarray,
    xyz: np.ndarray,
    rgb: np.ndarray,
    pc_feats: Optional[np.ndarray] = None,
    feats: Optional[np.ndarray] = None,
    sparse_voxel_size: float = 0.01,
    debug: bool = False,
) -> np.ndarray:
    """Tool to combine point clouds without duplicates. Concatenate, voxelize, and then return
    the finished results."""
    if pc_rgb is None:
        pc_rgb, pc_xyz = rgb, xyz
    else:
        pc_rgb = np.concatenate([pc_rgb, rgb], axis=0)
        pc_xyz = np.concatenate([pc_xyz, xyz], axis=0)
    # Optionally process extra point cloud features
    if feats is not None:
        if pc_feats is None:
            pc_feats = feats
        else:
            pc_feats = np.concatenate([pc_feats, feats], axis=0)

    # Create an Open3d point cloud object
    point_cloud = numpy_to_pcd(pc_xyz, pc_rgb)
    if debug:
        print("Showing union of input point clouds...")
        show_point_cloud(pc_xyz, pc_rgb / 255)

    # Apply downsampling
    if pc_feats is not None:
        (
            point_cloud,
            trace_matrix,
            trace_indices,
        ) = point_cloud.voxel_down_sample_and_trace(
            voxel_size=sparse_voxel_size,
            min_bound=np.min(pc_xyz, axis=0),
            max_bound=np.max(pc_xyz, axis=0),
        )
        # Features need to be downsampled using trace matrix. This is going to be slow.
        new_feats = np.zeros((len(trace_indices), feats.shape[-1]))
        for i, indices in enumerate(trace_indices):
            feature_vec = np.mean(pc_feats[indices], axis=0)
            new_feats[i] = feature_vec
    else:
        point_cloud = point_cloud.voxel_down_sample(voxel_size=sparse_voxel_size)
        new_feats = None
    return point_cloud, new_feats


def create_disk(radius, size):
    """create disk of the given size - helper function used to get explored areas"""

    # Create a grid of coordinates
    x = np.arange(0, size)
    y = np.arange(0, size)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # Compute the distance transform
    distance_map = np.sqrt((xx - size // 2) ** 2 + (yy - size // 2) ** 2)

    # Create the disk by thresholding the distance transform
    disk = distance_map <= radius

    return disk


class SparseVoxelMap(object):
    """Create a voxel map object which captures 3d information."""

    def __init__(
        self,
        resolution=0.01,
        feature_dim=3,
        grid_size: Tuple[int, int] = None,
        grid_resolution: float = 0.05,
        obs_min_height: float = 0.1,
        obs_max_height: float = 1.8,
        obs_min_density: float = 5,
        add_local_radius_points: bool = True,
        local_radius: float = 0.3,
        min_depth: float = 0.1,
        max_depth: float = 4.0,
    ):
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.observations = []
        self.obs_min_height = obs_min_height
        self.obs_max_height = obs_max_height
        self.obs_min_density = obs_min_density
        self.grid_resolution = grid_resolution
        self._seq = 0
        self._2d_last_updated = -1
        self.min_depth = min_depth
        self.max_depth = max_depth

        # Create an instance memory to associate bounding boxes in space
        self.instances = InstanceMemory(
            num_envs=1, du_scale=1, instance_association="bbox_iou"
        )

        # Create disk for mapping explored areas near the robot - since camera can't always see it
        self._disk_size = np.ceil(1.0 / self.grid_resolution)
        self._visited_disk = create_disk(
            1.0 / self.grid_resolution, (2 * self._disk_size) + 1
        )

        # Add points with local_radius to the voxel map at (0,0,0) unless we receive lidar points
        self.add_local_radius_points = add_local_radius_points
        self.local_radius = local_radius

        if grid_size is not None:
            self.grid_size = [grid_size[0], grid_size[1]]
        else:
            self.grid_size = DEFAULT_GRID_SIZE
        # Track the center of the grid - (0, 0) in our coordinate system
        # We then just need to update everything when we want to track obstacles
        self.grid_origin = np.array(self.grid_size + [0]) // 2
        # Init variables
        self.reset()

    def reset(self) -> None:
        """Clear out the entire voxel map."""
        self.observations = []
        # Create map here - just reset *some* variables
        self.reset_cache()

    def reset_cache(self):
        """Clear some tracked things"""

        # Stores points in 2d coords where robot has been
        self._visited = np.zeros(self.grid_size)

        # Store instances detected (all of them for now)
        self.instances.reset()

        # Store 2d map information
        # This is computed from our various point clouds
        self._map2d = None

        # Holds 3d data
        self._pcd = None
        self.xyz = None
        self.rgb = None
        self.feats = None

    def get_instances(self) -> List[InstanceView]:
        """Return a list of all viewable instances"""
        return self.instances.instance_views[0].values()

    def add(
        self,
        camera_pose: np.ndarray,
        xyz: np.ndarray,
        rgb: np.ndarray,
        feats: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        base_pose: Optional[np.ndarray] = None,
        obs: Optional[Observations] = None,
        **info,
    ):
        """Add this to our history of observations. Also update the current running map.

        Parameters:
            camera_pose(np.ndarray): necessary for measuring where the recording was taken
            xyz: N x 3 point cloud points
            rgb: N x 3 color points
            feats: N x D point cloud features; D == 3 for RGB is most common
            base_pose: optional location of robot base
            obs: observations
        """
        # TODO: we should remove the xyz/feats maybe? just use observations as input?
        # TODO: switch to using just Obs struct?
        assert (
            xyz.shape[-1] == 3
        ), "xyz must have last dimension = 3 for x, y, z position of points"
        W, H, _ = xyz.shape
        assert rgb.shape == xyz.shape, "rgb shape must match xyz"
        if feats is not None:
            assert (
                feats.shape[-1] == self.feature_dim
            ), f"features must match voxel feature dimenstionality of {self.feature_dim}"
            assert (
                xyz.shape[0] == feats.shape[0]
            ), "features must be available for each point"
        else:
            pass

        # add observations before we start changing things
        self.observations.append(
            Frame(camera_pose, xyz, rgb, feats, depth, base_pose, obs, info)
        )

        if feats is not None:
            feats = feats.reshape(-1, feats.shape[-1])
        rgb = rgb.reshape(-1, 3)
        xyz = xyz.reshape(-1, 3)
        full_world_xyz = trimesh.transform_points(xyz, camera_pose)

        if depth is not None:
            # Apply depth filter
            valid_depth = np.bitwise_and(depth > self.min_depth, depth < self.max_depth)
            valid_depth = valid_depth.reshape(-1)
            if feats is not None:
                feats = feats[valid_depth, :]
            rgb = rgb[valid_depth, :]
            xyz = xyz[valid_depth, :]
            world_xyz = full_world_xyz[valid_depth, :]
        else:
            valid_depth = None

        # TODO: tensorize earlier
        # Note that in process_instances_for_env, we assume that the background class is 0
        # In OvmmPerception, it is -1
        # This is why we add 1 to the image instance mask below.
        # TODO: it is very inefficient to do this conversion so late. We should switch to PyTorch tensors first instead of numpy matrices.
        if valid_depth is not None:
            instance = obs.instance.reshape(-1)
            instance[valid_depth == 0] = -1
            instance = instance.reshape(W, H)
        else:
            instance = obs.instance
        self.instances.process_instances_for_env(
            0,
            torch.Tensor(instance) + 1,
            torch.Tensor(full_world_xyz.reshape(W, H, 3)),
            torch.Tensor(obs.rgb).permute(2, 0, 1),
            torch.Tensor(obs.semantic),
            mask_out_object=False,  # Save the whole image here
        )
        self.instances.associate_instances_to_memory()

        # TODO: This is probably the only place we need to use Numpy for now. We can keep everything else as tensors.
        # Combine point clouds by adding in the current view to the previous ones and
        # voxelizing.
        self._pcd, self.feats = combine_point_clouds(
            self.xyz,
            self.rgb,
            world_xyz,
            rgb,
            self.feats,
            feats,
            sparse_voxel_size=self.resolution,
        )
        self.xyz, self.rgb = pcd_to_numpy(self._pcd)

        if base_pose is not None:
            self._update_visited(base_pose)

        # Increment sequence counter
        self._seq += 1

    def _update_visited(self, base_pose: np.ndarray):
        """Update 2d map of where robot has visited"""
        # Add exploration here
        # Base pose can be whatever, going to assume xyt for now
        map_xy = ((base_pose[:2] / self.grid_resolution) + self.grid_origin[:2]).astype(
            np.uint32
        )
        x0 = int(map_xy[0] - self._disk_size)
        x1 = int(map_xy[0] + self._disk_size + 1)
        y0 = int(map_xy[1] - self._disk_size)
        y1 = int(map_xy[1] + self._disk_size + 1)
        self._visited[x0:x1, y0:y1] += self._visited_disk

    def get_data(self, in_place: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return the current point cloud and features; optionally copying."""
        if in_place or self.xyz is None:
            return self.xyz, self.rgb, self.feats
        else:
            return self.xyz.copy(), self.rgb.copy(), self.feats.copy()

    def write_to_pickle(self, filename: str):
        """Write out to a pickle file. This is a rough, quick-and-easy output for debugging, not intended to replace the scalable data writer in data_tools for bigger efforts."""
        data = {}
        data["camera_poses"] = []
        data["base_poses"] = []
        data["xyz"] = []
        data["rgb"] = []
        data["depth"] = []
        data["feats"] = []
        data["obs"] = []
        for frame in self.observations:
            # add it to pickle
            # TODO: switch to using just Obs struct?
            data["camera_poses"].append(frame.camera_pose)
            data["base_poses"].append(frame.base_pose)
            data["xyz"].append(frame.xyz)
            data["rgb"].append(frame.rgb)
            data["depth"].append(frame.depth)
            data["feats"].append(frame.feats)
            data["obs"].append(frame.obs)
            for k, v in frame.info.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)
        data["combined_xyx"] = self.xyz
        data["combined_feats"] = self.feats
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def read_from_pickle(self, filename: str):
        """Read from a pickle file as above. Will clear all currently stored data first."""
        self.reset_cache()
        if isinstance(filename, str):
            filename = Path(filename)
        assert filename.exists(), f"No file found at {filename}"
        with filename.open("rb") as f:
            data = pickle.load(f)
        for camera_pose, xyz, rgb, feats, depth, base_pose, obs in zip(
            data["camera_poses"],
            data["xyz"],
            data["rgb"],
            data["feats"],
            data["depth"],
            data["base_poses"],
            data["obs"],
        ):
            self.add(camera_pose, xyz, rgb, feats, depth, base_pose, obs)

    def recompute_map(self):
        """Recompute the entire map from scratch instead of doing incremental updates.
        This is a helper function which recomputes everything from the beginning.

        Currently this will be slightly inefficient since it recreates all the objects incrementally."""
        old_observations = self.observations
        self.reset_cache()
        for frame in old_observations:
            self.add(
                frame.camera_pose,
                frame.xyz,
                frame.rgb,
                frame.feats,
                frame.depth,
                frame.base_pose,
                frame.obs,
                **frame.info,
            )

    def get_2d_map(
        self, debug: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get 2d map with explored area and frontiers."""

        # Is this already cached? If so we don't need to go to all this work
        if self._map2d is not None and self._seq == self._2d_last_updated:
            return self._map2d

        # Convert metric measurements to discrete
        # Gets the xyz correctly - for now everything is assumed to be within the correct distance of origin
        xyz = ((self.xyz / self.grid_resolution) + self.grid_origin).astype(np.uint32)

        # Crop to robot height
        min_height = int(self.obs_min_height / self.grid_resolution)
        max_height = int(self.obs_max_height / self.grid_resolution)
        # NOTE: keep this if we only care about obstacles
        # TODO: delete unused code
        # voxels = np.zeros(self.grid_size + [int(max_height - min_height)])
        # But we might want to track floor pixels as well
        voxels = np.zeros(self.grid_size + [max_height])
        # NOTE: you can use min_height for this if we only care about obstacles
        # TODO: delete unused code
        # obs_mask = np.bitwise_and(xyz[:, -1] > 0, xyz[:, -1] < max_height)
        obs_mask = xyz[:, -1] < max_height
        x_coords = xyz[obs_mask, 0]
        y_coords = xyz[obs_mask, 1]
        z_coords = xyz[obs_mask, 2]
        voxels[x_coords, y_coords, z_coords] = 1

        # Compute the obstacle voxel grid based on what we've seen
        obstacle_voxels = voxels[:, :, min_height:]
        obstacles_soft = np.sum(obstacle_voxels, axis=-1)
        obstacles = obstacles_soft > self.obs_min_density

        # Explored area = only floor mass
        floor_voxels = voxels[:, :, :min_height]
        explored_soft = np.sum(floor_voxels, axis=-1)

        # Add explored radius around the robot, up to min depth
        # TODO: make sure lidar is supported here as well; if we do not have lidar assume a certain radius is explored
        explored_soft += self._visited
        explored = explored_soft > 0

        # Frontier consists of floor voxels adjacent to empty voxels
        # TODO

        if debug:
            import matplotlib.pyplot as plt

            # TODO: uncomment to show the original world representation
            # from home_robot.utils.point_cloud import show_point_cloud
            # show_point_cloud(self.xyz, self.feats / 255., orig=np.zeros(3))
            # TODO: uncomment to show voxel point cloud
            # from home_robot.utils.point_cloud import show_point_cloud
            # show_point_cloud(xyz, self.feats/255., orig=self.grid_origin)

            plt.subplot(2, 2, 1)
            plt.imshow(obstacles_soft)
            plt.subplot(2, 2, 2)
            plt.imshow(explored_soft)
            plt.subplot(2, 2, 3)
            plt.imshow(obstacles)
            plt.subplot(2, 2, 4)
            plt.imshow(explored)
            plt.show()

        # Update cache
        self._map2d = (obstacles, explored)
        self._2d_last_updated = self._seq
        return obstacles, explored

    def get_kd_tree(self) -> open3d.geometry.KDTreeFlann:
        """Return kdtree for collision checks"""
        return open3d.geometry.KDTreeFlann(self._pcd)

    def show(self, instances: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Show and return bounding box information and rgb color information from an explored point cloud. Uses open3d."""

        # Create a combined point cloud
        # Do the other stuff we need to show instances
        pc_xyz, pc_rgb, pc_feats = self.get_data()
        pcd = numpy_to_pcd(pc_xyz, pc_rgb / 255.0)
        geoms = create_visualization_geometries(pcd=pcd, orig=np.zeros(3))
        if instances:
            for instance_view in self.get_instances():
                mins, maxs = instance_view.bounds
                if np.any(maxs - mins < 1e-5):
                    print("Warning: bad box:", mins, maxs)
                    continue
                width, height, depth = maxs - mins

                # Create a mesh to visualzie where the instances were seen
                mesh_box = open3d.geometry.TriangleMesh.create_box(
                    width=width, height=height, depth=depth
                )

                # Get vertex array from the mesh
                vertices = np.asarray(mesh_box.vertices)

                # Translate the vertices to the desired position
                vertices += mins
                triangles = np.asarray(mesh_box.triangles)

                # Create a wireframe mesh
                lines = []
                for tri in triangles:
                    lines.append([tri[0], tri[1]])
                    lines.append([tri[1], tri[2]])
                    lines.append([tri[2], tri[0]])

                # color = [1.0, 0.0, 0.0]  # Red color (R, G, B)
                color = np.random.random(3)
                colors = [color for _ in range(len(lines))]
                wireframe = open3d.geometry.LineSet(
                    points=open3d.utility.Vector3dVector(vertices),
                    lines=open3d.utility.Vector2iVector(lines),
                )
                # Get the colors and add to wireframe
                wireframe.colors = open3d.utility.Vector3dVector(colors)
                geoms.append(wireframe)

        # Show the geometries of where we have explored
        open3d.visualization.draw_geometries(geoms)
        return pc_xyz, pc_rgb


class SparseVoxelGridXYTSpace(XYT):
    """subclass for sampling XYT states from explored space"""

    def __init__(self, voxel_map: SparseVoxelMap):
        self.map = voxel_map

    def sample_uniform(self):
        """Sample any position that corresponds to an "explored" location. Goals are valid if they are within a reasonable distance of explored locations. Paths through free space are ok and don't collide."""
        # Extract 2d map from this - hopefully it is already cached
        obstacles, explored = self.map.get_2d_map()

        # Sample any point which is explored and not an obstacle

        # Sample a random orientation

        raise NotImplementedError()
