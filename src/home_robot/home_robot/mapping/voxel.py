# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import pickle
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import open3d as open3d
import torch
import trimesh
from pytorch3d.structures import Pointclouds
from torch import Tensor

from home_robot.core.interfaces import Observations
from home_robot.mapping.semantic.instance_tracking_modules import (
    InstanceMemory,
    InstanceView,
)
from home_robot.utils.bboxes_3d import BBoxes3D
from home_robot.utils.data_tools.dict import update
from home_robot.utils.point_cloud import (
    create_visualization_geometries,
    numpy_to_pcd,
    pcd_to_numpy,
    show_point_cloud,
)
from home_robot.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates
from home_robot.utils.visualization import create_disk
from home_robot.utils.voxel import VoxelizedPointcloud

Frame = namedtuple(
    "Frame",
    ["camera_pose", "camera_K", "xyz", "rgb", "feats", "depth", "base_pose", "info"],
)


DEFAULT_GRID_SIZE = [512, 512]


def ensure_tensor(arr):
    if isinstance(arr, np.ndarray):
        return Tensor(arr)
    if not isinstance(arr, Tensor):
        raise ValueError(f"arr of unknown type ({type(arr)}) cannot be cast to Tensor")


class SparseVoxelMap(object):
    """Create a voxel map object which captures 3d information."""

    DEFAULT_INSTANCE_MAP_KWARGS = dict(
        du_scale=1,
        instance_association="bbox_iou",
    )

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
        background_instance_label: int = -1,
        instance_memory_kwargs: Dict[str, Any] = {},
        voxel_kwargs: Dict[str, Any] = {},
    ):
        # TODO: We an use fastai.store_attr() to get rid of this boilerplate code
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.obs_min_height = obs_min_height
        self.obs_max_height = obs_max_height
        self.obs_min_density = obs_min_density
        self.grid_resolution = grid_resolution
        self.voxel_resolution = resolution
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.background_instance_label = background_instance_label
        self.instance_memory_kwargs = update(
            copy.deepcopy(self.DEFAULT_INSTANCE_MAP_KWARGS), instance_memory_kwargs
        )
        self.voxel_kwargs = voxel_kwargs

        # TODO: This 2D map code could be moved to another class or helper function
        #   This class could use that code via containment (same as InstanceMemory or VoxelizedPointcloud)

        # Create disk for mapping explored areas near the robot - since camera can't always see it
        self._disk_size = np.ceil(1.0 / self.grid_resolution)
        self._visited_disk = torch.from_numpy(
            create_disk(1.0 / self.grid_resolution, (2 * self._disk_size) + 1)
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
        self.grid_origin = Tensor(self.grid_size + [0]) // 2

        # Init variables
        self.reset()

    def reset(self) -> None:
        """Clear out the entire voxel map."""
        self.observations = []
        # Create an instance memory to associate bounding boxes in space
        self.instances = InstanceMemory(
            num_envs=1,
            **self.instance_memory_kwargs,
        )
        # Create voxelized pointcloud
        self.voxel_pcd = VoxelizedPointcloud(
            voxel_size=self.voxel_resolution,
            dim_mins=None,
            dim_maxs=None,
            feature_pool_method="mean",
            **self.voxel_kwargs,
        )

        self._seq = 0
        self._2d_last_updated = -1
        # Create map here - just reset *some* variables
        self.reset_cache()

    def reset_cache(self):
        """Clear some tracked things"""
        # Stores points in 2d coords where robot has been
        self._visited = torch.zeros(self.grid_size)

        # Store instances detected (all of them for now)
        self.instances.reset()

        self.voxel_pcd.reset()

        # Store 2d map information
        # This is computed from our various point clouds
        self._map2d = None

    def get_instances(self) -> List[InstanceView]:
        """Return a list of all viewable instances"""
        return tuple(self.instances.instances[0].values())

    def add(
        self,
        camera_pose: Tensor,
        rgb: Tensor,
        xyz: Optional[Tensor] = None,
        camera_K: Optional[Tensor] = None,
        feats: Optional[Tensor] = None,
        depth: Optional[Tensor] = None,
        base_pose: Optional[Tensor] = None,
        instance_image: Optional[Tensor] = None,
        instance_classes: Optional[Tensor] = None,
        instance_scores: Optional[Tensor] = None,
        # obs: Optional[Observations] = None,
        **info,
    ):
        """Add this to our history of observations. Also update the current running map.

        Parameters:
            camera_pose(Tensor): [4,4] cam_to_world matrix
            rgb(Tensor): N x 3 color points
            camera_K(Tensor): [3,3] camera instrinsics matrix -- usually pinhole model
            xyz(Tensor): N x 3 point cloud points in camera coordinates
            feats(Tensor): N x D point cloud features; D == 3 for RGB is most common
            base_pose(Tensor): optional location of robot base
            instance_image(Tensor): [H,W] image of ints where values at a pixel correspond to instance_id
            instance_classes(Tensor): [K] tensor of ints where class = instance_classes[instance_id]
            instance_scores: [K] of detection confidence score = instance_scores[instance_id]
            # obs: observations
        """
        # TODO: we should remove the xyz/feats maybe? just use observations as input?
        # TODO: switch to using just Obs struct?
        # Shape checking
        assert (
            rgb.ndim == 3 and rgb.shape[:2] == depth.shape
        ), f"{rgb.shape=} {depth.shape=}"
        H, W, _ = rgb.shape
        assert xyz is not None or (camera_K is not None and depth is not None)
        if xyz is not None:
            assert (
                xyz.shape[-1] == 3
            ), "xyz must have last dimension = 3 for x, y, z position of points"
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
        if depth is not None:
            assert depth.ndim == 2
        if camera_K is not None:
            assert camera_K.ndim == 2
        assert (
            camera_pose.ndim == 2
            and camera_pose.shape[0] == 4
            and camera_pose.shape[1] == 4
        )

        # Get full_world_xyz
        if xyz is not None:
            full_world_xyz = (
                torch.cat([xyz, torch.ones_like(xyz[..., [0]])]) @ camera_pose.T
            )
            # trimesh.transform_points(xyz, camera_pose)
        else:
            full_world_xyz = unproject_masked_depth_to_xyz_coordinates(  # Batchable!
                depth=depth.unsqueeze(0).unsqueeze(1),
                pose=camera_pose.unsqueeze(0),
                inv_intrinsics=torch.linalg.inv(camera_K[:3, :3]).unsqueeze(0),
            )

        # add observations before we start changing things
        self.observations.append(
            Frame(
                camera_pose,
                camera_K,
                full_world_xyz,
                rgb,
                feats,
                depth,
                base_pose,
                info,
            )
        )

        # filter depth
        valid_depth = torch.full_like(rgb[:, 0], fill_value=True, dtype=torch.bool)
        if depth is not None:
            valid_depth = (depth > self.min_depth) & (depth < self.max_depth)

        # Add instance views to memory
        instance = instance_image.clone()
        # instance[~valid_depth] = -1
        self.instances.process_instances_for_env(
            env_id=0,
            instance_seg=instance,
            point_cloud=full_world_xyz.reshape(H, W, 3),
            image=rgb.permute(2, 0, 1),
            instance_classes=instance_classes,
            instance_scores=instance_scores,
            mask_out_object=False,  # Save the whole image here? Or is this with background?
            background_instance_label=self.background_instance_label,
            valid_points=valid_depth,
        )
        self.instances.associate_instances_to_memory()

        # Add to voxel grid
        if feats is not None:
            feats = feats[valid_depth].reshape(-1, feats.shape[-1])
        rgb = rgb[valid_depth].reshape(-1, 3)
        world_xyz = full_world_xyz[valid_depth.flatten()]

        # TODO: weights could also be confidence, inv distance from camera, etc
        self.voxel_pcd.add(world_xyz, features=feats, rgb=rgb, weights=None)

        # Visited
        # TODO: just get this from camera_pose?
        if base_pose is not None:
            self._update_visited(base_pose)

        # Increment sequence counter
        self._seq += 1

    def _update_visited(self, base_pose: Tensor):
        """Update 2d map of where robot has visited"""
        # Add exploration here
        # Base pose can be whatever, going to assume xyt for now
        map_xy = ((base_pose[:2] / self.grid_resolution) + self.grid_origin[:2]).int()
        x0 = int(map_xy[0] - self._disk_size)
        x1 = int(map_xy[0] + self._disk_size + 1)
        y0 = int(map_xy[1] - self._disk_size)
        y1 = int(map_xy[1] + self._disk_size + 1)
        self._visited[x0:x1, y0:y1] += self._visited_disk

    def get_data(self, in_place: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return the current point cloud and features; optionally copying."""
        raise NotImplementedError("Should this return pointcloud? Instances?")

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
        (
            data["combined_xyz"],
            data["combined_feats"],
            data["combined_weights"],
            data["combined_rgb"],
        ) = self.voxel_pcd.get_pointcloud()
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
        xyz, _, _, _ = self.voxel_pcd.get_pointcloud()
        device = xyz.device
        xyz = ((self.xyz / self.grid_resolution) + self.grid_origin).int()

        # Crop to robot height
        min_height = int(self.obs_min_height / self.grid_resolution)
        max_height = int(self.obs_max_height / self.grid_resolution)
        # NOTE: keep this if we only care about obstacles
        # TODO: delete unused code
        # voxels = np.zeros(self.grid_size + [int(max_height - min_height)])
        # But we might want to track floor pixels as well
        voxels = torch.zeros(self.grid_size + [max_height], device=device)
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
        obstacles_soft = torch.sum(obstacle_voxels, dim=-1)
        obstacles = obstacles_soft > self.obs_min_density

        # Explored area = only floor mass
        floor_voxels = voxels[:, :, :min_height]
        explored_soft = torch.sum(floor_voxels, dim=-1)

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
            plt.imshow(obstacles_soft.detach().cpu().numpy())
            plt.subplot(2, 2, 2)
            plt.imshow(explored_soft.detach().cpu().numpy())
            plt.subplot(2, 2, 3)
            plt.imshow(obstacles.detach().cpu().numpy())
            plt.subplot(2, 2, 4)
            plt.imshow(explored.detach().cpu().numpy())
            plt.show()

        # Update cache
        self._map2d = (obstacles, explored)
        self._2d_last_updated = self._seq
        return obstacles, explored

    def get_kd_tree(self) -> open3d.geometry.KDTreeFlann:
        """Return kdtree for collision checks

        We could use Kaolin to get octree from pointcloud.
        Not hard to parallelize on GPU:
            Octree has K levels, each cube in level k corresponds to a  regular grid of "supervoxels"
            Occupancy can be done for each level in parallel.
        Hard part is converting to KDTreeFlann (or modifying the collision check to run on gpu)
        """
        points, _, _, rgb = self.voxel_pcd.get_pointcloud()
        pcd = numpy_to_pcd(points.detach().cpu().numpy(), rgb.detach().cpu().numpy())
        return open3d.geometry.KDTreeFlann(pcd)

    def show(
        self, instances: bool = True, backend: str = "open3d", **backend_kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Display the aggregated point cloud."""
        if backend == "open3d":
            return self._show_open3d(instances, **backend_kwargs)
        elif backend == "pytorch3d":
            return self._show_pytorch3d(instances, **backend_kwargs)
        else:
            raise NotImplementedError(
                f"Uknown backend {backend}, must be 'open3d' or 'pytorch3d"
            )

    def _show_pytorch3d(self, instances: bool = True, **plot_scene_kwargs):
        from pytorch3d.vis.plotly_vis import AxisArgs, plot_scene

        from home_robot.utils.bboxes_3d_plotly import plot_scene_with_bboxes

        points, _, _, rgb = self.voxel_pcd.get_pointcloud()

        traces = {}

        # Show points
        ptc = Pointclouds(points=[points], features=[rgb])
        traces["Points"] = ptc

        # Show instances
        if instances:
            bounds, names = zip(
                *[(v.bounds, v.category_id) for v in self.get_instances()]
            )
            detected_boxes = BBoxes3D(
                bounds=[torch.stack(bounds, dim=0)],
                # features = [colors],
                names=[torch.stack(names, dim=0).unsqueeze(-1)],
            )
            traces["IB"] = detected_boxes

        # Show cameras
        # "Fused boxes": global_boxes,
        # "cameras": cameras,

        _default_plot_args = dict(
            xaxis={"backgroundcolor": "rgb(200, 200, 230)"},
            yaxis={"backgroundcolor": "rgb(230, 200, 200)"},
            zaxis={"backgroundcolor": "rgb(200, 230, 200)"},
            axis_args=AxisArgs(showgrid=True),
            pointcloud_marker_size=3,
            pointcloud_max_points=200_000,
            boxes_plot_together=True,
            boxes_wireframe_width=3,
        )
        fig = plot_scene_with_bboxes(
            plots={"Global scene": traces},
            **update(_default_plot_args, plot_scene_kwargs),
        )
        return fig

    def _show_open3d(self, instances: bool = True, **backend_kwargs):
        """Show and return bounding box information and rgb color information from an explored point cloud. Uses open3d."""

        # Create a combined point cloud
        # Do the other stuff we need to show instances
        # pc_xyz, pc_rgb, pc_feats = self.get_data()
        points, _, _, rgb = self.voxel_pcd.get_pointcloud()
        pcd = numpy_to_pcd(points.detach().cpu().numpy(), rgb.detach().cpu().numpy())
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
        return points, rgb
