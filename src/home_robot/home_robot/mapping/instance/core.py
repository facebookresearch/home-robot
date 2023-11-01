# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from home_robot.utils.point_cloud_torch import get_bounds


@dataclass
class InstanceView:
    """
    Stores information about a single view of a single instance
    """

    # Required: 2D and 3D bbox. Should we unify the names?
    bbox: Tensor
    """ [4,] bbox: bounding box of instance in the current image """
    bounds: Tensor
    """[3, 2] xyz mins and maxes"""
    """TODO: rename to bounds_3d"""
    timestep: int
    """ timestep: timestep at which the current view was recorded """

    # View info
    cropped_image: Optional[Tensor] = None
    """ cropped_image: cropped image of instance in the current image"""
    embedding: Optional[Tensor] = None
    """ embedding: embedding of instance in the current image """
    mask: Tensor = None
    """ mask: mask of instance in the current (uncropped) image """
    image_instance_id: Optional[int] = None
    """ID of this instance in the image"""

    # Detection info
    global_instance_id: Optional[int] = None
    """ID of this instance in the scene (across views)"""
    category_id: Optional[int] = None
    """category_id: category id of instance in the current image"""
    score: Optional[float] = None
    """score: confidence of the detection (used for NMS)"""

    # 3D info (rename point_cloud to points_3d?)
    point_cloud: Tensor = None
    """point_cloud: 3d locations of world points corresponding to instance view pixels"""
    point_cloud_rgb: Tensor = None
    """point_cloud_rgb: rgb colors corrsponding to point_cloud"""
    point_cloud_features: Tensor = None
    """point_cloud_features: features corresponding to point_clouds"""
    cam_to_world: Tensor = None
    """[4,4] Tensor pose matrix mapping camera space to world space"""

    # Where did we observe this from
    pose: Tensor = None
    """ Base pose of the robot when this view was collected"""

    @cached_property
    def object_coverage(self):
        return float(self.mask.sum()) / self.mask.size

    def show(self, backend="folder", **backend_kwargs):
        assert backend in ["folder"], backend
        self._show_folder(self, **backend_kwargs)

    def _show_folder(self, folder_path: Union[Path, str]):
        import os

        import cv2

        full_image = self.cropped_image
        full_image = (
            (full_image * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        )
        # overlay mask on image
        mask = np.zeros(full_image.shape, full_image.dtype)
        mask[:, :] = (0, 0, 255)
        mask = cv2.bitwise_and(mask, mask, mask=self.mask.astype(np.uint8))
        masked_image = cv2.addWeighted(mask, 1, full_image, 1, 0)
        os.makedirs(folder_path, exist_ok=True)
        cv2.imwrite(
            f"{folder_path}/{self.timestep}_{self.image_instance_id}_cat_{self.category_id}.png",
            masked_image,
        )


@dataclass
class Instance:
    """
    A single instance found in the environment. Each instance is composed of a list of InstanceView objects, each of which is a view of the instance at a particular timestep.
    TODO: make instances a VoxelizedPointcloud?
    """

    name: str = None
    category_id: int = None
    """Integer indicating the category"""
    point_cloud: Tensor = None
    """point_cloud: aggregated point cloud for the instance """
    point_cloud_rgb: Tensor = None
    """point_cloud_rgb: aggregated point cloud colors for the instance """
    point_cloud_features: Tensor = None
    """point_cloud_features: aggregated point cloud features for the instance """
    bounds: Tensor = None
    """ 3 x 2 mins and maxes """
    instance_views: List[InstanceView] = field(default_factory=list)
    """List of all instance views"""
    score: float = None
    """Confidence score of bbox detection"""
    score_aggregation_method: str = "max"

    def get_image_embedding(self, aggregation_method="max", normalize: bool = True):
        """Get the combined image embedding across all views"""
        view_embeddings = [view.embedding for view in self.instance_views]
        if len(view_embeddings) > 0 and view_embeddings[0] is None:
            return [None] * len(view_embeddings)
        # Create one tensor for all of these
        view_embeddings = torch.concatenate(view_embeddings, dim=0)
        if aggregation_method == "concatenate":
            emb = view_embeddings
        elif aggregation_method == "max":
            emb = view_embeddings.max(dim=0).values
        elif aggregation_method == "mean":
            emb = view_embeddings.mean(dim=0)
        else:
            raise RuntimeError(
                f"Unsupported aggregation method {aggregation_method}. Options: max, mean."
            )
        if normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def get_best_view(self, metric: str = "area") -> InstanceView:
        """Get best view by some metric."""
        best_view = None
        if metric == "area":
            best_area = 0
            for view in self.instance_views:
                if view.cropped_image is None:
                    continue
                h, w = view.cropped_image.shape[:2]
                area = h * w
                if area > best_area:
                    best_area = area
                    best_view = view
        else:
            raise NotImplementedError(f"metric {metric} not supported")
        return best_view

    def add_instance_view(self, instance_view: InstanceView):
        if len(self.instance_views) == 0:
            # instantiate from instance
            self.category_id = instance_view.category_id
            self.instance_views.append(instance_view)
            self.bounds = instance_view.bounds
            self.point_cloud = instance_view.point_cloud
            self.point_cloud_rgb = instance_view.point_cloud_rgb
            self.point_cloud_features = instance_view.point_cloud_features
            self.score = instance_view.score
        else:
            # Right now we concatenate point clouds
            # To keep the number of points manageable, we could make the pointcloud a VoxelizedPointcloud class
            self.point_cloud = torch.cat(
                [self.point_cloud, instance_view.point_cloud], dim=0
            )
            if self.point_cloud_rgb is not None:
                self.point_cloud_rgb = torch.cat(
                    [self.point_cloud_rgb, instance_view.point_cloud_rgb],
                    dim=0,
                )
            if self.point_cloud_features is not None:
                self.point_cloud_features = torch.cat(
                    [
                        self.point_cloud_features,
                        instance_view.point_cloud_features,
                    ],
                    dim=0,
                )
            if self.score is None:
                self.score = instance_view.score
            elif self.score_aggregation_method == "max":
                self.score = max(self.score, instance_view.score)
            elif self.score_aggregation_method == "mean":
                self.score = (
                    self.score * len(self.instance_views) + instance_view.score
                ) / (len(self.instance_views) + 1)
            else:
                raise NotImplementedError(
                    f'Unknown score_aggregation_method "{self.score_aggregation_method}"'
                )

            # add instance view to global instance
            # do this last because we use the current length for computing average score above
            self.instance_views.append(instance_view)

            self.bounds = get_bounds(self.point_cloud)

    def _show_point_cloud_open3d(self, **kwargs):
        from home_robot.utils.point_cloud import show_point_cloud

        show_point_cloud(self.point_cloud, self.point_cloud_rgb / 255.0, **kwargs)

    def _show_point_cloud_pytorch3d(self, **plot_scene_kwargs):
        """Visualize an instance in the map

        Args:
            idx (int): Instance index

        Returns:
            ptc_fig: Plotly visualization of pointcloud
        """
        from pytorch3d.structures import Pointclouds
        from pytorch3d.vis.plotly_vis import AxisArgs, plot_scene

        from home_robot.utils.bboxes_3d_plotly import plot_scene_with_bboxes
        from home_robot.utils.data_tools.dict import update

        # Show points
        features = [self.point_cloud_rgb] if self.point_cloud_rgb is not None else None
        ptc = Pointclouds(points=[self.point_cloud], features=features)

        _default_plot_args = dict(
            xaxis={"backgroundcolor": "rgb(230, 200, 200)"},
            yaxis={"backgroundcolor": "rgb(200, 230, 200)"},
            zaxis={"backgroundcolor": "rgb(200, 200, 230)"},
            axis_args=AxisArgs(showgrid=True),
            pointcloud_marker_size=3,
            pointcloud_max_points=200_000,
        )
        fig = plot_scene_with_bboxes(
            plots={
                f"Name {self.name}: (category: {self.category_id}) -- {len(self.instance_views)} views": {
                    "Points": ptc,
                    # "Instance boxes": detected_boxes,
                    # "Fused boxes": global_boxes,
                    # "cameras": cameras,
                },
                # Could add keyframes or instances here.
            },
            **update(_default_plot_args, plot_scene_kwargs),
        )
        return fig

    def _show_instance_view_frames(self, mask_out_opacity=0.3, imsize=3, n_col=2):
        import matplotlib.pyplot as plt

        n_views = len(self.instance_views)
        n_rows = (
            n_views + n_col - 1
        ) // n_col  # Calculate the number of rows based on n_col

        plt.figure(figsize=(n_col * imsize, n_rows * imsize))

        n_views = len(self.instance_views)
        f, axarr = plt.subplots(n_rows, n_col)  # n_rows rows, n_col columns

        for i, view in enumerate(self.instance_views):
            if n_col > 1:
                row = i // n_col  # Calculate the row index
                col = i % n_col  # Calculate the column index
                ax = axarr[row, col]
            elif n_rows > 1:
                ax = axarr[i]
            else:
                ax = axarr
            cropped_image_np = torch.from_numpy(
                view.cropped_image.detach().cpu().numpy()
            )
            mask_np = torch.from_numpy(view.mask.detach().cpu().numpy())
            if (
                mask_np.shape[0] != cropped_image_np.shape[0]
                or mask_np.shape[1] != cropped_image_np.shape[1]
            ):
                mask_np = mask_np[
                    view.bbox[0, 0] : view.bbox[1, 0], view.bbox[0, 1] : view.bbox[1, 1]
                ]
            display_im = cropped_image_np * mask_out_opacity + (
                1 - mask_out_opacity
            ) * (cropped_image_np * (mask_np))
            ax.imshow((display_im))
            ax.set_title(f"View {i}")

        # Remove any empty subplots
        for i in range(n_views, n_rows * n_col):
            axarr.flatten()[i].axis("off")

        plt.tight_layout()
        plt.show()
        return f
