# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import numpy as np
import seaborn as sns
import torch
from omegaconf import DictConfig
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from sklearn.cluster import DBSCAN
from torch import Tensor
from torchvision import transforms
from tqdm import trange

import home_robot.utils.image as im
from home_robot.mapping.instance import Instance, InstanceView
from home_robot.perception.encoders import BaseImageTextEncoder
from home_robot.utils.bboxes_3d import box3d_volume_from_bounds
from home_robot.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates
from home_robot.utils.voxel import VoxelizedPointcloud

COLOR_LIST = [
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
    [0.0, 0.0, 0.5],
    [0.0, 0.5, 1.0],
]


custom_palette = sns.color_palette("viridis", 24)

logger = logging.getLogger(__name__)


def get_bounding_boxes(clusterer: DBSCAN, data: Tensor):
    """
    Clusters points into instances and returns bounding box around each cluster

    Args:
        clusterer: calls clusterer.fit(data) and returns db.labels_
        data: [N, D]

    Returns:
        box_bounds: [K, D, 2] (mins and maxes)
        data_labels: [N] (cluster idx)
    """
    # db = DBSCAN(eps=args.epsilon, min_samples=args.min_samples).fit(data.cpu().numpy())
    db = clusterer.fit(data.cpu().numpy())
    labels = db.labels_

    num_clusters = len(np.unique(labels))
    num_noise = np.sum(np.array(labels) == -1, axis=0)

    logger.debug("Estimated no. of clusters: %d" % num_clusters)
    logger.debug("Estimated no. of noisy points: %d" % num_noise)

    # get max and min x, y, z values for each cluster
    bb_coords = torch.zeros((num_clusters, 3, 2), device=data.device)
    for i in range(num_clusters - 1):
        cluster = data[labels == i]
        bb_coords[i, :, 0] = torch.min(cluster, axis=0)[0]
        bb_coords[i, :, 1] = torch.max(cluster, axis=0)[0]

        # use different bright colors for points in different clusters
        # color_data[labels == i] = torch.tensor(COLOR_LIST[i % len(COLOR_LIST)], device=data.device).float()
    return bb_coords, labels


class ConceptFusion:
    def __init__(
        self,
        mask_generator: SamAutomaticMaskGenerator,
        image_text_encoder: BaseImageTextEncoder,
        voxel_ptc: VoxelizedPointcloud,
        clusterer: DBSCAN,
        similarity_params: DictConfig,
        file_params: DictConfig,
        device: Optional[str] = None,
        min_depth: float = 0.1,
        max_depth: float = 4.0,
    ) -> None:
        """
        Initialize concept fusion model.

        Args:
            device (str): Device to use.
            sam_params (DictConfig): SAM parameters.
            open_clip_params (DictConfig): OpenCLIP parameters.
            data_params (DictConfig): Data parameters.
            dbscan_params (DictConfig): DBSCAN parameters.
            similarity_params (DictConfig): Similarity parameters.
            file_params (DictConfig): File parameters.
        """
        self.device = device
        self.mask_generator = mask_generator
        self.image_text_encoder = image_text_encoder
        self.voxel_map = voxel_ptc
        if clusterer is None:
            clusterer = DBSCAN()
        self.clusterer = clusterer
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.similarity_params = similarity_params
        self.file_params = file_params

        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        self.feat_dim = None
        self.class_id_to_class_names = None
        self.class_names_to_class_id = None

    def clear(self):
        self.voxel_map.reset()
        self.feat_dim = None

    def generate_mask(self, img: np.ndarray):
        masks = self.mask_generator.generate(img)

        # remove masks with zero area
        masks = list(filter(lambda x: x["bbox"][2] * x["bbox"][3] != 0, masks))

        return masks

    def set_vocabulary(self, class_id_to_class_names: Dict[int, str]):
        self.class_id_to_class_names = class_id_to_class_names
        self.class_names_to_class_id = {
            v: k for k, v in class_id_to_class_names.items()
        }

    def generate_global_features(
        self,
        img: np.ndarray,
    ):
        # CLIP features global
        global_feat = None
        with torch.cuda.amp.autocast():
            global_feat = self.image_text_encoder.encode_image(img)
            global_feat /= global_feat.norm(dim=-1, keepdim=True)

        global_feat = global_feat.half().to(self.device)
        global_feat = torch.nn.functional.normalize(
            global_feat, dim=-1
        )  # --> (1, 1024)

        if self.feat_dim is None:
            self.feat_dim = global_feat.shape[-1]

        return global_feat

    def generate_local_features(
        self,
        img: np.ndarray,
        masks: List[dict],
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate concept fusion features.

        Args:
            img (Image): Original image.
            masks (list[dict]): List of segmentation masks.
            global_feat (torch.Tensor): CLIP features global.

        Returns:
            torch.Tensor: Concept fusion features.
        """
        load_image_height, load_image_width = img.shape[0], img.shape[1]

        # CLIP features per ROI
        feat_per_roi = []
        roi_nonzero_inds = []
        similarity_scores = []
        outfeat = torch.zeros(
            load_image_height,
            load_image_width,
            self.feat_dim,
            dtype=torch.half,
            device=self.device,
        )

        for mask in masks:
            _x, _y, _w, _h = tuple(mask["bbox"])  # xywh bounding box

            # make sure _x, _y, _w, _h are ints
            _x, _y, _w, _h = int(_x), int(_y), int(_w), int(_h)

            nonzero_inds = torch.argwhere(torch.from_numpy(mask["segmentation"]))

            # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
            img_roi = img[_y : _y + _h, _x : _x + _w, :]

            roifeat = self.image_text_encoder.encode_image(img_roi)
            roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(nonzero_inds)
            _sim = self.cosine_similarity(global_feat, roifeat)
            similarity_scores.append(_sim)

        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)

        for maskidx in range(len(masks)):

            _weighted_feat = (
                softmax_scores[maskidx] * global_feat
                + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
            )
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
            outfeat[
                roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
            ] += (_weighted_feat[0].detach().half())
            outfeat[
                roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
            ] = torch.nn.functional.normalize(
                outfeat[
                    roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
                ].float(),
                dim=-1,
            ).half()

        outfeat = outfeat.unsqueeze(
            0
        ).float()  # interpolate is not implemented for float yet in pytorch
        outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        outfeat = torch.nn.functional.interpolate(
            outfeat, [load_image_height, load_image_width], mode="nearest"
        )
        outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
        outfeat = outfeat[0].half()  # --> H, W, feat_dim

        return outfeat

    def query_similarity(
        self,
        query: str,
        map_features: torch.Tensor,
    ):
        # text = self.tokenizer([query])
        textfeat = self.image_text_encoder.encode_text(query)
        textfeat = torch.nn.functional.normalize(textfeat, dim=-1)
        textfeat = textfeat.unsqueeze(0)

        # make sure features are on cuda
        map_features = map_features.to(self.device)
        map_features = torch.nn.functional.normalize(map_features, dim=-1)

        similarity = self.cosine_similarity(textfeat, map_features)

        # We use relative similarity
        similarity = (similarity - similarity.min()) / (
            similarity.max() - similarity.min() + 1e-12
        )

        if self.similarity_params.viz_type == "topk":
            # Viz topk points
            _, topk_ind = torch.topk(similarity, self.similarity_params.topk)
            selected_inds = topk_ind

        elif self.similarity_params.viz_type == "thresh":
            # Viz thresholded "relative" attention scores
            similarity[similarity < self.similarity_params.similarity_thresh] = 0.0
            selected_inds = torch.nonzero(similarity.squeeze()).squeeze(1)

        return selected_inds, similarity.squeeze()

    def text_queries(
        self,
        queries: Sequence[str],
    ):
        """
        Args:
            queries: List of string queries

        Returns:
            Dict[query, List[Instances]]
        """
        pc_xyz, pc_feat, _, pc_rgb = self.voxel_map.get_pointcloud()

        textfeat = self.image_text_encoder.encode_text(queries)
        textfeat = torch.nn.functional.normalize(textfeat, dim=-1)

        pc_feat = pc_feat.to(self.device)
        pc_feat = torch.nn.functional.normalize(pc_feat, dim=-1)

        # match dimensions to do cosine similarity
        textfeat = textfeat.unsqueeze(1).repeat(1, pc_feat.shape[0], 1)
        pc_feat_expanded = pc_feat.unsqueeze(0).repeat(textfeat.shape[0], 1, 1)

        similarity = self.cosine_similarity(
            textfeat, pc_feat_expanded
        )  # [n_points, n_classes]

        del pc_feat_expanded

        point_class = similarity.argmax(dim=0)

        instances_dict = {}
        for idx, class_name in enumerate(queries):
            selected_inds = torch.nonzero(
                (point_class == idx)
                & (similarity[idx] > self.similarity_params.similarity_thresh)
            ).squeeze(1)

            pc_feat_objects = pc_feat[selected_inds]
            pc_rgb_objects = pc_rgb[selected_inds]
            pc_xyz_objects = pc_xyz[selected_inds]
            similarity_score_objects = similarity[idx, selected_inds].squeeze()

            bounding_boxes, labels = get_bounding_boxes(self.clusterer, pc_xyz_objects)

            instances_per_class = []
            for idx in range(len(bounding_boxes)):
                bounds = bounding_boxes[idx]

                volume = float(box3d_volume_from_bounds(bounds).squeeze())
                min_dim = (bounds[:, 1] - bounds[:, 0]).min()

                if volume < 1e-6 or min_dim < 0.01:
                    warnings.warn(
                        f"Skipping box with bounding box {bounds}, volume {volume} and min_dim {min_dim}",
                        UserWarning,
                    )
                else:
                    instance = Instance(score_aggregation_method="max")

                    # TODO: get rid of the instance view altogether
                    instance_view = InstanceView(
                        bbox=None,
                        bounds=bounds,
                        timestep=0,
                        embedding=pc_feat_objects[labels == idx].mean(axis=0),
                        point_cloud=pc_xyz_objects[labels == idx],
                        point_cloud_rgb=pc_rgb_objects[labels == idx],
                        category_id=self.class_names_to_class_id[class_name],
                        score=similarity_score_objects[labels == idx].max(),
                    )
                    # append instance view to list of instance views
                    instance.add_instance_view(instance_view)
                    instances_per_class.append(instance)

            instances_dict[class_name] = instances_per_class

        return instances_dict

    def build_scene(self, scene_obs: Dict[str, Any]):
        """
        Build scene and get pointcloud.

        Args:
            scene_obs (Dict[str, Any]): Scene observations.

        Returns:
            torch.Tensor: Pointcloud xyz.
            torch.Tensor: Pointcloud features.
            torch.Tensor: Pointcloud TODO.
            torch.Tensor: Pointcloud rgb.
        """
        for i in trange(len(scene_obs["images"])):
            img = (scene_obs["images"][i].cpu().numpy() * 255).astype(np.uint8)
            depth = scene_obs["depths"][i].permute(2, 0, 1)

            camera_pose = scene_obs["poses"][i].float()
            # original_image_size = img.shape[:2]
            # img, depth = self.resize_images(img, depth)
            # adjusted_intrinsics = adjust_intrinsics_matrix(
            #     original_intrinsics,
            #     original_image_size,
            #         (self.data_params.desired_height, self.data_params.desired_width),
            # )
            masks = self.generate_mask(img)

            # CLIP features global
            global_feat = self.generate_global_features(img)

            # CLIP features per ROI
            outfeat = self.generate_local_features(img, masks, global_feat)

            original_intrinsics = scene_obs["intrinsics"][i]

            xyz = unproject_masked_depth_to_xyz_coordinates(
                depth.unsqueeze(0),
                camera_pose.unsqueeze(0),
                original_intrinsics.inverse()[:3, :3].unsqueeze(0),
            )
            valid_depth = torch.full_like(
                depth.squeeze(0), fill_value=True, dtype=torch.bool
            )
            if depth is not None:
                valid_depth = (depth > self.min_depth) & (depth < self.max_depth)
            valid_depth = valid_depth.flatten()
            xyz = xyz[valid_depth]
            features = outfeat.reshape(-1, outfeat.shape[-1])[valid_depth].to(
                self.device
            )
            rgb = (
                torch.tensor(img, device=self.device)
                .reshape(-1, 3)[valid_depth]
                .to(self.device)
                .float()
            )
            self.voxel_map.add(points=xyz, features=features, rgb=rgb)

        return self.voxel_map.get_pointcloud()

    def get_instances_for_queries(self, queries: Sequence[str]):
        """
        Args:
            queries: List of string queries

        Returns:
            Dict[query, List[Instances]]
        """

        pc_xyz, pc_feat, _, pc_rgb = self.voxel_map.get_pointcloud()
        instances_dict = {}
        for class_name in queries:
            logger.debug("Querying for class: {}".format(class_name))
            selected_inds, similarity_score = self.query_similarity(class_name, pc_feat)

            if selected_inds.shape[0] <= 1:
                warnings.warn(
                    f"No points found for class {class_name}",
                    UserWarning,
                )
                continue

            pc_feat_objects = pc_feat[selected_inds]
            # pc_rgb_objects = pc_rgb_after_query[selected_inds].squeeze()
            pc_xyz_objects = pc_xyz[selected_inds].squeeze()
            similarity_score_objects = similarity_score[selected_inds].squeeze()
            bounding_boxes, labels = get_bounding_boxes(self.clusterer, pc_xyz_objects)

            instances_per_class = []
            for idx in range(len(bounding_boxes)):
                bounds = bounding_boxes[idx]

                volume = float(box3d_volume_from_bounds(bounds).squeeze())
                min_dim = (bounds[:, 1] - bounds[:, 0]).min()

                if volume < 1e-6 or min_dim < 0.01:
                    warnings.warn(
                        f"Skipping box with bounding box {bounds}, volume {volume} and min_dim {min_dim}",
                        UserWarning,
                    )
                else:
                    instance = Instance(score_aggregation_method="max")

                    # TODO: get rid of the instance view altogether
                    instance_view = InstanceView(
                        bbox=None,
                        bounds=bounds,
                        timestep=0,
                        embedding=pc_feat_objects[labels == idx].mean(axis=0),
                        point_cloud=pc_xyz_objects[labels == idx],
                        point_cloud_rgb=pc_rgb[selected_inds][labels == idx],
                        category_id=self.class_names_to_class_id[class_name],
                        score=similarity_score_objects[labels == idx].max(),
                    )
                    # append instance view to list of instance views
                    instance.add_instance_view(instance_view)
                    instances_per_class.append(instance)

            instances_dict[class_name] = instances_per_class

        return instances_dict

    def build_scene_and_get_instances_for_queries(
        self, scene_obs: Dict[str, Any], queries: Sequence[str]
    ):
        self.build_scene(scene_obs)

        if self.similarity_params.viz_type == "classify_all":
            return self.text_queries(queries)
        else:
            return self.get_instances_for_queries(queries)

    def show_point_cloud_query_pytorch3d(self, query, **plot_scene_kwargs):
        from pytorch3d.structures import Pointclouds
        from pytorch3d.vis.plotly_vis import AxisArgs

        from home_robot.utils.bboxes_3d import BBoxes3D
        from home_robot.utils.bboxes_3d_plotly import plot_scene_with_bboxes
        from home_robot.utils.data_tools.dict import update

        pc_xyz, pc_feat, _, pc_rgb = self.voxel_map.get_pointcloud()
        inds, similarity = self.query_similarity("chair", pc_feat)
        cmap = matplotlib.colormaps["jet"]
        similarity_colormap = cmap(similarity.detach().cpu().numpy())[:, :3]
        map_colors = torch.tensor(similarity_colormap)

        traces = {}
        traces["Points"] = Pointclouds(points=[pc_xyz.cpu()], features=[map_colors])
        _default_plot_args = dict(
            xaxis={"backgroundcolor": "rgb(200, 200, 230)"},
            yaxis={"backgroundcolor": "rgb(230, 200, 200)"},
            zaxis={"backgroundcolor": "rgb(200, 230, 200)"},
            axis_args=AxisArgs(showgrid=True),
            pointcloud_marker_size=3,
            pointcloud_max_points=500_000,
            boxes_plot_together=False,
            boxes_wireframe_width=3,
        )
        fig = plot_scene_with_bboxes(
            plots={"Conceptfusion Pointcloud": traces},
            **update(_default_plot_args, plot_scene_kwargs),
        )
        fig.update_layout(
            height=800,
            width=1600,
        )
        return fig

    def show_point_cloud_pytorch3d(self, instances_dict=None, **plot_scene_kwargs):
        """Visualize the created pointcloud using pytorch3d.

        Args:
            idx (int): Instance index

        Returns:
            ptc_fig: Plotly visualization of pointcloud
        """
        from pytorch3d.structures import Pointclouds
        from pytorch3d.vis.plotly_vis import AxisArgs

        from home_robot.utils.bboxes_3d import BBoxes3D
        from home_robot.utils.bboxes_3d_plotly import plot_scene_with_bboxes
        from home_robot.utils.data_tools.dict import update

        traces = {}

        pc_xyz, _, _, pc_rgb = self.voxel_map.get_pointcloud()

        traces["Points"] = Pointclouds(points=[pc_xyz], features=[pc_rgb / 255.0])

        # Show instances
        if instances_dict:
            for class_name, instances in instances_dict.items():
                if len(instances) == 0:
                    continue
                bounds, names, colors = [], [], []
                for instance in instances:
                    bounds.append(instance.bounds)
                    names.append(torch.tensor(instance.category_id, device=self.device))
                    colors.append(
                        torch.tensor(
                            COLOR_LIST[instance.category_id % len(COLOR_LIST)],
                            device=self.device,
                        )
                    )
                detected_boxes = BBoxes3D(
                    bounds=[torch.stack(bounds, dim=0)],
                    features=[torch.stack(colors, dim=0)],
                    names=[torch.stack(names, dim=0).unsqueeze(-1)],
                )
                traces[
                    class_name + "_bbox_" + str(round(instance.score.item(), 2))
                ] = detected_boxes

        _default_plot_args = dict(
            xaxis={"backgroundcolor": "rgb(200, 200, 230)"},
            yaxis={"backgroundcolor": "rgb(230, 200, 200)"},
            zaxis={"backgroundcolor": "rgb(200, 230, 200)"},
            axis_args=AxisArgs(showgrid=True),
            pointcloud_marker_size=3,
            pointcloud_max_points=500_000,
            boxes_plot_together=False,
            boxes_wireframe_width=3,
        )
        fig = plot_scene_with_bboxes(
            plots={"Conceptfusion Pointcloud": traces},
            **update(_default_plot_args, plot_scene_kwargs),
        )
        fig.update_layout(
            height=800,
            width=1600,
        )
        return fig

    def save_input_data(
        self,
        original_image: torch.Tensor,
        masks: List[dict],
        idx: int,
        save_path: str,
    ):
        """
        Save original image and segmentation masks for debugging.

        Args:
            original_image (torch.Tensor): Original image.
            masks (list[dict]): List of segmentation masks.
            outfeat (torch.Tensor): Concept fusion features.
            idx (int): Index of image.
            save_path (str): Path to save data.
        """
        # create directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save original image
        original_image = Image.fromarray(original_image)
        file_path = os.path.join(save_path, "original_" + str(idx) + ".png")
        original_image.save(file_path)

        # save segmentation masks
        segmentation_image = torch.zeros(
            original_image.size[0], original_image.size[0], 3
        )
        for i, mask in enumerate(masks):
            segmentation_image += (
                torch.from_numpy(mask["segmentation"])
                .unsqueeze(-1)
                .repeat(1, 1, 3)
                .float()
                * torch.tensor(custom_palette[i % 24])
                * 255.0
            )

        mask = Image.fromarray(segmentation_image.numpy().astype("uint8"))
        file_path = os.path.join(save_path, "mask_" + str(idx) + ".png")
        mask.save(file_path)


def write_pointcloud(filename, xyz_points, rgb_points=None):
    """creates a .pkl file of the point clouds generated"""
    import pytorch3d

    assert xyz_points.shape[1] == 3, "Input XYZ points should be Nx3 float array"
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8) * 255
    assert (
        xyz_points.shape == rgb_points.shape
    ), "Input RGB colors should be Nx3 float array and have same size as input XYZ points"

    rgb_points = [rgb_points] if rgb_points is not None else []
    pointcloud = pytorch3d.structures.Pointclouds(
        points=[xyz_points], features=rgb_points
    )
    pytorch3d.io.IO().save_pointcloud(pointcloud)
