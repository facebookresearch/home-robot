import logging
import os
from pathlib import Path
from tqdm import trange
from typing import List, Sequence, Dict, Any
import warnings

import matplotlib
import numpy as np
from omegaconf import DictConfig
import open_clip
from PIL import Image
import seaborn as sns
from sklearn.cluster import DBSCAN
import torch
from torchvision import transforms

from home_robot.utils.voxel import VoxelizedPointcloud
import home_robot.utils.image as im
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

from home_robot.utils.bboxes_3d import box3d_volume_from_bounds
from home_robot.mapping.semantic.instance_tracking_modules import InstanceView, Instance
from home_robot.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates

from utils import adjust_intrinsics_matrix, COLOR_LIST

custom_palette = sns.color_palette("viridis", 24)

logger = logging.getLogger(__name__)

def convert_pose_to_real_world_axis(hab_pose):
    """Update axis convention of habitat pose to match the real-world axis convention"""
    hab_pose[[1, 2]] = hab_pose[[2, 1]]
    hab_pose[:, [1, 2]] = hab_pose[:, [2, 1]]

    return hab_pose

def get_bounding_boxes(args, data, color_data):
    db = DBSCAN(eps=args.epsilon, min_samples=args.min_samples).fit(data.cpu().numpy())
    labels = db.labels_

    num_clusters = len(np.unique(labels))
    num_noise = np.sum(np.array(labels) == -1, axis=0)

    logger.debug('Estimated no. of clusters: %d' % num_clusters)
    logger.debug('Estimated no. of noisy points: %d' % num_noise)

    # get max and min x, y, z values for each cluster
    bb_coords = torch.zeros((num_clusters, 3, 2), device=data.device)
    for i in range(num_clusters-1):
        cluster = data[labels == i]
        bb_coords[i, :, 0] = torch.min(cluster, axis=0)[0]
        bb_coords[i, :, 1] = torch.max(cluster, axis=0)[0]

        # use different bright colors for points in different clusters
        color_data[labels == i] = torch.tensor(COLOR_LIST[i % len(COLOR_LIST)], device=data.device).float()
    return bb_coords, color_data, labels

class TorchCamera:
    def __init__(
        self,
        width,
        height,
        fov_degrees,
    ):
        self.width = width
        self.height = height
        horizontal_fov_radians = fov_degrees * np.pi / 180.0
        self.px = (width - 1.0) / 2.0
        self.py = (height - 1.0) / 2.0
        self.fx = (width - 1.0) / (2.0 * np.tan(horizontal_fov_radians / 2.0))
        self.fy = self.fx
    
    def update_intrinsics(self, intrinsics):
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]

    def depth_to_xyz(self, depth):
        indices = torch.stack(
            torch.meshgrid(
                torch.arange(self.height, dtype=torch.float32),
                torch.arange(self.width, dtype=torch.float32),
            ),
            dim=-1,
        ).to(depth.device)
        z = depth
        x = (indices[:, :, 1] - self.px) * (z / self.fx)
        y = (indices[:, :, 0] - self.py) * (z / self.fy)
        xyz = torch.stack([x, y, z], axis=-1)
        return xyz

    def get_intrinsics(self, inverse=False, device='cuda'):
        intrinsics = torch.tensor(
            [
                [self.fx, 0, self.px],
                [0, self.fy, self.py],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        if inverse:
            intrinsics = torch.inverse(intrinsics)
        return intrinsics

class ConceptFusion:
    def __init__(
            self,
            device: str,
            sam_params: DictConfig,
            open_clip_params: DictConfig,
            data_params: DictConfig,
            dbscan_params: DictConfig,
            similarity_params: DictConfig,
            file_params: DictConfig,
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
        self.sam_params = sam_params
        self.data_params = data_params
        self.open_clip_params = open_clip_params
        self.dbscan_params = dbscan_params
        self.similarity_params = similarity_params
        self.file_params = file_params

        sam = sam_model_registry[self.sam_params.model_type](checkpoint=Path(self.sam_params.SAM_checkpoint_path))
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self.sam_params.points_per_side,
            pred_iou_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )

        logger.info("Initializing OpenCLIP model: {} pre-trained on {}...".format(
            self.open_clip_params.open_clip_model,
            self.open_clip_params.open_clip_pretrained_dataset
        ))

        self.CLIP_model, _, self.CLIP_preprocess = open_clip.create_model_and_transforms(
            self.open_clip_params.open_clip_model, self.open_clip_params.open_clip_pretrained_dataset
        )
        self.CLIP_model.to(device=self.device)
        self.CLIP_model.eval()

        self.tokenizer = open_clip.get_tokenizer(self.open_clip_params.open_clip_model)

        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        self.feat_dim = None

        self.camera = TorchCamera(
            width=self.data_params.desired_width,
            height=self.data_params.desired_height,
            fov_degrees=self.data_params.camera_fov_degrees,
        )

        self.voxel_map = VoxelizedPointcloud(
            voxel_size=self.data_params.voxel_size,
        )

        self.transform = transforms.Resize((self.data_params.desired_height, self.data_params.desired_width), interpolation=Image.Resampling.NEAREST)

        self.class_id_to_class_names = None
        self.class_names_to_class_id = None

    def clear(self):
        self.voxel_map.reset()
        self.feat_dim = None

    def generate_mask(self, img):
        masks = self.mask_generator.generate(img)
        
        # remove masks with zero area
        masks = list(filter(lambda x: x["bbox"][2] * x["bbox"][3] != 0, masks))
        
        return masks
    
    def set_vocabulary(self, class_id_to_class_names):
        self.class_id_to_class_names = class_id_to_class_names
        self.class_names_to_class_id = {v: k for k, v in class_id_to_class_names.items()}
    
    def resize_images(self, rgb, depth):
        """Resize images using torch."""
        # RGB images have the weird requirement that they need to be in numpy format for input to concept fusion
        rgb = Image.fromarray(rgb)

        rgb = self.transform(rgb)
        depth = self.transform(depth)

        rgb = np.array(rgb)

        return rgb, depth
    
    def generate_global_features(
        self,
        img: Image,
    ):
        # CLIP features global
        global_feat = None
        with torch.cuda.amp.autocast():
            _img = self.CLIP_preprocess(Image.fromarray(img)).unsqueeze(0)
            global_feat = self.CLIP_model.encode_image(_img.to(self.device))
            global_feat /= global_feat.norm(dim=-1, keepdim=True)

        global_feat = global_feat.half().to(self.device)
        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)  # --> (1, 1024)

        if self.feat_dim is None:
            self.feat_dim = global_feat.shape[-1]

        return global_feat
    
    def generate_local_features(
        self,
        img: Image,
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
        outfeat = torch.zeros(load_image_height, load_image_width, self.feat_dim, dtype=torch.half, device=self.device)

        for mask in masks:
            _x, _y, _w, _h = tuple(mask["bbox"])  # xywh bounding box

            # make sure _x, _y, _w, _h are ints
            _x, _y, _w, _h = int(_x), int(_y), int(_w), int(_h)

            nonzero_inds = torch.argwhere(torch.from_numpy(mask["segmentation"]))

            # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
            img_roi = img[_y : _y + _h, _x : _x + _w, :]

            img_roi = Image.fromarray(img_roi)
            img_roi = self.CLIP_preprocess(img_roi).unsqueeze(0).to(self.device)
            roifeat = self.CLIP_model.encode_image(img_roi)
            roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(nonzero_inds)
            _sim = self.cosine_similarity(global_feat, roifeat)
            similarity_scores.append(_sim)
        
        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)

        for maskidx in range(len(masks)):

            _weighted_feat = softmax_scores[maskidx] * global_feat + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[0].detach().half()
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
                outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1
            ).half()

        outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
        outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        outfeat = torch.nn.functional.interpolate(outfeat, [self.data_params.desired_height, self.data_params.desired_width], mode="nearest")
        outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
        outfeat = outfeat[0].half() # --> H, W, feat_dim

        return outfeat

    def text_query_original(
        self,
        query: str,
        map_features: torch.Tensor,
    ):
        text = self.tokenizer([query])
        textfeat = self.CLIP_model.encode_text(text.to(self.device))
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

        map_colors = np.zeros((similarity.shape[1], 3))

        if self.similarity_params.viz_type == "topk":
            # Viz topk points
            _, topk_ind = torch.topk(similarity, self.similarity_params.topk)
            map_colors[topk_ind.detach().cpu().numpy()] = np.array([1.0, 0.0, 0.0])
            selected_inds = topk_ind

        elif self.similarity_params.viz_type == "thresh":
            # Viz thresholded "relative" attention scores
            similarity[similarity < self.similarity_params.similarity_thresh] = 0.0
            selected_inds = torch.nonzero(similarity.squeeze()).squeeze(1)

            cmap = matplotlib.colormaps["jet"]
            similarity_colormap = cmap(similarity[0].detach().cpu().numpy())[:, :3]

            map_colors = 0.5 * map_colors + 0.5 * similarity_colormap

        return map_colors, selected_inds.detach().cpu().numpy(), similarity.squeeze()

    def text_queries(
        self,
        queries: Sequence[str],
    ):
        pc_xyz, pc_feat, _, pc_rgb = self.voxel_map.get_pointcloud()

        text = self.tokenizer(queries)
        textfeat = self.CLIP_model.encode_text(text.to(self.device))
        textfeat = torch.nn.functional.normalize(textfeat, dim=-1)

        pc_feat = pc_feat.to(self.device)
        pc_feat = torch.nn.functional.normalize(pc_feat, dim=-1)

        # match dimensions to do cosine similarity
        textfeat = textfeat.unsqueeze(1).repeat(1, pc_feat.shape[0], 1)
        pc_feat_expanded = pc_feat.unsqueeze(0).repeat(textfeat.shape[0], 1, 1)

        similarity = self.cosine_similarity(textfeat, pc_feat_expanded) # [n_points, n_classes]

        del pc_feat_expanded

        point_class = similarity.argmax(dim=0)

        instances_dict = {}
        for idx, class_name in enumerate(queries):
            selected_inds = torch.nonzero((point_class == idx) & (similarity[idx] > self.similarity_params.similarity_thresh)).squeeze(1)

            pc_feat_objects = pc_feat[selected_inds]
            pc_rgb_objects = pc_rgb[selected_inds]
            pc_xyz_objects = pc_xyz[selected_inds]
            similarity_score_objects = similarity[idx, selected_inds].squeeze()

            bounding_boxes, pc_rgb_objects, labels = get_bounding_boxes(self.dbscan_params, pc_xyz_objects, pc_rgb_objects)

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
                    instance = Instance(score_aggregation_method='max')

                    # TODO: get rid of the instance view altogether
                    instance_view = InstanceView(
                        bbox=None,
                        bounds=bounds,
                        timestep=0,
                        embedding=pc_feat_objects[labels == idx].mean(axis=0),
                        point_cloud=pc_xyz_objects[labels == idx],
                        point_cloud_rgb=pc_rgb_objects[labels == idx],
                        category_id=self.class_names_to_class_id[class_name],
                        score=similarity_score_objects[labels == idx].max()
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

            original_image_size = img.shape[:2]
            camera_pose = scene_obs["poses"][i].float()
            img, depth = self.resize_images(img, depth)

            masks = self.generate_mask(img)

            # CLIP features global
            global_feat = self.generate_global_features(img)

            # CLIP features per ROI
            outfeat = self.generate_local_features(img, masks, global_feat)

            if self.data_params.habitat_dataset:
                camera_pose = convert_pose_to_real_world_axis(camera_pose)

                xyz = self.camera.depth_to_xyz(depth).reshape(-1, 3)[:, [0, 2, 1]]
                xyz[:, 1] *= -1 if self.data_params.habitat_dataset else 1

                xyz = (
                    torch.cat([xyz, torch.ones_like(xyz[..., [0]])], axis=1) @ camera_pose.T
                )
            else:
                original_intrinsics = scene_obs["intrinsics"][i]
                adjusted_intrinsics = adjust_intrinsics_matrix(
                    original_intrinsics,
                    original_image_size,
                    (self.data_params.desired_height, self.data_params.desired_width),
                )
                xyz = unproject_masked_depth_to_xyz_coordinates(
                    depth.unsqueeze(0),
                    camera_pose.unsqueeze(0),
                    adjusted_intrinsics.inverse()[:3, :3].unsqueeze(0),
                )

            self.voxel_map.add(
                points=xyz[:, :3],
                features=outfeat.reshape(-1, 1024).to(device=self.device),
                rgb=torch.tensor(img, device=self.device).reshape(-1, 3),
            )

        return self.voxel_map.get_pointcloud()

    def get_instances_for_queries(
        self,
        queries: Sequence[str]
    ):
        pc_xyz, pc_feat, _, pc_rgb = self.voxel_map.get_pointcloud()
        instances_dict = {}
        for class_name in queries:
            logger.debug("Querying for class: {}".format(class_name))
            pc_rgb_after_query, selected_inds, similarity_score = self.text_query_original(class_name, pc_feat)

            if selected_inds.shape[0] <= 1:
                warnings.warn(
                    f"No points found for class {class_name}",
                    UserWarning,
                )
                continue

            pc_feat_objects = pc_feat[selected_inds]
            pc_rgb_objects = pc_rgb_after_query[selected_inds].squeeze()
            pc_xyz_objects = pc_xyz[selected_inds].squeeze()
            similarity_score_objects = similarity_score[selected_inds].squeeze()
            bounding_boxes, pc_rgb_after_query[selected_inds], labels = get_bounding_boxes(self.dbscan_params, pc_xyz_objects, pc_rgb_objects)

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
                    instance = Instance(score_aggregation_method='max')

                    # TODO: get rid of the instance view altogether
                    instance_view = InstanceView(
                        bbox=None,
                        bounds=bounds,
                        timestep=0,
                        embedding=pc_feat_objects[labels == idx].mean(axis=0),
                        point_cloud=pc_xyz_objects[labels == idx],
                        point_cloud_rgb=pc_rgb_objects[labels == idx],
                        category_id=self.class_names_to_class_id[class_name],
                        score=similarity_score_objects[labels == idx].max()
                    )
                    # append instance view to list of instance views
                    instance.add_instance_view(instance_view)
                    instances_per_class.append(instance)

            instances_dict[class_name] = instances_per_class

        return instances_dict

    def build_scene_and_get_instances_for_queries(
        self,
        scene_obs: Dict[str, Any],
        queries: Sequence[str]
    ):
        self.build_scene(scene_obs)

        if self.similarity_params.viz_type == 'classify_all':
            return self.text_queries(queries)
        else:
            return self.get_instances_for_queries(queries)

    def show_point_cloud_pytorch3d(
            self,
            instances_dict = None,
            **plot_scene_kwargs
        ):
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
                    colors.append(torch.tensor(COLOR_LIST[instance.category_id % len(COLOR_LIST)], device=self.device))
                detected_boxes = BBoxes3D(
                    bounds=[torch.stack(bounds, dim=0)],
                    features=[torch.stack(colors, dim=0)],
                    names=[torch.stack(names, dim=0).unsqueeze(-1)],
                )
                traces[class_name + "_bbox_" + str(round(instance.score.item(), 2))] = detected_boxes

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
            plots={f"Conceptfusion Pointcloud": traces},
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
        segmentation_image = torch.zeros(original_image.size[0], original_image.size[0], 3)
        for i, mask in enumerate(masks):
            segmentation_image += torch.from_numpy(mask["segmentation"]).unsqueeze(-1).repeat(1, 1, 3).float() * \
                torch.tensor(custom_palette[i%24]) * 255.0

        mask = Image.fromarray(segmentation_image.numpy().astype("uint8"))
        file_path = os.path.join(save_path, "mask_" + str(idx) + ".png")
        mask.save(file_path)