from pathlib import Path
from tqdm import trange
from typing import List, Sequence, Dict, Any
import warnings

import matplotlib
import numpy as np
from omegaconf import DictConfig
import open_clip
from PIL import Image
from sklearn.cluster import DBSCAN
import torch
from torchvision import transforms

from home_robot.utils.voxel import VoxelizedPointcloud
import home_robot.utils.image as im
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

from home_robot.utils.bboxes_3d import box3d_volume_from_bounds
from home_robot.mapping.semantic.instance_tracking_modules import InstanceView, Instance

from utils import COLOR_LIST

def convert_pose_to_real_world_axis(hab_pose):
    """Update axis convention of habitat pose to match the real-world axis convention"""
    hab_pose[[1, 2]] = hab_pose[[2, 1]]
    hab_pose[:, [1, 2]] = hab_pose[:, [2, 1]]

    return hab_pose

def get_bounding_boxes(args, data, color_data):
    db = DBSCAN(eps=args.epsilon, min_samples=args.min_samples).fit(data)
    labels = db.labels_

    num_clusters = len(np.unique(labels))
    num_noise = np.sum(np.array(labels) == -1, axis=0)

    print('Estimated no. of clusters: %d' % num_clusters)
    print('Estimated no. of noisy points: %d' % num_noise)

    # get max and min x, y, z values for each cluster
    bb_coords = np.zeros((num_clusters, 3, 2))
    for i in range(num_clusters-1):
        cluster = data[labels == i]
        bb_coords[i, :, 0] = torch.min(cluster, axis=0)[0]
        bb_coords[i, :, 1] = torch.max(cluster, axis=0)[0]

        # use different bright colors for points in different clusters
        color_data[labels == i] = torch.tensor(COLOR_LIST[i % len(COLOR_LIST)]).float()
    return bb_coords, color_data, labels

class ConceptFusion:
    def __init__(
            self,
            args: DictConfig,
        ) -> None:
        """
        Initialize concept fusion model.

        Args:
            args (DictConfig): Hydra config.
        """
        self.args = args

        sam = sam_model_registry[args.model_type](checkpoint=Path(args.SAM_checkpoint_path))
        sam.to(device=args.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=args.points_per_side,
            pred_iou_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )

        print(
            f"Initializing OpenCLIP model: {args.open_clip_model}"
            f" pre-trained on {args.open_clip_pretrained_dataset}..."
        )

        self.CLIP_model, _, self.CLIP_preprocess = open_clip.create_model_and_transforms(
            args.open_clip_model, args.open_clip_pretrained_dataset
        )
        self.CLIP_model.cuda()
        self.CLIP_model.eval()

        self.tokenizer = open_clip.get_tokenizer(args.open_clip_model)

        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        self.feat_dim = None

        self.camera = im.Camera.from_width_height_fov(
            width=args.desired_width,
            height=args.desired_height,
            fov_degrees=90,
            near_val=0.1,
            far_val=4.0,
        )

        self.voxel_map = VoxelizedPointcloud(
            voxel_size=args.voxel_size,
        )

        self.transform = transforms.Resize((self.args.desired_height, self.args.desired_width), interpolation=Image.NEAREST)

    
    def generate_mask(self, img):
        masks = self.mask_generator.generate(img)
        
        # remove masks with zero area
        masks = list(filter(lambda x: x["bbox"][2] * x["bbox"][3] != 0, masks))
        
        return masks
    
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
            global_feat = self.CLIP_model.encode_image(_img.cuda())
            global_feat /= global_feat.norm(dim=-1, keepdim=True)

        global_feat = global_feat.half().cuda()
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
        outfeat = torch.zeros(load_image_height, load_image_width, self.feat_dim, dtype=torch.half)

        for mask in masks:
            _x, _y, _w, _h = tuple(mask["bbox"])  # xywh bounding box

            # make sure _x, _y, _w, _h are ints
            _x, _y, _w, _h = int(_x), int(_y), int(_w), int(_h)

            nonzero_inds = torch.argwhere(torch.from_numpy(mask["segmentation"]))

            # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
            img_roi = img[_y : _y + _h, _x : _x + _w, :]

            img_roi = Image.fromarray(img_roi)
            img_roi = self.CLIP_preprocess(img_roi).unsqueeze(0).cuda()
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
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[0].detach().cpu().half()
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
                outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1
            ).half()

        outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
        outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        outfeat = torch.nn.functional.interpolate(outfeat, [self.args.desired_height, self.args.desired_width], mode="nearest")
        outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
        outfeat = outfeat[0].half() # --> H, W, feat_dim

        return outfeat

    def text_query(
        self,
        query: str,
        map_features: torch.Tensor,
    ):
        text = self.tokenizer([query])
        textfeat = self.CLIP_model.encode_text(text.cuda())
        textfeat = torch.nn.functional.normalize(textfeat, dim=-1)
        textfeat = textfeat.unsqueeze(0)

        # make sure features are on cuda
        map_features = map_features.cuda()
        map_features = torch.nn.functional.normalize(map_features, dim=-1)

        similarity = self.cosine_similarity(textfeat, map_features)

        # We use relative similarity
        similarity = (similarity - similarity.min()) / (
            similarity.max() - similarity.min() + 1e-12
        )

        map_colors = np.zeros((similarity.shape[1], 3))

        if self.args.viz_type == "topk":
            # Viz topk points
            _, topk_ind = torch.topk(similarity, self.args.topk)
            map_colors[topk_ind.detach().cpu().numpy()] = np.array([1.0, 0.0, 0.0])
            selected_inds = topk_ind

        elif self.args.viz_type == "thresh":
            # Viz thresholded "relative" attention scores
            similarity[similarity < self.args.similarity_thresh] = 0.0
            selected_inds = torch.nonzero(similarity.squeeze()).squeeze(1)

            cmap = matplotlib.cm.get_cmap("jet")
            similarity_colormap = cmap(similarity[0].detach().cpu().numpy())[:, :3]

            map_colors = 0.5 * map_colors + 0.5 * similarity_colormap

        return map_colors, selected_inds.detach().cpu().numpy(), similarity.squeeze()

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
        print("Building scene with {} images".format(len(scene_obs["images"])))
        for i in trange(len(scene_obs["images"])):
            img = scene_obs["images"][i]
            depth = torch.tensor(scene_obs["depths"][i]).unsqueeze(0)
            camera_pose = torch.tensor(scene_obs["poses"][i]).float()
            img, depth = self.resize_images(img, depth)

            masks = self.generate_mask(img)

            # CLIP features global
            global_feat = self.generate_global_features(img)

            # CLIP features per ROI
            outfeat = self.generate_local_features(img, masks, global_feat)

            camera_pose = convert_pose_to_real_world_axis(camera_pose)

            xyz = torch.Tensor(self.camera.depth_to_xyz(depth.numpy())).reshape(-1, 3)[:, [0, 2, 1]]
            xyz[:, 1] *= -1
            xyz = (
                torch.cat([xyz, torch.ones_like(xyz[..., [0]])], axis=1) @ camera_pose.T
            )
            self.voxel_map.add(
                points=xyz[:, :3],
                features=outfeat.reshape(-1, 1024),
                rgb=torch.tensor(img).reshape(-1, 3),
            )

        return self.voxel_map.get_pointcloud()

    def get_instances_for_queries(
        self,
        queries: Sequence[str]
    ):
        pc_xyz, pc_feat, _, pc_rgb = self.voxel_map.get_pointcloud()
        instances_dict = {}
        category_id = 0 # TODO: Fix this hardcoding
        for class_name in queries:
            print("Querying for class: {}".format(class_name))
            pc_rgb_after_query, selected_inds, similarity_score = self.text_query(class_name, pc_feat)

            if selected_inds.shape[0] <= 1:
                warnings.warn(
                    f"No points found for class {class_name}",
                    UserWarning,
                )
                continue

            pc_feat_objects = pc_feat[selected_inds]
            pc_rgb_objects = torch.tensor(pc_rgb_after_query[selected_inds].squeeze()).float()
            pc_xyz_objects = pc_xyz[selected_inds].squeeze()
            similarity_score_objects = similarity_score[selected_inds].squeeze()
            bounding_boxes, pc_rgb_after_query[selected_inds], labels  = get_bounding_boxes(self.args, pc_xyz_objects, pc_rgb_objects)

            instances_per_class = []
            for idx in range(len(bounding_boxes)):
                bounds = torch.tensor(bounding_boxes[idx])

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
                        category_id=category_id,
                        score=similarity_score_objects[labels == idx].max()
                    )
                    # append instance view to list of instance views
                    instance.add_instance_view(instance_view)
                    instances_per_class.append(instance)

            instances_dict[class_name] = instances_per_class

            category_id += 1

        return instances_dict

    def build_scene_and_get_instances_for_queries(
        self,
        scene_obs: Dict[str, Any],
        queries: Sequence[str]
    ):
        self.build_scene(scene_obs)

        return self.get_instances_for_queries(queries)

    def _show_point_cloud_pytorch3d(
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
            idx = 0
            for class_name, instances in instances_dict.items():
                if len(instances) == 0:
                    continue
                bounds, names, colors = [], [], []
                for instance in instances:
                    bounds.append(instance.bounds)
                    names.append(torch.tensor(instance.category_id))
                    colors.append(torch.tensor(COLOR_LIST[idx % len(COLOR_LIST)]))
                detected_boxes = BBoxes3D(
                    bounds=[torch.stack(bounds, dim=0)],
                    features=[torch.stack(colors, dim=0)],
                    names=[torch.stack(names, dim=0).unsqueeze(-1)],
                )
                traces[class_name + "_bbox"] = detected_boxes
                idx += 1

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
