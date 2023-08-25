# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from tqdm import tqdm


class ScanNetDataset(object):
    # Segmentation data
    METAINFO = {
        "classes": (
            "wall",
            "floor",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "bookshelf",
            "picture",
            "counter",
            "desk",
            "curtain",
            "refrigerator",
            "showercurtrain",
            "toilet",
            "sink",
            "bathtub",
            "otherfurniture",
        ),
        "palette": [
            [174, 199, 232],
            [152, 223, 138],
            [31, 119, 180],
            [255, 187, 120],
            [188, 189, 34],
            [140, 86, 75],
            [255, 152, 150],
            [214, 39, 40],
            [197, 176, 213],
            [148, 103, 189],
            [196, 156, 148],
            [23, 190, 207],
            [247, 182, 210],
            [219, 219, 141],
            [255, 127, 14],
            [158, 218, 229],
            [44, 160, 44],
            [112, 128, 144],
            [227, 119, 194],
            [82, 84, 163],
        ],
        "seg_valid_class_ids": (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            14,
            16,
            24,
            28,
            33,
            34,
            36,
            39,
        ),
        "seg_all_class_ids": tuple(range(41)),
    }
    DEPTH_SCALE_FACTOR = 0.001  # to MM
    DEFAULT_HEIGHT = 968.0
    DEFAULT_WIDTH = 1296.0

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        keep_only_scenes: Optional[List[str]] = None,
        frame_skip: int = 1,
        height: Optional[int] = 480,
        width: Optional[int] = 640,
        modalities: Tuple[str] = ("rgb", "depth", "pose", "intrinsics"),
        show_load_progress: bool = False,
    ):
        """
        The directory structure after pre-processing should be as below.
        For dowloading and preprocessing scripts, see https://mmdetection3d.readthedocs.io/en/v0.15.0/datasets/scannet_det.html

        scannet
        ├── meta_data
        ├── batch_load_scannet_data.py
        ├── load_scannet_data.py
        ├── scannet_utils.py
        ├── README.md
        ├── scans
        ├── scans_test
        ├── scannet_instance_data
        ├── scannet_2d_instance_data
        │   ├── scene_yyyy_yy
        │   │   ├── labels
        │   │   ├── labels-filt
        │   │   ├── instance
        │   │   ├── instance-filt
        ├── points
        │   ├── xxxxx.bin
        ├── instance_mask
        │   ├── xxxxx.bin
        ├── semantic_mask
        │   ├── xxxxx.bin
        ├── seg_info
        │   ├── train_label_weight.npy
        │   ├── train_resampled_scene_idxs.npy
        │   ├── val_label_weight.npy
        │   ├── val_resampled_scene_idxs.npy
        ├── posed_images
        │   ├── scenexxxx_xx
        │   │   ├── xxxxxx.txt # pose
        │   │   ├── xxxxxx.png # depth
        │   │   ├── xxxxxx.jpg # color
        │   │   ├── intrinsic.txt
        ├── scannet_infos_train.pkl
        ├── scannet_infos_val.pkl
        ├── scannet_infos_test.pkl

        """
        assert split in ["train", "val", "test"]
        self.root_dir = Path(root_dir)
        self.posed_dir = self.root_dir / "posed_images"
        self.instance_dir = self.root_dir / "scannet_instance_data"
        self.instance_2d_dir = self.root_dir / "scannet_instance_data"
        self.scan_dir = self.root_dir / "scannet_instance_data"
        self.split = split
        self.height = height
        self.width = width
        assert (self.height is None) == (self.width is None)  # Neither or both
        self.frame_skip = frame_skip

        with open(self.root_dir / "meta_data" / f"scannetv2_{split}.txt", "rb") as f:
            self.scene_list = [line.rstrip().decode() for line in f]
        if keep_only_scenes is not None:
            self.scene_list = [s for s in self.scene_list if s in keep_only_scenes]
        self.scene_list = natsorted(self.scene_list)
        assert len(self.scene_list) > 0

    def find_data(self, scan_name):

        # RGBD + pose
        scene_pose_dir = self.posed_dir / scan_name
        scene_posed_files = list([str(s) for s in scene_pose_dir.iterdir()])
        img_names = list(
            natsorted([s for s in scene_posed_files if s.endswith(".jpg")])
        )[:: self.frame_skip]
        depth_names = list(
            natsorted([s for s in scene_posed_files if s.endswith(".png")])
        )[:: self.frame_skip]
        pose_names = list(
            natsorted(
                [
                    s
                    for s in scene_posed_files
                    if s.endswith(".txt") and not s.endswith("intrinsic.txt")
                ]
            )
        )[:: self.frame_skip]
        assert len(img_names) == len(depth_names)
        assert len(img_names) == len(pose_names)

        # 2D instance masks

        intrinsic_name = self.posed_dir / scan_name / "intrinsic.txt"
        return {
            "img_paths": [scene_pose_dir / f for f in img_names],
            "depth_paths": [scene_pose_dir / f for f in depth_names],
            "pose_paths": [scene_pose_dir / f for f in pose_names],
            "intrinsic_path": intrinsic_name,
            "bboxs_unaligned_path": self.instance_dir
            / f"{scan_name}_unaligned_bbox.npy",
            "bboxs_aligned_path": self.instance_dir / f"{scan_name}_aligned_bbox.npy",
            "axis_align_path": self.instance_dir / f"{scan_name}_axis_align_matrix.npy",
        }

    def __getitem__(self, idx: Union[str, int], show_progress: bool = False):
        if isinstance(idx, str):
            idx = self.scene_list.index(idx)
        scan_name = self.scene_list[idx]

        data = self.find_data(scan_name)
        K = torch.from_numpy(np.loadtxt(data["intrinsic_path"]).astype(np.float32))
        axis_align_mat = torch.from_numpy(np.load(data["axis_align_path"])).float()

        K[0] *= float(self.width) / self.DEFAULT_WIDTH  # scale_x
        K[1] *= float(self.height) / self.DEFAULT_HEIGHT  # scale_y

        poses, intrinsics, images, depths = [], [], [], []
        boxes_aligned, axis_align_mats = [], []
        image_paths = []
        for i, (img, depth, pose) in enumerate(
            maybe_show_progress(
                zip(data["img_paths"], data["depth_paths"], data["pose_paths"]),
                description=f"Loading scene {scan_name}",
                length=len(data["img_paths"]),
                show=show_progress,
            )
        ):
            pose = np.loadtxt(pose)
            pose = np.array(pose).reshape(4, 4)
            # pose[:3, 1] *= -1
            # pose[:3, 2] *= -1
            pose = torch.from_numpy(pose.astype(np.float32)).float()
            # We cannot accept files directly, as some of the poses are invalid
            if np.isinf(pose).any():
                continue

            image_paths.append(img)
            img = get_image_from_path(img, height=self.height, width=self.width)
            depth = get_depth_image_from_path(
                depth,
                height=self.height,
                width=self.width,
                scale_factor=self.DEPTH_SCALE_FACTOR,
            )

            axis_align_mats.append(axis_align_mat)
            poses.append(pose)
            intrinsics.append(K)
            images.append(img)
            depths.append(depth)
        poses = torch.stack(poses)
        intrinsics = torch.stack(intrinsics)
        images = torch.stack(images)
        depths = torch.stack(depths)
        axis_align_mats = torch.stack(axis_align_mats)
        boxes_aligned, box_classes = load_3d_bboxes(data["bboxs_aligned_path"])
        return dict(
            poses=poses,
            intrinsics=intrinsics,
            images=images,
            depths=depths,
            image_paths=image_paths,
            boxes_aligned=boxes_aligned,
            box_classes=box_classes,
            axis_align_mats=axis_align_mats,
            scan_name=scan_name,
        )


def maybe_show_progress(iterable, description, length, show=False):
    if not show:
        return iterable
    for x in tqdm(iterable, desc=description, total=length):
        yield x


def load_pose(path):
    pose = np.loadtxt(path)
    pose = np.array(pose).reshape(4, 4)
    pose[:3, 1] *= -1
    pose[:3, 2] *= -1
    pose = torch.from_numpy(pose).float()
    return pose


def load_intrinsics(path):
    pass


def load_semantic_masks(path):
    pass


def get_image_from_path(
    image_path: Union[str, Path],
    height: Optional[int] = None,
    width: Optional[int] = None,
    keep_alpha: bool = False,
) -> torch.Tensor:
    """Returns a 3 channel image.
    # Adapted from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/scannet_dataparser.py
    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_path)

    assert (height is None) == (width is None)  # Neither or both
    if height is not None:
        pil_image = pil_image.resize((width, height), resample=Image.BILINEAR)
    image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
    if len(image.shape) == 2:
        image = image[:, :, None].repeat(3, axis=2)
    assert len(image.shape) == 3
    assert image.dtype == np.uint8
    assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
    image = torch.from_numpy(image.astype("float32") / 255.0)
    if not keep_alpha and image.shape[-1] == 4:
        image = image[:, :, :3]
        # image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
    return image


def get_depth_image_from_path(
    filepath: Path,
    height: Optional[int] = None,
    width: Optional[int] = None,
    scale_factor: float = 1.0,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.
    # Adapted from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/scannet_dataparser.py
    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [height, width, 1].
    """
    assert (height is None) == (width is None)  # Neither or both
    do_resize = height is not None
    if filepath.suffix == ".npy":
        image = np.load(filepath) * scale_factor
        assert (height is None) == (width is None)  # Neither or both
        if do_resize:
            image = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float64) * scale_factor
        if do_resize:
            image = cv2.resize(image, (width, height), interpolation=interpolation)
    return torch.from_numpy(image[:, :, np.newaxis])


def load_3d_bboxes(path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads {scene_id}_aligned_bbox.npy or {scene_id}_unaligned_bbox.npy

    Returns:
        bounds: FloatTensor of box bounds (xyz mins and maxes)
        label_id: IntTensor of label IDs
    """
    bboxes = np.load(path)
    bbox_coords = torch.from_numpy(bboxes[:, :6])
    labels = torch.from_numpy(bboxes[:, -1])
    centers, lengths = bbox_coords[:, :3], bbox_coords[:, 3:6]
    mins = centers - lengths / 2.0
    maxs = centers + lengths / 2.0
    return torch.stack([mins, maxs], dim=-1), labels


if __name__ == "__main__":
    import open3d

    from home_robot.utils.point_cloud import (
        numpy_to_pcd,
        pcd_to_numpy,
        show_point_cloud,
    )
    from home_robot.utils.point_cloud_torch import get_xyz_coordinates

    data = ScanNetDataset(
        root_dir="/private/home/ssax/home-robot/projects/eval_scannet/scannet",
        frame_skip=30,
    )
    result = data.__getitem__(0, show_progress=True)
    K = result["intrinsics"][0][:3, :3]
    depth = result["depths"][0].squeeze().unsqueeze(0).unsqueeze(1)
    valid_depth = (0.1 < depth) & (depth < 4.0)

    xyz = get_xyz_coordinates(
        depth=depth,
        mask=~valid_depth,
        pose=torch.eye(4).unsqueeze(0),
        inv_intrinsics=torch.linalg.inv(K).unsqueeze(0),  # K.unsqueeze(0), #
    )
    rgb = result["images"][0].reshape(-1, 3)[valid_depth.flatten()]
    print(result["image_paths"][0])
    print("prop valid depths", valid_depth.float().mean())
    print(
        "depths",
        depth.flatten()[valid_depth.flatten()].min(),
        depth.flatten()[valid_depth.flatten()].max(),
    )
    for i in range(3):
        print(xyz[:, i].min(), xyz[:, i].max())
    point_subsample = 8
    show_point_cloud(
        xyz[::point_subsample], rgb[::point_subsample], orig=np.zeros(3), save="ptc.png"
    )
