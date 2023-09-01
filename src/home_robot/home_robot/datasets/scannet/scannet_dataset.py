# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
import os
import warnings
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

from .referit3d_data import ReferIt3dDataConfig, load_referit3d_data
from .scanrefer_data import ScanReferDataConfig, load_scanrefer_data


class ScanNetDataset(object):
    # Segmentation data
    METAINFO = {
        "classes": (
            # "wall",
            # "floor",
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
            "shower curtain",
            "toilet",
            "sink",
            "bathtub",
            "furniture",  # other furniture
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
            # 1,
            # 2,
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
        # modalities: Tuple[str] = ("rgb", "depth", "pose", "intrinsics"),
        referit3d_config: Optional[ReferIt3dDataConfig] = None,
        scanrefer_config: Optional[ScanReferDataConfig] = None,
        show_load_progress: bool = False,
    ):
        """

        frame_skip: Only use every frame_skip frames


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
        ├── referit3d
        │   ├── nr3d.csv
        │   ├── sr3d.csv
        │   ├── sr3d+.csv
        ├── scanrefer
        │   ├── ScanRefer_filtered_<SPLIT>.csv
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

        # Referit3d
        self.referit_data: Optional[pd.DateFrame] = None
        if referit3d_config is not None:
            if split != "train":
                warnings.warn(RuntimeWarning("ReferIt3D not evaluated on test set"))
            r3d_config_copy = copy.deepcopy(referit3d_config)
            if not os.path.isabs(r3d_config_copy.nr3d_csv_fpath):
                r3d_config_copy.nr3d_csv_fpath = (
                    self.root_dir / r3d_config_copy.nr3d_csv_fpath
                )
            if r3d_config_copy.sr3d_csv_fpath is not None and not os.path.isabs(
                r3d_config_copy.sr3d_csv_fpath
            ):
                r3d_config_copy.sr3d_csv_fpath = (
                    self.root_dir / r3d_config_copy.sr3d_csv_fpath
                )
            self.referit_data = load_referit3d_data(
                scans_split={"train": self.scene_list},
                **dataclasses.asdict(r3d_config_copy),
            )

        # ScanRefer
        self.scanrefer_data: Optional[pd.DateFrame] = None
        if scanrefer_config is not None:
            json_fpath = (
                self.root_dir
                / scanrefer_config.json_dir
                / f"ScanRefer_filtered_{split}.json"
            )
            self.scanrefer_data = load_scanrefer_data(json_fpath)
        # '/private/home/ssax/home-robot/src/home_robot/home_robot/datasets/scannet/data/scanrefer/ScanRefer_filtered_val.json'

    def find_data(self, scan_name: str):
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
        #   Not implemented yet

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

        # 2D information
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
            pose = axis_align_mat @ torch.from_numpy(pose.astype(np.float32)).float()
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
        poses = torch.stack(poses).float()
        intrinsics = torch.stack(intrinsics).float()
        images = torch.stack(images).float()
        depths = torch.stack(depths).float()
        axis_align_mats = torch.stack(axis_align_mats).float()

        # 3D information
        boxes_aligned, box_classes, box_obj_ids = load_3d_bboxes(
            data["bboxs_aligned_path"]
        )

        # Referring expressions
        column_names = [
            "utterance",
            "instance_type",
            "target_id",
            "stimulus_id",
            "dataset",
        ]
        ref_expr_df = pd.DataFrame(columns=column_names)

        # Referit
        if self.referit_data is not None:
            r3d_expr = self.referit_data[self.referit_data.scan_id == scan_name][
                column_names
            ]
            ref_expr_df = pd.concat([ref_expr_df, r3d_expr])

        # ScanRefer
        if self.scanrefer_data is not None:
            scanrefer_expr = self.scanrefer_data[
                self.scanrefer_data.scan_id == scan_name
            ][column_names]
            ref_expr_df = pd.concat([scanrefer_expr, r3d_expr])

        return dict(
            # Pose
            poses=poses,
            intrinsics=intrinsics,
            axis_align_mats=axis_align_mats,
            # Frames
            images=images,
            depths=depths,
            image_paths=image_paths,
            # 3d boxes
            boxes_aligned=boxes_aligned,
            box_classes=box_classes,
            box_target_ids=box_obj_ids,
            # Scene metadata
            scan_name=scan_name,
            # Referring expressions,
            ref_expr=ref_expr_df,
        )


def maybe_show_progress(iterable, description, length, show=False):
    if not show:
        return iterable
    for x in tqdm(iterable, desc=description, total=length):
        yield x


##################################
# Load different modalities
def load_pose_opengl(path):
    pose = np.loadtxt(path)
    pose = np.array(pose).reshape(4, 4)
    pose[:3, 1] *= -1
    pose[:3, 2] *= -1
    pose = torch.from_numpy(pose).float()
    return pose


def load_cam_intrinsics(path):
    raise NotImplementedError


def load_semantic_masks(path):
    raise NotImplementedError


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
        label_id: IntTensor of class IDs
        obj_id: IntTensor of object IDs
    """
    bboxes = np.load(path)
    bbox_coords = torch.from_numpy(bboxes[:, :6])
    labels = torch.from_numpy(bboxes[:, -2]).int()
    obj_ids = torch.from_numpy(bboxes[:, -1]).int()
    centers, lengths = bbox_coords[:, :3], bbox_coords[:, 3:6]
    mins = centers - lengths / 2.0
    maxs = centers + lengths / 2.0
    return torch.stack([mins, maxs], dim=-1), labels, obj_ids


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
