# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
import logging
import os
import warnings
from enum import Enum, auto
from functools import partial
from numbers import Number
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from .referit3d_data import ReferIt3dDataConfig, load_referit3d_data
from .scannet_constants import (
    MAX_CLASS_IDX,
    NUM_CLASSES,
    SCANNET_DATASET_CLASS_IDS,
    SCANNET_DATASET_CLASS_LABELS,
    SCANNET_DATASET_COLOR_MAPS,
)
from .scanrefer_data import ScanReferDataConfig, load_scanrefer_data

logger = logging.getLogger(__name__)


class ScanNetModalities(Enum):
    RGB = auto()
    DEPTH = auto()
    POSE = auto()
    INTRINSICS = auto()
    INSTANCE_2D = auto()
    BBOX_3D = auto()
    ALL = auto()
    # BBOX_3D = auto()


class ScanNetDataset(object):

    DEPTH_SCALE_FACTOR = 0.001  # to MM
    DEFAULT_HEIGHT = 968.0
    DEFAULT_WIDTH = 1296.0

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        keep_only_scenes: Optional[List[str]] = None,
        keep_only_first_k_scenes: int = -1,
        skip_first_k_scenes: int = 0,
        frame_skip: int = 1,
        height: Optional[int] = 480,
        width: Optional[int] = 640,
        modalities: Union[str, Tuple[ScanNetModalities]] = ScanNetModalities.ALL,
        referit3d_config: Optional[ReferIt3dDataConfig] = None,
        scanrefer_config: Optional[ScanReferDataConfig] = None,
        show_load_progress: bool = False,
        n_classes: int = 20,
        load_only_first_k_frames: Optional[int] = None,
        skipnan: bool = True,
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
        assert (
            n_classes in SCANNET_DATASET_COLOR_MAPS
        ), f"{n_classes=} must be in {SCANNET_DATASET_COLOR_MAPS.keys()}"

        # Set up directories and metadata
        assert split in ["train", "val", "test"]
        self.root_dir = Path(root_dir)
        self.posed_dir = self.root_dir / "posed_images"
        self.instance_dir = self.root_dir / "scannet_instance_data"
        self.instance_2d_dir = self.root_dir / "scannet_2d_instance_data"
        self.scan_dir = self.root_dir / "scannet_instance_data"

        # Metainfo
        self.METAINFO = {
            "COLOR_MAP": SCANNET_DATASET_COLOR_MAPS[n_classes],
            "CLASS_NAMES": SCANNET_DATASET_CLASS_LABELS[n_classes],
            "CLASS_IDS": SCANNET_DATASET_CLASS_IDS[n_classes],
        }
        # Load class names
        labels_pd = pd.read_csv(
            self.root_dir / "meta_data" / "scannetv2-labels.combined.tsv",
            sep="\t",
            header=0,
        )
        labels_pd.loc[labels_pd.raw_category == "stick", ["category"]] = "object"
        labels_pd.loc[labels_pd.category == "wardrobe ", ["category"]] = "wardrobe"
        self.ALL_CLASS_IDS_TO_CLASS_NAMES = dict(
            zip(labels_pd["id"], labels_pd["category"])
        )
        self.ALL_CLASS_NAMES_TO_CLASS_IDS = dict(
            zip(labels_pd["category"], labels_pd["id"])
        )
        # self.METAINFO['CLASS_NAMES'] = [self.ALL_CLASS_IDS_TO_CLASS_NAMES[k] for k in self.METAINFO['CLASS_IDS']]
        self.METAINFO["CLASS_IDS"] = [
            self.ALL_CLASS_NAMES_TO_CLASS_IDS[k] for k in self.METAINFO["CLASS_NAMES"]
        ]
        # Create tensor lookup table
        self.class_ids_ten = torch.tensor(self.METAINFO["CLASS_IDS"])
        self.DROP_CLASS_VAL = -1
        self.class_ids_lookup = make_lookup_table(
            self.class_ids_ten,
            self.class_ids_ten,
            missing_key_value=self.DROP_CLASS_VAL,
            key_max=MAX_CLASS_IDX + 1,
        )

        # Image metadata
        self.split = split
        self.height = height
        self.width = width
        assert (self.height is None) == (self.width is None)  # Neither or both
        self.frame_skip = frame_skip

        # Modalities
        if modalities == ScanNetModalities.ALL:
            self.modalities = [
                getattr(ScanNetModalities, v)
                for v in ScanNetModalities.__members__
                if v != "ALL"
            ]
        else:
            self.modalities = modalities

        # Create scene list
        with open(self.root_dir / "meta_data" / f"scannetv2_{split}.txt", "rb") as f:
            self.scene_list = [line.rstrip().decode() for line in f]
        if keep_only_scenes is not None:
            self.scene_list = [s for s in self.scene_list if s in keep_only_scenes]
        self.scene_list = natsorted(self.scene_list)
        logger.info(
            f"ScanNetDataset: Keeping next {keep_only_first_k_scenes} scenes starting at idx {skip_first_k_scenes}"
        )
        self.scene_list = self.scene_list[skip_first_k_scenes:][
            :keep_only_first_k_scenes
        ]
        assert len(self.scene_list) > 0

        self.load_only_first_k_frames = load_only_first_k_frames
        self.skipnan = skipnan

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

    def find_data(self, scan_name: str):
        # RGBD + pose
        scene_pose_dir = self.posed_dir / scan_name
        scene_posed_files = list([str(s) for s in scene_pose_dir.iterdir()])

        def get_endswith(f_list, endswith):
            return list(natsorted([s for s in f_list if s.endswith(endswith)]))

        # RGB
        img_names = get_endswith(scene_posed_files, ".jpg")[:: self.frame_skip]
        assert len(img_names) > 0, f"Found zero images for scene {scan_name}"

        # Depth
        depth_names = get_endswith(scene_posed_files, ".png")[:: self.frame_skip]
        assert len(depth_names) == len(
            img_names
        ), f"Unequal number of color and depth images for scene {scan_name} ({len(img_names)} != ({len(depth_names)}))"

        # Pose
        pose_names = list(
            natsorted(
                [
                    s
                    for s in scene_posed_files
                    if s.endswith(".txt") and not s.endswith("intrinsic.txt")
                ]
            )
        )[:: self.frame_skip]
        assert len(pose_names) == len(
            img_names
        ), f"Unequal number of color and poses for scene {scan_name} ({len(img_names)} != ({len(pose_names)}))"

        # 2D Instance
        inst2d_names = self.find_instance_2d(scan_name)[:: self.frame_skip]
        assert len(inst2d_names) == len(
            img_names
        ), f"Unequal number of color and poses for scene {scan_name} ({len(img_names)} != ({len(pose_names)}))"

        # img_names = list(
        #     natsorted([s for s in scene_posed_files if s.endswith(".jpg")])
        # )[:: self.frame_skip]
        # depth_names = list(
        #     natsorted([s for s in scene_posed_files if s.endswith(".png")])
        # )[:: self.frame_skip]

        # Camera Intrinsics
        intrinsic_name = self.posed_dir / scan_name / "intrinsic.txt"

        return {
            "img_paths": [scene_pose_dir / f for f in img_names],
            "depth_paths": [scene_pose_dir / f for f in depth_names],
            "pose_paths": [scene_pose_dir / f for f in pose_names],
            "intrinsic_path": intrinsic_name,
            "instance_2d_paths": inst2d_names,
            "bboxs_unaligned_path": self.instance_dir
            / f"{scan_name}_unaligned_bbox.npy",
            "bboxs_aligned_path": self.instance_dir / f"{scan_name}_aligned_bbox.npy",
            "axis_align_path": self.instance_dir / f"{scan_name}_axis_align_matrix.npy",
        }

    def find_instance_2d(self, scan_name: str):
        file_list = list((self.instance_2d_dir / scan_name / "instance-filt").iterdir())
        return natsorted(file_list)

    def __getitem__(self, idx: Union[str, int], show_progress: bool = False):
        if isinstance(idx, str):
            idx = self.scene_list.index(idx)
        scan_name = self.scene_list[idx]

        data = self.find_data(scan_name)
        axis_align_mat = torch.from_numpy(np.load(data["axis_align_path"])).float()

        # Intrinsics shared across images
        K = torch.from_numpy(np.loadtxt(data["intrinsic_path"]).astype(np.float32))
        K[0] *= float(self.width) / self.DEFAULT_WIDTH  # scale_x
        K[1] *= float(self.height) / self.DEFAULT_HEIGHT  # scale_y

        poses, intrinsics, images, depths = [], [], [], []
        boxes_aligned, axis_align_mats = [], []
        image_paths = []
        instance_2ds = []

        # Load all images
        for i, (img_path, depth_path, pose_path, instance_2d_path) in enumerate(
            maybe_show_progress(
                list(
                    zip(
                        data["img_paths"],
                        data["depth_paths"],
                        data["pose_paths"],
                        data["instance_2d_paths"],
                    )
                ),
                description=f"Loading scene {scan_name}",
                length=len(data["img_paths"]),
                show=show_progress,
            )
        ):
            if (
                self.load_only_first_k_frames is not None
                and i >= self.load_only_first_k_frames
            ):
                continue
            pose = np.loadtxt(pose_path)
            pose = np.array(pose).reshape(4, 4)
            # pose[:3, 1] *= -1
            # pose[:3, 2] *= -1
            pose = axis_align_mat @ torch.from_numpy(pose.astype(np.float32)).float()
            # We cannot accept files directly, as some of the poses are invalid
            if self.skipnan and torch.any(torch.isnan(pose)):
                # print(f"Found inf pose in {scan_name}")
                continue
            poses.append(pose)
            axis_align_mats.append(axis_align_mat)
            intrinsics.append(K)

            if ScanNetModalities.RGB in self.modalities:
                image_paths.append(img_path)
                img = get_image_from_path(
                    img_path, height=self.height, width=self.width
                )
                images.append(img)

            if ScanNetModalities.DEPTH in self.modalities:
                depth = get_depth_image_from_path(
                    depth_path,
                    height=self.height,
                    width=self.width,
                    scale_factor=self.DEPTH_SCALE_FACTOR,
                )
                depths.append(depth)

            if ScanNetModalities.INSTANCE_2D in self.modalities:
                instance_2d = get_instance_image_from_path(
                    instance_2d_path,
                    height=self.height,
                    width=self.width,
                )
                instance_2ds.append(instance_2d)

        poses = torch.stack(poses).float()
        intrinsics = torch.stack(intrinsics).float()
        axis_align_mats = torch.stack(axis_align_mats).float()
        if ScanNetModalities.RGB in self.modalities:
            images = torch.stack(images).float()

        if ScanNetModalities.DEPTH in self.modalities:
            depths = torch.stack(depths).float()

        if ScanNetModalities.INSTANCE_2D in self.modalities:
            instance_2ds = torch.stack(instance_2ds)

        # Load bounding boxes
        if ScanNetModalities.BBOX_3D in self.modalities:
            boxes_aligned, box_classes, box_obj_ids = load_3d_bboxes(
                data["bboxs_aligned_path"]
            )
            # keep_boxes = (box_classes.unsqueeze(1) == self.class_ids_ten.unsqueeze(0)).any(
            #     dim=1
            # )
            keep_boxes = (
                self.class_ids_lookup[box_classes.long()] != self.DROP_CLASS_VAL
            )
            boxes_aligned = boxes_aligned[keep_boxes]
            box_classes = box_classes[keep_boxes]
            box_obj_ids = box_obj_ids[keep_boxes]

            if len(boxes_aligned) == 0:
                raise RuntimeError(f"No GT boxes for scene {scan_name}")

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

        if len(ref_expr_df) > 0:
            ref_expr_df = filter_ref_exp_by_class(
                ref_expr_df, box_obj_ids, box_classes, drop_class=self.DROP_CLASS_VAL
            )

        # Return as dict
        return dict(
            # Pose
            poses=poses,
            intrinsics=intrinsics,
            axis_align_mats=axis_align_mats,
            # Frames
            images=images,
            depths=depths,
            instance_2ds=instance_2ds,
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

    def __len__(self):
        return len(self.scene_list)


##################################
# Load different modalities
#################################
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


def load_instance_masks(path):
    raise NotImplementedError


def get_image_from_path(
    image_path: Union[str, Path],
    height: Optional[int] = None,
    width: Optional[int] = None,
    keep_alpha: bool = False,
    resample=Image.BILINEAR,
) -> torch.Tensor:
    """Returns a 3 channel image.
    # Adapted from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/scannet_dataparser.py
    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_path)

    assert (height is None) == (width is None)  # Neither or both
    if height is not None:
        pil_image = pil_image.resize((width, height), resample=resample)
    image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
    if len(image.shape) == 2:
        image = image[:, :, None].repeat(3, axis=2)
    assert len(image.shape) == 3
    assert image.dtype == np.uint8
    assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is incorrect."
    image = torch.from_numpy(image.astype("float32") / 255.0)
    if not keep_alpha and image.shape[-1] == 4:
        image = image[:, :, :3]
        # image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
    return image


def get_instance_image_from_path(
    image_path: Union[str, Path],
    height: Optional[int] = None,
    width: Optional[int] = None,
    resample=Image.NEAREST,
) -> torch.Tensor:
    """Returns a 1 channel image.
    # Adapted from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/scannet_dataparser.py
    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_path)

    assert (height is None) == (width is None)  # Neither or both
    if height is not None:
        pil_image = pil_image.resize((width, height), resample=resample)
    image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
    assert image.dtype == np.uint8
    image = torch.from_numpy(image)  # .byte()
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


def filter_ref_exp_by_class(
    ref_expr_df: pd.DataFrame,
    box_target_ids: Tensor,
    box_classes: Tensor,
    drop_class: int = -1,
) -> pd.DataFrame:
    """Keeps only referring expressions where referring expression"""
    ref_exp_target_ids = torch.from_numpy(
        ref_expr_df.target_id.to_numpy().astype(np.int64)
    )

    # Make lookuptable of lookuptable[target_ids] -> target_class
    max_key = max(box_target_ids.max(), ref_exp_target_ids.max()) + 1
    ids_to_classes = make_lookup_table(
        box_target_ids.long(), box_classes, missing_key_value=-1, key_max=max_key
    )

    # Keep referring expressions who have targets where class != -1 (i.e. where target is in box_target_ids)
    ref_exp_classes = ids_to_classes[ref_exp_target_ids]
    df = ref_expr_df.copy()
    df["target_class_id"] = ref_exp_classes.cpu().numpy()
    keep_exp = ref_exp_classes != drop_class
    df = df.loc[keep_exp.cpu().numpy()]

    # # Map to class name with something like:
    # df['instance_type2'] = [class_id_to_name[class_idx] for class_idx in df['target_class_id']]
    return df


#############################################################
# Utils
#############################################################


def maybe_show_progress(iterable, description, length, show=False):
    if show:
        for x in tqdm(iterable, desc=description, total=length):
            yield x
    else:
        for x in iterable:
            yield x


def make_lookup_table(
    keys: Tensor,
    values: Tensor,
    key_max: Optional[int] = None,
    missing_key_value: Number = torch.nan,
) -> Tensor:
    """
    Create a lookup table using keys and values tensors.

    This function creates a 1D tensor (lookup table) using keys and values.
    The length of the lookup table is determined by `key_max`. The `keys` tensor
    specifies the indices in the lookup table that will be populated with the corresponding
    values from the `values` tensor. Indices not present in `keys` will be filled with
    `missing_key_value`.

    Parameters:
    -----------
    keys : torch.Tensor
        1D tensor of long integers specifying the indices in the lookup table
        where values should be placed. Must have dtype of torch.long.
    values : torch.Tensor
        1D tensor containing the values to be placed in the lookup table.
        Must have the same length as `keys`.
    key_max : int, optional
        The maximum key value + 1, which determines the length of the lookup table.
        If None, it is set to the maximum value in `keys` + 1. Default is None.
    missing_key_value : Number, optional
        The value to fill in for missing keys in the lookup table. Default is NaN.

    Returns:
    --------
    keys_expanded : torch.Tensor
        The populated lookup table. The dtype will match that of `values`.

    Raises:
    -------
    AssertionError
        If the dtype of the `keys` is not torch.long.

    Example:
    --------
    >>> keys = torch.tensor([1, 3, 5], dtype=torch.long)
    >>> values = torch.tensor([10.0, 30.0, 50.0])
    >>> make_lookup_table(keys, values)
    tensor([nan, 10.0, nan, 30.0, nan, 50.0])
    """
    if key_max is None:
        key_max = keys.max().item() + 1
    assert (
        keys.dtype == torch.long
    ), f"keys must have dtype torch.long -- not {keys.dtype}"
    keys_expanded = torch.full(
        [key_max],
        fill_value=missing_key_value,
        device=values.device,
        dtype=values.dtype,
    )
    keys_expanded.scatter_(dim=0, index=keys, src=values)
    return keys_expanded


if __name__ == "__main__":
    import open3d

    from home_robot.utils.point_cloud import (
        numpy_to_pcd,
        pcd_to_numpy,
        show_point_cloud,
    )
    from home_robot.utils.point_cloud_torch import get_xyz_coordinates

    data = ScanNetDataset(
        root_dir="./data/",
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
