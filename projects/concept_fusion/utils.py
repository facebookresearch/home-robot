import os
import numpy as np
import struct

import torch
from PIL import Image
from typing import List
import seaborn as sns


custom_palette = sns.color_palette("viridis", 24)

COLOR_LIST = [
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
]

def write_pointcloud(filename, xyz_points, rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8) * 255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(
            bytearray(
                struct.pack(
                    "fffccc",
                    xyz_points[i,0],
                    xyz_points[i,1],
                    xyz_points[i,2],
                    rgb_points[i,0].astype(np.uint8).tostring(),
                    rgb_points[i,1].astype(np.uint8).tostring(),
                    rgb_points[i,2].astype(np.uint8).tostring()
                )
            )
        )
    fid.close()


def save_data(
        original_image: torch.Tensor,
        masks: List[dict],
        outfeat: torch.Tensor,
        idx: int,
        save_path: str,
    ):
    """
    Save original image, segmentation masks, and concept fusion features.

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

    # save concept fusion features
    file_path = os.path.join(save_path, "concept_fusion_features_" + str(idx) + ".pt")
    torch.save(outfeat.detach().cpu(), file_path)