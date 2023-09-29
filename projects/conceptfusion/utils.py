from copy import deepcopy
import os
import struct
from typing import List

import numpy as np
from PIL import Image
import pytorch3d
import torch



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

def write_pointcloud(filename, xyz_points, rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8) * 255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    rgb_points = [rgb_points] if rgb_points is not None else []
    pointcloud = pytorch3d.structures.Pointclouds(points=[xyz_points], features=rgb_points)
    pytorch3d.io.IO().save_pointcloud(pointcloud)

def adjust_intrinsics_matrix(K, old_size, new_size):
    """
    Adjusts the camera intrinsics matrix after resizing an image.

    Args:
        K (np.ndarray): the original 3x3 intrinsics matrix.
        old_size (list[int]): the original size of the image in (width, height).
        new_size (list[int]): the new size of the image in (width, height).
    Returns:
        np.ndarray: the adjusted 3x3 intrinsics matrix.

    :example:
    >>> K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    >>> old_size = (640, 480)
    >>> new_size = (320, 240)
    >>> K_new = adjust_intrinsics_matrix(K, old_size, new_size)
    """
    # Calculate the scale factors for width and height
    scale_x = new_size[0] / old_size[0]
    scale_y = new_size[1] / old_size[1]

    # Adjust the intrinsics matrix
    K_new = deepcopy(K)
    K_new[0, 0] *= scale_x  # Adjust f_x
    K_new[1, 1] *= scale_y  # Adjust f_y
    K_new[0, 2] *= scale_x  # Adjust c_x
    K_new[1, 2] *= scale_y  # Adjust c_y

    return K_new