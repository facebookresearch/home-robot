# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def show_image(rgb):
    """Simple helper function to show images"""
    plt.figure()
    plt.imshow(rgb)
    plt.show()


def show_image_with_mask(rgb, mask):
    """tool for showing a mask and some other stuff"""
    plt.figure()
    plt.subplot(131)
    plt.imshow(rgb)
    plt.subplot(132)
    plt.imshow(mask)
    plt.subplot(133)
    _mask = mask[:, :, None]
    _mask = np.repeat(_mask, 3, axis=-1)
    plt.imshow(_mask * rgb / 255.0)
    plt.show()


def get_contour_points(
    pos: Tuple[float, float, float],
    origin: Tuple[float, float],
    size: int = 20,
) -> np.ndarray:
    x, y, o = pos
    pt1 = (int(x) + origin[0], int(y) + origin[1])
    pt2 = (
        int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1],
    )
    pt3 = (int(x + size * np.cos(o)) + origin[0], int(y + size * np.sin(o)) + origin[1])
    pt4 = (
        int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1],
    )

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(
    start: Tuple[int, int],
    end: Tuple[int, int],
    mat: np.ndarray,
    steps: int = 25,
    w: int = 1,
) -> np.ndarray:
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w : x + w, y - w : y + w] = 1
    return mat


def create_disk(radius: float, size: int):
    """Create image of a disk of the given size - helper function used to get explored areas. Image will be size x size."""

    # Create a grid of coordinates
    x = np.arange(0, size)
    y = np.arange(0, size)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # Compute the distance transform
    distance_map = np.sqrt((xx - size // 2) ** 2 + (yy - size // 2) ** 2)

    # Create the disk by thresholding the distance transform
    disk = distance_map <= radius

    return disk


def get_x_and_y_from_path(path: List[torch.Tensor]) -> Tuple[List[float]]:
    x_list, y_list = zip(
        *[
            (t[0].item(), t[1].item())
            if t.dim() == 1
            else (t[0, 0].item(), t[0, 1].item())
            for t in path
        ]
    )
    assert len(x_list) == len(y_list), "problem parsing tensors"
    return x_list, y_list
