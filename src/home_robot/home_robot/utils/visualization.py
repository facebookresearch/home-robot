# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


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
