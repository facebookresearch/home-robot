# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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
