# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
import torch.nn.functional as F


def binary_dilation(binary_image, kernel):
    """
    Arguments:
        binary_image: binary image tensor of shape (bs, 1, H1, W1)
        kernel: binary structuring element tensor of shape (1, 1, H2, W2)

    Returns:
        binary image tensor of the same shape as input
    """
    return torch.clamp(
        torch.nn.functional.conv2d(binary_image, kernel, padding=kernel.shape[-1] // 2),
        0,
        1,
    )


def binary_erosion(binary_image, kernel):
    """
    Arguments:
        binary_image: binary image tensor of shape (bs, 1, H1, W1)
        kernel: binary structuring element tensor of shape (1, 1, H2, W2)

    Returns:
        binary image tensor of the same shape as input
    """
    return 1 - torch.clamp(
        torch.nn.functional.conv2d(
            1 - binary_image, kernel, padding=kernel.shape[-1] // 2
        ),
        0,
        1,
    )


def binary_opening(binary_image, kernel):
    return binary_dilation(binary_erosion(binary_image, kernel), kernel)


def binary_closing(binary_image, kernel):
    return binary_erosion(binary_dilation(binary_image, kernel), kernel)


def binary_denoising(binary_image, kernel):
    return binary_opening(binary_closing(binary_image, kernel), kernel)


def get_edges(mask: torch.Tensor, threshold: Optional[float] = 0.5) -> torch.Tensor:
    """Extract edges from a torch tensor.

    Args:
        threshold(float): what derivative determines its an edge. If none, returns derivative."""

    mask = mask.float()

    # Define the Sobel filter kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=mask.device
    )
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=mask.device
    )

    # Calculate padding for convolution to preserve the original size
    # Sobel x and y operators are the same size
    padding_x = sobel_x.size(0) // 2
    padding_y = sobel_x.size(1) // 2

    # Apply Sobel filter to detect edges in x and y directions
    edges_x = F.conv2d(
        mask.unsqueeze(0).unsqueeze(0),
        sobel_x.unsqueeze(0).unsqueeze(0),
        padding=(padding_x, padding_y),
    )
    edges_y = F.conv2d(
        mask.unsqueeze(0).unsqueeze(0),
        sobel_y.unsqueeze(0).unsqueeze(0),
        padding=(padding_x, padding_y),
    )

    # Combine x and y edge responses to get the magnitude of edges
    edges = torch.sqrt(edges_x**2 + edges_y**2)
    edges = edges[0, 0]
    if threshold is not None:
        edges = edges > threshold
    assert (
        edges.shape == mask.shape
    ), "something went wrong when computing padding, most likely - shape not preserved"
    return edges


def expand_mask(mask: torch.Tensor, radius: int, threshold: float = 0.5):
    """Expand a mask by some radius in pytorch"""

    # Needs to be converted to a float to work
    mask = mask.float()

    # Create a disk-shaped structuring element
    x, y = torch.meshgrid(
        torch.arange(-radius, radius + 1),
        torch.arange(-radius, radius + 1),
        indexing="ij",
    )
    selem = (x**2 + y**2 <= radius**2).to(torch.float32)

    # Calculate padding for convolution to preserve the original size
    padding_x = selem.size(0) // 2
    padding_y = selem.size(1) // 2

    # Apply binary dilation to expand the mask
    expanded_mask = F.conv2d(
        mask.unsqueeze(0).unsqueeze(0),
        selem.unsqueeze(0).unsqueeze(0),
        padding=(padding_x, padding_y),
    )

    # Binarize the expanded mask (optional)
    expanded_mask = (expanded_mask > 0).to(torch.float32)
    expanded_mask = expanded_mask[0, 0] > threshold
    assert (
        expanded_mask.shape == mask.shape
    ), "something went wrong when computing padding, most likely - shape not preserved"
    return expanded_mask


def find_closest_point_on_mask(mask: torch.Tensor, point: torch.Tensor):
    """
    Find the closest point on a binary mask to another point.

    Parameters:
    - mask: Binary mask where 1 represents the region of interest (PyTorch tensor).
    - point: Coordinates of the target point (PyTorch tensor).

    Returns:
    - closest_point: Coordinates of the closest point on the mask (PyTorch tensor).
    """
    # Ensure the input mask is binary (0 or 1)
    mask = (mask > 0).to(torch.float32)

    # Find all nonzero (1) pixels in the mask
    nonzero_pixels = torch.nonzero(mask, as_tuple=False)

    if nonzero_pixels.size(0) == 0:
        # If the mask has no nonzero pixels, return None
        return None

    # Calculate the Euclidean distance between the target point and all nonzero pixels
    distances = torch.norm(nonzero_pixels - point, dim=1)

    # Find the index of the closest pixel
    closest_index = torch.argmin(distances)

    # Get the closest point
    closest_point = nonzero_pixels[closest_index]

    return closest_point
