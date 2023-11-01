# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh.transformations as tra
from torch import Tensor


class Camera(object):
    """
    Simple pinhole camera model. Contains parameters for projecting from depth to xyz, and saving information about camera position for planning.
    TODO: Move this to utils/cameras.py?
    """

    @staticmethod
    def from_width_height_fov(
        width: float,
        height: float,
        fov_degrees: float,
        near_val: float = 0.1,
        far_val: float = 4.0,
    ):
        """Create a simple pinhole camera given minimal information only. Fov is in degrees"""
        horizontal_fov_rad = np.radians(fov_degrees)
        h_focal_length = width / (2 * np.tan(horizontal_fov_rad / 2))
        v_focal_length = width / (
            2 * np.tan(horizontal_fov_rad / 2) * float(height) / width
        )
        principal_point_x = (width - 1.0) / 2
        principal_point_y = (height - 1.0) / 2
        return Camera(
            (0, 0, 0),
            (0, 0, 0, 1),
            height,
            width,
            h_focal_length,
            v_focal_length,
            principal_point_x,
            principal_point_y,
            near_val,
            far_val,
            np.eye(4),
            None,
            None,
            horizontal_fov_rad,
        )

    @staticmethod
    def from_K(K: np.ndarray, width: float, height: float):
        """return camera created from a 3x3 camera intrinsics matrix K"""
        assert K.shape == (3, 3)
        return Camera(
            (0, 0, 0),
            (0, 0, 0, 1),
            height,
            width,
            K[0, 0],
            K[1, 1],
            K[0, 2],
            K[1, 2],
            0,
            5,
            np.eye(4),
            None,
            None,
            None,
        )

    def __init__(
        self,
        pos,
        orn,
        height,
        width,
        fx,
        fy,
        px,
        py,
        near_val,
        far_val,
        pose_matrix,
        proj_matrix,
        view_matrix,
        fov,
        *args,
        **kwargs,
    ):
        self.pos = pos
        self.orn = orn
        self.height = height
        self.width = width
        self.px = px
        self.py = py
        self.fov = fov
        self.near_val = near_val
        self.far_val = far_val
        self.fx = fx
        self.fy = fy
        self.pose_matrix = pose_matrix
        self.pos = pos
        self.orn = orn
        self.K = np.array([[self.fx, 0, self.px], [0, self.fy, self.py], [0, 0, 1]])

    def to_dict(self):
        """create a dictionary so that we can extract the necessary information for
        creating point clouds later on if we so desire"""
        info = {}
        info["pos"] = self.pos
        info["orn"] = self.orn
        info["height"] = self.height
        info["width"] = self.width
        info["near_val"] = self.near_val
        info["far_val"] = self.far_val
        info["proj_matrix"] = self.proj_matrix
        info["view_matrix"] = self.view_matrix
        info["max_depth"] = self.max_depth
        info["pose_matrix"] = self.pose_matrix
        info["px"] = self.px
        info["py"] = self.py
        info["fx"] = self.fx
        info["fy"] = self.fy
        info["fov"] = self.fov
        return info

    def get_pose(self):
        return self.pose_matrix.copy()

    def depth_to_xyz(self, depth):
        """get depth from numpy using simple pinhole self model"""
        indices = np.indices((self.height, self.width), dtype=np.float32).transpose(
            1, 2, 0
        )
        z = depth
        # pixel indices start at top-left corner. for these equations, it starts at bottom-left
        x = (indices[:, :, 1] - self.px) * (z / self.fx)
        y = (indices[:, :, 0] - self.py) * (z / self.fy)
        # Should now be height x width x 3, after this:
        xyz = np.stack([x, y, z], axis=-1)
        return xyz

    def fix_depth(self, depth):
        if isinstance(depth, np.ndarray):
            depth = depth.copy()
        else:
            # Assuming it's a torch tensor instead
            depth = depth.clone()

        depth[depth > self.far_val] = 0
        depth[depth < self.near_val] = 0
        return depth


def z_from_opengl_depth(depth, camera: Camera):
    near = camera.near_val
    far = camera.far_val
    # return (2.0 * near * far) / (near + far - depth * (far - near))
    return (near * far) / (far - depth * (far - near))


# We apply this correction to xyz when computing it in sim
# R_CORRECTION = R1 @ R2
T_CORRECTION = tra.euler_matrix(0, 0, np.pi / 2)
R_CORRECTION = T_CORRECTION[:3, :3]


def opengl_depth_to_xyz(depth, camera: Camera):
    """get depth from numpy using simple pinhole camera model"""
    indices = np.indices((camera.height, camera.width), dtype=np.float32).transpose(
        1, 2, 0
    )
    z = depth
    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    # indices[..., 0] = np.flipud(indices[..., 0])
    x = (indices[:, :, 1] - camera.px) * (z / camera.fx)
    y = (indices[:, :, 0] - camera.py) * (z / camera.fy)  # * -1
    # Should now be height x width x 3, after this:
    xyz = np.stack([x, y, z], axis=-1) @ R_CORRECTION
    return xyz


def depth_to_xyz(depth, camera: Camera):
    """get depth from numpy using simple pinhole camera model"""
    indices = np.indices((camera.height, camera.width), dtype=np.float32).transpose(
        1, 2, 0
    )
    z = depth
    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    x = (indices[:, :, 1] - camera.px) * (z / camera.fx)
    y = (indices[:, :, 0] - camera.py) * (z / camera.fy)
    # Should now be height x width x 3, after this:
    xyz = np.stack([x, y, z], axis=-1)
    return xyz


def smooth_mask(mask, kernel=None, num_iterations=3):
    """Dilate and then erode.

    Arguments:
        mask: the mask to clean up

    Returns:
        mask: the dilated mask
        mask2: dilated, then eroded mask
    """
    if kernel is None:
        kernel = np.ones((5, 5))
    mask = mask.astype(np.uint8)
    mask1 = cv2.dilate(mask, kernel, iterations=num_iterations)
    # second step
    mask2 = mask
    mask2 = cv2.erode(mask2, kernel, iterations=num_iterations)
    mask2 = np.bitwise_and(mask, mask2)
    return mask1, mask2


def rotate_image(imgs: List[np.ndarray]) -> List[np.ndarray]:
    """stretch specific routine to flip and rotate sideways images for normal viewing"""
    imgs = [np.rot90(np.fliplr(np.flipud(x))) for x in imgs]
    return imgs


def build_mask(
    target: Tensor, val: float = 0.0, tol: float = 1e-3, mask_extra_radius: int = 5
) -> Tensor:
    """Build mask where all channels are (val - tol) <= target <= (val + tol)
        Optionally, dilate by mask_extra_radius

    Args:
        target (Tensor): [B, N_channels, H, W] input tensor
        val (float): Value to use for masking. Defaults to 0.0.
        tol (float): Tolerance for mask. Defaults to 1e-3.
        mask_extra_radius (int, optional): Dilate by mask_extra_radius pix . Defaults to 5.

    Returns:
        _type_: Mask of shape target.shape
    """
    assert (
        target.ndim == 4
    ), f"target should be of shape [B, N_channels, H, W], was {target.shape}"
    if target.shape[1] == 1:
        masks = [target[:, t] for t in range(target.shape[1])]
        masks = [(t >= val - tol) & (t <= val + tol) for t in masks]
        mask = functools.reduce(lambda a, b: a & b, masks).unsqueeze(1)
    else:
        mask = (target >= val - tol) & (target <= val + tol)
    mask = 0 != F.conv2d(
        mask.float(),
        torch.ones(1, 1, mask_extra_radius, mask_extra_radius, device=mask.device),
        padding=(mask_extra_radius // 2),
    )
    return (~mask).expand_as(target)


def dilate_or_erode_mask(mask: Tensor, radius: int, num_iterations=1) -> Tensor:
    """
    Dilate or erode a binary mask using a square kernel.

    This function either dilates or erodes a 2D binary mask based on the given radius
    and number of iterations. A positive radius value will dilate the mask, while a
    negative radius value will erode it.

    Parameters:
    -----------
    mask : torch.Tensor
        A 2D binary mask of shape (H, W), where H is the height and W is the width.
        The dtype must be torch.bool.
    radius : int
        The radius of the square kernel used for dilation or erosion. A positive value
        will dilate the mask, while a negative value will erode it.
    num_iterations : int, optional
        The number of times the dilation or erosion operation should be applied.
        Default is 1.

    Returns:
    --------
    Tensor : torch.Tensor
        A dilated or eroded 2D binary mask of the same shape as the input mask.

    Raises:
    -------
    AssertionError
        If the dtype of the input mask is not torch.bool.

    Example:
    --------
    >>> mask = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.bool)
    >>> dilated_mask = dilate_or_erode_mask(mask, radius=1)
    >>> eroded_mask = dilate_or_erode_mask(mask, radius=-1)

    """
    assert mask.dtype == torch.bool, mask.dtype
    abs_radius = abs(radius)
    erode = radius < 0
    if erode:
        mask = ~mask
    mask = mask.half()
    conv_kernel = torch.ones(
        (1, 1, abs_radius, abs_radius), dtype=mask.dtype, device=mask.device
    )
    for _ in range(num_iterations):
        mask = mask.half()
        mask = F.conv2d(mask, conv_kernel, padding="same")
        mask = mask > 0.0
    if erode:
        mask = ~mask
    return mask


def get_cropped_image_with_padding(self, image, bbox, padding: float = 1.0):
    """
    Crop an image based on a bounding box with optional padding.

    Given an image and a bounding box, this function returns a cropped version of
    the image. Padding can be applied to extend the area of the cropped region.

    Parameters:
    -----------
    image : torch.Tensor
        Input image tensor of shape (C, H, W), where C is the number of channels,
        H is the height, and W is the width.
    bbox : torch.Tensor
        A bounding box tensor of shape (2, 2), where the first row contains the
        (y, x) coordinates of the top-left corner, and the second row contains the
        (y, x) coordinates of the bottom-right corner.
    padding : float, optional
        Padding factor applied to the bounding box dimensions. Default is 1.0, which
        means no padding. A value greater than 1.0 will increase the cropped area.

    Returns:
    --------
    cropped_image : torch.Tensor
        The cropped image tensor of shape (C, H', W'), where H' and W' are the
        dimensions of the cropped region.

    Example:
    --------
    >>> image = torch.rand(3, 100, 100)
    >>> bbox = torch.tensor([[10, 20], [50, 60]])
    >>> cropped_image = get_cropped_image_with_padding(image, bbox, padding=1.2)

    Notes:
    ------
    The function ensures that the cropped region does not exceed the original image
    dimensions. If the padded bounding box does, it will be clipped to fit within
    the image.

    """
    im_h = image.shape[1]
    im_w = image.shape[2]
    # bbox = iv.bbox
    x = bbox[0, 1]
    y = bbox[0, 0]
    w = bbox[1, 1] - x
    h = bbox[1, 0] - y
    x = 0 if (x - (padding - 1) * w / 2) < 0 else int(x - (padding - 1) * w / 2)
    y = 0 if (y - (padding - 1) * h / 2) < 0 else int(y - (padding - 1) * h / 2)
    y2 = im_h if y + int(h * padding) >= im_h else y + int(h * padding)
    x2 = im_w if x + int(w * padding) >= im_w else x + int(w * padding)
    cropped_image = image[
        :,
        y:y2,
        x:x2,
    ]
    return cropped_image


def interpolate_image(image: Tensor, scale_factor: float = 1.0, mode: str = "nearest"):
    """
    Interpolates images by the specified scale_factor using the specific interpolation mode.
    This method uses `torch.nn.functional.interpolate` by temporarily adding batch dimension and channel dimension for 2D inputs.
    image (Tensor): image of shape [3, H, W] or [H, W]
    scale_factor (float): multiplier for spatial size
    mode: (str): algorithm for interpolation: 'nearest' (default), 'bicubic' or other interpolation modes at https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    """

    if len(image.shape) == 2:
        image = image.unsqueeze(0)

    image_downsampled = (
        torch.nn.functional.interpolate(
            image.unsqueeze(0).float(),
            scale_factor=scale_factor,
            mode=mode,
        )
        .squeeze()
        .squeeze()
        .bool()
    )
    return image_downsampled
