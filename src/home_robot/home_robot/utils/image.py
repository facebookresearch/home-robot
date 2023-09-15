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
        **kwargs
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
        val (float, optional): Value to use for masking. Defaults to 0.0.
        tol (float, optional): Tolerance for mask. Defaults to 1e-3.
        mask_extra_radius (int, optional): Dilate by mask_extra_radius pix . Defaults to 5.

    Returns:
        _type_: Mask of shape target.shape
    """
    assert target.ndim == 4, target.shape
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
    #     mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2) != 0
    return (~mask).expand_as(target)


def dilate_or_erode_mask(mask: Tensor, radius: int, num_iterations=1) -> Tensor:
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
