# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import cv2
import numpy as np
import trimesh.transformations as tra


class Camera(object):
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
