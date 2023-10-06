# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF

import home_robot.utils.bullet as hrb


class Footprint:
    """contains information about robot footprint"""

    def __init__(
        self,
        length: float,
        width: float,
        length_offset: float = 0.0,
        width_offset: float = 0.0,
    ):
        self.length = length
        self.width = width
        self.length_offset = length_offset
        self.width_offset = width_offset

    def get_mask(
        self, resolution: float, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Get a single mask for this robot"""
        size = int(
            np.ceil(
                np.sqrt(
                    (self.width + abs(self.width_offset)) ** 2
                    + (self.length + abs(self.length_offset)) ** 2
                )
                / resolution
            )
        )
        # Now try
        width = int(np.ceil(self.width / resolution))
        length = int(np.ceil(self.length / resolution))
        l0_offset = int(np.floor(self.length_offset / resolution))
        l1_offset = int(np.ceil(self.length_offset / resolution))
        w0_offset = int(np.floor(self.width_offset / resolution))
        w1_offset = int(np.ceil(self.width_offset / resolution))
        mask = torch.zeros((size, size))
        center = size // 2
        if size % 2 == 0:
            size += 1
        else:
            # Unequal, will look goofy
            w1_offset += 1
            l1_offset += 1
        x0 = center - (width // 2) + w0_offset
        x1 = center + (width // 2) + w1_offset
        y0 = center - (length // 2) + l0_offset
        y1 = center + (length // 2) + 1 + l1_offset
        mask[y0:y1, x0:x1] = 1
        return mask.bool()

    def get_rotated_mask(
        self,
        resolution: float,
        angle_radians: float,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Get a single mask for collision checking against the robot's footprint, and then rotate it"""
        mask = self.get_mask(resolution, device).unsqueeze(0).unsqueeze(0)
        mask = TF.rotate(mask, np.rad2deg(angle_radians))
        return mask.squeeze(0).squeeze(0).bool()


class Robot(abc.ABC):
    """placeholder"""

    def __init__(
        self,
        name="robot",
        urdf_path: Optional[str] = None,
        visualize=False,
        assets_path=None,
    ):
        # Load and create planner
        self.backend = hrb.PbClient(visualize=visualize)
        if urdf_path is not None:
            # Create object reference
            self.ref = self.backend.add_articulated_object(
                name, urdf_path, assets_path=assets_path
            )

    def get_backend(self) -> hrb.PbClient:
        """Return model of the robot in bullet - environment for 3d collision checks"""
        return self.backend

    def get_dof(self) -> int:
        """return degrees of freedom of the robot"""
        return self.dof

    @abc.abstractmethod
    def set_config(self, q):
        """put the robot in the right position for bullet planning"""
        raise NotImplementedError

    def get_config(self):
        """turn current state into a vector"""
        raise NotImplementedError

    def set_head_config(self, q):
        """just for the head"""
        raise NotImplementedError

    def set_camera_to_head(self, camera, q=None):
        """take a bullet camera and put it on the robot's head"""
        if q is not None:
            self.set_head_config(q)
        raise NotImplementedError

    @abc.abstractmethod
    def get_footprint(self) -> torch.Tensor:
        """return a footprint mask that we can check 2d collisions against"""
        raise NotImplementedError
