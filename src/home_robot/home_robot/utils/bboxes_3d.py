# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from pytorch3d.structures.Pointclouds which has license:
# BSD License

# For PyTorch3D software

# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

#  * Neither the name Meta nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.

import warnings

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from itertools import zip_longest
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from pytorch3d.common.datatypes import Device, make_device
from pytorch3d.ops import box3d_overlap
from pytorch3d.structures import utils as struct_utils
from torch import Tensor


class BBoxes3D:
    _INTERNAL_TENSORS = [
        "_bounds_packed",  # bounds
        "_bounds_padded",
        "_names_packed",
        "_names_padded",
        "_features_packed",
        "_features_padded",
        "_packed_to_scene_idx",
        "_scene_to_packed_first_idx",
        "_num_boxes_per_scene",
        "_padded_to_packed_idx",
        "valid",
        "equisized",
    ]

    def __init__(
        self,
        bounds: List[torch.Tensor],
        features: Optional[List[torch.Tensor]] = None,
        names: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """
        Args:
            bounds:
                Can be either

                - List where each element is a tensor of shape (num_boxes, 3)
                  containing the (x, y, z) coordinates of each point.
                Not Implemented: - Padded float tensor with shape (num_scenes, num_boxes, 3).
            names:
                Can be either

                - None
                - List where each element is a tensor of shape (num_boxes, 3)
                  containing the normal vector for each point.
                Not Implemented: - Padded float tensor of shape (num_scenes, num_boxes, 3).
            features:
                Can be either

                - None
                - List where each element is a tensor of shape (num_boxes, C)
                  containing the features for the boxes in the scene.
                Not Implemented: - Padded float tensor of shape (num_scenes, num_boxes, C).
                where C is the number of channels in the features.
                For example 3 for RGB color.

        Refer to comments above for descriptions of List and Padded
        representations.

        # Internally, boxes are represented as bounds.
        # But we might later want to add orientation Pytorch3d.transforms.Rotation or Pytorch3d.transforms.Transform (e.g. if we want to encode viewing angle. )
        """
        self.device = torch.device("cpu")

        # Indicates whether the scenes in the list/batch have the same number
        # of boxes.
        self.equisized = False

        # Boolean indicator for each scene in the batch.
        # True if scene has non zero number of boxes, False otherwise.
        self.valid = None

        self._N = 0  # batch size (number of scenes)
        self._P = 0  # (max) number of boxes per scene
        self._C = None  # number of channels in the features

        # List of Tensors of boxes and features.
        self._bounds_list = None
        self._names_list = None
        self._features_list = None

        # Number of boxes per scene.
        self._num_boxes_per_scene = None  # N

        # Packed representation.
        self._bounds_packed = None  # (sum(P_n), 3)
        self._names_packed = None  # (sum(P_n), 3)
        self._features_packed = None  # (sum(P_n), C)

        self._packed_to_scene_idx = None  # sum(P_n)

        # Index of each scene's first point in the packed boxes.
        # Assumes packing is sequential.
        self._scene_to_packed_first_idx = None  # N

        # Padded representation.
        self._bounds_padded = None  # (N, max(P_n), 3)
        self._names_padded = None  # (N, max(P_n), 3)
        self._features_padded = None  # (N, max(P_n), C)

        # Index to convert boxes from flattened padded to packed.
        self._padded_to_packed_idx = None  # N * max_P

        # Identify type of boxes.
        if isinstance(bounds, list):
            self._bounds_list = bounds
            self._N = len(self._bounds_list)
            self.valid = torch.zeros((self._N,), dtype=torch.bool, device=self.device)

            if self._N > 0:
                self.device = self._bounds_list[0].device
                for p in self._bounds_list:
                    if len(p) > 0 and (
                        p.dim() != 3 or p.shape[1] != 3 or p.shape[2] != 2
                    ):
                        raise ValueError(
                            "BBoxes in list must be of shape Px3x2 or empty"
                        )
                    if p.device != self.device:
                        raise ValueError("All boxes must be on the same device")

                num_boxes_per_scene = torch.tensor(
                    [len(p) for p in self._bounds_list], device=self.device
                )
                self._P = int(num_boxes_per_scene.max())
                self.valid = torch.tensor(
                    [len(p) > 0 for p in self._bounds_list],
                    dtype=torch.bool,
                    device=self.device,
                )

                if len(num_boxes_per_scene.unique()) == 1:
                    self.equisized = True
                self._num_boxes_per_scene = num_boxes_per_scene
            else:
                self._num_boxes_per_scene = torch.tensor([], dtype=torch.int64)
        elif torch.is_tensor(bounds):
            if bounds.dim() != 4 or bounds.shape[2] != 3 or bounds.shape[3] != 2:
                raise ValueError("bounds tensor has incorrect dimensions.")
            self._bounds_padded = bounds
            self._N = self._bounds_padded.shape[0]
            self._P = self._bounds_padded.shape[1]
            self.device = self._bounds_padded.device
            self.valid = torch.ones((self._N,), dtype=torch.bool, device=self.device)
            self._num_boxes_per_scene = torch.tensor(
                [self._P] * self._N, device=self.device
            )
            self.equisized = True
        else:
            raise ValueError(
                "BBoxes must be either a list or a tensor with \
                    shape (batch_size, K, 3, 2) where K is the maximum number of \
                    boxes in a scene."
            )

        # parse names
        names_parsed = self._parse_auxiliary_input(names)
        self._names_list, self._names_padded, names_C = names_parsed
        if names_C is not None and names_C != 1:
            raise ValueError("Names are expected to be 1-dimensional")

        # parse features
        features_parsed = self._parse_auxiliary_input(features)
        self._features_list, self._features_padded, features_C = features_parsed
        if features_C is not None:
            self._C = features_C

    def _parse_auxiliary_input(
        self, aux_input
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor], Optional[int]]:
        """
        Interpret the auxiliary inputs (names, features) given to __init__.

        Args:
            aux_input:
              Can be either

                - List where each element is a tensor of shape (num_boxes, C)
                  containing the features for the boxes in the scene.
                - Padded float tensor of shape (num_scenes, numboxes, C).
              For names, C = 3

        Returns:
            3-element tuple of list, padded, num_channels.
            If aux_input is list, then padded is None. If aux_input is a tensor,
            then list is None.
        """
        if aux_input is None or self._N == 0:
            return None, None, None

        aux_input_C = None

        if isinstance(aux_input, list):
            return self._parse_auxiliary_input_list(aux_input)
        if torch.is_tensor(aux_input):
            if aux_input.dim() != 3:
                raise ValueError("Auxiliary input tensor has incorrect dimensions.")
            if self._N != aux_input.shape[0]:
                raise ValueError("Boxes and inputs must be the same length.")
            if self._P != aux_input.shape[1]:
                raise ValueError(
                    "Inputs tensor must have the right maximum \
                    number of boxes in each scene."
                )
            if aux_input.device != self.device:
                raise ValueError(
                    "All auxiliary inputs must be on the same device as the boxes."
                )
            aux_input_C = aux_input.shape[2]
            return None, aux_input, aux_input_C
        else:
            raise ValueError(
                "Auxiliary input must be either a list or a tensor with \
                    shape (batch_size, P, C) where P is the maximum number of \
                    boxes in a scene."
            )

    def _parse_auxiliary_input_list(
        self, aux_input: list
    ) -> Tuple[Optional[List[torch.Tensor]], None, Optional[int]]:
        """
        Interpret the auxiliary inputs (names, features) given to __init__,
        if a list.

        Args:
            aux_input:
                - List where each element is a tensor of shape (num_boxes, C)
                  containing the features for the boxes in the scene.
              For names, C = 3

        Returns:
            3-element tuple of list, padded=None, num_channels.
            If aux_input is list, then padded is None. If aux_input is a tensor,
            then list is None.
        """
        aux_input_C = None
        good_empty = None
        needs_fixing = False

        if len(aux_input) != self._N:
            raise ValueError("Bboxes and auxiliary input must be the same length.")
        for p, d in zip(self._num_boxes_per_scene, aux_input):
            valid_but_empty = p == 0 and d is not None and d.ndim == 2
            if p > 0 or valid_but_empty:
                if p != d.shape[0]:
                    raise ValueError(
                        "A scene has mismatched numbers of boxes and inputs"
                    )
                if d.dim() != 2:
                    raise ValueError(
                        "A scene auxiliary input must be of shape PxC or empty"
                    )
                if aux_input_C is None:
                    aux_input_C = d.shape[1]
                elif aux_input_C != d.shape[1]:
                    raise ValueError("The scenes must have the same number of channels")
                if d.device != self.device:
                    raise ValueError(
                        "All auxiliary inputs must be on the same device as the boxes."
                    )
            else:
                needs_fixing = True

        if aux_input_C is None:
            # We found nothing useful
            return None, None, None

        # If we have empty but "wrong" inputs we want to store "fixed" versions.
        if needs_fixing:
            if good_empty is None:
                good_empty = torch.zeros((0, aux_input_C), device=self.device)
            aux_input_out = []
            for p, d in zip(self._num_boxes_per_scene, aux_input):
                valid_but_empty = p == 0 and d is not None and d.ndim == 2
                if p > 0 or valid_but_empty:
                    aux_input_out.append(d)
                else:
                    aux_input_out.append(good_empty)
        else:
            aux_input_out = aux_input

        return aux_input_out, None, aux_input_C

    def get_world_to_box_rotation(self):
        pass

    def __len__(self) -> int:
        return self._N

    def __getitem__(
        self,
        index: Union[int, List[int], slice, torch.BoolTensor, torch.LongTensor],
    ) -> "BBoxes3D":
        """
        Args:
            index: Specifying the index of the scene to retrieve.
                Can be an int, slice, list of ints or a boolean tensor.

        Returns:
            BBoxes3D object with selected scenes. The tensors are not cloned.
        """
        names, features = None, None
        names_list = self.names_list()
        features_list = self.features_list()
        if isinstance(index, int):
            bounds = [self.bounds_list()[index]]
            if names_list is not None:
                names = [names_list[index]]
            if features_list is not None:
                features = [features_list[index]]
        elif isinstance(index, slice):
            bounds = self.bounds_list()[index]
            if names_list is not None:
                names = names_list[index]
            if features_list is not None:
                features = features_list[index]
        elif isinstance(index, list):
            bounds = [self.bounds_list()[i] for i in index]
            if names_list is not None:
                names = [names_list[i] for i in index]
            if features_list is not None:
                features = [features_list[i] for i in index]
        elif isinstance(index, torch.Tensor):
            if index.dim() != 1 or index.dtype.is_floating_point:
                raise IndexError(index)
            # NOTE consider converting index to cpu for efficiency
            if index.dtype == torch.bool:
                # advanced indexing on a single dimension
                index = index.nonzero()
                index = index.squeeze(1) if index.numel() > 0 else index
                index = index.tolist()
            bounds = [self.bounds_list()[i] for i in index]
            if names_list is not None:
                names = [names_list[i] for i in index]
            if features_list is not None:
                features = [features_list[i] for i in index]
        else:
            raise IndexError(index)

        return self.__class__(bounds=bounds, names=names, features=features)

    def isempty(self) -> bool:
        """
        Checks whether any scene is valid.

        Returns:
            bool indicating whether there is any data.
        """
        return self._N == 0 or self.valid.eq(False).all()

    def bounds_list(self) -> List[torch.Tensor]:
        """
        Get the list representation of the boxes.

        Returns:
            list of tensors of boxes of shape (P_n, 3).
        """
        if self._bounds_list is None:
            assert (
                self._bounds_padded is not None
            ), "bounds_padded is required to compute bounds_list."
            bounds_list = []
            for i in range(self._N):
                bounds_list.append(
                    self._bounds_padded[i, : self.num_boxes_per_scene()[i]]
                )
            self._bounds_list = bounds_list
        return self._bounds_list

    def names_list(self) -> Optional[List[torch.Tensor]]:
        """
        Get the list representation of the names,
        or None if there are no names.

        Returns:
            list of tensors of names of shape (P_n, 3).
        """
        if self._names_list is None:
            if self._names_padded is None:
                # No names provided so return None
                return None
            self._names_list = struct_utils.padded_to_list(
                self._names_padded, self.num_boxes_per_scene().tolist()
            )
        return self._names_list

    def features_list(self) -> Optional[List[torch.Tensor]]:
        """
        Get the list representation of the features,
        or None if there are no features.

        Returns:
            list of tensors of features of shape (P_n, C).
        """
        if self._features_list is None:
            if self._features_padded is None:
                # No features provided so return None
                return None
            self._features_list = struct_utils.padded_to_list(
                self._features_padded, self.num_boxes_per_scene().tolist()
            )
        return self._features_list

    def bounds_packed(self) -> torch.Tensor:
        """
        Get the packed representation of the boxes.

        Returns:
            tensor of boxes of shape (sum(P_n), 3).
        """
        self._compute_packed()
        return self._bounds_packed

    def names_packed(self) -> Optional[torch.Tensor]:
        """
        Get the packed representation of the names.

        Returns:
            tensor of names of shape (sum(P_n), 3),
            or None if there are no names.
        """
        self._compute_packed()
        return self._names_packed

    def features_packed(self) -> Optional[torch.Tensor]:
        """
        Get the packed representation of the features.

        Returns:
            tensor of features of shape (sum(P_n), C),
            or None if there are no features
        """
        self._compute_packed()
        return self._features_packed

    def packed_to_scene_idx(self):
        """
        Return a 1D tensor x with length equal to the total number of boxes.
        packed_to_scene_idx()[i] gives the index of the scene which contains
        bounds_packed()[i].

        Returns:
            1D tensor of indices.
        """
        self._compute_packed()
        return self._packed_to_scene_idx

    def packed_to_scene_first_idx(self):
        """
        Return a 1D tensor x with length equal to the number of scenes such that
        the first point of the ith scene is bounds_packed[x[i]].

        Returns:
            1D tensor of indices of first items.
        """
        self._compute_packed()
        return self._scene_to_scene_first_idx

    def num_boxes_per_scene(self) -> torch.Tensor:
        """
        Return a 1D tensor x with length equal to the number of scenes giving
        the number of boxes in each scene.

        Returns:
            1D tensor of sizes.
        """
        return self._num_boxes_per_scene

    def bounds_padded(self) -> torch.Tensor:
        """
        Get the padded representation of the boxes.

        Returns:
            tensor of boxes of shape (N, max(P_n), 3).
        """
        self._compute_padded()
        return self._bounds_padded

    def names_padded(self) -> Optional[torch.Tensor]:
        """
        Get the padded representation of the names,
        or None if there are no names.

        Returns:
            tensor of names of shape (N, max(P_n), 3).
        """
        self._compute_padded()
        return self._names_padded

    def features_padded(self) -> Optional[torch.Tensor]:
        """
        Get the padded representation of the features,
        or None if there are no features.

        Returns:
            tensor of features of shape (N, max(P_n), 3).
        """
        self._compute_padded()
        return self._features_padded

    def padded_to_packed_idx(self):
        """
        Return a 1D tensor x with length equal to the total number of boxes
        such that bounds_packed()[i] is element x[i] of the flattened padded
        representation.
        The packed representation can be calculated as follows.

        .. code-block:: python

            p = bounds_padded().reshape(-1, 3)
            bounds_packed = p[x]

        Returns:
            1D tensor of indices.
        """
        if self._padded_to_packed_idx is not None:
            return self._padded_to_packed_idx
        if self._N == 0:
            self._padded_to_packed_idx = []
        else:
            self._padded_to_packed_idx = torch.cat(
                [
                    torch.arange(v, dtype=torch.int64, device=self.device) + i * self._P
                    for (i, v) in enumerate(self.num_boxes_per_scene())
                ],
                dim=0,
            )
        return self._padded_to_packed_idx

    def _compute_padded(self, refresh: bool = False):
        """
        Computes the padded version from bounds_list, names_list and features_list.

        Args:
            refresh: whether to force the recalculation.
        """
        if not (refresh or self._bounds_padded is None):
            return

        self._namess_padded, self._features_padded = None, None
        if self.isempty():
            self._bounds_padded = torch.zeros((self._N, 0, 3), device=self.device)
        else:
            self._bounds_padded = struct_utils.list_to_padded(
                self.bounds_list(),
                (self._P, 3, 3),
                pad_value=0.0,
                equisized=self.equisized,
            )
            names_list = self.names_list()
            if names_list is not None:
                self._names_padded = struct_utils.list_to_padded(
                    names_list,
                    (self._P, 1),
                    pad_value=0.0,
                    equisized=self.equisized,
                )
            features_list = self.features_list()
            if features_list is not None:
                self._features_padded = struct_utils.list_to_padded(
                    features_list,
                    (self._P, self._C),
                    pad_value=0.0,
                    equisized=self.equisized,
                )

    def _compute_packed(self, refresh: bool = False):
        """
        Computes the packed version from bounds_list, names_list and
        features_list and sets the values of auxiliary tensors.

        Args:
            refresh: Set to True to force recomputation of packed
                representations. Default: False.
        """

        if not (
            refresh
            or any(
                v is None
                for v in [
                    self._bounds_packed,
                    self._packed_to_scene_idx,
                    self._scene_to_packed_first_idx,
                ]
            )
        ):
            return

        # Packed can be calculated from padded or list, so can call the
        # accessor function for the lists.
        bounds_list = self.bounds_list()
        names_list = self.names_list()
        features_list = self.features_list()
        if self.isempty():
            self._bounds_packed = torch.zeros(
                (0, 3), dtype=torch.float32, device=self.device
            )
            self._packed_to_scene_idx = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )
            self._scene_to_packed_first_idx = torch.zeros(
                (0,), dtype=torch.int64, device=self.device
            )
            self._names_packed = None
            self._features_packed = None
            return

        bounds_list_to_packed = struct_utils.list_to_packed(bounds_list)
        self._bounds_packed = bounds_list_to_packed[0]
        if not torch.allclose(self._num_boxes_per_scene, bounds_list_to_packed[1]):
            raise ValueError("Inconsistent list to packed conversion")
        self._scene_to_packed_first_idx = bounds_list_to_packed[2]
        self._packed_to_scene_idx = bounds_list_to_packed[3]

        self._names_packed, self._features_packed = None, None
        if names_list is not None:
            names_list_to_packed = struct_utils.list_to_packed(names_list)
            self._names_packed = names_list_to_packed[0]

        if features_list is not None:
            features_list_to_packed = struct_utils.list_to_packed(features_list)
            self._features_packed = features_list_to_packed[0]

    def clone(self):
        """
        Deep copy of BBoxes3D object. All internal tensors are cloned
        individually.

        Returns:
            new BBoxes3D object.
        """
        # instantiate new boxescene with the representation which is not None
        # (either list or tensor) to save compute.
        new_bounds, new_names, new_features = None, None, None
        if self._bounds_list is not None:
            new_bounds = [v.clone() for v in self.bounds_list()]
            names_list = self.names_list()
            features_list = self.features_list()
            if names_list is not None:
                new_names = [n.clone() for n in names_list]
            if features_list is not None:
                new_features = [f.clone() for f in features_list]
        elif self._bounds_padded is not None:
            new_bounds = self.bounds_padded().clone()
            names_padded = self.names_padded()
            features_padded = self.features_padded()
            if names_padded is not None:
                new_names = self.names_padded().clone()
            if features_padded is not None:
                new_features = self.features_padded().clone()
        other = self.__class__(
            bounds=new_bounds, names=new_names, features=new_features
        )
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        return other

    def detach(self):
        """
        Detach BBoxes3D object. All internal tensors are detached
        individually.

        Returns:
            new BBoxes3D object.
        """
        # instantiate new boxescene with the representation which is not None
        # (either list or tensor) to save compute.
        new_bounds, new_names, new_features = None, None, None
        if self._bounds_list is not None:
            new_bounds = [v.detach() for v in self.bounds_list()]
            names_list = self.names_list()
            features_list = self.features_list()
            if names_list is not None:
                new_names = [n.detach() for n in names_list]
            if features_list is not None:
                new_features = [f.detach() for f in features_list]
        elif self._bounds_padded is not None:
            new_bounds = self.bounds_padded().detach()
            names_padded = self.names_padded()
            features_padded = self.features_padded()
            if names_padded is not None:
                new_names = self.names_padded().detach()
            if features_padded is not None:
                new_features = self.features_padded().detach()
        other = self.__class__(
            bounds=new_bounds, names=new_names, features=new_features
        )
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.detach())
        return other

    def to(self, device: Device, copy: bool = False):
        """
        Match functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
          device: Device (as str or torch.device) for the new tensor.
          copy: Boolean indicator whether or not to clone self. Default False.

        Returns:
          BBoxes3D object.
        """
        device_ = make_device(device)

        if not copy and self.device == device_:
            return self

        other = self.clone()
        if self.device == device_:
            return other

        other.device = device_
        if other._N > 0:
            other._bounds_list = [v.to(device_) for v in other.bounds_list()]
            if other._names_list is not None:
                other._names_list = [n.to(device_) for n in other.names_list()]
            if other._features_list is not None:
                other._features_list = [f.to(device_) for f in other.features_list()]
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.to(device_))
        return other

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def get_scene(self, index: int):
        """
        Get tensors for a single scene from the list representation.

        Args:
            index: Integer in the range [0, N).

        Returns:
            boxes: Tensor of shape (P, 3).
            names: Tensor of shape (P, 3)
            features: LongTensor of shape (P, C).
        """
        if not isinstance(index, int):
            raise ValueError("scene index must be an integer.")
        if index < 0 or index > self._N:
            raise ValueError(
                "scene index must be in the range [0, N) where \
            N is the number of scenes in the batch."
            )
        bounds = self.bounds_list()[index]
        names, features = None, None
        names_list = self.names_list()
        if names_list is not None:
            names = names_list[index]
        features_list = self.features_list()
        if features_list is not None:
            features = features_list[index]
        return bounds, names, features

    def split(self, split_sizes: list):
        """
        Splits BBoxes3D object of size N into a list of BBoxes3D objects
        of size len(split_sizes), where the i-th BBoxes3D object is of size
        split_sizes[i]. Similar to torch.split().

        Args:
            split_sizes: List of integer sizes of BBoxes3D objects to be
            returned.

        Returns:
            list[BBoxes3D].
        """
        if not all(isinstance(x, int) for x in split_sizes):
            raise ValueError("Value of split_sizes must be a list of integers.")
        scenelist = []
        curi = 0
        for i in split_sizes:
            scenelist.append(self[curi : curi + i])
            curi += i
        return scenelist

    def extend(self, N: int):
        """
        Create new BBoxes3D which contains each scene N times.

        Args:
            N: number of new copies of each scene.

        Returns:
            new BBoxes3D object.
        """
        if not isinstance(N, int):
            raise ValueError("N must be an integer.")
        if N <= 0:
            raise ValueError("N must be > 0.")

        new_bounds_list, new_names_list, new_features_list = [], None, None
        for bounds in self.bounds_list():
            new_bounds_list.extend(bounds.clone() for _ in range(N))
        names_list = self.names_list()
        if names_list is not None:
            new_names_list = []
            for names in names_list:
                new_names_list.extend(names.clone() for _ in range(N))
        features_list = self.features_list()
        if features_list is not None:
            new_features_list = []
            for features in features_list:
                new_features_list.extend(features.clone() for _ in range(N))
        return self.__class__(
            bounds=new_bounds_list, names=new_names_list, features=new_features_list
        )


def join_boxes_as_batch(boxes: Sequence[BBoxes3D]) -> BBoxes3D:
    """
    Merge a list of BBoxes3D objects into a single batched BBoxes3D
    object. All BBoxes3D must be on the same device.

    Args:
        batch: List of BBoxes3D objects each with batch dim [b1, b2, ..., bN]
    Returns:
        boxescene: Poinscenes object with all input BBoxes3D collated into
            a single object with batch dim = sum(b1, b2, ..., bN)
    """
    if isinstance(boxes, BBoxes3D) or not isinstance(boxes, Sequence):
        raise ValueError("Wrong first argument to join_boxes_as_batch.")

    device = boxes[0].device
    if not all(p.device == device for p in boxes):
        raise ValueError("BBoxes3D must all be on the same device")

    kwargs = {}
    for field in ("bounds", "names", "features"):
        field_list = [getattr(p, field + "_list")() for p in boxes]
        if None in field_list:
            if field == "bounds":
                raise ValueError("boxes cannot have their bounds set to None!")
            if not all(f is None for f in field_list):
                raise ValueError(
                    f"boxes in the batch have some fields '{field}'"
                    + " defined and some set to None."
                )
            field_list = None
        else:
            field_list = [p for boxes in field_list for p in boxes]
            if field == "features" and any(
                p.shape[1] != field_list[0].shape[1] for p in field_list[1:]
            ):
                raise ValueError("BBoxes3D must have the same number of features")
        kwargs[field] = field_list

    return BBoxes3D(**kwargs)


def join_boxes_as_scene(boxes: Union[BBoxes3D, List[BBoxes3D]]) -> BBoxes3D:
    """
    Joins a batch of point scene in the form of a BBoxes3D object or a list of BBoxes3D
    objects as a single point scene. If the input is a list, the BBoxes3D objects in the
    list must all be on the same device, and they must either all or none have features and
    all or none have names.

    Args:
        BBoxes3D: BBoxes3D object that contains a batch of point scenes, or a list of
                    BBoxes3D objects.

    Returns:
        new BBoxes3D object containing a single point scene
    """
    if isinstance(boxes, list):
        boxes = join_boxes_as_batch(boxes)

    if len(boxes) == 1:
        return boxes
    bounds = boxes.bounds_packed()
    features = boxes.features_packed()
    names = boxes.names_packed()
    boxescene = BBoxes3D(
        bounds=bounds[None],
        features=None if features is None else features[None],
        names=None if names is None else names[None],
    )
    return boxescene


##############################################################
# Box utilities


def get_box_verts_from_bounds(
    bounds: torch.Tensor,  # (N, 3, 2)
    R: Optional[torch.Tensor] = None,
):  # pragma: no cover
    """
    Returns corners of a bounding box.
    If R provided, rotates about the center of the box
                    v4_____________________v5
                    /|                    /|
                   / |                   / |
                  /  |                  /  |
                 /___|_________________/   |
              v0|    |                 |v1 |
                |    |                 |   |
                |    |                 |   |
                |    |                 |   |
                |    |_________________|___|
                |   / v7               |   /v6
                |  /                   |  /
                | /                    | /
                |/_____________________|/
                v3                     v2


    """
    assert bounds.shape[1] == 3 and bounds.shape[2] == 2
    minx, miny, minz = bounds[:, :, 0].unbind(1)
    maxx, maxy, maxz = bounds[:, :, 1].unbind(1)

    v0 = torch.stack([minx, miny, minz], axis=-1)
    v1 = torch.stack([maxx, miny, minz], axis=-1)
    v2 = torch.stack([maxx, maxy, minz], axis=-1)
    v3 = torch.stack([minx, maxy, minz], axis=-1)
    v4 = torch.stack([minx, miny, maxz], axis=-1)
    v5 = torch.stack([maxx, miny, maxz], axis=-1)
    v6 = torch.stack([maxx, maxy, maxz], axis=-1)
    v7 = torch.stack([minx, maxy, maxz], axis=-1)

    # Either one works
    # v0 = torch.stack([minx, miny, minz], axis=-1)
    # v1 = torch.stack([minx, miny, maxz], axis=-1)
    # v2 = torch.stack([maxx, miny, maxz], axis=-1)
    # v3 = torch.stack([maxx, miny, minz], axis=-1)
    # v4 = torch.stack([minx, maxy, minz], axis=-1)
    # v5 = torch.stack([minx, maxy, maxz], axis=-1)
    # v6 = torch.stack([maxx, maxy, maxz], axis=-1)
    # v7 = torch.stack([maxx, maxy, minz], axis=-1)
    verts = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1)

    means = bounds[:, :, 1] - bounds[:, :, 0]
    if R is not None:
        # rotate
        verts = R @ (verts - means) + means
    return verts


def get_box_bounds_from_verts(
    verts: torch.Tensor, R: Optional[torch.Tensor] = None  # (N, 8, 3)
) -> torch.Tensor:
    """
    Returns the bounds of a bbox given vertices. Assumes axis-aligned for now.

    Args:
        verts: [N, 8, 3] corner vertices
        R: (N, 3, 3) Rotation of the bounding box (not yet implemented)

    Returns:
        bounds: [N, 3, 2] mins and maxes along each axis
    """
    assert verts.shape[1] == 8 and verts.shape[2] == 3
    if R is not None:
        # rotate
        # verts = R @ (verts - means) + means
        raise NotImplementedError
    return torch.stack([verts.min(dim=1)[0], verts.max(dim=1)[0]], dim=-1)


def box3d_overlap_from_bounds(bounds1: Tensor, bounds2: Tensor, eps=1e-4):
    """Calculates box overlap

    Args:
        bounds1 (Tensor): [N, 3, 2] mins and maxes along each axis
        bounds2 (Tensor): [M, 3, 2] mins and maxes along each axis

    Returns:
        vol: [N, M] volume of intersection
        iou: [N, M] intersection over union
    """
    corners1 = get_box_verts_from_bounds(bounds1)
    corners2 = get_box_verts_from_bounds(bounds2)
    return box3d_overlap(corners1, corners2, eps=eps)


def box3d_intersection_from_bounds(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-4):
    """
    Calculate Intersection over Union (IoU) scores between a local 3D bounding box and a list of global 3D bounding boxes.

    Args:
        boxes1 (Tuple[np.ndarray, np.ndarray]): Bounding box of a point cloud obtained from the local instance in the current frame.
        boxes2 (List[Tuple[np.ndarray, np.ndarray]]): List of bounding boxes of instances obtained by aggregating point clouds across different views.

    Returns:
        vol_intersection:
        ious (np.ndarray): IoU scores between the boxes1 and each of the boxes2.
        intersection_bounds
    """
    if boxes1.ndim == 2:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.ndim == 2:
        boxes2 = boxes2.unsqueeze(0)
    n_boxes1 = len(boxes1)
    n_boxes2 = len(boxes2)

    assert (
        boxes1.ndim == 3 and boxes1.shape[-1] == 2 and boxes1.shape[-2] == 3
    ), boxes1.shape
    assert (
        boxes2.ndim == 3 and boxes2.shape[-1] == 2 and boxes2.shape[-2] == 3
    ), boxes2.shape
    boxes1_min, boxes1_max = torch.unbind(boxes1, dim=-1)
    boxes2_min, boxes2_max = torch.unbind(boxes2, dim=-1)
    intersection_min = torch.maximum(
        boxes1_min.unsqueeze(1).expand(n_boxes1, n_boxes2, 3),
        boxes2_min.unsqueeze(0).expand(n_boxes1, n_boxes2, 3),
    )
    intersection_max = torch.minimum(
        boxes1_max.unsqueeze(1).expand(n_boxes1, n_boxes2, 3),
        boxes2_max.unsqueeze(0).expand(n_boxes1, n_boxes2, 3),
    )
    zero_iou = (intersection_min > intersection_max).any(dim=-1)

    intersection_bounds = torch.stack([intersection_min, intersection_max], dim=-1)
    intersection = torch.prod(intersection_max - intersection_min, dim=-1)
    union = (
        torch.prod(boxes1_max - boxes1_min, dim=-1).unsqueeze(1)
        + torch.prod(boxes2_max - boxes2_min, dim=-1).unsqueeze(0)
        - intersection
    )

    intersection[zero_iou] = 0.0
    intersection[torch.isnan(intersection)] = 0.0
    ious = intersection / union
    return intersection, ious, intersection_bounds


def box3d_volume_from_bounds(bounds: Tensor):
    assert bounds.shape[-1] == 2 and bounds.shape[-2] == 3, bounds.shape
    if bounds.ndim == 2:
        bounds = bounds.unsqueeze(0)
    mins, maxs = bounds.unbind(dim=-1)
    return torch.prod(maxs - mins, dim=-1)


def box3d_nms(bounding_boxes, confidence_score, iou_threshold=0.3):
    """
    Non-max suppression

    Args:
      bounding_boxes: (N, 8, 3) vertex coordinates. Must be in order specified by box3d_overlap
      confidence_score: (N,)
      iou_threshold: Suppress boxes whose IoU > iou_threshold

    Returns:
      keep, vol, iou, assignments
      keep: indexes into N of which bounding boxes to keep
      vol: (N, N) tensor of the volume of the intersecting convex shapes
      iou: (N, M) tensor of the intersection over union which is
          defined as: `iou = vol / (vol1 + vol2 - vol)`
      assignments: superbox_idx -> List[boxes_to_delete]
    """
    assert len(bounding_boxes) > 0, bounding_boxes.shape

    order = torch.argsort(confidence_score)

    vol, iou = box3d_overlap(bounding_boxes, bounding_boxes)
    keep = []
    assignments = {}
    while len(order) > 0:

        idx = order[-1]  # Highest confidence (S)

        # push S in filtered predictions list
        keep.append(idx)

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # keep the boxes with IoU less than thresh_iou
        # _, iou = box3d_overlap(bounding_boxes[idx].unsqueeze(0), bounding_boxes[order])
        mask = iou[idx, order] < iou_threshold
        assignments[idx] = order[~mask]
        order = order[mask]
    return torch.tensor(keep), vol, iou, assignments


# def get_cuboid_verts_faces(box3d=None, R=None):
#     """
#     Computes vertices and faces from a 3D cuboid representation.
#     Args:
#         bbox3d (flexible): [[X Y Z W H L]]
#         R (flexible): [np.array(3x3)]
#     Returns:
#         verts: the 3D vertices of the cuboid in camera space
#         faces: the vertex indices per face
#     """
#     if box3d is None:
#         box3d = [0, 0, 0, 1, 1, 1]

#     # make sure types are correct
#     box3d = to_float_tensor(box3d)

#     if R is not None:
#         R = to_float_tensor(R)

#     squeeze = len(box3d.shape) == 1

#     if squeeze:
#         box3d = box3d.unsqueeze(0)
#         if R is not None:
#             R = R.unsqueeze(0)

#     n = len(box3d)

#     x3d = box3d[:, 0].unsqueeze(1)
#     y3d = box3d[:, 1].unsqueeze(1)
#     z3d = box3d[:, 2].unsqueeze(1)
#     w3d = box3d[:, 3].unsqueeze(1)
#     h3d = box3d[:, 4].unsqueeze(1)
#     l3d = box3d[:, 5].unsqueeze(1)

#     '''
#                     v4_____________________v5
#                     /|                    /|
#                    / |                   / |
#                   /  |                  /  |
#                  /___|_________________/   |
#               v0|    |                 |v1 |
#                 |    |                 |   |
#                 |    |                 |   |
#                 |    |                 |   |
#                 |    |_________________|___|
#                 |   / v7               |   /v6
#                 |  /                   |  /
#                 | /                    | /
#                 |/_____________________|/
#                 v3                     v2
#     '''

#     verts = to_float_tensor(torch.zeros([n, 3, 8], device=box3d.device))

#     # setup X
#     verts[:, 0, [0, 3, 4, 7]] = -l3d / 2
#     verts[:, 0, [1, 2, 5, 6]] = l3d / 2

#     # setup Y
#     verts[:, 1, [0, 1, 4, 5]] = -h3d / 2
#     verts[:, 1, [2, 3, 6, 7]] = h3d / 2

#     # setup Z
#     verts[:, 2, [0, 1, 2, 3]] = -w3d / 2
#     verts[:, 2, [4, 5, 6, 7]] = w3d / 2

#     if R is not None:

#         # rotate
#         verts = R @ verts

#     # translate
#     verts[:, 0, :] += x3d
#     verts[:, 1, :] += y3d
#     verts[:, 2, :] += z3d

#     verts = verts.transpose(1, 2)

#     faces = torch.tensor([
#         [0, 1, 2], # front TR
#         [2, 3, 0], # front BL

#         [1, 5, 6], # right TR
#         [6, 2, 1], # right BL

#         [4, 0, 3], # left TR
#         [3, 7, 4], # left BL

#         [5, 4, 7], # back TR
#         [7, 6, 5], # back BL

#         [4, 5, 1], # top TR
#         [1, 0, 4], # top BL

#         [3, 2, 6], # bottom TR
#         [6, 7, 3], # bottom BL
#     ]).float().unsqueeze(0).repeat([n, 1, 1])

#     if squeeze:
#         verts = verts.squeeze()
#         faces = faces.squeeze()

#     return verts, faces.to(verts.device)
