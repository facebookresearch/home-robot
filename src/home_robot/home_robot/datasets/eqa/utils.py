# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/concept-graphs/concept-graphs/blob/25c54e3a022ed8d01e1f7eca0f18a6935f3b863a/LICENSE#L1C1-L21C10
# MIT License

# Copyright (c) 2023 concept-graphs

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import time
from typing import Union

import numpy as np

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
from kornia.geometry.linalg import compose_transformations, inverse_transformation


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def to_tensor(numpy_array, device=None):
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)


def to_scalar(d: Union[np.ndarray, torch.Tensor, float]) -> Union[int, float]:
    """
    Convert the d to a scalar
    """
    if isinstance(d, float):
        return d

    elif "numpy" in str(type(d)):
        assert d.size == 1
        return d.item()

    elif isinstance(d, torch.Tensor):
        assert d.numel() == 1
        return d.item()

    else:
        raise TypeError(f"Invalid type for conversion: {type(d)}")


def relative_transformation(
    trans_01: torch.Tensor, trans_02: torch.Tensor, orthogonal_rotations: bool = False
) -> torch.Tensor:
    r"""Function that computes the relative homogenous transformation from a
    reference transformation :math:`T_1^{0} = \begin{bmatrix} R_1 & t_1 \\
    \mathbf{0} & 1 \end{bmatrix}` to destination :math:`T_2^{0} =
    \begin{bmatrix} R_2 & t_2 \\ \mathbf{0} & 1 \end{bmatrix}`.

    .. note:: Works with imperfect (non-orthogonal) rotation matrices as well.

    The relative transformation is computed as follows:

    .. math::

        T_1^{2} = (T_0^{1})^{-1} \cdot T_0^{2}

    Arguments:
        trans_01 (torch.Tensor): reference transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.
        trans_02 (torch.Tensor): destination transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.
        orthogonal_rotations (bool): If True, will invert `trans_01` assuming `trans_01[:, :3, :3]` are
            orthogonal rotation matrices (more efficient). Default: False

    Shape:
        - Output: :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        torch.Tensor: the relative transformation between the transformations.

    Example::
        >>> trans_01 = torch.eye(4)  # 4x4
        >>> trans_02 = torch.eye(4)  # 4x4
        >>> trans_12 = gradslam.geometry.geometryutils.relative_transformation(trans_01, trans_02)  # 4x4
    """
    if not torch.is_tensor(trans_01):
        raise TypeError(
            "Input trans_01 type is not a torch.Tensor. Got {}".format(type(trans_01))
        )
    if not torch.is_tensor(trans_02):
        raise TypeError(
            "Input trans_02 type is not a torch.Tensor. Got {}".format(type(trans_02))
        )
    if not trans_01.dim() in (2, 3) and trans_01.shape[-2:] == (4, 4):
        raise ValueError(
            "Input must be a of the shape Nx4x4 or 4x4."
            " Got {}".format(trans_01.shape)
        )
    if not trans_02.dim() in (2, 3) and trans_02.shape[-2:] == (4, 4):
        raise ValueError(
            "Input must be a of the shape Nx4x4 or 4x4."
            " Got {}".format(trans_02.shape)
        )
    if not trans_01.dim() == trans_02.dim():
        raise ValueError(
            "Input number of dims must match. Got {} and {}".format(
                trans_01.dim(), trans_02.dim()
            )
        )
    trans_10: torch.Tensor = (
        inverse_transformation(trans_01)
        if orthogonal_rotations
        else torch.inverse(trans_01)
    )
    trans_12: torch.Tensor = compose_transformations(trans_10, trans_02)
    return trans_12
