# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from numpy import ndarray
from torch import Tensor


class BaseImageTextEncoder:
    """
    Encodes images, encodes text, and allows comparisons between the two encoding.
    """

    def encode_image(self, image: Union[ndarray, Tensor]):
        raise NotImplementedError

    def encode_text(self, text: str):
        raise NotImplementedError
