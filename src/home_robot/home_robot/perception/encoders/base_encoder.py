from typing import Union

from numpy import ndarray
from torch import Tensor


class BaseImageTextEncoder:
    def encode_image(self, image: Union[ndarray, Tensor]):
        raise NotImplementedError

    def encode_text(self, text: str):
        raise NotImplementedError
