# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from typing import Any

from .interfaces import Observations


class PerceptionModule(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, obs: Observations) -> Any:
        pass
