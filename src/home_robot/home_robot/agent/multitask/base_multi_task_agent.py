# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Sequence

from home_robot.mapping.instance import Instance


class BaseMultiTaskAgent(ABC):
    """
    Basic interface that a VLM needs to implement
    """

    @abstractmethod
    def set_vocabulary(self, vocab: Mapping[str, int]):
        pass

    @abstractmethod
    def build_scene_and_get_instances_for_queries(
        self, scene_obs: Mapping[str, Any], queries: Sequence[str], reset: bool = True
    ) -> Dict[str, List[Instance]]:
        pass
