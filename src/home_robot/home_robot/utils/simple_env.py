# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from home_robot.core.abstract_env import Env
from home_robot.core.interfaces import Action, Observations


class SimpleEnv(Env):
    """Simple 2D environment for testing."""

    def __init__(self, seed: int = 0):
        self.reset(seed)

    @abstractmethod
    def reset(self, seed: int = 0):
        self.seed = seed
        np.random.seed(seed)
        # TODO: regenerate start, goal and obstacles

    @abstractmethod
    def apply_action(
        self,
        action: Action,
        info: Optional[Dict[str, Any]] = None,
        prev_obs: Optional[Observations] = None,
    ):
        pass

    @abstractmethod
    def get_observation(self) -> Observations:
        pass

    @property
    @abstractmethod
    def episode_over(self) -> bool:
        pass

    @abstractmethod
    def get_episode_metrics(self) -> Dict:
        pass
