# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import habitat
from habitat.core.simulator import Observations

import home_robot


class HabitatAgent(habitat.Agent, ABC):
    def __init__(self, home_robot_agent: home_robot.core.Agent):
        self.home_robot_agent = home_robot_agent

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def act(self, obs: Observations) -> Union[int, str, Dict[str, Any]]:
        pass
