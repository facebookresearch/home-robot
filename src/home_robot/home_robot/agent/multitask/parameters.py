# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, Optional, Tuple

from home_robot.utils.config import get_config


class Parameters(object):
    """Wrapper class for handling parameters safely"""

    def __init__(self, **kwargs):
        self.data = kwargs

    def get(self, key: str, default: Any = None):
        """Safe wrapper to dictionary, with defaults"""
        if "/" in key:
            keys = key.split("/")
            data = self.data[keys[0]]
            key = keys[-1]
            if len(keys) > 2:
                raise NotImplementedError(
                    f"we dont yet support nested parameters to this depth: {len(keys)}"
                )
        else:
            data = self.data
        if default is not None and key not in data:
            return default
        return data[key]

    def __getitem__(self, key: str) -> Any:
        """Just a wrapper to the dictionary"""
        return self.data[key]

    def __setitem__(self, key: str, value: Any):
        """Just a wrapper to the dictionary"""
        self.data[key] = value

    def __str__(self):
        result = ""
        for i, (key, value) in enumerate(self.data.items()):
            if i > 0:
                result += "\n"
            result += f"{key}: {value}"
        return result

    def get_task_goals(parameters) -> Tuple[str, str]:
        """Helper for extracting task information: returns the two different task goals for a very simple OVMM-style (pick, place) task."""
        if "object_to_find" in parameters.data:
            object_to_find = parameters["object_to_find"]
            if len(object_to_find) == 0:
                object_to_find = None
        else:
            object_to_find = None
        if "location_to_place" in parameters.data:
            location_to_place = parameters["location_to_place"]
            if len(location_to_place) == 0:
                location_to_place = None
        else:
            location_to_place = None
        return object_to_find, location_to_place

    @staticmethod
    def load(path: str):
        """Load it from the path"""
        return Parameters(**get_config(path)[0])

    @property
    def guarantee_instance_is_reachable(self) -> bool:
        """Should we use planning to check if we can get to things? Defautls to False"""
        if "guarantee_instance_is_reachable" in self.data:
            return self.data["guarantee_instance_is_reachable"]
        else:
            return False


def get_parameters(path: str):
    """Load parameters from a path"""
    return Parameters.load(path)
