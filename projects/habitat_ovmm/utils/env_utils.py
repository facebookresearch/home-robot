# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import TYPE_CHECKING

from habitat import make_dataset
from habitat.core.environments import get_env_class
from habitat.utils.gym_definitions import _get_env_name

from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import (
    HabitatOpenVocabManipEnv,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


def create_ovmm_env_fn(config: "DictConfig") -> HabitatOpenVocabManipEnv:
    """
    Creates an environment for the OVMM task.

    Creates habitat environment from config and wraps it into HabitatOpenVocabManipEnv.

    :param config: configuration for the environment.
    :return: environment instance.
    """
    habitat_config = config.habitat
    dataset = make_dataset(habitat_config.dataset.type, config=habitat_config.dataset)
    env_class_name = _get_env_name(config)
    env_class = get_env_class(env_class_name)
    habitat_env = env_class(config=habitat_config, dataset=dataset)
    habitat_env.seed(habitat_config.seed)
    env = HabitatOpenVocabManipEnv(habitat_env, config, dataset=dataset)
    return env
