# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Optional, Tuple

from habitat_baselines.config.default import _BASELINES_CFG_DIR
from habitat_baselines.config.default import get_config as _get_habitat_config
from omegaconf import DictConfig, OmegaConf


def get_habitat_config(
    config_path: str,
    overrides: Optional[list] = None,
    configs_dir: str = _BASELINES_CFG_DIR,
) -> Tuple[DictConfig, str]:
    config = _get_habitat_config(
        config_path, overrides=overrides, configs_dir=configs_dir
    )
    return config, ""


def get_ovmm_baseline_config(baseline_config_path: str) -> DictConfig:
    config = OmegaConf.load(baseline_config_path)
    OmegaConf.set_readonly(config, True)
    return config


def merge_configs(habitat_config, baseline_config):
    config = DictConfig({**habitat_config, **baseline_config})

    visualize = config.VISUALIZE or config.PRINT_IMAGES
    if not visualize:
        if "robot_third_rgb" in config.habitat.gym.obs_keys:
            config.habitat.gym.obs_keys.remove("robot_third_rgb")
        if "third_rgb_sensor" in config.habitat.simulator.agents.main_agent.sim_sensors:
            config.habitat.simulator.agents.main_agent.sim_sensors.pop(
                "third_rgb_sensor"
            )

    episode_ids_range = config.habitat.dataset.episode_indices_range
    if episode_ids_range is not None:
        config.EXP_NAME = os.path.join(
            config.EXP_NAME, f"{episode_ids_range[0]}_{episode_ids_range[1]}"
        )

    OmegaConf.set_readonly(config, True)
    return config
