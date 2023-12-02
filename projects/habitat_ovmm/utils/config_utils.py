# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Optional, Tuple

from habitat_baselines.config.default import _BASELINES_CFG_DIR
from habitat_baselines.config.default import get_config as _get_habitat_config
from omegaconf import DictConfig, OmegaConf


def split_config_overrides(overrides):
    agent_overrides = {}
    env_overrides = {}
    habitat_overrides = {}
    return agent_overrides, env_overrides, overrides


def get_habitat_config(
    config_path: str,
    overrides: Optional[list] = None,
    configs_dir: str = _BASELINES_CFG_DIR,
) -> Tuple[DictConfig, str]:
    """Returns habitat config object composed of configs from yaml file (config_path) and overrides."""
    config = _get_habitat_config(
        config_path, overrides=overrides, configs_dir=configs_dir
    )
    return config, ""


def get_omega_config(config_path: str) -> DictConfig:
    """Returns the baseline configuration."""
    config = OmegaConf.load(config_path)
    OmegaConf.set_readonly(config, True)
    return config


def create_env_config(
    habitat_config: DictConfig, env_config: DictConfig, evaluation_type: str = "local"
) -> DictConfig:
    """
    Merges habitat and env configurations.

    Adjusts the configuration based on the provided arguments:
    1. Removes third person sensors to improve speed if visualization is not required.
    2. Processes the episode range if specified and updates the EXP_NAME accordingly.
    3. Adds paths to test objects in case of remote evaluation

    :param habitat_config: habitat configuration.
    :param env_config: baseline configuration.
    :param evaluation_type: one of ["local", "remote", "local_vectorized"]
    :return: merged env configuration
    """

    env_config = DictConfig({**habitat_config, **env_config})
    record_videos = env_config.get("EVAL_VECTORIZED", {}).get("record_videos", False)
    visualize = env_config.VISUALIZE or env_config.PRINT_IMAGES or record_videos
    if not visualize:
        if "third_rgb" in env_config.habitat.gym.obs_keys:
            env_config.habitat.gym.obs_keys.remove("third_rgb")
        if (
            "third_rgb_sensor"
            in env_config.habitat.simulator.agents.main_agent.sim_sensors
        ):
            env_config.habitat.simulator.agents.main_agent.sim_sensors.pop(
                "third_rgb_sensor"
            )
    if (
        getattr(env_config.ENVIRONMENT, "evaluate_instance_tracking", False)
        or env_config.GROUND_TRUTH_SEMANTICS
    ) and "head_panoptic" not in env_config.habitat.gym.obs_keys:
        env_config.habitat.gym.obs_keys.append("head_panoptic")

    if env_config.NO_GPU:
        env_config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1

    episode_ids_range = env_config.habitat.dataset.episode_indices_range
    if episode_ids_range is not None:
        env_config.EXP_NAME = os.path.join(
            env_config.EXP_NAME, f"{episode_ids_range[0]}_{episode_ids_range[1]}"
        )

    if evaluation_type == "remote":
        # in case of remote evaluation, add test object config paths
        train_val_object_config_paths = (
            env_config.habitat.simulator.additional_object_paths
        )
        test_object_config_paths = [
            path.replace("train_val", "test") for path in train_val_object_config_paths
        ]
        env_config.habitat.simulator.additional_object_paths = (
            train_val_object_config_paths + test_object_config_paths
        )
    OmegaConf.set_readonly(env_config, True)
    return env_config


def create_agent_config(
    env_config: DictConfig, baseline_config: DictConfig
) -> DictConfig:
    """
    Merges habitat and baseline configurations.

    :param env_config: env configuration.
    :param baseline_config: baseline configuration.
    :return: merged agent configuration
    """
    agent_config = DictConfig({**env_config, "AGENT": baseline_config})
    OmegaConf.set_readonly(agent_config, True)
    return agent_config
