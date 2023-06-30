# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Tuple

from habitat_baselines.config.default import _BASELINES_CFG_DIR
from habitat_baselines.config.default import get_config as get_habitat_config
from argparse import Namespace
from omegaconf import DictConfig, OmegaConf


def get_config(
    path: str, opts: Optional[list] = None, configs_dir: str = _BASELINES_CFG_DIR
) -> Tuple[DictConfig, str]:
    config = get_habitat_config(path, overrides=opts, configs_dir=configs_dir)
    return config, ""

def process_and_adjust_config(args: Namespace) -> DictConfig:
    """
    Process and adjust the configuration based on the provided arguments.

    This function takes a Namespace object containing parsed command-line arguments and performs the following steps:
    1. Merges the habitat and baseline configurations.
    2. Removes third person sensors to improve speed if visualization is not required.
    3. Processes the episode range if specified and updates the EXP_NAME accordingly.

    Args:
        args (Namespace): The parsed command-line arguments.

    Returns:
        DictConfig: The processed and adjusted configuration.
    """

    config, _ = get_config(args.habitat_config_path, opts=args.opts)
    baseline_config = OmegaConf.load(args.baseline_config_path)
    config = DictConfig({**config, **baseline_config})
    visualize = config.VISUALIZE or config.PRINT_IMAGES

    if not visualize:
        # TODO: not seeing any speed improvements when removing these sensors
        if "robot_third_rgb" in config.habitat.gym.obs_keys:
            config.habitat.gym.obs_keys.remove("robot_third_rgb")
        if "third_rgb_sensor" in config.habitat.simulator.agents.main_agent.sim_sensors:
            config.habitat.simulator.agents.main_agent.sim_sensors.pop(
                "third_rgb_sensor", None
            )

    episode_ids_range = config.habitat.dataset.episode_indices_range
    if episode_ids_range is not None:
        config.EXP_NAME = os.path.join(
            config.EXP_NAME, f"{episode_ids_range[0]}_{episode_ids_range[1]}"
        )
    OmegaConf.set_readonly(config, True)
    return config
