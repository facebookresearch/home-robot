import json
from typing import Optional, Tuple

import habitat.config.default
import yaml
from habitat_baselines.config.default import _BASELINES_CFG_DIR
from habitat_baselines.config.default import get_config as get_habitat_config
from omegaconf import DictConfig, OmegaConf


def get_config(
    habitat_config_path: str,
    baseline_config_path: str,
    opts: Optional[list] = None,
    configs_dir: str = _BASELINES_CFG_DIR,
) -> Tuple[DictConfig, str]:
    """Get configuration and ensure consistency between configurations
    inherited from the task and defaults and our code's configuration.

    Arguments:
        path: path to our code's config
        opts: command line arguments overriding the config
    """
    habitat_config = get_habitat_config(
        habitat_config_path, overrides=opts, configs_dir=configs_dir
    )
    baseline_config = OmegaConf.load(baseline_config_path)
    config = DictConfig({**habitat_config, **baseline_config})

    # Ensure consistency between configurations inherited from the task
    # # and defaults and our code's configuration

    sim_sensors = config.habitat.simulator.agents.main_agent.sim_sensors

    rgb_sensor = sim_sensors.rgb_sensor
    depth_sensor = sim_sensors.depth_sensor
    semantic_sensor = sim_sensors.semantic_sensor
    frame_height = config.ENVIRONMENT.frame_height
    assert rgb_sensor.height == depth_sensor.height
    if semantic_sensor:
        assert rgb_sensor.height == semantic_sensor.height
    assert rgb_sensor.height >= frame_height and rgb_sensor.height % frame_height == 0

    frame_width = config.ENVIRONMENT.frame_width
    assert rgb_sensor.width == depth_sensor.width
    if semantic_sensor:
        assert rgb_sensor.width == semantic_sensor.width
    assert rgb_sensor.width >= frame_width and rgb_sensor.width % frame_width == 0

    camera_height = config.ENVIRONMENT.camera_height
    assert camera_height == rgb_sensor.position[1]
    assert camera_height == depth_sensor.position[1]
    if semantic_sensor:
        assert camera_height == semantic_sensor.position[1]

    hfov = config.ENVIRONMENT.hfov
    assert hfov == rgb_sensor.hfov
    assert hfov == depth_sensor.hfov
    if semantic_sensor:
        assert hfov == semantic_sensor.hfov

    assert config.ENVIRONMENT.min_depth == depth_sensor.min_depth
    assert config.ENVIRONMENT.max_depth == depth_sensor.max_depth
    assert config.ENVIRONMENT.turn_angle == config.habitat.simulator.turn_angle

    return config
