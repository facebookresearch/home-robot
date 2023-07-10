# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Tuple

from habitat import get_config as get_habitat_config
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from omegaconf import DictConfig, OmegaConf


def get_config(path: str, opts: Optional[list] = None) -> Tuple[DictConfig, str]:
    """Get configuration and ensure consistency between configurations inherited from
    the task and defaults and our code's configuration. Assume Stretch embodiment.

    Arguments:
        path: path to our code's config
        opts: command line arguments overriding the config
    """
    config = get_habitat_config(path, opts)

    with read_write(config):
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = config.simulator_gpu_id
        del config.habitat.task.measurements.distance_to_goal_reward

        agent_height, agent_radius = (1.41, 0.17)
        cam_height, cam_width, cam_hfov = (640, 480, 42)
        cam_position = [0, 1.31, 0]

        config.habitat.simulator.agents.main_agent.height = agent_height
        config.habitat.simulator.agents.main_agent.radius = agent_radius

        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = (
            cam_height
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = (
            cam_width
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = (
            cam_position
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = (
            cam_hfov
        )

        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = (
            cam_height
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = (
            cam_width
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = (
            cam_position
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = (
            cam_hfov
        )

        if config.generate_videos:
            config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        type="MyTopDownMap",
                        max_episode_steps=config.habitat.environment.max_episode_steps,
                        map_padding=3,
                        map_resolution=512,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True, visibility_dist=5.0, fov=cam_hfov
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

    return config, OmegaConf.to_yaml(config)
