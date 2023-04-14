from typing import Optional, Tuple

from habitat import get_config as get_habitat_config
from omegaconf import DictConfig, OmegaConf


def get_config(path: str, opts: Optional[list] = None) -> Tuple[DictConfig, str]:
    """Get configuration and ensure consistency between configurations inherited from
    the task defaults and our code's configuration. Assume Stretch embodiment.

    Arguments:
        path: path to our code's config
        opts: command line arguments overriding the config
    """
    config = get_habitat_config(path, opts)

    config.defrost()
    config.habitat.simulator.habitat_sim_v0.gpu_device_id = config.simulator_gpu_id
    del config.habitat.task.measurements.distance_to_goal_reward

    agent_height, agent_radius = (1.41, 0.17)
    cam_height, cam_width, cam_hfov = (640, 480, 42)
    cam_position = [0, 1.31, 0]

    config.habitat.simulator.agents.main_agent.height = agent_height
    config.habitat.simulator.agents.main_agent.radius = agent_radius

    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = cam_height
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = cam_width
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = cam_position
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = cam_hfov

    config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = cam_height
    config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = cam_width
    config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = cam_position
    config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = cam_hfov

    config.freeze()

    return config, OmegaConf.to_yaml(config)
