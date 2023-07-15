# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import habitat_extensions  # noqa
from config_utils import get_config
from habitat.core.env import Env

from home_robot.agent.imagenav_agent.imagenav_agent import ImageNavAgent
from home_robot.agent.imagenav_agent.visualizer import record_video
from home_robot_sim.env.habitat_imagenav_env.habitat_imagenav_env import (
    HabitatImageNavEnv,
)

if __name__ == "__main__":
    config, config_str = get_config("configs/instance_imagenav_hm3d.yaml")
    print("Config:\n", config_str, "\n", "-" * 100)

    env = HabitatImageNavEnv(Env(config=config.habitat), config=config)
    agent = ImageNavAgent(config=config)

    env.reset()
    agent.reset()

    t = 0
    while not env.episode_over:
        t += 1
        print("STEP =", t)
        obs = env.get_observation()
        action = agent.act(obs)
        env.apply_action(action)

    metrics = {
        k: v
        for k, v in env.get_episode_metrics().items()
        if k not in ["top_down_map", "collisions"]
    }
    print(metrics)

    record_video(
        target_dir=f"{config.dump_location}/videos/{config.exp_name}",
        image_dir=f"{config.dump_location}/images/{config.exp_name}",
    )
