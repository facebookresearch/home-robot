import sys
from pathlib import Path

# TODO Install home_robot, home_robot_sim and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot_sim"),
)

from spot_wrapper.spot import Spot

from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
from home_robot.utils.config import get_config
from home_robot_hw.env.spot_teleop_env import SpotTeleopEnv


def main():
    config_path = "projects/spot_teleop/configs/config.yaml"
    config, config_str = get_config(config_path)

    env = SpotTeleopEnv(spot)
    env.env.power_robot()
    env.env.initialize_arm()
    env.reset()

    agent = ObjectNavAgent(config=config)
    env = SpotTeleopEnv(config=config)

    agent.reset()
    env.reset()

    t = 0
    while not env.episode_over:
        t += 1
        obs = env.get_observation()
        action, info = agent.act(obs)
        print("STEP =", t)
        env.apply_action(action, info=info)

    print(env.get_episode_metrics())


if __name__ == "__main__":
    spot = Spot("RealNavEnv")
    with spot.get_lease(hijack=True):
        main(spot)
