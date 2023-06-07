import click
import numpy as np
import rospy
from slap_manipulation.agents.slap_agent import SLAPAgent
from slap_manipulation.env.general_language_env import GeneralLanguageEnv

from home_robot.core.interfaces import ContinuousEndEffectorAction
from home_robot_hw.env.stretch_pick_and_place_env import load_config


@click.command()
@click.option("--task-id", default=-1)
def main(task_id, **kwargs):
    config = load_config(
        visualize=True,
        config_path="projects/slap_manipulation/configs/language_agent.yaml",
        **kwargs
    )
    if config.model_type == "slap":
        rospy.init_node("eval_slap")
        agent = SLAPAgent(config, task_id)

    env = GeneralLanguageEnv(
        config=config,
        segmentation_method="detic",
    )

    env.reset()
    agent.reset()

    goal_info = agent.get_goal_info()
    env.set_goal(goal_info)
    obs = env.get_observation()
    obs.task_observations.update(goal_info)
    result, info = agent.predict(obs)
    if result is not None:
        action = ContinuousEndEffectorAction(
            result[:, :3], result[:, 3:7], np.expand_dims(result[:, 7], -1)
        )
    else:
        action = ContinuousEndEffectorAction(
            np.random.rand(1, 3), np.random.rand(1, 4), np.random.rand(1, 1)
        )
    env.apply_action(action, info=info)
