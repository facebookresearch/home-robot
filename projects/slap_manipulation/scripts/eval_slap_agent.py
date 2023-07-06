# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import click
import numpy as np
import rospy
import trimesh.transformations as tra
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
    rospy.init_node("eval_slap")
    agent = SLAPAgent(config, task_id=task_id)
    agent.load_models()

    env = GeneralLanguageEnv(
        config=config,
        segmentation_method="detic",
    )

    env.reset()
    agent.reset()

    goal_info = agent.get_goal_info()
    env.set_goal(goal_info)
    if not env.robot.in_manipulation_mode():
        env._switch_to_manip_mode(grasp_only=True, pre_demo_pose=True)
        env.robot.manip.open_gripper()
    res = input("Press Y/y to close the gripper")
    if res == "y" or res == "Y":
        env.robot.manip.close_gripper()
        rospy.sleep(7.0)
    obs = env.get_observation()
    obs.task_observations.update(goal_info)
    camera_pose = obs.task_observations["base_camera_pose"]
    obs.xyz = tra.transform_points(obs.xyz.reshape(-1, 3), camera_pose)
    result, info = agent.predict(
        obs, visualize=config.SLAP.visualize, save_logs=config.SLAP.save_logs
    )
    if result is not None:
        action = ContinuousEndEffectorAction(
            result[:, :3], result[:, 3:7], np.expand_dims(result[:, 7], -1)
        )
    else:
        action = ContinuousEndEffectorAction(
            np.random.rand(1, 3), np.random.rand(1, 4), np.random.rand(1, 1)
        )
    env.apply_action(action, info=info)


if __name__ == "__main__":
    main()
