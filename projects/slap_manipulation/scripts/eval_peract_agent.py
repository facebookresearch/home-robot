# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""Main script for running per-skill manipulation using PerActAgent. Takes in
task-id just as an identifier for which skill to run (read from config)"""

import click
import numpy as np
import rospy
import trimesh.transformations as tra
from slap_manipulation.agents.peract_agent import PeractAgent
from slap_manipulation.env.general_language_env import GeneralLanguageEnv

from home_robot.core.interfaces import ContinuousEndEffectorAction
from home_robot_hw.utils.config import load_slap_config


@click.command()
@click.option("--test-pick", default=False, is_flag=True)
@click.option("--dry-run", default=False, is_flag=True)
# @click.option("--testing/--no-testing", default=False, is_flag=True)
@click.option("--object", default="cup")
@click.option("--task-id", default=0)
@click.option(
    "--cat-map-file",
    default="projects/stretch_ovmm/configs/example_cat_map.json",
)
def main(
    task_id: int,
    cat_map_file: str,
    test_pick: bool = False,
    dry_run: bool = False,
    **kwargs
):
    config = load_slap_config(
        visualize=True,
        config_path="projects/slap_manipulation/configs/language_agent.yaml",
        **kwargs
    )
    rospy.init_node("eval_peract")
    agent = PeractAgent(config, task_id=task_id)
    agent.load_models()

    env = GeneralLanguageEnv(
        config=config,
        test_grasping=test_pick,
        dry_run=dry_run,
        # segmentation_method="detic",
        cat_map_file=cat_map_file,
    )

    env.reset(None, None, None, set_goal=False, open_gripper=False)
    agent.reset()

    goal_info = agent.get_goal_info()
    env._set_goal(goal_info)
    if not env.robot.in_manipulation_mode():
        env._switch_to_manip_mode(grasp_only=True, pre_demo_pose=True)
    res = input("Press Y/y to close the gripper")
    if res == "y" or res == "Y":
        env._handle_gripper_action(1)
    else:
        env._handle_gripper_action(-1)
    rospy.sleep(3.0)
    for i in range(goal_info["num-actions"]):
        obs = env.get_observation()
        obs.task_observations.update(goal_info)
        camera_pose = obs.task_observations["base_camera_pose"]
        obs.xyz = tra.transform_points(obs.xyz.reshape(-1, 3), camera_pose)
        result, info = agent.predict(obs)
        if result is not None:
            action = ContinuousEndEffectorAction(
                np.expand_dims(result["predicted_pos"], 0),
                np.expand_dims(result["predicted_quat"], 0),
                np.expand_dims(result["gripper_act"], 0),
            )
        else:
            pos, quat = env.robot.manip.get_ee_pose()
            # check if pos and quat are not numpy ndarrays
            if not isinstance(pos, np.ndarray):
                pos = np.array(pos)
            if not isinstance(quat, np.ndarray):
                quat = np.array(quat)
            action = ContinuousEndEffectorAction(
                pos.reshape(1, 3), quat.reshape(1, 4), np.random.rand(1, 1)
            )
        env.apply_action(action, info=info)


if __name__ == "__main__":
    main()
