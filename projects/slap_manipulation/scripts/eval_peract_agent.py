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
    "--cat-map-file", default="projects/stretch_ovmm/configs/example_cat_map.json"
)
def main(task_id, cat_map_file, test_pick=False, dry_run=False, **kwargs):
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

    env.reset()
    agent.reset()

    goal_info = agent.get_goal_info()
    env._set_goal(goal_info)
    if not env.robot.in_manipulation_mode():
        env._switch_to_manip_mode(grasp_only=True, pre_demo_pose=True)
    res = input("Press Y/y to close the gripper")
    if res == "y" or res == "Y":
        env._handle_gripper_action(1)
        # env.robot.manip.close_gripper()
    else:
        env._handle_gripper_action(-1)
        # env.robot.manip.open_gripper()
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
            action = ContinuousEndEffectorAction(
                np.random.rand(1, 3), np.random.rand(1, 4), np.random.rand(1, 1)
            )
        env.apply_action(action, info=info)


if __name__ == "__main__":
    main()
