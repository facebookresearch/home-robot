import click
import rospy
from slap_manipulation.agents.general_language_agent import GeneralLanguageAgent
from slap_manipulation.env.general_language_env import GeneralLanguageEnv

from home_robot_hw.env.stretch_pick_and_place_env import load_config


@click.command()
@click.option("--test-pick", default=False, is_flag=True)
@click.option("--dry-run", default=False, is_flag=True)
@click.option("--testing", default=False, is_flag=True)
@click.option("--object", default="cup")
@click.option("--task-id", default=0)
def main(task_id, test_pick=False, dry_run=False, testing=False, **kwargs):
    TASK = task_id
    # TODO: add logic here to read task_id automatically for experiments
    rospy.init_node("eval_episode_lang_ovmm")

    config = load_config(
        visualize=True,
        config_path="projects/slap_manipulation/configs/language_agent.yaml",
        **kwargs
    )

    env = GeneralLanguageEnv(
        config=config,
        test_grasping=test_pick,
        dry_run=dry_run,
        segmentation_method="detic",
    )
    agent = GeneralLanguageAgent(cfg=config, debug=True, skip_gaze=True)
    # robot = env.get_robot()

    agent.reset()
    env.reset()

    t = 0
    while not agent.task_is_done():
        t += 1
        print("TIMESTEP = ", t)
        obs = env.get_observation()
        action, info = agent.act(obs, TASK)
        print("ACTION = ", action)
        env.apply_action(action, info=info)


if __name__ == "__main__":
    main()
