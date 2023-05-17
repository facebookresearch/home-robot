import click
import rospy
from slap_manipulation.agents.language_ovmm_agent import LangAgent
from slap_manipulation.env.language_planner_env import LanguagePlannerEnv

from home_robot_hw.env.stretch_pick_and_place_env import (
    REAL_WORLD_CATEGORIES,
    load_config,
)


@click.command()
@click.option("--test-pick", default=False, is_flag=True)
@click.option("--dry-run", default=False, is_flag=True)
@click.option("--testing", default=False, is_flag=True)
@click.option("--object", default="cup")
@click.option("--start-recep", default="table")
@click.option("--goal-recep", default="chair")
def main(
    test_pick=False,
    dry_run=False,
    testing=False,
    object="cup",
    start_recep="table",
    goal_recep="chair",
    **kwargs
):
    if testing:
        TASK = "3"
    else:
        TASK = "bring me a cup from the table"
    rospy.init_node("eval_episode_lang_ovmm")

    config = load_config(
        visualize=True,
        config_path="projects/slap_manipulation/configs/language_agent.yaml",
        **kwargs
    )

    env = LanguagePlannerEnv(
        config=config,
        test_grasping=test_pick,
        dry_run=dry_run,
    )
    agent = LangAgent(cfg=config, debug=True, skip_gaze=True)
    # robot = env.get_robot()

    agent.reset()
    env.reset()

    t = 0
    while not agent.task_is_done():
        t += 1
        print("TIMESTEP = ", t)
        while not agent.task_is_done():
            obs = env.get_observation()
            action, info = agent.act(obs, TASK)
            print("ACTION = ", action)
            env.apply_action(action, info=info)


if __name__ == "__main__":
    main()
