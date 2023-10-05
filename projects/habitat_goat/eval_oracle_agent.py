import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from pprint import pprint

import numpy as np
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from tqdm import tqdm

# TODO Install home_robot, home_robot_sim and remove this
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot_sim"),
)

from config_utils import get_config
from habitat.core.env import Env
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

from home_robot.agent.goat_agent.goat_agent import GoatAgent
from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot_sim.env.habitat_goat_env.habitat_goat_env import HabitatGoatEnv


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["goat_top_down_map"], output_size
    )


def get_reachable_viewpoint(sim, agent_pos, current_goals, follower):
    current_goal_reachable_viewpoint = None
    for goal in current_goals:
        goal_vp = goal.view_points[0].agent_state.position
        gd = sim.geodesic_distance(goal_vp, agent_pos)
        if gd != math.inf:
            if follower.get_next_action(goal_vp) != 0:
                current_goal_reachable_viewpoint = goal_vp
                break
    return current_goal_reachable_viewpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="goat/modular_goat_hm3d.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/habitat_goat/configs/agent/hm3d_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    print("Arguments:")
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    print("-" * 100)

    config = get_config(args.habitat_config_path, args.baseline_config_path)
    config.PRINT_IMAGES = 0

    habitat_env = Env(config)
    env = HabitatGoatEnv(habitat_env, config=config)
    agent = GoatAgent(config=config)

    results_dir = os.path.join(config.DUMP_LOCATION, "results", config.EXP_NAME)
    os.makedirs(results_dir, exist_ok=True)

    videos_dir = os.path.join(config.DUMP_LOCATION, "videos", config.EXP_NAME)
    os.makedirs(videos_dir, exist_ok=True)

    metrics = {}

    for i in range(len(env.habitat_env.episodes)):
        env.reset()
        agent.reset()

        goal_radius = config.habitat.simulator.forward_step_size

        follower = ShortestPathFollower(env.habitat_env.sim, goal_radius, False)

        t = 0

        scene_id = env.habitat_env.current_episode.scene_id.split("/")[-1].split(".")[0]
        episode = env.habitat_env.current_episode
        episode_id = episode.episode_id

        pbar = tqdm(total=config.AGENT.max_steps)

        all_subtask_metrics = []
        images = []

        current_task_idx = 0

        agent_pos = env.habitat_env.sim.get_agent_state().position
        current_goal = env.habitat_env.current_episode.goals[
            env.habitat_env.task.current_task_idx
        ]
        current_goal_reachable_viewpoint = get_reachable_viewpoint(
            env.habitat_env.sim, agent_pos, current_goal, follower
        )

        if current_goal_reachable_viewpoint is None:
            print("Goal viewpoint not reachable")
            warnings.warn(f"Goal viewpoint not reachable for episode ID {episode_id}.")
            continue

        while (
            not env.habitat_env.episode_over
            and current_goal_reachable_viewpoint is not None
        ):
            t += 1

            obs = env.get_observation()
            if t == 1:
                obs_tasks = []
                for task in obs.task_observations["tasks"]:
                    obs_task = {}
                    for key, value in task.items():
                        if key == "image":
                            continue
                        obs_task[key] = value
                    obs_tasks.append(obs_task)

            best_action = follower.get_next_action(current_goal_reachable_viewpoint)
            if best_action is None:
                break

            env.apply_action(best_action)

            if config.PRINT_IMAGES:
                im = obs.rgb
                ep_metrics = env.get_episode_metrics()
                top_down_map = draw_top_down_map(ep_metrics, im.shape[0])
                output_im = np.concatenate((im, top_down_map), axis=1)
                images.append(output_im)

            pbar.set_description(
                f"Action: {str(best_action).split('.')[-1]} (sub-task: {env.habitat_env.task.current_task_idx})"
            )
            pbar.update(1)
            if best_action == 0 or env.habitat_env.episode_over:
                print("Stopping")
                ep_metrics = env.get_episode_metrics()
                ep_metrics.pop("goat_top_down_map", None)
                print(ep_metrics)
                all_subtask_metrics.append(ep_metrics)

                if config.PRINT_IMAGES:
                    images_to_video(
                        images, videos_dir, f"{episode_id}_{current_task_idx}"
                    )
                    images = []

                if not env.episode_over:
                    current_task_idx = env.habitat_env.task.current_task_idx
                    current_goal = env.habitat_env.current_episode.goals[
                        env.habitat_env.task.current_task_idx
                    ]
                    current_goal_reachable_viewpoint = get_reachable_viewpoint(
                        env.habitat_env.sim, agent_pos, current_goal, follower
                    )
                    if current_goal_reachable_viewpoint is None:
                        warnings.warn(
                            f"Goal viewpoint not reachable for episode ID {episode_id}."
                        )
                        continue
                    pbar.reset()

        pbar.close()

        ep_metrics = env.get_episode_metrics()
        scene_ep_id = f"{scene_id}_{episode_id}"
        metrics[scene_ep_id] = {"metrics": all_subtask_metrics}
        metrics[scene_ep_id]["total_num_steps"] = t
        metrics[scene_ep_id]["sub_task_timesteps"] = agent.sub_task_timesteps[0]
        metrics[scene_ep_id]["tasks"] = obs_tasks

        for metric in list(metrics.values())[0]["metrics"][0].keys():
            metrics[scene_ep_id][f"{metric}_mean"] = np.round(
                np.nanmean(
                    np.array([y[metric] for y in metrics[scene_ep_id]["metrics"]])
                ),
                4,
            )
            metrics[scene_ep_id][f"{metric}_median"] = np.round(
                np.nanmedian(
                    np.array([y[metric] for y in metrics[scene_ep_id]["metrics"]])
                ),
                4,
            )

        with open(os.path.join(results_dir, "per_episode_metrics.json"), "w") as fp:
            json.dump(metrics, fp, indent=4)

        stats = {}

        for metric in list(metrics.values())[0]["metrics"][0].keys():
            stats[f"{metric}_mean"] = np.round(
                np.nanmean(
                    np.array(
                        [
                            y[metric]
                            for scene_ep_id in metrics.keys()
                            for y in metrics[scene_ep_id]["metrics"]
                        ]
                    )
                ),
                4,
            )
            stats[f"{metric}_median"] = np.round(
                np.nanmedian(
                    np.array(
                        [
                            y[metric]
                            for scene_ep_id in metrics.keys()
                            for y in metrics[scene_ep_id]["metrics"]
                        ]
                    )
                ),
                4,
            )

        with open(os.path.join(results_dir, "cumulative_metrics.json"), "w") as fp:
            json.dump(stats, fp, indent=4)
