import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
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

from home_robot.agent.languagenav_agent.languagenav_agent import LanguageNavAgent
from home_robot_sim.env.habitat_languagenav_env.habitat_languagenav_env import (
    HabitatLanguageNavEnv,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="languagenav/modular_languagenav_hm3d.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/habitat_languagenav/configs/agent/hm3d_eval.yaml",
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

    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1
    config.habitat.dataset.split = "val"

    agent = LanguageNavAgent(config=config)
    habitat_env = Env(config)
    env = HabitatLanguageNavEnv(habitat_env, config=config)

    results_dir = os.path.join(config.DUMP_LOCATION, "results", config.EXP_NAME)
    os.makedirs(results_dir, exist_ok=True)

    metrics = {}

    for i in range(len(env.habitat_env.episodes)):
        env.reset()
        agent.reset()
        t = 0

        scene_id = env.habitat_env.current_episode.scene_id.split("/")[-1].split(".")[0]
        agent.planner.set_vis_dir(scene_id, env.habitat_env.current_episode.episode_id)
        episode_id = env.habitat_env.current_episode.episode_id

        pbar = tqdm(total=config.AGENT.max_steps)
        while not env.episode_over:
            t += 1
            obs = env.get_observation()
            if t == 1:
                print(env.habitat_env.current_episode.instructions[0])
                print("Target:", obs.task_observations["target"])
                print("Landmarks:", obs.task_observations["landmarks"])

            action, info = agent.act(obs)
            env.apply_action(action, info=info)
            pbar.set_description(f"Action: {str(action).split('.')[-1]}")
            pbar.update(1)

        pbar.close()

        ep_metrics = env.get_episode_metrics()
        scene_ep_id = f"{scene_id}_{episode_id}"
        metrics[scene_ep_id] = ep_metrics
        metrics[scene_ep_id][
            "num_goal_candidates_visited"
        ] = agent.num_goal_candidates_visited
        metrics[scene_ep_id]["num_steps"] = t
        metrics[scene_ep_id]["target"] = obs.task_observations["target"]
        metrics[scene_ep_id]["landmarks"] = obs.task_observations["landmarks"]
        metrics[scene_ep_id]["caption"] = obs.task_observations["caption"]

        print(f"{scene_id}_{episode_id}", ep_metrics)

        with open(os.path.join(results_dir, "per_episode_metrics.json"), "w") as fp:
            json.dump(metrics, fp, indent=4)

        stats = {}

        for metric in list(metrics.values())[0].keys():
            if metric in ["target", "landmarks", "caption"]:
                continue

            stats[f"{metric}_mean"] = np.nanmean(
                np.array([metrics[x][metric] for x in metrics.keys()])
            )
            stats[f"{metric}_median"] = np.nanmedian(
                np.array([metrics[x][metric] for x in metrics])
            )

        with open(os.path.join(results_dir, "cumulative_metrics.json"), "w") as fp:
            json.dump(stats, fp, indent=4)
