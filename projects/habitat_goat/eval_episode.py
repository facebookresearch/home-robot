import argparse
import json
import os
import sys
from pathlib import Path
from pprint import pprint

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

from home_robot.agent.goat_agent.goat_agent import GoatAgent
from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot_sim.env.habitat_goat_env.habitat_goat_env import HabitatGoatEnv

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
        "--scene_idx",
        type=int,
        default=0,
        help="Scene indices (for parallel eval)",
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

    all_scenes = os.listdir(os.path.dirname(config.habitat.dataset.data_path.format(split=config.habitat.dataset.split)) + "/content/")
    all_scenes = sorted([x.split('.')[0] for x in all_scenes])

    if args.scene_idx != -1:
        scene_start = args.scene_idx * 5
        config.habitat.dataset.content_scenes = all_scenes[scene_start:scene_start+5]

    # config.habitat.dataset.content_scenes = ["5cdEh9F2hJL"] # TODO: for debugging. REMOVE later.

    config.NUM_ENVIRONMENTS = 1
    config.PRINT_IMAGES = 1

    config.EXP_NAME = f"{config.EXP_NAME}_{args.scene_idx}"

    agent = GoatAgent(config=config)
    habitat_env = Env(config)
    env = HabitatGoatEnv(habitat_env, config=config)

    results_dir = os.path.join(config.DUMP_LOCATION, "results", config.EXP_NAME)
    os.makedirs(results_dir, exist_ok=True)

    metrics = {}

    for i in range(len(env.habitat_env.episodes)):
        env.reset()
        agent.reset()

        t = 0

        scene_id = env.habitat_env.current_episode.scene_id.split("/")[-1].split(".")[0]
        episode = env.habitat_env.current_episode
        episode_id = episode.episode_id

        # if episode_id in ["7", "15", "19", "0"]:
        #     continue

        if episode_id != "1":
            continue

        # if i < 48:
        #     continue

        # if scene_id != "HkseAnWCgqk":
        #     continue
        # else:
        #     if episode_id != "1":
        #         continue

        # if scene_id != 'XYyR54sxe6b':
        #     continue


        # vlp ->
        # if episode_id != "15":
        #     continue

        # if scene_id != 'oEPjPNSPmzL': 
        # if scene_id != 'qk9eeNeR4vw': 
        # if scene_id != 'HkseAnWCgqk': 
        # if scene_id != 'vLpv2VX547B': 
        #     continue

        agent.planner.set_vis_dir(scene_id, f"{episode_id}_{env.habitat_env.task.current_task_idx}")
        agent.imagenav_visualizer.set_vis_dir(
            f"{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}"
        )
        agent.matching.set_vis_dir(f"{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}")
        env.visualizer.set_vis_dir(scene_id, f"{episode_id}_{env.habitat_env.task.current_task_idx}")

        all_subtask_metrics = []
        pbar = tqdm(total=config.AGENT.max_steps)
        
        while not env.episode_over:
            current_task_idx = env.habitat_env.task.current_task_idx
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

                pprint(obs_tasks)

            action, info = agent.act(obs)
            env.apply_action(action, info=info)
            pbar.set_description(
                f"{scene_id}_{episode_id}_{current_task_idx}"
            )
            pbar.update(1)
            # print(env.get_episode_metrics()["goat_distance_to_sub-goal"])

            if action == DiscreteNavigationAction.STOP:
                ep_metrics = env.get_episode_metrics()
                ep_metrics.pop("goat_top_down_map", None)
                print('-------------------------')
                print(f"{scene_id}_{episode_id}_{current_task_idx}", ep_metrics)
                print('-------------------------')

                all_subtask_metrics.append(ep_metrics)
                if not env.episode_over:
                    agent.imagenav_visualizer.set_vis_dir(
                        f"{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}"
                    )
                    agent.matching.set_vis_dir(
                        f"{scene_id}_{episode_id}_{env.habitat_env.task.current_task_idx}"
                    )
                    agent.planner.set_vis_dir(
                        scene_id, f"{episode_id}_{env.habitat_env.task.current_task_idx}"
                    )
                    env.visualizer.set_vis_dir(
                        scene_id, f"{episode_id}_{env.habitat_env.task.current_task_idx}"
                    )
                    pbar.reset()

        pbar.close()

        ep_metrics = env.get_episode_metrics()
        scene_ep_id = f"{scene_id}_{episode_id}"
        metrics[scene_ep_id] = {"metrics": all_subtask_metrics}
        metrics[scene_ep_id]["total_num_steps"] = t
        metrics[scene_ep_id]["sub_task_timesteps"] = agent.sub_task_timesteps[0]
        metrics[scene_ep_id]["tasks"] = obs_tasks

        try:
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
        except Exception as e:
            print(e)
            import pdb

            pdb.set_trace()

        print("---------------------------------")

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
