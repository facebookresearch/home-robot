import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from config_utils import get_config
from omegaconf import DictConfig, OmegaConf

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot"),
)
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "src/home_robot_sim"),
)

from habitat import make_dataset
from habitat.core.environments import get_env_class
from habitat.core.vector_env import VectorEnv
from habitat.utils.gym_definitions import _get_env_name
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import (
    HabitatOpenVocabManipEnv,
)


def create_ovmm_env_fn(config):
    """Create habitat environment using configsand wrap HabitatOpenVocabManipEnv around it. This function is used by VectorEnv for creating the individual environments"""
    habitat_config = config.habitat
    dataset = make_dataset(habitat_config.dataset.type, config=habitat_config.dataset)
    env_class_name = _get_env_name(config)
    env_class = get_env_class(env_class_name)
    habitat_env = env_class(config=habitat_config, dataset=dataset)
    habitat_env.seed(habitat_config.seed)
    env = HabitatOpenVocabManipEnv(habitat_env, config, dataset=dataset)
    return env


class VectorizedEvaluator(PPOTrainer):
    """Class for creating vectorized environments, evaluating OpenVocabManipAgent on an episode dataset and returning metrics"""

    def __init__(self, config, config_str: str):
        self.config = config
        self.results_dir = self.config.habitat_baselines.eval_ckpt_path_dir
        self.videos_dir = self.config.habitat_baselines.video_dir
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        super().__init__(config)

    def eval(self, num_episodes_per_env=10):
        self._init_envs(
            config=self.config, is_eval=True, make_env_fn=create_ovmm_env_fn
        )
        agent = OpenVocabManipAgent(
            config=self.config,
            obs_spaces=self.envs.observation_spaces,
            action_spaces=self.envs.orig_action_spaces,
        )
        self._eval(
            agent,
            self.envs,
            num_episodes_per_env=num_episodes_per_env,
            episode_keys=None,
        )

    def write_results(self, episode_metrics):
        aggregated_metrics = defaultdict(list)
        metrics = set(
            [k for k in list(episode_metrics.values())[0].keys() if k != "goal_name"]
        )
        for v in episode_metrics.values():
            for k in metrics:
                aggregated_metrics[f"{k}/total"].append(v[k])
                aggregated_metrics[f"{k}/{v['goal_name']}"].append(v[k])

        aggregated_metrics = dict(
            sorted(
                {
                    k2: v2
                    for k1, v1 in aggregated_metrics.items()
                    for k2, v2 in {
                        f"{k1}/mean": np.mean(v1),
                        f"{k1}/min": np.min(v1),
                        f"{k1}/max": np.max(v1),
                    }.items()
                }.items()
            )
        )
        with open(f"{self.results_dir}/aggregated_results.json", "w") as f:
            json.dump(aggregated_metrics, f, indent=4)
        with open(f"{self.results_dir}/episode_results.json", "w") as f:
            json.dump(episode_metrics, f, indent=4)

    def _eval(
        self,
        agent: OpenVocabManipAgent,
        envs: VectorEnv,
        num_episodes_per_env=None,
        episode_keys=None,
    ):
        # The stopping condition is either specified through
        # num_episodes_per_env (stop after each environment
        # finishes a certain number of episodes) or episode_keys
        # (stop after we iterate through a list of specific episodes)
        assert (num_episodes_per_env is not None and episode_keys is None) or (
            num_episodes_per_env is None and episode_keys is not None
        )

        def stop():
            if num_episodes_per_env is not None:
                return all([i >= num_episodes_per_env for i in episode_idxs]) or len(
                    episode_metrics
                ) == sum(self.envs.number_of_episodes)
            elif episode_keys is not None:
                return done_episode_keys == episode_keys

        start_time = time.time()
        episode_metrics = {}
        episode_idxs = [0] * envs.num_envs
        done_episode_keys = set()

        obs = envs.call(["reset"] * envs.num_envs)

        agent.reset_vectorized()
        while not stop():
            current_episodes_info = self.envs.current_episodes()
            # TODO: Currently agent can work with only 1 env, Parallelize act across envs
            actions, infos = zip(*[agent.act(ob) for ob in obs])

            outputs = envs.call(
                ["apply_action"] * envs.num_envs,
                [{"action": a, "info": i} for a, i in zip(actions, infos)],
            )

            obs, dones, hab_infos = [list(x) for x in zip(*outputs)]
            for e, (done, info, hab_info) in enumerate(zip(dones, infos, hab_infos)):
                if done:
                    episode_key = (
                        f"{current_episodes_info[e].scene_id.split('/')[-1].split('.')[0]}_"
                        f"{current_episodes_info[e].episode_id}"
                    )
                    metrics = self._extract_scalars_from_info(hab_info)
                    # If the episode keys we care about are specified,
                    #  ignore all other episodes
                    if episode_keys is not None:
                        if episode_key in episode_keys:
                            done_episode_keys.add(episode_key)
                            episode_metrics[episode_key] = {
                                **metrics,
                                "goal_name": info["goal_name"],
                            }
                            print(
                                f"Finished episode {episode_key} after "
                                f"{round(time.time() - start_time, 2)} seconds"
                            )

                    elif num_episodes_per_env is not None:
                        if episode_idxs[e] < num_episodes_per_env:
                            episode_metrics[episode_key] = {
                                **metrics,
                                "goal_name": info["goal_name"],
                            }
                        episode_idxs[e] += 1
                        print(
                            f"Episode indexes {episode_idxs} / {num_episodes_per_env} "
                            f"after {round(time.time() - start_time, 2)} seconds"
                        )

                        # [temporary] to print running metrics
                        aggregated_metrics = defaultdict(list)
                        metrics = set(
                            [
                                k
                                for k in list(episode_metrics.values())[0].keys()
                                if k != "goal_name"
                            ]
                        )
                        for v in episode_metrics.values():
                            k = "nav_to_obj_success"
                            aggregated_metrics[f"{k[:3]}/total"].append(v[k])
                            aggregated_metrics[f"{k[:3]}/{v['goal_name'][:16]}"].append(
                                v[k]
                            )

                        aggregated_metrics = dict(
                            sorted(
                                {
                                    k2: v2
                                    for k1, v1 in aggregated_metrics.items()
                                    for k2, v2 in {
                                        f"{k1}/mean": np.round(np.mean(v1), 2),
                                    }.items()
                                }.items()
                            )
                        )

                    agent.reset_vectorized_for_env(e)

                    if len(episode_metrics) % 5 == 0:
                        self.write_results(episode_metrics)
                    if not stop():
                        obs[e] = envs.call_at(e, "reset")

        envs.close()
        self.write_results(episode_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="rearrange/modular_nav.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/agent/floorplanner_eval.yaml",
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

    print("Configs:")
    config, config_str = get_config(args.habitat_config_path, opts=args.opts)
    OmegaConf.set_readonly(config, True)
    baseline_config = OmegaConf.load(args.baseline_config_path)
    config = DictConfig({**config, **baseline_config})
    evaluator = VectorizedEvaluator(config, config_str)
    print(config_str)
    print("-" * 100)
    evaluator.eval(
        num_episodes_per_env=config.EVAL_VECTORIZED.num_episodes_per_env,
    )
