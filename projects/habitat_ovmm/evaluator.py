# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
import time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
from habitat import make_dataset
from habitat.core.environments import get_env_class
from habitat.core.vector_env import VectorEnv
from habitat.utils.gym_definitions import _get_env_name
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from omegaconf import DictConfig
from tqdm import tqdm

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


class OVMMEvaluator(PPOTrainer):
    """Class for creating vectorized environments, evaluating OpenVocabManipAgent on an episode dataset and returning metrics"""

    def __init__(self, eval_config: DictConfig) -> None:
        self.metrics_save_freq = eval_config.EVAL_VECTORIZED.metrics_save_freq
        self.results_dir = os.path.join(
            eval_config.DUMP_LOCATION, "results", eval_config.EXP_NAME
        )
        self.videos_dir = eval_config.habitat_baselines.video_dir
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

        super().__init__(eval_config)

    def local_evaluate_vectorized(self, agent, num_episodes_per_env=10):
        self._init_envs(
            config=self.config, is_eval=True, make_env_fn=create_ovmm_env_fn
        )
        self._evaluate_vectorized(
            agent,
            self.envs,
            num_episodes_per_env=num_episodes_per_env,
        )

    def _evaluate_vectorized(
        self,
        agent: OpenVocabManipAgent,
        envs: VectorEnv,
        num_episodes_per_env=None,
    ):
        # The stopping condition is either specified through
        # num_episodes_per_env (stop after each environment
        # finishes a certain number of episodes)
        print(f"Running eval on {envs.number_of_episodes} episodes")

        if num_episodes_per_env is None:
            num_episodes_per_env = envs.number_of_episodes
        else:
            num_episodes_per_env = [num_episodes_per_env] * envs.num_envs

        episode_metrics = {}

        def stop():
            return all(
                [
                    episode_idxs[i] >= num_episodes_per_env[i]
                    for i in range(envs.num_envs)
                ]
            )

        start_time = time.time()
        episode_idxs = [0] * envs.num_envs
        obs = envs.call(["reset"] * envs.num_envs)

        agent.reset_vectorized(self.envs.current_episodes())
        while not stop():
            current_episodes_info = self.envs.current_episodes()
            # TODO: Currently agent can work with only 1 env, Parallelize act across envs
            actions, infos, _ = zip(*[agent.act(ob) for ob in obs])

            outputs = envs.call(
                ["apply_action"] * envs.num_envs,
                [{"action": a, "info": i} for a, i in zip(actions, infos)],
            )

            obs, dones, hab_infos = [list(x) for x in zip(*outputs)]
            for e, (done, info, hab_info) in enumerate(zip(dones, infos, hab_infos)):
                episode_key = (
                    f"{current_episodes_info[e].scene_id.split('/')[-1].split('.')[0]}_"
                    f"{current_episodes_info[e].episode_id}"
                )
                if episode_key not in episode_metrics:
                    episode_metrics[episode_key] = {}
                if info["skill_done"] != "":
                    metrics = self._extract_scalars_from_info(hab_info)
                    metrics_at_skill_end = {
                        f"{info['skill_done']}." + k: v for k, v in metrics.items()
                    }
                    episode_metrics[episode_key] = {
                        **metrics_at_skill_end,
                        **episode_metrics[episode_key],
                    }
                    episode_metrics[episode_key]["goal_name"] = info["goal_name"]
                if done:  # environment times out
                    metrics = self._extract_scalars_from_info(hab_info)
                    if episode_idxs[e] < num_episodes_per_env[e]:
                        metrics_at_episode_end = {
                            f"END." + k: v for k, v in metrics.items()
                        }
                        episode_metrics[episode_key] = {
                            **metrics_at_episode_end,
                            **episode_metrics[episode_key],
                        }
                        episode_metrics[episode_key]["goal_name"] = info["goal_name"]
                        episode_idxs[e] += 1
                        print(
                            f"Episode indexes {episode_idxs[e]} / {num_episodes_per_env[e]} "
                            f"after {round(time.time() - start_time, 2)} seconds"
                        )
                    if len(episode_metrics) % self.metrics_save_freq == 0:
                        aggregated_metrics = self._aggregare_metrics(episode_metrics)
                        self._write_results(episode_metrics, aggregated_metrics)
                    if not stop():
                        obs[e] = envs.call_at(e, "reset")
                        agent.reset_vectorized_for_env(
                            e, self.envs.current_episodes()[e]
                        )

        envs.close()

        aggregated_metrics = self._aggregare_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        return aggregated_metrics

    def _aggregare_metrics(self, episode_metrics):
        aggregated_metrics = defaultdict(list)
        metrics = set(
            [
                k
                for metrics_per_episode in episode_metrics.values()
                for k in metrics_per_episode
                if k != "goal_name"
            ]
        )
        for v in episode_metrics.values():
            for k in metrics:
                if k in v:
                    aggregated_metrics[f"{k}/total"].append(v[k])

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

        return aggregated_metrics

    def _write_results(self, episode_metrics, aggregated_metrics):
        with open(f"{self.results_dir}/aggregated_results.json", "w") as f:
            json.dump(aggregated_metrics, f, indent=4)
        with open(f"{self.results_dir}/episode_results.json", "w") as f:
            json.dump(episode_metrics, f, indent=4)

    def local_evaluate(
        self, agent, num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = self._env.number_of_episodes
        else:
            assert num_episodes <= self._env.number_of_episodes, (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(num_episodes, self._env.number_of_episodes)
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        episode_metrics: Dict = {}

        count_episodes: int = 0

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            observations, done = self._env.reset(), False
            current_episode = self._env.get_current_episode()
            agent.reset_vectorized([current_episode])

            current_episode_key = (
                f"{current_episode.scene_id.split('/')[-1].split('.')[0]}_"
                f"{current_episode.episode_id}"
            )
            current_episode_metrics = {}

            while not done:
                action, info, _ = agent.act(observations)
                observations, done, hab_info = self._env.apply_action(action, info)

                if info["skill_done"] != "":
                    metrics = self._extract_scalars_from_info(hab_info)
                    metrics_at_skill_end = {
                        f"{info['skill_done']}." + k: v for k, v in metrics.items()
                    }
                    current_episode_metrics = {
                        **metrics_at_skill_end,
                        **current_episode_metrics,
                    }
                    current_episode_metrics["goal_name"] = info["goal_name"]

            metrics = self._extract_scalars_from_info(hab_info)
            metrics_at_episode_end = {"END." + k: v for k, v in metrics.items()}
            current_episode_metrics = {
                **metrics_at_episode_end,
                **current_episode_metrics,
            }
            current_episode_metrics["goal_name"] = info["goal_name"]

            episode_metrics[current_episode_key] = current_episode_metrics
            if len(episode_metrics) % self.metrics_save_freq == 0:
                aggregated_metrics = self._aggregare_metrics(episode_metrics)
                self._write_results(episode_metrics, aggregated_metrics)

            count_episodes += 1
            pbar.update(1)

        self._env.close()
        aggregated_metrics = self._aggregare_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        return aggregated_metrics

    def remote_evaluate(self, agent, num_episodes: Optional[int] = None):
        # The modules imported below are specific to habitat-challenge remote evaluation.
        # These modules are not part of the habitat-lab repository.
        import pickle
        import time

        import evalai_environment_habitat  # noqa: F401
        import evaluation_pb2
        import evaluation_pb2_grpc
        import grpc

        time.sleep(60)

        def pack_for_grpc(entity):
            return pickle.dumps(entity)

        def unpack_for_grpc(entity):
            return pickle.loads(entity)

        def remote_ep_over(stub):
            res_env = unpack_for_grpc(
                stub.episode_over(evaluation_pb2.Package()).SerializedEntity
            )
            return res_env["episode_over"]

        env_address_port = os.environ.get("EVALENV_ADDPORT", "localhost:8085")
        channel = grpc.insecure_channel(env_address_port)
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        base_num_episodes = unpack_for_grpc(
            stub.num_episodes(evaluation_pb2.Package()).SerializedEntity
        )
        num_episodes = base_num_episodes["num_episodes"]

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0

        while count_episodes < num_episodes:
            agent.reset()
            res_env = unpack_for_grpc(
                stub.reset(evaluation_pb2.Package()).SerializedEntity
            )

            while not remote_ep_over(stub):
                obs = res_env["observations"]
                action = agent.act(obs)

                res_env = unpack_for_grpc(
                    stub.act_on_environment(
                        evaluation_pb2.Package(SerializedEntity=pack_for_grpc(action))
                    ).SerializedEntity
                )

            metrics = unpack_for_grpc(
                stub.get_metrics(
                    evaluation_pb2.Package(SerializedEntity=pack_for_grpc(action))
                ).SerializedEntity
            )

            for m, v in metrics["metrics"].items():
                agg_metrics[m] += v
            count_episodes += 1

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        stub.evalai_update_submission(evaluation_pb2.Package())

        return avg_metrics

    def evaluate(
        self, agent, num_episodes: Optional[int] = None, remote: bool = False
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """
        if remote:
            self._env = None
            return self.remote_evaluate(agent, num_episodes)
        else:
            self._env = create_ovmm_env_fn(self.config)
            return self.local_evaluate(agent, num_episodes)
