# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import time
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd
from habitat.utils.visualizations.utils import build_text_image, images_to_video
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from omegaconf import DictConfig
from tqdm import tqdm
from utils.env_utils import create_ovmm_env_fn
from utils.metrics_utils import get_stats_from_episode_metrics
from utils.video_utils import get_snapshots_from_disk, record_video

if TYPE_CHECKING:
    from habitat.core.dataset import BaseEpisode
    from habitat.core.vector_env import VectorEnv

    from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
    from home_robot.core.abstract_agent import Agent


class EvaluationType(Enum):
    LOCAL = "local"
    LOCAL_VECTORIZED = "local_vectorized"
    REMOTE = "remote"


class OVMMEvaluator(PPOTrainer):
    """Class for creating vectorized environments, evaluating OpenVocabManipAgent on an episode dataset and returning metrics"""

    def __init__(self, eval_config: DictConfig) -> None:
        self.metrics_save_freq = eval_config.EVAL_VECTORIZED.metrics_save_freq
        self.results_dir = os.path.join(
            eval_config.DUMP_LOCATION, "results", eval_config.EXP_NAME
        )
        self.images_dir = os.path.join(
            eval_config.DUMP_LOCATION, "images", eval_config.EXP_NAME
        )
        self.videos_dir = eval_config.habitat_baselines.video_dir
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
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

    def _summarize_metrics(self, episode_metrics: Dict) -> Dict:
        """Gets stats from episode metrics"""
        # convert to a dataframe
        episode_metrics_df = pd.DataFrame.from_dict(episode_metrics, orient="index")
        episode_metrics_df["start_idx"] = 0
        stats = get_stats_from_episode_metrics(episode_metrics_df)
        return stats

    def _print_summary(self, summary: dict):
        """Prints the summary of metrics"""
        print("=" * 50)
        print("Averaged metrics")
        print("=" * 50)
        for k, v in summary.items():
            print(f"{k}: {v}")
        print("=" * 50)

    def _check_set_planner_vis_dir(
        self, agent: "Agent", current_episode: "BaseEpisode"
    ):
        """
        Sets vis_dir for storing planner's debug visualisations if the agent has a planner.
        """
        if hasattr(agent, "planner"):
            agent.planner.set_vis_dir(
                current_episode.scene_id.split("/")[-1].split(".")[0],
                current_episode.episode_id,
            )

    def _evaluate_vectorized(
        self,
        agent: "OpenVocabManipAgent",
        envs: "VectorEnv",
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

        agent.reset_vectorized()
        self._check_set_planner_vis_dir(agent, self.envs.current_episodes()[0])
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
                # Record metrics after each skill finishes. This is useful for debugging.
                if "skill_done" in info and info["skill_done"] != "":
                    metrics = self._extract_scalars_from_info(hab_info)
                    metrics_at_skill_end = {
                        f"{info['skill_done']}." + k: v for k, v in metrics.items()
                    }
                    episode_metrics[episode_key] = {
                        **metrics_at_skill_end,
                        **episode_metrics[episode_key],
                    }
                    if "goal_name" in episode_metrics[episode_key]:
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
                        if "goal_name" in episode_metrics[episode_key]:
                            episode_metrics[episode_key]["goal_name"] = info[
                                "goal_name"
                            ]
                        episode_idxs[e] += 1
                        print(
                            f"Episode indexes {episode_idxs[e]} / {num_episodes_per_env[e]} "
                            f"after {round(time.time() - start_time, 2)} seconds"
                        )
                    if len(episode_metrics) % self.metrics_save_freq == 0:
                        aggregated_metrics = self._aggregate_metrics(episode_metrics)
                        self._write_results(episode_metrics, aggregated_metrics)
                    if not stop():
                        obs[e] = envs.call_at(e, "reset")
                        agent.reset_vectorized_for_env(e)
                        self._check_set_planner_vis_dir(
                            envs, envs.current_episodes()[e]
                        )

        envs.close()

        aggregated_metrics = self._aggregate_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        average_metrics = self._summarize_metrics(episode_metrics)
        self._print_summary(average_metrics)

        return average_metrics

    def _aggregate_metrics(self, episode_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Aggregates metrics tracked by environment."""
        aggregated_metrics = defaultdict(list)
        metrics = set(
            [
                k
                for metrics_per_episode in episode_metrics.values()
                for k, v in metrics_per_episode.items()
                if not isinstance(v, str)
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
                        f"{k1}/sum": np.sum(v1),
                    }.items()
                }.items()
            )
        )

        return aggregated_metrics

    def _write_results(
        self,
        episode_metrics: Dict[str, Dict],
        aggregated_metrics: Dict[str, float],
        average_metrics: Dict[str, float],
    ) -> None:
        """Writes metrics tracked by environment to a file."""
        with open(f"{self.results_dir}/aggregated_results.json", "w") as f:
            json.dump(aggregated_metrics, f, indent=4)
        with open(f"{self.results_dir}/episode_results.json", "w") as f:
            json.dump(episode_metrics, f, indent=4)
        with open(f"{self.results_dir}/summary_results.json", "w") as f:
            json.dump(average_metrics, f, indent=4)

    def _get_episode_completion_stage(self, metrics_at_episode_end):
        # TODO: temporary
        if metrics_at_episode_end["END.ovmm_place_success"] == 1:
            return "0_overall_success"
        elif metrics_at_episode_end["END.obj_anywhere_on_goal.0"] == 1:
            return "1_place_anywhere_on_goal_success"
        elif metrics_at_episode_end["END.ovmm_find_recep_phase_success"] == 1:
            return "2_nav_to_goal_success_but_place_failure"
        elif metrics_at_episode_end["END.ovmm_pick_object_phase_success"] == 1:
            return "3_pick_success_but_nav_to_goal_failure"
        elif metrics_at_episode_end["END.ovmm_find_object_phase_success"] == 1:
            return "4_nav_to_object_success_but_pick_failure"
        return "5_nav_to_object_failure"

    def get_episode_completion_stage(self, metrics_at_episode_end, coarse=False):
        episode_completion_stage = self._get_episode_completion_stage(
            metrics_at_episode_end
        )
        if coarse:
            coarse_stage = (
                "success"
                if episode_completion_stage
                in ["0_overall_success", "1_place_anywhere_on_goal_success"]
                else "failure"
            )
            return coarse_stage
        return episode_completion_stage

    def get_all_episode_completion_stages(self):
        # TODO: temporary: should be an Enum if productionized
        return [
            "0_overall_success",
            "1_place_anywhere_on_goal_success",
            "2_nav_to_goal_success_but_place_failure",
            "3_pick_success_but_nav_to_goal_failure",
            "4_nav_to_object_success_but_pick_failure",
            "5_nav_to_object_failure",
        ]

    def initialize_episode_and_agent(self, agent):
        observations, done = self._env.reset(), False
        current_episode = self._env.get_current_episode()
        agent.reset()
        self._check_set_planner_vis_dir(agent, current_episode)
        print(
            f"Starting evaluation for {self.config.EVAL_VECTORIZED.split} episode {current_episode.episode_id}"
        )
        print(f"Using strategy: {self.config.EXPERIMENT.type}")
        agent.set_oracle_info(self._env)
        return observations, done, current_episode

    def local_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluates the agent in the local environment.

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        env_num_episodes = self._env.number_of_episodes
        if num_episodes is None:
            num_episodes = env_num_episodes
        else:
            assert num_episodes <= env_num_episodes, (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(num_episodes, env_num_episodes)
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        episode_metrics: Dict = {}

        count_episodes: int = 0
        computed_episodes = []
        skip_computed = False  # TODO: temporary

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            observations, done, current_episode = self.initialize_episode_and_agent(
                agent
            )
            current_scene_name = current_episode.scene_id.split("/")[-1].split(".")[0]
            current_episode_key = f"{current_scene_name}_{current_episode.episode_id}"
            current_episode_metrics = {}

            if skip_computed:
                computed_episodes = []
                for completion_stage in self.get_all_episode_completion_stages():
                    target_dir_experiment = os.path.join(
                        self.videos_dir, completion_stage, self.config.EXPERIMENT.type
                    )
                    target_file_experiment = f"split_{self.config.EVAL_VECTORIZED.split}_scene_{current_scene_name}_episode_{current_episode.episode_id}"
                    if os.path.exists(
                        f"{target_dir_experiment}/{target_file_experiment}.json"
                    ):
                        computed_episodes.append(current_episode_key)
                        break
                if current_episode_key in computed_episodes:
                    try:
                        with open(
                            f"{target_dir_experiment}/{target_file_experiment}.json",
                            "r",
                        ) as f:
                            episode_metrics[current_episode_key] = json.load(f)
                        print(
                            f"Skipping episode {current_episode.episode_id} because it has already been computed"
                        )
                        count_episodes += 1
                        pbar.update(1)

                        if self.config.EVAL_VECTORIZED.record_videos:
                            source_dir = os.path.join(
                                self.images_dir, current_episode_key
                            )
                            target_dir_annotation = os.path.join(
                                "video_dir_annotation",
                                self.get_episode_completion_stage(
                                    episode_metrics[current_episode_key],
                                    coarse=True,
                                ),
                                f"scene_{current_scene_name}",
                            )
                            target_file_annotation = f"split_{self.config.EVAL_VECTORIZED.split}_episode_{current_episode.episode_id}"
                            os.makedirs(target_dir_annotation, exist_ok=True)
                            with open(
                                f"{target_dir_annotation}/{target_file_annotation}.json",
                                "w",
                            ) as f:
                                json.dump(
                                    episode_metrics[current_episode_key], f, indent=4
                                )

                            import shutil

                            shutil.copyfile(
                                f"{target_dir_experiment}/{target_file_experiment}.mp4",
                                f"{target_dir_annotation}/{target_file_annotation}.mp4",
                            )

                        continue
                    except Exception as e:
                        print(
                            f"Error {e} loading metrics for {current_episode_key}. Not skipping. Recomputing..."
                        )

            steps, max_steps = -1, 2000
            max_nav_obj_steps = 800
            start_time = time.time()

            while not done and steps < max_steps:
                steps += 1
                action, info, _ = agent.act(observations)
                observations, done, hab_info = self._env.apply_action(action, info)
                print(
                    f"Timestep:\t{steps}\t{info['curr_skill']}\t({hab_info['ovmm_dist_to_pick_goal']:.4f},\t{hab_info['ovmm_dist_to_place_goal']:.4f})",
                    end="\r",
                )
                # print(f"Current skill: {info['curr_skill']}")
                # print(
                #     f"info['ovmm_dist_to_pick_goal']:\t{hab_info['ovmm_dist_to_pick_goal']:.4f}"
                # )
                # print(
                #     f"info['ovmm_dist_to_keep_goal']:\t{hab_info['ovmm_dist_to_place_goal']:.4f}"
                # )

                if info["curr_skill"] == "NAV_TO_OBJ" and steps > max_nav_obj_steps:
                    print("Nav to obj is taking too long, moving to next episode")
                    break

                if "skill_done" in info and info["skill_done"] != "":
                    metrics = self._extract_scalars_from_info(hab_info)
                    metrics_at_skill_end = {
                        f"{info['skill_done']}." + k: v for k, v in metrics.items()
                    }
                    if (
                        info["curr_skill"] == "NAV_TO_REC"
                        and info["skill_done"] == "PICK"
                        and metrics_at_skill_end["PICK.ovmm_pick_object_phase_success"]
                        == 0
                    ):
                        print("Pick failure, the rest of the episode is moot")
                        break

                    current_episode_metrics = {
                        **metrics_at_skill_end,
                        **current_episode_metrics,
                    }
                    if "goal_name" in info:
                        current_episode_metrics["goal_name"] = info["goal_name"]

            end_time = time.time()
            print(f"Episode took {end_time - start_time} seconds")

            metrics = self._extract_scalars_from_info(hab_info)
            metrics["total_time_in_seconds"] = end_time - start_time
            metrics["done"] = 1.0 if done else 0.0
            metrics_at_episode_end = {"END." + k: v for k, v in metrics.items()}
            current_episode_metrics = {
                **metrics_at_episode_end,
                **current_episode_metrics,
            }
            current_episode_metrics["data_split"] = self.config.EVAL_VECTORIZED.split
            current_episode_metrics["scene_name"] = current_scene_name
            current_episode_metrics["episode_id"] = current_episode.episode_id
            current_episode_metrics[
                "episode_completion_stage"
            ] = self.get_episode_completion_stage(current_episode_metrics)
            current_episode_metrics["experiment_name"] = self.config.EXP_NAME
            current_episode_metrics["experiment_type"] = self.config.EXPERIMENT.type
            if "goal_name" in info:
                current_episode_metrics["goal_name"] = info["goal_name"]

            source_dir = os.path.join(self.images_dir, current_episode_key)
            target_dir_experiment = os.path.join(
                self.videos_dir,
                self.get_episode_completion_stage(
                    current_episode_metrics, coarse=False
                ),
                self.config.EXPERIMENT.type,
            )
            target_file_experiment = f"split_{self.config.EVAL_VECTORIZED.split}_scene_{current_scene_name}_episode_{current_episode.episode_id}"

            target_dir_annotation = os.path.join(
                "video_dir_annotation",
                self.get_episode_completion_stage(current_episode_metrics, coarse=True),
                f"scene_{current_scene_name}",
            )
            target_file_annotation = f"split_{self.config.EVAL_VECTORIZED.split}_episode_{current_episode.episode_id}"

            save_down_videos = self.config.EVAL_VECTORIZED.record_videos
            # try:
            #     # if the episode has already been computed, and the new episode run is not better than the older, don't save down videos
            #     # the definition of better is (1) new computation results in overall success, (2) new computation has less steps than older one.
            #     metrics_from_earlier_run_of_same_episode = None
            #     with open(f"""{os.path.join("video_dir_annotation", "success", f"scene_{current_scene_name}")}/{target_file_annotation}.json""", "r") as f:
            #         metrics_from_earlier_run_of_same_episode = json.load(f)
            #     if (
            #         metrics_from_earlier_run_of_same_episode is not None and (
            #             metrics_from_earlier_run_of_same_episode["END.num_steps"] <= current_episode_metrics["END.num_steps"]
            #             or self.get_episode_completion_stage(current_episode_metrics, coarse=True) != "success"
            #         )
            #     ):
            #         save_down_videos = False
            # except Exception:
            #     pass

            if (
                "v8" not in self.config.EXP_NAME
                and self.get_episode_completion_stage(
                    current_episode_metrics, coarse=True
                )
                == "failure"
            ):
                save_down_videos = False

            if save_down_videos:
                os.makedirs(target_dir_experiment, exist_ok=True)
                with open(
                    f"{target_dir_experiment}/{target_file_experiment}.json", "w"
                ) as f:
                    json.dump(current_episode_metrics, f, indent=4)
                # frames = get_snapshots_from_disk(source_dir, snapshot_file_prefix="tp_snapshot")
                # if frames is not None and len(frames) > 0:
                #     images_to_video(frames, target_dir_experiment, target_file_experiment, fps=24, quality=5)

                os.makedirs(target_dir_annotation, exist_ok=True)
                with open(
                    f"{target_dir_annotation}/{target_file_annotation}.json", "w"
                ) as f:
                    json.dump(current_episode_metrics, f, indent=4)

                episode_frames = self._env.habitat_env.env._env._env._task._frames
                if episode_frames is not None and len(episode_frames) > 0:
                    robot_goal_text = build_text_image(
                        episode_frames[0],
                        f"Robot's goal: {current_episode_metrics['goal_name'].replace('_', ' ')}",
                        color="black",
                    )
                    human_goal_text = build_text_image(
                        episode_frames[0],
                        "Your goal: Say the actions the robot is performing in natural language.",
                        color="black",
                    )
                    episode_frames = [
                        np.concatenate(
                            (robot_goal_text, frame, human_goal_text), axis=0
                        )
                        for frame in episode_frames
                    ]
                    images_to_video(
                        episode_frames,
                        target_dir_annotation,
                        target_file_annotation,
                        fps=24,
                        quality=5,
                    )

            episode_metrics[current_episode_key] = current_episode_metrics
            count_episodes += 1
            pbar.update(1)

        self._env.close()

        aggregated_metrics = self._aggregate_metrics(episode_metrics)
        average_metrics = self._summarize_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics, average_metrics)
        self._print_summary(average_metrics)

        return average_metrics

    def remote_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluates the agent in the remote environment.

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        # The modules imported below are specific to challenge remote evaluation.
        # These modules are not part of the home-robot repository.
        import pickle
        import time

        import evalai_environment_habitat  # noqa: F401
        import evaluation_pb2
        import evaluation_pb2_grpc
        import grpc

        # Wait for the remote environment to be up and running
        time.sleep(60)

        def grpc_dumps(entity):
            return pickle.dumps(entity)

        def grpc_loads(entity):
            return pickle.loads(entity)

        env_address_port = os.environ.get("EVALENV_ADDPORT", "localhost:8085")
        channel = grpc.insecure_channel(
            target=env_address_port,
            compression=grpc.Compression.Gzip,
            options=[
                (
                    "grpc.max_receive_message_length",
                    -1,
                )  # Unlimited message length that the channel can receive
            ],
        )
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        stub.init_env(evaluation_pb2.Package())

        env_num_episodes = grpc_loads(
            stub.number_of_episodes(evaluation_pb2.Package()).SerializedEntity
        )
        if num_episodes is None:
            num_episodes = env_num_episodes
        else:
            assert num_episodes <= env_num_episodes, (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(num_episodes, env_num_episodes)
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        episode_metrics: Dict = {}

        count_episodes: int = 0

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            observations, done = (
                grpc_loads(stub.reset(evaluation_pb2.Package()).SerializedEntity),
                False,
            )
            current_episode = grpc_loads(
                stub.get_current_episode(evaluation_pb2.Package()).SerializedEntity
            )
            agent.reset()
            self._check_set_planner_vis_dir(agent, current_episode)

            current_episode_key = (
                f"{current_episode.scene_id.split('/')[-1].split('.')[0]}_"
                f"{current_episode.episode_id}"
            )
            current_episode_metrics = {}

            while not done:
                action, info, _ = agent.act(observations)
                observations, done, hab_info = grpc_loads(
                    stub.apply_action(
                        evaluation_pb2.Package(
                            SerializedEntity=grpc_dumps((action, info))
                        )
                    ).SerializedEntity
                )

                # record metrics if the current skill finishes
                if "skill_done" in info and info["skill_done"] != "":
                    metrics = self._extract_scalars_from_info(hab_info)
                    metrics_at_skill_end = {
                        f"{info['skill_done']}." + k: v for k, v in metrics.items()
                    }
                    current_episode_metrics = {
                        **metrics_at_skill_end,
                        **current_episode_metrics,
                    }
                    if "goal_name" in info:
                        current_episode_metrics["goal_name"] = info["goal_name"]

            metrics = self._extract_scalars_from_info(hab_info)
            metrics_at_episode_end = {"END." + k: v for k, v in metrics.items()}
            current_episode_metrics = {
                **metrics_at_episode_end,
                **current_episode_metrics,
            }
            if "goal_name" in info:
                current_episode_metrics["goal_name"] = info["goal_name"]

            episode_metrics[current_episode_key] = current_episode_metrics
            if len(episode_metrics) % self.metrics_save_freq == 0:
                aggregated_metrics = self._aggregate_metrics(episode_metrics)
                self._write_results(episode_metrics, aggregated_metrics)

            count_episodes += 1
            pbar.update(1)

        stub.close(evaluation_pb2.Package())
        stub.evalai_update_submission(evaluation_pb2.Package())

        aggregated_metrics = self._aggregate_metrics(episode_metrics)
        self._write_results(episode_metrics, aggregated_metrics)

        average_metrics = self._summarize_metrics(episode_metrics)
        self._print_summary(average_metrics)

        return average_metrics

    def evaluate(
        self,
        agent: "Agent",
        num_episodes: Optional[int] = None,
        evaluation_type: str = "local",
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """
        if evaluation_type == EvaluationType.LOCAL.value:
            self._env = create_ovmm_env_fn(self.config)
            return self.local_evaluate(agent, num_episodes)
        elif evaluation_type == EvaluationType.LOCAL_VECTORIZED.value:
            self._env = create_ovmm_env_fn(self.config)
            return self.local_evaluate_vectorized(agent, num_episodes)
        elif evaluation_type == EvaluationType.REMOTE.value:
            self._env = None
            return self.remote_evaluate(agent, num_episodes)
        else:
            raise ValueError(
                "Invalid evaluation type. Please choose from 'local', 'local_vectorized', 'remote'"
            )
