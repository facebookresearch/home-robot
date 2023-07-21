# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pickle
import numpy as np

from habitat.tasks.rearrange.utils import get_robot_spawns
from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import HabitatOpenVocabManipEnv
from matplotlib import pyplot as plt
from evaluator import create_ovmm_env_fn
from utils.config_utils import (
    create_env_config,
    get_habitat_config,
    get_omega_config,
)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

all_receptacles = ['cabinet', 'stool', 'trunk', 'shoe_rack', 'chest_of_drawers',
                   'table', 'toilet', 'serving_cart',
                   'bed', 'washer_dryer', 'hamper', 'stand', 'bathtub',
                   'couch', 'counter', 'shelves', 'chair', 'bench']


def extract_scene_id(scene_id: str) -> str:
    """extracts scene id from string containing the scene id"""

    before, _ = scene_id.split('.scene')
    _, after = before.split('uncluttered/')
    return after


def get_init_scene_episode_count_dict(env: HabitatOpenVocabManipEnv) -> dict:
    """Returns a dictionary containing entries for all (scene, episode) pairs 
        with count value initialized as 0"""
    count_dict = {}
    for episode in env._dataset.episodes:
        scene_id = extract_scene_id(episode.scene_id)
        hash_str = f'ep_{episode.episode_id}_scene_{scene_id}'
        count_dict[hash_str] = 0
    return count_dict


def receptacle_position_aggregate(scene_info_dir: str, env: HabitatOpenVocabManipEnv):
    """Aggregates receptacles position by scene using all episodes"""

    # This is for iterating through all episodes once using only one env
    count_dict = get_init_scene_episode_count_dict(env)

    receptacle_positions = {}

    # Ideally, we can make it like an iterator to make it feel more intuitive
    while True:
        # Get a new episode
        env.reset()
        episode = env.get_current_episode()
        scene_id = extract_scene_id(episode.scene_id)

        # Check if you have iterated through all episodes and if yes, break the loop
        hash_str = f'ep_{episode.episode_id}_scene_{scene_id}'
        if count_dict[hash_str] == 0:
            count_dict[hash_str] += 1
        elif count_dict[hash_str] == 1:
            break
        else:
            raise NotImplementedError  # Technically, this shouldn't happen

        if not scene_id in receptacle_positions:
            receptacle_positions[scene_id] = {}

        if not episode.goal_recep_category in receptacle_positions[scene_id]:
            receptacle_positions[scene_id][episode.goal_recep_category] = set()
        for recep in episode.candidate_goal_receps:
            recep_position = list(recep.position)
            view_point_position = list(recep.view_points[0].agent_state.position)
            receptacle_positions[scene_id][episode.goal_recep_category].add(
                tuple(recep_position + view_point_position))

    os.makedirs(f'./{scene_info_dir}', exist_ok=True)
    with open(f'./{scene_info_dir}/recep_position.pickle', 'wb') as handle:
        pickle.dump(receptacle_positions, handle, protocol=pickle.HIGHEST_PROTOCOL)


def gen_receptacle_images(scene_info_dir: str, data_dir: str, env: HabitatOpenVocabManipEnv):
    """Generates images of receptacles by episode for all scenes"""

    sim = env.habitat_env.env._env._env._sim

    # This is for iterating through all episodes once using only one env
    count_dict = get_init_scene_episode_count_dict(env)

    # Also, creating folders for storing dataset
    os.makedirs(f'./{data_dir}', exist_ok=True)
    for episode in env._dataset.episodes:
        scene_id = extract_scene_id(episode.scene_id)
        os.makedirs(f'./{data_dir}/scene_{scene_id}', exist_ok=True)

    with open(f'./{scene_info_dir}/recep_position.pickle', 'rb') as handle:
        receptacle_positions = pickle.load(handle)

    # Ideally, we can make it like an iterator to make it feel more intuitive
    while True:
        # Get a new episode
        obs = env.reset()
        episode = env.get_current_episode()
        scene_id = extract_scene_id(episode.scene_id)

        # Check if you have iterated through all episodes and if yes, break the loop
        hash_str = f'ep_{episode.episode_id}_scene_{scene_id}'
        if count_dict[hash_str] == 0:
            count_dict[hash_str] += 1
        elif count_dict[hash_str] == 1:
            break
        else:
            raise NotImplementedError  # Technically, this shouldn't happen

        os.makedirs(
            f'./{data_dir}/scene_{scene_id}/ep_{episode.episode_id}', exist_ok=True)

        for recep in receptacle_positions[scene_id]:
            os.makedirs(
                f'./{data_dir}/scene_{scene_id}/ep_{episode.episode_id}/{recep}', exist_ok=True)
            recep_vals = list(receptacle_positions[scene_id][recep])

            if len(recep_vals) > 4:  # Too many views around same receptacle can be unneccassary
                np.random.shuffle(recep_vals)
                recep_len = np.random.randint(1, 5)
                recep_vals = recep_vals[:recep_len]

            idx = 0
            for pos_pair in recep_vals:
                pos_pair_lst = list(pos_pair)
                recep_position = np.array(pos_pair_lst[:3])
                view_point_position = np.array(pos_pair_lst[3:]).astype(np.float32)
                start_position, start_rotation, _ = get_robot_spawns(target_positions=view_point_position[None],
                                                                     rotation_perturbation_noise=0,
                                                                     distance_threshold=0,
                                                                     sim=sim,
                                                                     num_spawn_attempts=100,
                                                                     physics_stability_steps=100,
                                                                     orient_positions=recep_position[None],
                                                                     )
                sim.robot.base_pos = start_position
                sim.robot.base_rot = start_rotation
                sim.maybe_update_robot()
                obs = sim.get_sensor_observations()
                plt.imsave(
                    f'./{data_dir}/scene_{scene_id}/ep_{episode.episode_id}/{recep}/img_{idx}.png',
                    obs['robot_head_rgb'][:, :, :3]
                )
                idx += 1


def gen_dataset_question(scene_info_dir: str, data_dir: str, env: HabitatOpenVocabManipEnv):
    """Generates templated Q/A per episode for all scenes"""

    # This is for iterating through all episodes once using only one env
    count_dict = get_init_scene_episode_count_dict(env)

    with open(f'./{scene_info_dir}/recep_position.pickle', 'rb') as handle:
        receptacle_positions = pickle.load(handle)

    # Ideally, we can make it like an iterator to make it feel more intuitive
    while True:
        # Get a new episode
        env.reset()
        episode = env.get_current_episode()
        scene_id = extract_scene_id(episode.scene_id)

        # Check if you have iterated through all episodes and if yes, break the loop
        hash_str = f'ep_{episode.episode_id}_scene_{scene_id}'
        if count_dict[hash_str] == 0:
            count_dict[hash_str] += 1
        elif count_dict[hash_str] == 1:
            break
        else:
            raise NotImplementedError  # Technically, this shouldn't happen

        scene_receptacles = list(receptacle_positions[scene_id].keys())
        scene_receptacles_id = {}

        recep_idx = 0
        for recep in scene_receptacles:
            scene_dir = f'./{data_dir}/scene_{scene_id}/ep_{episode.episode_id}/{recep}/'
            scene_receptacles_id[recep] = []

            for path in os.listdir(scene_dir):
                if os.path.isfile(os.path.join(scene_dir, path)):
                    scene_receptacles_id[recep].append(recep_idx)
                    recep_idx += 1

        for (q_idx, recep) in enumerate(all_receptacles):
            question_str = f"We show images from different locations inside a home. Which location(s) contain {recep.replace('_', ' ')}? You can choose multiple options\n"
            options_str = "Options: "
            for idx in range(recep_idx):
                options_str += f"image {idx}: <tok></tok>, "
            options_str = options_str[:-2] + "\n"
            answer_str = "Answer with image index: "

            if recep in scene_receptacles:
                for idx in scene_receptacles_id[recep]:
                    answer_str += f"<{idx}> "
                answer_str = answer_str[:-1]
            else:
                answer_str += "Not found"

            prompt_str = question_str + options_str + answer_str
            with open(f"./{data_dir}/scene_{scene_id}/ep_{episode.episode_id}/question_{q_idx}.txt", "w") as f_text:
                f_text.write(prompt_str)
                f_text.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation_type",
        type=str,
        choices=["local", "local_vectorized", "remote"],
        default="local",
    )
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--scene_info_dir",
        type=str,
        default="scene_info",
        help="Path to saving scene info",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_out",
        help="Path to saving data",
    )
    parser.add_argument(
        "--env_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/env/hssd_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "overrides",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    # get habitat config
    habitat_config, _ = get_habitat_config(
        args.habitat_config_path, overrides=args.overrides
    )

    # get env config
    env_config = get_omega_config(args.env_config_path)

    # merge habitat and env config to create env config
    env_config = create_env_config(
        habitat_config, env_config, evaluation_type=args.evaluation_type
    )

    # Create an env
    env = create_ovmm_env_fn(env_config)

    # Aggregate receptacles position by scene using all episodes
    receptacle_position_aggregate(args.scene_info_dir, env)

    # Generate images of receptacles by episode
    gen_receptacle_images(args.scene_info_dir, args.data_dir, env)

    # Generate templated Q/A per episode
    gen_dataset_question(args.scene_info_dir, args.data_dir, env)
