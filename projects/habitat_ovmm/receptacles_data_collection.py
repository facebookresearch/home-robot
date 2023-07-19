import argparse
import os
import pickle
import numpy as np
from copy import deepcopy

from config_utils import get_habitat_config, get_ovmm_baseline_config
from evaluator import OVMMEvaluator
from omegaconf import DictConfig, OmegaConf
from habitat.tasks.rearrange.utils import get_robot_spawns
from matplotlib import pyplot as plt

from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from evaluator import create_ovmm_env_fn

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

all_receptacles = ['cabinet', 'stool', 'trunk', 'shoe_rack', 'chest_of_drawers', 'table', 'toilet', 'serving_cart', 'bed', 'washer_dryer', 'hamper', 'stand', 'bathtub', 'couch', 'counter', 'shelves', 'chair', 'bench']

def extract_scene_id(scene_id):
    before, _ = scene_id.split('.scene')
    _, after = before.split('uncluttered/')
    return after

def merge_configs(habitat_config, baseline_config):
    config = DictConfig({**habitat_config, **baseline_config})

    visualize = config.VISUALIZE or config.PRINT_IMAGES
    if not visualize:
        if "robot_third_rgb" in config.habitat.gym.obs_keys:
            config.habitat.gym.obs_keys.remove("robot_third_rgb")
        if "third_rgb_sensor" in config.habitat.simulator.agents.main_agent.sim_sensors:
            config.habitat.simulator.agents.main_agent.sim_sensors.pop(
                "third_rgb_sensor"
            )

    episode_ids_range = config.habitat.dataset.episode_indices_range
    if episode_ids_range is not None:
        config.EXP_NAME = os.path.join(
            config.EXP_NAME, f"{episode_ids_range[0]}_{episode_ids_range[1]}"
        )

    OmegaConf.set_readonly(config, True)
    return config

def receptacle_position_aggregate(env):
    # This is for iterating through all episodes once using only one env
    count_dict = {}
    for episode in env._dataset.episodes:
        scene_id = extract_scene_id(episode.scene_id)
        hash_str = f'ep_{episode.episode_id}_scene_{scene_id}'
        count_dict[hash_str] = 0
        
    receptacle_positions = dict()    

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
            raise NotImplementedError # Technically, this shouldn't happen

        if not scene_id in receptacle_positions:
            receptacle_positions[scene_id] = dict()

        if not episode.goal_recep_category in receptacle_positions[scene_id]:
            receptacle_positions[scene_id][episode.goal_recep_category] = set() 
        for recep in episode.candidate_goal_receps:
            recep_position = list(recep.position)
            view_point_position = list(recep.view_points[0].agent_state.position)
            receptacle_positions[scene_id][episode.goal_recep_category].add(tuple(recep_position + view_point_position))    

    os.makedirs('./scene_info', exist_ok=True)
    with open('./scene_info/recep_position.pickle', 'wb') as handle:
        pickle.dump(receptacle_positions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return receptacle_positions

def gen_receptacle_images(env):
    sim = env.habitat_env.env._env._env._sim 
    
    # This is for iterating through all episodes once using only one env
    # Also, creating folders for storing dataset
    count_dict = {}
    os.makedirs('./data_out', exist_ok=True)
    for episode in env._dataset.episodes:
        scene_id = extract_scene_id(episode.scene_id)
        os.makedirs(f'./data_out/scene_{scene_id}', exist_ok=True)
        hash_str = f'ep_{episode.episode_id}_scene_{scene_id}'
        count_dict[hash_str] = 0

    with open('./scene_info/recep_position.pickle', 'rb') as handle:
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
            raise NotImplementedError #Technically, this shouldn't happen

        os.makedirs(f'./data_out/scene_{scene_id}/ep_{episode.episode_id}', exist_ok=True)

        for recep in receptacle_positions[scene_id]:
            os.makedirs(f'./data_out/scene_{scene_id}/ep_{episode.episode_id}/{recep}', exist_ok=True)
            recep_vals = list(receptacle_positions[scene_id][recep])
            
            if len(recep_vals) > 4: # Too many views around same receptacle can be unneccassary 
                np.random.shuffle(recep_vals)
                recep_len = np.random.randint(1,5)
                recep_vals = recep_vals[:recep_len]
            
            idx = 0
            for pos_pair in recep_vals:
                pos_pair_lst = list(pos_pair)
                recep_position = np.array(pos_pair_lst[:3])
                view_point_position = np.array(pos_pair_lst[3:]).astype(np.float32)
                start_position, start_rotation, success = get_robot_spawns(target_positions=view_point_position[None], 
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
                plt.imsave(f'./data_out/scene_{scene_id}/ep_{episode.episode_id}/{recep}/img_{idx}.png', obs['robot_head_rgb'][:,:,:3])
                idx += 1

def gen_dataset_question(env):
    # This is for iterating through all episodes once using only one env
    count_dict = {}
    for episode in env._dataset.episodes:
        scene_id = extract_scene_id(episode.scene_id)
        hash_str = f'ep_{episode.episode_id}_scene_{scene_id}'
        count_dict[hash_str] = 0
    
    with open('./scene_info/recep_position.pickle', 'rb') as handle:
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
            raise NotImplementedError #Technically, this shouldn't happen

        scene_receptacles = list(receptacle_positions[scene_id].keys())
        scene_receptacles_id = dict()

        recep_idx = 0
        for recep in scene_receptacles:
            scene_dir = f'./data_out/scene_{scene_id}/ep_{episode.episode_id}/{recep}/'
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
            text_file = open(f"./data_out/scene_{scene_id}/ep_{episode.episode_id}/question_{q_idx}.txt", "w")
            text_file.write(prompt_str)
            text_file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, default="local", choices=["local", "remote"]
    )
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/agent/hssd_eval.yaml",
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
    habitat_config_path = os.environ.get(
        "CHALLENGE_CONFIG_FILE", args.habitat_config_path
    )
    habitat_config, _ = get_habitat_config(
        habitat_config_path, overrides=args.overrides
    )

    # get baseline config
    baseline_config = get_ovmm_baseline_config(args.baseline_config_path)

    # merge habitat and baseline configs
    eval_config = merge_configs(habitat_config, baseline_config)
    env = create_ovmm_env_fn(eval_config)

    # Aggregate receptacles position by scene using all episodes
    receptacle_position_aggregate(env)

    # Generate images of receptacles by episode
    gen_receptacle_images(env)

    # Generate templated Q/A per episode
    gen_dataset_question(env)




    

