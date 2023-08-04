import os
import sys
os.chdir('/private/home/sergioarnaud/ovmm/')
sys.path.append('projects/habitat_ovmm/')
from utils.config_utils import (
    create_agent_config,
    create_env_config,
    get_habitat_config,
    get_omega_config,
)
from utils.env_utils import create_ovmm_env_fn
from home_robot.agent.ovmm_agent.random_agent import RandomAgent
from home_robot.agent.ovmm_agent.motor_skills import OracleNavSkill, OraclePickSkill, OraclePlaceSkill
import numpy as np
# omega conf load
from omegaconf import OmegaConf
import torch
from tqdm import tqdm


habitat_config_path = 'ovmm/ovmm_eval.yaml'
env_config_path = 'projects/habitat_ovmm/configs/env/hssd_eval.yaml'
baseline_config_path = 'projects/habitat_ovmm/configs/agent/heuristic_agent.yaml'

overrides = []

# get habitat config
habitat_config, _ = get_habitat_config(habitat_config_path, overrides=overrides)

# get baseline config
baseline_config = get_omega_config(baseline_config_path)

# get env config
env_config = get_omega_config(env_config_path)

# merge habitat and env config to create env config
env_config = create_env_config(habitat_config, env_config)

# merge env config and baseline config to create agent config
agent_config = create_agent_config(env_config, baseline_config)
env = create_ovmm_env_fn(env_config)
env.habitat_env.env._env._env._sim
agent = RandomAgent(agent_config)
env.reset()

num_episodes = 1200
count_episodes = 0
pbar = tqdm(total=num_episodes)
for _ in range(len(os.listdir('results_oracle_spot_27_jul_run1/movies'))):
    count_episodes += 1
    env.reset()
    pbar.update(1)

while count_episodes < num_episodes:

    path = 'src/home_robot/home_robot/agent/ovmm_agent/motor_skills/conf/motor_skills/oracle_nav.yaml'
    conf_nav = OmegaConf.load(path)
    conf_nav = conf_nav.oracle_nav.skill_config

    path = 'src/home_robot/home_robot/agent/ovmm_agent/motor_skills/conf/motor_skills/oracle_pick.yaml'
    conf_pick = OmegaConf.load(path)
    conf_pick = conf_pick.oracle_pick.skill_config

    path = 'src/home_robot/home_robot/agent/ovmm_agent/motor_skills/conf/motor_skills/oracle_place.yaml'
    conf_place = OmegaConf.load(path)
    conf_place = conf_place.oracle_place.skill_config

    observations, done = env.reset(), False
    e = env.habitat_env.env._env._env
    action, info, other = agent.act(observations)

    nav = OracleNavSkill(
        conf_nav,
        env.observation_space,
        env.original_action_space,
        1,
        e,
    )
    pick = OraclePickSkill(
        conf_pick,
        env.observation_space,
        env.original_action_space,
        1,
        e,
    )
    place = OraclePlaceSkill(
        conf_place,
        env.observation_space,
        env.original_action_space,
        1,
        e,
    )
    a = torch.zeros(1, 11, device='cpu', dtype=torch.bool,)
    e.sim._load_navmesh(e.sim.ep_info)

    imgs = []
    nav.target_pos = env.habitat_env.env._env._env._sim.ep_info.candidate_objects[0].position
    nav.on_enter([0], None)
    # env.habitat_env.env._env._env._sim.ep_info.candidate_start_receps
    for k in range(1000):
        action, done_1 = nav.act({}, a, a, a, [0])
        ac = action[0] 
        ac = ac.numpy().astype(dtype=np.float32)
        observations, done, haba_info = env.apply_action(ac, info)
        imgs.append(observations.third_person_image)
        # if k % 10 == 0:
        #     print('obj to pick', haba_info['ovmm_dist_to_pick_goal'])
        if done_1:
            break
    n1tm = nav.termination_message
    pick.on_enter([0], None)
    pick.target_pos = env.habitat_env.env._env._env._sim.ep_info.candidate_objects[0].position
    action, done_1 = pick.act({}, a, a, a, [0])
    pitm = pick.termination_message

    posible_receptacles = env.habitat_env.env._env._env._sim.ep_info.candidate_goal_receps

    # sample one
    import random
    receptacle = random.choice(posible_receptacles)
    nav.target_pos = receptacle.position
    for k in range(1000):
        action, done_1 = nav.act({}, a, a, a, [0])
        ac = action[0] 
        ac = ac.numpy().astype(dtype=np.float32)
        observations, done, haba_info = env.apply_action(ac, info)
        imgs.append(observations.third_person_image)
        # if k % 10 == 0:
        #     print('obj to place', haba_info['ovmm_object_to_place_goal_distance.0'])
        if done_1:
            break
    n2tm = nav.termination_message

    place.on_enter([0], None)
    place.target_pos = receptacle.position
    action, done_1 = place.act({}, a, a, a, [0])
    pltm = place.termination_message

    for _ in range(5):
        final_action = np.zeros(11).astype(dtype=np.float32)
        observations, done, haba_info = env.apply_action(final_action, info)
        imgs.append(observations.third_person_image)

    # make movie 
    import imageio
    ep = env.get_current_episode().episode_id
    ep = str(ep).zfill(4)
    imageio.mimsave(f'results_oracle_spot_27_jul_run1/movies/movie_{ep}.mp4', imgs, fps=30)
    pbar.update(1)
    count_episodes += 1
    x = env.habitat_env.env._env._env._sim.ep_info.candidate_objects[0].object_category
    y = env.habitat_env.env._env._env._sim.ep_info.candidate_start_receps[0].object_category
    z = env.habitat_env.env._env._env._sim.ep_info.candidate_goal_receps[0].object_category
    haba_info['instr'] = f'Move {x} from {y} to {z}'
    haba_info['nav_1_termination_message'] = n1tm
    haba_info['nav_2_termination_message'] = n2tm
    haba_info['pick_termination_message'] = pitm
    haba_info['place_termination_message'] = pltm
    # write haba info
    with open(f'results_oracle_spot_27_jul_run1/results/res_{ep}.json', 'w') as f:
        f.write(str(haba_info))