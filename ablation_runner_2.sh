#!/bin/bash

python projects/habitat_ovmm/eval_baselines_agent.py --env_config_path projects/habitat_ovmm/configs/env/hssd_eval_v1.yaml habitat.dataset.episode_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,151,182,193]
python projects/habitat_ovmm/eval_baselines_agent.py --env_config_path projects/habitat_ovmm/configs/env/hssd_eval_v1b.yaml habitat.dataset.episode_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,151,182,193]
python projects/habitat_ovmm/eval_baselines_agent.py --env_config_path projects/habitat_ovmm/configs/env/hssd_eval_v2.yaml habitat.dataset.episode_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,151,182,193]
python projects/habitat_ovmm/eval_baselines_agent.py --env_config_path projects/habitat_ovmm/configs/env/hssd_eval_v2b.yaml habitat.dataset.episode_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,151,182,193]
python projects/habitat_ovmm/eval_baselines_agent.py --env_config_path projects/habitat_ovmm/configs/env/hssd_eval_v3.yaml habitat.dataset.episode_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,151,182,193]
python projects/habitat_ovmm/eval_baselines_agent.py --env_config_path projects/habitat_ovmm/configs/env/hssd_eval_v3b.yaml habitat.dataset.episode_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,151,182,193]