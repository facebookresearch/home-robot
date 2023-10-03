#!/bin/bash

python projects/habitat_ovmm/eval_baselines_agent.py --env_config_path projects/habitat_ovmm/configs/env/hssd_eval_v6.yaml habitat.dataset.episode_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,151,182,193]
python projects/habitat_ovmm/eval_baselines_agent.py --env_config_path projects/habitat_ovmm/configs/env/hssd_eval_v6b.yaml habitat.dataset.episode_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,151,182,193]
python projects/habitat_ovmm/eval_baselines_agent.py --env_config_path projects/habitat_ovmm/configs/env/hssd_eval_v7.yaml habitat.dataset.episode_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,151,182,193]
