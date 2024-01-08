# Habitat OVMM

## Table of contents
   1. [Dataset setup](#dataset-setup)
   2. [Demo setup](#demo-setup)
   3. [Training DD-PPO skills](#training-dd-ppo-skills)
   4. [Running evaluations](#running-evaluations)


## Dataset Setup

Run `git lfs install` to install Git LFS, which is used to manage the OVMM dataset. Please sign in [here](https://huggingface.co/datasets/hssd/hssd-hab/tree/ovmm) and accept the license for using HSSD scenes.

You can then use the following command to download data:
```
$HOME_ROBOT_ROOT/projects/habitat_ovmm/install.sh
```

### Detailed Explanation

If anything goes wrong, you can check out this explanation of the different steps.

#### Scene dataset setup 

Please sign in [here](https://huggingface.co/datasets/hssd/hssd-hab/tree/ovmm) and accept the license for using HSSD scenes before proceeding to download them using the [data download script](../../download_data.sh). You will need your login information to proceed.
```
cd $HOME_ROBOT_ROOT
./download_data.sh
```

If this didn't trigger a download of the datasets, you may be running an older version of git. Either upgrade your git version, or try the following commands:
```
cd data/hssd-hab
git lfs pull
cd -

cd data/objects
git lfs pull
cd -
```

#### Download the Episodes

These describe where objects are and where the robot starts:

```
git submodule update --init data/datasets/ovmm
```

Similar to the scene dataset setup, you may need to run the following commands if this didn't download the episodes:
```
cd data/datasets/ovmm
git lfs pull
cd -
```


#### Download the Robot Model

Download and unzip the robot model:
```
mkdir -p data/robots/hab_stretch
cd data/robots/hab_stretch

wget http://dl.fbaipublicfiles.com/habitat/robots/hab_stretch_v1.0.zip
unzip hab_stretch_v1.0.zip
```

## Demo setup

Run
```
python projects/habitat_ovmm/eval_baselines_agent.py --env_config projects/habitat_ovmm/configs/env/hssd_demo.yaml
```

Results are saved to `datadump/images/eval_hssd/`.


## Training DD-PPO skills

First setup data directory
```
cd /path/to/home-robot/src/third_party/habitat-lab/

# create soft link to data/ directory
ln -s /path/to/home-robot/data data
```

To train on a single machine use the following script:
```
#/bin/bash

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

set -x
python -u -m habitat_baselines.run \
   --exp-config habitat-baselines/habitat_baselines/config/ovmm/rl_skill.yaml \
   --run-type train benchmark/ovmm=<skill_name> \
   habitat_baselines.checkpoint_folder=data/new_checkpoints/ovmm/<skill_name>
```
Here `<skill_name>` should be one of `gaze`, `place`, `nav_to_obj` or `nav_to_rec`.

To run on a cluster with SLURM using distributed training run the following script. While this is not necessary, if you have access to a cluster, it can significantly speed up training. To run on multiple machines in a SLURM cluster run the following script: change `#SBATCH --nodes $NUM_OF_MACHINES` to the number of machines and `#SBATCH --ntasks-per-node $NUM_OF_GPUS` and `$SBATCH --gres $NUM_OF_GPUS` to specify the number of GPUS to use per requested machine.

```
#!/bin/bash
#SBATCH --job-name=ddppo
#SBATCH --output=logs.ddppo.out
#SBATCH --error=logs.ddppo.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --mem=60GB
#SBATCH --time=12:00
#SBATCH --signal=USR1@600
#SBATCH --partition=dev

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
set -x
python -u -m habitat_baselines.run \
   --exp-config habitat-baselines/habitat_baselines/config/ovmm/rl_skill.yaml \
   --run-type train benchmark/ovmm=<skill_name> \
   habitat_baselines.checkpoint_folder=data/new_checkpoints/ovmm/<skill_name>
```


## Running evaluations


### Evaluate with ground truth semantics
```
# Evaluation on complete episode dataset with GT semantics
python projects/habitat_ovmm/eval_baselines_agent.py

# Print out the metrics
python projects/habitat_ovmm/scripts/summarize_metrics.py
```

### Evaluate with DETIC
Ensure `GROUND_TRUTH_SEMANTICS:0` in `configs/env/hssd_eval.yaml` before running the above command

### Evaluate on specific episodes
```
python projects/habitat_ovmm/eval_baselines_agent.py habitat.dataset.episode_ids="[151,182]"
```

### Evaluate all baseline variants
1. First generate all possible configs using the base config `configs/agent/hssd_eval.yaml`. Configs will be saved under `projects/habitat_ovmm/configs/agent/generated`
```
python projects/habitat_ovmm/scripts/gen_configs.py
```

2. Run evaluation using the generated config files
```
python projects/habitat_ovmm/eval_baselines_agent.py --baseline_config_path projects/habitat_ovmm/configs/agent/generated/<dir_name>/<manip>_m_<nav>_n_<perception><viz?>.yaml
```

Here `<manip>/<nav>` are to be set to 'h' or 'r' for heuristic and RL skills respectively. `<perception>` is one of 'gt'/'detic'. Append `<viz?>=_viz` for saving images.

In case you run into issues, please prepend your python command with `HABITAT_ENV_DEBUG=1` to get a better error message.
