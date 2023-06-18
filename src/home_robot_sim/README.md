## Table of contents
   1. [Environment setup](#environment-setup)
   2. [Dataset setup](#dataset-setup)
   3. [Demo setup](#demo-setup)
   4. [Run!](#run)

## Environment Setup

These setup instructions are meant to be followed after reaching step 7 in the main [README.md](../../README.md) file. If you haven't completed those instructions yet, please refer to the main [README.md](../../README.md) and complete the steps mentioned there before continuing.

### On an Ubuntu machine with GPU:

1. Install `habitat_sim` and other dependencies

```
mamba env update -f src/home_robot_sim/environment.yml
pip install "git+https://github.com/facebookresearch/habitat-sim.git@ovmm_challenge_2023"
```

2. Install dependencies.
```
pip install -e src/third_party/habitat-lab/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-baselines
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

3. Install home_robot_sim library
```
# Install home robot sim interfaces
pip install -e src/home_robot_sim
```


## Dataset Setup

### Scene dataset setup 

```
mkdir data && cd data

# Download the scenes
git clone https://huggingface.co/datasets/fpss/fphab --branch ovmm-old-scenes

# Download the objects and metadata
git clone https://huggingface.co/datasets/osmm/objects --branch release
```

### Download the Episodes

These describe where objects are and where the robot starts:
```
git clone https://huggingface.co/datasets/osmm/episodes --branch release
```

### Download the Robot Model

Download and unzip the robot model:
```
mkdir -p robots/hab_stretch
cd robots/hab_stretch

wget http://dl.fbaipublicfiles.com/habitat/robots/hab_stretch_v1.0.zip
unzip hab_stretch_v1.0.zip
```

## Demo setup

Run
```
python projects/habitat_ovmm/eval_dataset.py
```

Results are saved to `datadump/images/eval_floorplanner/`.


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


## Run

```
cd /path/to/home-robot


# Evaluation on complete episode dataset with GT semantics
python projects/habitat_ovmm/eval_dataset.py

# Evaluation on complete episode dataset with DETIC
Ensure `GROUND_TRUTH_SEMANTICS:0` in `configs/agent/hssd_eval.yaml` before running the above command

# Evaluation on specific episodes
python projects/habitat_ovmm/eval_dataset.py habitat.dataset.episode_ids="[151,182]"

# Evaluating all baseline variants
# 1. First generate all possible configs using the base config `configs/agent/hssd_eval.yaml`. Configs will be saved under `projects/habitat_ovmm/configs/agent/generated`
python projects/habitat_ovmm/scripts/gen_configs.py

# 2. Run evaluation using the generated config files
python projects/habitat_ovmm/eval_dataset.py --baseline_config_path projects/habitat_ovmm/configs/agent/generated/<dir_name>/<manip>_m_<nav>_n_<perception><viz?>.yaml
Here <manip>/<nav> are to be set to 'h' or 'r' for heuristic and RL skills respectively. <perception> is one of 'gt'/'detic'. Append <viz?>='_viz' for saving images.

```
