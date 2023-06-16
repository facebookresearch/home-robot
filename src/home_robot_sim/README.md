## Table of contents
   1. [Environment setup](#environment-setup)
   2. [Dataset setup](#dataset-setup)
   3. [Demo setup](#demo-setup)
   4. [DETIC setup](#install-detic)
   5. [Run!](#run)

## Environment Setup

### On an Ubuntu machine with GPU:

1. Clone this github repository.

```
git clone git@github.com:facebookresearch/home-robot.git
cd home-robot
```

2. Setup virtual environment.

```
conda env create -n home-robot --file=src/home_robot/environment.yml
conda activate home-robot

# Download third-party packages
git submodule update --init --recursive

cd src/third_party/habitat-sim
```

3. Build habitat-sim from source.

```
pip install -r requirements.txt

sudo apt-get update || true
# These are fairly ubiquitous packages and your system likely has them already,
# but if not, let's get the essentials for EGL support:
sudo apt-get install -y --no-install-recommends \
     libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev

# Build habitat with bullet physics
python setup.py install --bullet
```

4. Install dependencies.
```
cd -
pip install -e src/third_party/habitat-lab/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-baselines
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Install the core home_robot package
pip install -e src/home_robot

# Install home_robot_hw
pip install -e src/home_robot_hw

# Install home robot sim interfaces
pip install -e src/home_robot_sim
```

### On Mac:
```
conda create -n home-robot python=3.10 cmake
conda activate home-robot

conda install -y pytorch torchvision -c pytorch

conda activate home-robot

# Download third-party packages
git submodule update --init --recursive

cd src/third_party/habitat-sim
```
Follow the instructions [here](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md) to build habitat-sim from source.
```
cd -

pip install -e src/third_party/habitat-lab/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-baselines

pip install natsort scikit-image scikit-fmm pandas trimesh scikit-learn
conda install -c pytorch3d pytorch3d
```

## Dataset Setup

### Scene dataset setup 

```
mkdir data && cd data

# Download the scenes
git clone https://huggingface.co/datasets/fpss/fphab --branch ovmm

# Download the objects and metadata
git clone https://huggingface.co/datasets/osmm/objects
```

### Download the Episodes

These describe where objects are and where the robot starts:
```
git clone https://huggingface.co/datasets/osmm/episodes
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

## Install Detic

```sh
git submodule update --init --recursive src/third_party/detectron2 src/home_robot/home_robot/perception/detection/detic/Detic
pip install -e src/third_party/detectron2

cd src/home_robot/home_robot/perception/detection/detic/Detic
pip install -r requirements.txt

mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

# Test it with
wget https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg
python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out.jpg --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

cd -
```

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
   --exp-config habitat-baselines/habitat_baselines/config/rearrange/rl_skill.yaml \
   --run-type train benchmark/rearrange=<skill_name> \
   habitat_baselines.checkpoint_folder=data/new_checkpoints/ovmm/<skill_name>
```
Here `<skill_name>` should be one of `cat_gaze`, `cat_place`, `cat_nav_to_obj` or `cat_nav_to_rec`.

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
   --exp-config habitat-baselines/habitat_baselines/config/rearrange/rl_skill.yaml \
   --run-type train benchmark/rearrange=<skill_name> \
   habitat_baselines.checkpoint_folder=data/new_checkpoints/ovmm/<skill_name>
```


## Run evaluation

```
cd /path/to/home-robot


# Evaluation on complete episode dataset with GT semantics
python projects/habitat_ovmm/eval_dataset.py

# Evaluation on complete episode dataset with DETIC
Ensure `GROUND_TRUTH_SEMANTICS:0` in `configs/agent/hssd_eval.yaml` before running the above command

# Evaluation on specific episodes
python projects/habitat_ovmm/eval_dataset.py habitat.dataset.episode_ids="[151,182]"
```
