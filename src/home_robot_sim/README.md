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
# Build habitat with bullet physics
python setup.py install --bullet
```

4. Install dependencies.
```
cd -
pip install -e src/third_party/habitat-lab/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-baselines
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"

conda install -c pytorch torchvision
conda install -c anaconda scikit-image
conda install pybind11 -c conda-forge
pip install sophuspy
pip install scikit-fmm
pip install -U scikit-learn

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
python projects/habitat_ovmm/eval_vectorized.py
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

## Run

> Note: Ensure `GROUND_TRUTH_SEMANTICS:0` in `configs/agent/floorplanner_eval.yaml` to test DETIC perception.

```
cd /path/to/home-robot


# Evaluation on complete episode dataset with GT semantics
python projects/habitat_ovmm/eval_vectorized.py

# Evaluation on complete episode dataset with DETIC
python projects/habitat_ovmm/eval_vectorized.py  --baseline_config_path projects/habitat_ovmm/configs/agent/floorplanner_detic_eval.yaml

# Evaluation on specific episodes
python projects/habitat_ovmm/eval_vectorized.py habitat.dataset.episode_ids="[151,182]"
```
