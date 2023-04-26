## Table of contents
   1. [Environment setup](#environment-setup)
   2. [Dataset setup](#dataset-setup)
   3. [Demo setup](#demo-setup)
   4. [DETIC setup](#install-detic)
   5. [Run!](#run)

## Environment Setup

On an Ubuntu machine with GPU:
```
conda env create -n home-robot --file=src/home_robot/environment.yml
conda activate home-robot

git clone https://github.com/facebookresearch/habitat-sim
cd habitat-sim
git checkout 7b99db753272079d609b88e00f24ca0ad0ef23aa # latest main forces Python > 3.9
python -m pip install -r requirements.txt
python setup.py install --headless --with-bullet
# (if the above commands runs out of memory) 
# python setup.py build_ext --parallel 8 install --headless
cd ..

git clone --branch modular_nav_obj_on_rec https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
python -m pip install -e ./habitat-baselines
cd habitat-lab
python -m pip install -r requirements.txt
python -m pip install -e .
cd ../..

python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

On Mac:
```
conda create -n home-robot python=3.10 cmake
conda activate home-robot

conda install -y pytorch torchvision -c pytorch

git clone https://github.com/facebookresearch/habitat-sim
cd habitat-sim
git checkout 7b99db753272079d609b88e00f24ca0ad0ef23aa # latest main forces Python > 3.9
pip install -r requirements.txt
python setup.py install --with-bullet
cd ..

git clone --branch modular_nav_obj_on_rec https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-baselines
cd habitat-lab
pip install -r requirements.txt
# Not clear if this should have --all or just be a pip install .
python setup.py develop
cd ../..

pip install natsort scikit-image scikit-fmm pandas trimesh scikit-learn
conda install -c pytorch3d pytorch3d
```

## Dataset Setup

### Scene dataset setup 

```
cd `HOME_ROBOT_ROOT/data/`
# Download the scenes
git clone https://huggingface.co/datasets/osmm/fpss --branch osmm
# Download the objects and metadata
git clone https://huggingface.co/datasets/osmm/objects
```

### Download the Robot Model

Download and unzip the robot model:
```
mkdir -p $HOME_ROBOT_ROOT/data/robots/hab_stretch
cd $HOME_ROBOT_ROOT/data/robots/hab_stretch
wget http://dl.fbaipublicfiles.com/habitat/robots/hab_stretch_v1.0.zip
unzip hab_stretch_v1.0.zip
```

### Download the Episodes

These describe where objects are and where the robot starts:
```
cd $HOME_ROBOT_ROOT/data/
git clone https://huggingface.co/datasets/osmm/episodes
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
