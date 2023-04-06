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

**[IMPORTANT]: Add habitat-lab path to PYTHONPATH**:

```
export PYTHONPATH=$PYTHONPATH:/path/to/home-robot-dev/habitat-lab/
```

## Dataset Setup

### Scene dataset setup 

```
cd `HOME_ROBOT_ROOT/data/`
git clone https://huggingface.co/datasets/osmm/fpss --branch osmm
```

The google scanned objects and amazon berkeley objects will need to be in `data/objects/google_object_dataset` and `data/objects/amazon_berkeley` respectively. These datasets can be downloaded from [here](https://drive.google.com/drive/u/0/folders/1Qs99bMMC7ZpZwksZYDC_IkNqK_IB6ONU). They are also available on Skynet at: `/srv/flash1/aramacha35/habitat-lab/data/objects`.

TODO: Download these using git clone https://huggingface.co/datasets/osmm/objects

### Other instructions

Rough notes; some things were missing for configuring a new environment:
  - Download the objects into `HOME_ROBOT_ROOT/data`
  - Download the urdf into `HOME_ROBOT_ROOT/data/robots/hab_stretch` - robot should be at `data/robots/hab_stretch/urdf/` - robot from [the FAIR distribution here in zip format](http://dl.fbaipublicfiles.com/habitat/robots/hab_stretch_v1.0.zip)


### Episode dataset setup
```

cd `HOME_ROBOT_ROOT/data/`
git clone https://huggingface.co/datasets/osmm/episodes
```

### Download CLIP embeddings
Download from `https://drive.google.com/file/d/1sSDSKZgYeIPPk8OM4oWhLtAf4Z-zjAVy/view?usp=sharing` and place them under `HOME_ROBOT_ROOT/data/objects` directory.

TODO: Remove this after we start downloading `objects` folder from huggingface.

## Demo setup

Run
```
python projects/habitat_ovmm/eval_vectorized.py
```

Results are saved to `datadump/images/eval_floorplanner/`.

## Install Detic
TODO Fix these instructions to start by downloading submodule
```
cd /path/to/home-robot-dev/src/home_robot/home_robot/agent/perception/detection/detic
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install -r requirements.txt
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

# Test it with
wget https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg
python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out.jpg --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

## Run

> Note: Ensure `GROUND_TRUTH_SEMANTICS:0` in `configs/agent/floorplanner_eval.yaml` to test DETIC perception.

```
cd /path/to/home-robot


# Evaluation on complete episode dataset
python projects/habitat_ovmm/eval_vectorized.py

# Evaluation on specific episodes
python projects/habitat_ovmm/eval_vectorized.py habitat.dataset.episode_ids="[151,182]"
```
