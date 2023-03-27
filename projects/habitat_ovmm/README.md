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

git clone https://github.com/3dlg-hcvc/habitat-sim --branch floorplanner
cd habitat-sim
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
```

On Mac:
```
conda create -n home-robot python=3.10 cmake
conda activate home-robot

conda install pytorch torchvision torchaudio -c pytorch

git clone https://github.com/3dlg-hcvc/habitat-sim --branch floorplanner
cd habitat-sim
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
```

**[IMPORTANT]: Add habitat-lab path to PYTHONPATH**:

```
export PYTHONPATH=$PYTHONPATH:/path/to/home-robot-dev/habitat-lab/
```

[TEMPORARY]: Until we port to habitat v0.2.3.

> Comment out L36 in habitat-lab/habitat/tasks/rearrange/rearrange_sim.py

## Dataset Setup

### Scene dataset setup (v0.2.0)

```
wget --no-check-certificate https://aspis.cmpt.sfu.ca/projects/scenebuilder/fphab/v0.2.0/fphab-v0.2.0.zip -O datasets/scene_datasets/fphab-v0.2.0.zip
unzip datasets/scene_datasets/fphab-v0.2.0.zip -d datasets/scene_datasets/
mkdir -p datasets/scene_datasets/floorplanner
mv datasets/scene_datasets/fphab-v0.2.0 datasets/scene_datasets/floorplanner/v0.2.0
```

The google scanned objects and amazon berkeley objects will need to be in `data/objects/google_object_dataset` and `data/objects/amazon_berkeley` respectively.

### Other instructions

Rough notes; some things were missing for configuring a new environment:
  - Download the objects into `HOME_ROBOT_ROOT/data`
  - Download the urdf into `HOME_ROBOT_ROOT/data/robots/hab_stretch` - robot should be at `data/robots/hab_stretch/urdf/` - robot from [the FAIR distribution here in zip format](http://dl.fbaipublicfiles.com/habitat/robots/hab_stretch_v1.0.zip)


### Episode dataset setup

```
mkdir -p datasets/episode_datasets/floorplanner/indoor_only/
wget https://www.dropbox.com/s/n1g1s6uvowo4tbm/v0.2.0_receptacle_cat_indoor_only_val.zip -O datasets/episode_datasets/floorplanner/indoor_only/v0.2.0_receptacle_cat_indoor_only_val.zip
unzip datasets/episode_datasets/floorplanner/indoor_only/v0.2.0_receptacle_cat_indoor_only_val.zip -d datasets/episode_datasets/floorplanner/indoor_only/
```

## Create episode symlink

For example:
```
ln -s ~/src/habitat-lab/data/datasets/floorplanner/v0.2.0/ ~/src/home-robot/data/datasets/floorplanner/v0.2.0
```


## Demo setup

Run
```
python eval_episode.py
```

Results are saved to `datadump/images/debug`.

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

# Single episode to debug (ensuring)
export HABITAT_SIM_LOG=quiet
python project/habitat_ovmm/eval_episode.py

# Vectorized evaluation
sbatch eval_vectorized.sh --config_path configs/agent/floorplanner_eval.yaml
```
