## Table of contents
   1. [Environment setup](#environment-setup)
   2. [Dataset setup](#dataset-setup)
   3. [Run evaluation](#run-evaluation)
   4. [DETIC setup](#install-detic)

## Environment Setup

```
conda create -n home-robot python=3.10 cmake pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda activate home-robot

git submodule update --init src/third_party/habitat-lab src/third_party/habitat-sim

cd src/third_party/habitat-sim
pip install -r requirements.txt
python setup.py install --headless --bullet
# or `python setup.py build_ext --parallel 8 install --headless` (if the above command runs out of memory)


cd ../habitat-lab
git checkout home-robot_objectnav_support
pip install -e habitat-lab
pip install -e habitat-baselines
cd ../../..

pip install scikit-learn scikit-image sophuspy scikit-fmm pandas
# sophuspy might require `conda install -c conda-forge pybind11`

```

## Dataset Setup

### HM3D

#### Scene dataset

Download to `data/scene_datasets/hm3d_v0.2` through instructions specified [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#downloading-hm3d-with-the-download-utility).

#### Episode dataset
```sh
mkdir -p data/datasets/objectnav/hm3d
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip -O data/datasets/objectnav/hm3d/objectnav_hm3d_v2.zip
unzip data/datasets/objectnav/hm3d/objectnav_hm3d_v2.zip -d data/datasets/objectnav/hm3d
mv data/datasets/objectnav/hm3d/objectnav_hm3d_v2 data/datasets/objectnav/hm3d/v2
```

### Floorplanner (OUTDATED)

#### Scene dataset (v0.2.0)

```
wget --no-check-certificate https://aspis.cmpt.sfu.ca/projects/scenebuilder/fphab/v0.2.0/fphab-v0.2.0.zip -O datasets/scene_datasets/fphab-v0.2.0.zip
unzip datasets/scene_datasets/fphab-v0.2.0.zip -d datasets/scene_datasets/
mkdir -p datasets/scene_datasets/floorplanner
mv datasets/scene_datasets/fphab-v0.2.0 datasets/scene_datasets/floorplanner/v0.2.0
```

#### Episode dataset

```
mkdir -p datasets/episode_datasets/floorplanner/indoor_only/
wget https://www.dropbox.com/s/eex8v8un6r6ru3z/v0.2.0_receptacle_cat_indoor_only_val.zip -O datasets/episode_datasets/floorplanner/indoor_only/v0.2.0_receptacle_cat_indoor_only_val.zip
unzip datasets/episode_datasets/floorplanner/indoor_only/v0.2.0_receptacle_cat_indoor_only_val.zip -d datasets/episode_datasets/floorplanner/indoor_only/
```

[TEMPORARY] Floorplanner dataset episodes need to point to the right scene dataset config for scenes to load correctly:

> Add the below line after L93 of `habitat-lab/habitat/core/env.py`

```
self.current_episode.scene_dataset_config = "/path/to/datasets/scene_datasets/floorplanner/v0.2.0/hab-fp.scene_dataset_config.json"
```


## Run evaluation!

Run
```
python projects/habitat_objectnav/eval_episode.py
```

Results are saved to `datadump/images/debug`.

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

