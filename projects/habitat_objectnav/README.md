# Habitat ObjectNav

Please follow the setup instructions on main [README](../../README.md) and [home-robot-sim](../../src/home_robot_sim/README.md)

## Table of contents
   1. [Environment setup](#environment-setup)
   2. [Dataset setup](#dataset-setup)
   3. [Run evaluation](#run-evaluation)

## Environment Setup

```
cd src/third_party/habitat-lab
git checkout home-robot_objectnav_support
pip install -e habitat-lab
pip install -e habitat-baselines
cd ../../..
```

## Dataset Setup

### HM3D

#### Scene dataset

Download to `data/scene_datasets/hm3d_v0.2` through instructions specified [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#downloading-hm3d-with-the-download-utility).

#### Episode dataset
Run the following command to download episodes and to place them in the right location
```
./projects/habitat_objectnav/download_hm3d_episodes.sh
```

## Run evaluation!

Run
```
python projects/habitat_objectnav/eval_episode.py
```

Results are saved to `datadump/images/eval_hm3d`.

