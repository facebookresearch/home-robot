# Instance ImageNav on Stretch

This directory contains scripts to demo our modular InstanceImageNav model (Mod-IIN) in Habitat and on a Stretch robot.

In short, our method explores the environment using **frontier-based exploration (FBE)**. A SuperGLue-based **keypoint matching** system is used to detect if the goal instance is visible. Upon positive detection, keypoint correspondences localize the goal instance with the aid of **Detic instance segmentation**. Finally, an **analytic local navigator** conveys the agent to the goal.

## Environment Setup

```bash
git clone --recurse-submodules git@github.com:facebookresearch/home-robot.git

# if you have already cloned home-robot, without the submodules, run:
git submodule update --recursive --init

conda create -n iin_demo python=3.8 cmake=3.14.0
conda activate iin_demo

# -------------------- Habitat Sim v0.2.3 --------------------
git clone --branch v0.2.3 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --headless --with-cuda
cd ..
# ------------------------------------------------------------


# -------------------- Habitat Lab v0.2.3 --------------------
git clone --branch v0.2.3 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..
# ------------------------------------------------------------


# -------------------- Install Home-Robot --------------------
cd src/home_robot
conda env update --file environment.yml --prune
pip install -e .
cd ../src/home_robot_sim
pip install -e .
cd ../..
# ------------------------------------------------------------


# -------------------- Detic + Detectron2 --------------------
# Install Detectron2
pip install -e src/thrid_party/detectron2
# Install Detic requirements
pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt
# download Detic model weights
mkdir -p src/home_robot/home_robot/perception/detection/detic/Detic/models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
  -O src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
# ------------------------------------------------------------


# ----------------------- This Project -----------------------
cd projects/instanceimagenav
pip install -r requirements.txt
mkdir -p data/scene_datasets
# --------------------  SCENE DATASET  -----------------------
#   Dataset: HM3D (w/ Semantics v0.2)
#   Download the HM3D scenes from https://aihabitat.org/datasets/hm3d/ 
#   Alternatively, link a local copy.
#   Extract to data/scene_datasets/hm3d_v0.2/{SPLIT}/...
# ------------------------------------------------------------

mkdir -p data/datasets/instance_imagenav/hm3d/v3
# -------------------- EPISODE DATASET -----------------------
#   Dataset: Habitat ImageNav Challenge Dataset (2023) 
#   Extract to data/datasets/instance_imagenav/hm3d/v3/{SPLIT}/content/{SCENE}.json.gz
cd data/datasets
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip
unzip instance_imagenav_hm3d_v3.zip
mv instance_imagenav_hm3d_v3/* instance_imagenav/hm3d/v3
rm -r instance_imagenav_hm3d_v3 instance_imagenav_hm3d_v3.zip
cd ..
# ------------------------------------------------------------
```

At this point, the simulation demo should run successfully. If you want to run the physical Stretch demo, you will additionally need to install `home_robot_hw` to this environment.

## Run Demo: Simulation

```bash
# in directory: /path/to/home-robot/projects/instanceimagenav
python eval_episode_habitat.py
```

This will generate step-wise images and a video for a single episode. The results are saved to `projects/instanceimagenav/datadump/[images|videos]/debug`.

If the enviornment installation went properly, the resulting video should look something like [this](https://drive.google.com/file/d/1g8RJNdQPGKkYRHWaYc56v5TksmeGR_ra/view?usp=share_link). This episode's metrics were:

```json
{
    "success": 1.00,
    "spl": 0.61,
    "soft_spl": 0.61,
    "distance_to_goal": 0.05,
    "num_steps": 137,
}
```

## Run Demo: Stretch

The InstanceImageNav model is setup to demo on Stretch.

### Collecting Image Goals

The model needs to be provided an image goal depicting an object instance in the physical scene. Capture a 512x512 image and save it in png format. Look at the [InstanceImageNav paper](https://arxiv.org/abs/2211.15876) to see simulation examples of what goal images should look like. Goal images can be saved as `./image_goals/{$GOAL_NAME}.png`.

Update `configs/instance_imagenav_hm3d.yaml` to point to the goal image path:

```yaml
stretch_goal_image_path: image_goals/0_tst_chair.png
```

### Running on Stretch

The episode can now be run:

```bash
# in directory: /path/to/home-robot/projects/instanceimagenav
python eval_episode_stretch.py
```

Step-wise images and a video will be generated. The results are saved to `projects/instanceimagenav/datadump/[images|videos]/debug`.
