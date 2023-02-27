# Instance ImageNav on Stretch

This directory contains scripts to demo our modular InstanceImageNav model in Habitat and on a Stretch robot.

In short, our method explores the environment using **frontier-based exploration (FBE)**. A SuperGLue-based **keypoint matching** system is used to detect if the goal instance is visible. Upon positive detection, keypoint correspondences localize the goal instance with the aid of **Detic instance segmentation**. Finally, an **analytic local navigator** conveys the agent to the goal.

## Environment Setup

These are the installation directions that work for me.

Components:

- Habitat v0.2.3 (Sim & Lab)
- Detic
- SuperGlue
- HM3D Scene Dataset
- 2023 Habitat Instance ImageNav Challenge Dataset

```bash
git clone -b imagenav --recurse-submodules git@github.com:jacobkrantz/home-robot.git

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


cd home-robot


# -------------------- Detic + Detectron2 --------------------
# Install a compatible pytorch.
#   NOTE: I'm using cuda version 10.2 (cu102).
#     replace with your version, as long as its detectron2-compatible.
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
# Install Detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
# Install Detic requirements
pip install src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt
# download Detic model weights
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
  -O src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
# ------------------------------------------------------------


cd projects/instanceimagenav
pip install -r requirements.txt

# now within this directory, get the scene dataset and episode dataset:


# --------------------  SCENE DATASET  --------------------
#   Dataset: HM3D (w/ Semantics v0.2)
#   Download the HM3D scenes from https://aihabitat.org/datasets/hm3d/ 
#   Alternatively, link a local copy.
#   Extract to data/scene_datasets/hm3d/{SPLIT}/...

# -------------------- EPISODE DATASET --------------------
#   Dataset: Habitat ImageNav Challenge Dataset (2023)
#   Download: https://drive.google.com/file/d/1GXHa0TpqJjvV1OO4Epu0aaU7CY1J-iFW/view?usp=share_link
#   Extract to data/datasets/instance_imagenav/hm3d/v3/{SPLIT}/content/{SCENE}.json.gz
```

At this point, the simulation demo should run successfully. If you want to run the physical Stretch demo, you will additionally need to install `home_robot_hw` and `home_robot` to this environment.

## Run Demo: Simulation

```bash
# in directory: /path/to/home-robot/projects/instanceimagenav
python projects/instanceimagenav/eval_episode_habitat.py
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
python projects/instanceimagenav/eval_episode_stretch.py
```

Step-wise images and a video will be generated. The results are saved to `projects/instanceimagenav/datadump/[images|videos]/debug`.
