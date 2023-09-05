# Spot Setup with demo branch

## Installation

### Requirements

+ Home robot - [Demo branch](https://github.com/facebookresearch/home-robot/tree/demo)
+ Spot sim2real - [Jay's fork with no_habitat branhc](https://github.com/jdvakil/spot-sim2real)

### Steps

#### Home-Robot

+ `git clone https://github.com/facebookresearch/home-robot.git --recursive`

+ `git submodule update --init --recursive src/home_robot/home_robot/perception/detection/detic/Detic src/third_party/detectron2 src/third_party/contact_graspnet  src/third_party/habitat-lab src/third_party/spot-sim2real src/third_party/MiDaS src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork` 

    - If this command doesn't add all the submodules

    - `git submodule update -f src/home_robot/home_robot/perception/detection/detic/Detic src/third_party/detectron2 src/third_party/contact_graspnet  src/third_party/habitat-lab src/third_party/spot-sim2real src/third_party/MiDaS src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork` 

+ `mamba env create -n home-robot -f src/home_robot_hw/environment.yml`
    + If `mamba` not found, `conda install -c conda-forge mamba --yes`
+ `conda activate home-robot` 

+ Change this env variable to your cuda -- `CUDA_HOME=/usr/local/cuda-11.7`


#### Install spot sim2real

```
cd src/third_party/spot-sim2real
git checkout no_habitat
```
##### Installing spot sim2real packages

```
cd bd_spot_wrapper/
python generate_executables.py
pip install -e .
```
```
cd ../spot_rl_experiments
python generate_executables.py
pip install -e .
```
`pip install bosdyn-api transforms3d einops gym==0.23.1 vtk`

#### Install MiDaS
+ Make sure you are on the master branch
```
cd src/third_party/MiDaS
git checkout master
git pull origin
```
+ Install weights
```
cd weights
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
```

#### Install Habitat-sim
```
cd src/third_party/habitat-lab
pip install -e habitat-lab
mamba install habitat-sim headless -c conda-forge -c aihabitat --yes
```

#### Install Boston Dynamics packages

```
pip install bosdyn-api bosdyn-client
```

#### Install SuperGLUE
```
pip install -r src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork/requirements.txt
```

#### Install Detic
```
cd src/home_robot/home_robot/perception/detection/detic/Detic
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
wget https://eecs.engin.umich.edu/~fouhey/fun/desk/desk.jpg
python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out.jpg --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```
