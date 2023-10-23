# Spot Setup

## Installation

### Basic

#### 0. Requirements

- **home-robot branch:** `goat`
- Access to [Matthew's fork](https://github.com/MatthewChang/spot-sim2real/tree/b66f4dd399efeca16a115ef5d106759fb871adc7) of spotsim2real is required.

#### 1. Install Home-robot as usual.

> Make sure to initialize all submodules needed by goat. Notice the addition of spotsim2real, habitat-lab, MiDaS
> and SuperGlue

```
git submodule sync
git submodule update --init --recursive src/home_robot/home_robot/perception/detection/detic/Detic src/third_party/detectron2 src/third_party/contact_graspnet  src/third_party/habitat-lab src/third_party/spot-sim2real src/third_party/MiDaS src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork
```

#### 2. Install spotsim2real

The pinned branch in the submodule is wrong so do:

```
cd src/third_party/spot-sim2real
git checkout no_habitat
```

Install the spotsim2real packages

```
# Spot Wrapper
cd bd_spot_wrapper/
python generate_executables.py
pip install -e .

# Spot RL
cd ../spot_rl_experiments
python generate_executables.py
pip install -e .

### The following is a minimal set of dependencies needed to use the functionalities that GOAT uses from spotsim2real + some home-robot deps
pip install bosdyn-api bosdyn-client transforms3d einops gym==0.23.1 vtk scikit-image open3d natsort scikit-fmm imutils
```

#### MiDaS

```
git submodule update --init --recursive src/home_robot/home_robot/perception/detection/detic/Detic src/third_party/detectron2 src/third_party/contact_graspnet  src/third_party/habitat-lab src/third_party/spot-sim2real src/third_party/MiDaS src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork
```
cd $HOME_ROBOT_ROOT/src/third_party/MiDaS/weights
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
cd $HOME_ROBOT_ROOT
```


### GOAT - Old Instructions

0. Requirements

- **home-robot branch:** `goat`
- Access to [Mathews fork](https://github.com/MatthewChang/spot-sim2real/tree/b66f4dd399efeca16a115ef5d106759fb871adc7) of spotsim2real is required.

1. Install Home-robot as usual.

> Make sure to initialize all submodules needed by goat. Notice the addition of spotsim2real, habitat-lab, MiDaS
> and SuperGlue

```
git submodule update --init --recursive src/home_robot/home_robot/perception/detection/detic/Detic src/third_party/detectron2 src/third_party/contact_graspnet  src/third_party/habitat-lab src/third_party/spot-sim2real src/third_party/MiDaS src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork
```
cd $HOME_ROBOT_ROOT/src/third_party/MiDaS/weights
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
```

2. Install spotsim2real

The pinned branch in the submodule is wrong so do:

```
cd src/third_party/spot-sim2real
git checkout no_habitat
```

Install the spotsim2real packages

```
# Spot Wrapper
cd bd_spot_wrapper/
python generate_executables.py
pip install -e .

# Spot RL
cd ../spot_rl_experiments
python generate_executables.py
pip install -e .

### The following is a minimal set of dependencies needed to use the functionalities that GOAT uses from spotsim2real
pip install bosdyn-api==3.2.3 transforms3d einops gym==0.23.1 vtk

```

3. Install habitat

```
### Install habitat. We should be able to remove all pointers to habitat-lab from the goat entrypoint on the future.
cd src/third_party/habitat-lab
pip install -e habitat-lab
conda install habitat-sim headless -c conda-forge -c aihabitat
```

4. Make sure that detic was indeed installed. Run the detic demo.py
5. Superglue: we might be missing a step here

```
pip install -r src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork/requirements.txt
```

6. MiDas: checkout to master, pull and add models

```
cd src/third_party/MiDaS
git checkout master
git pull origin/master
# download the following model https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
# and move it to src/third_party/MiDaS/weights
```

7. Spot Network configuration
   1. See https://github.com/facebookresearch/spot-sim2real/blob/main/installation/SETUP_INSTRUCTIONS.md#setup-spot-robot. Make sure you can ping the SPOT_IP

8. Entrypoints
   1. Run `python projects/spot/objectnav.py` for the objectnav entrypoint
   2. Run `python projects/spot/goat` for the goat entrypoint

### Basic

```
pip install bosdyn-api
```
Version should be >= 3.3 - tested with 3.3.0 and 3.3.1

### ROS - Deprecated

#### Catkin Environment

You need a full real catkin env for this.
```
conda deactivate
cd $HOME
mkdir spot_ws
catkin_make -DPYTHON_EXECUTABLE=`which python3`
```

#### ROS Packages

You need some additional packages
```
sudo apt install ros-noetic-twist-mux ros-noetic-joy-node ros-noetic-joy ros-noetic-interactive-marker-twist-server ros-noetic-teleop-twist-joy
```
Alternately
```
mamba install ros-noetic-twist-mux ros-noetic-joy ros-noetic-interactive-marker-twist-server ros-noetic-teleop-twist-joy -c robostack
```

### Example data

This contains some trajectory examples:
```
https://drive.google.com/file/d/195z0DoyxIdT47zN_E44gogPgAeIn3krq/view?usp=drive_link
```
