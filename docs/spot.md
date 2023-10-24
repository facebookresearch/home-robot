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
