# Spot Setup with demo branch

## Installation

### Requirements

+ Home robot - [Demo branch](https://github.com/facebookresearch/home-robot/tree/demo)
+ Spot sim2real - [Jay's fork with no_habitat branch](https://github.com/jdvakil/spot-sim2real)

### Steps

#### Home-Robot
```
git clone https://github.com/facebookresearch/home-robot.git --recursive
```

```
echo 'alias HOME_ROBOT_ROOT=<path/to/home-robot>' >> ~/.bashrc 
source ~/.bashrc
```

```
git submodule update --init --recursive src/home_robot/home_robot/perception/detection/detic/Detic src/third_party/detectron2 src/third_party/contact_graspnet  src/third_party/habitat-lab src/third_party/spot-sim2real src/third_party/MiDaS src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork
``` 

- If this command doesn't add all the submodules

```
git submodule update -f src/home_robot/home_robot/perception/detection/detic/Detic src/third_party/detectron2 src/third_party/contact_graspnet  src/third_party/habitat-lab src/third_party/spot-sim2real src/third_party/MiDaS src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork
```

```
mamba env create -n home-robot -f src/home_robot_hw/environment.yml
```

 If `mamba` not found, `conda install -c conda-forge mamba --yes`
 
```
conda activate home-robot
pip install -e src/home_robot_hw/.
```

Change this env variable to your cuda -- `CUDA_HOME=/usr/local/cuda-<version_tag>`


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
```
pip install bosdyn-api  bosdyn-client transforms3d einops gym==0.23.1 vtk scikit-image open3d natsort scikit-fmm
```

#### Install MiDaS
+ Make sure you are on the master branch
```
cd $HOME_ROBOT_ROOT/src/third_party/MiDaS
git checkout master
git pull origin
```
+ Install weights
```
cd weights
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
```
+ Packages
```
cd ../
pip install -e .
pip install imutils

```

#### Install Habitat-sim
```
cd $HOME_ROBOT_ROOT/src/third_party/habitat-lab
pip install -e habitat-lab
mamba install habitat-sim headless -c conda-forge -c aihabitat --yes
```


#### Install SuperGLUE
```
pip install -r src/home_robot/home_robot/agent/imagenav_agent/SuperGluePretrainedNetwork/requirements.txt
```

#### Install Detic
```
cd src/home_robot/home_robot/perception/detection/detic/Detic
pip install -r requirements.txt
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
wget https://eecs.engin.umich.edu/~fouhey/fun/desk/desk.jpg
python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out.jpg --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```


#### Additional installtions

```
cd $HOME_ROBOT_ROOT
pip install -e src/third_party/detectron2/.
```
##### Issues with installing detectron2
Set `CUDA_HOME=/usr/local/cuda-<VERSION>/` and install again

###### If getting a version_mismatch error
+ Install cuda from this [website](https://developer.nvidia.com/cuda-downloads) and select the correct version tag when installing cuda. 
##### For `qObject: movToThread` errors/warnings when running the demo

```
pip install --no-binary opencv-python opencv-python
```

##### Attribute error for SE3 with sophuspy
```
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install/
make -j8
make install
pybind11_DIR=$PWD/../install/share/cmake/pybind11/ pip3 install --user sophuspy
```
