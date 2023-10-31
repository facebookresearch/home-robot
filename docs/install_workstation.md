## Detailed Workstation Instructions

If you have trouble using the [install script](../install_deps.sh), you can try to follow these instructions.

#### 1. Create Your Environment

If necessary, [install mamba](https://mamba.readthedocs.io/en/latest/installation.html) in your base conda environment. Optionally: [install ROS noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) on your workstation.

```
# Create a conda env - use the version in home_robot_hw if you want to run on the robot
mamba env create -n home-robot -f src/home_robot_hw/environment.yml

# Otherwise, use the version in src/home_robot
mamba env create -n home-robot -f src/home_robot/environment.yml

conda activate home-robot
```

This should install pytorch; if you run into trouble, you may need to edit the installation to make sure you have the right CUDA version. See the [pytorch install notes](docs/install_pytorch.md) for more.

Optionally, setup a [catkin workspace](catkin.md) to use improved ROS visualizations.

##### Installing Pinocchio

[Pinocchio](https://stack-of-tasks.github.io/pinocchio/) is an optional fast inverse kinematics library used in HomeRobot for some manipulation tasks on Stretch. It's an optional dependency which requires a version of libboost that may conflict with the installation above. As such it's best to install separately:

```
conda install pinocchio>=2.6.17 -c conda-forge
```

This will automatically be called in the [install script](../install_deps.sh).


#### 2. Install Home Robot Packages
```
conda activate home-robot

# Install the core home_robot package
python -m pip install -e src/home_robot

Skip to step 4 if you do not have a real robot setup or if you only want to use our simulation stack.

# Install home_robot_hw
python -m pip install -e src/home_robot_hw
```

_Testing Real Robot Setup:_ Now you can run a couple commands to test your connection. If the `roscore` and the robot controllers are running properly, you can run `rostopic list` and should see a list of topics - streams of information coming from the robot. You can then run RVIZ to visualize the robot sensor output:

```
rviz -d $HOME_ROBOT_ROOT/src/home_robot_hw/launch/mapping_demo.rviz
```

#### 3. Download third-party packages
```
git submodule update --init --recursive src/home_robot/home_robot/perception/detection/detic/Detic src/third_party/detectron2 src/third_party/contact_graspnet
```

#### 4. Hardware Testing

Run the hardware manual test to make sure you can control the robot remotely. Ensure the robot has one meter of free space before running the script.

```
python tests/hw_manual_test.py
```

Follow the on-screen instructions. The robot should move through a set of configurations.

#### 5. Install Detic

Install [detectron2](https://detectron2.readthedocs.io/tutorials/install.html). If you installed our default environment above, you may need to [download CUDA11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive).


Download Detic checkpoint as per the instructions [on the Detic github page](https://github.com/facebookresearch/Detic):

```bash
cd $HOME_ROBOT_ROOT/src/home_robot/home_robot/perception/detection/detic/Detic/
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth --no-check-certificate
```

You should be able to run the Detic demo script as per the Detic instructions to verify your installation was correct:
```bash
wget https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg
python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out2.jpg --vocabulary custom --custom_vocabulary headphone,webcam,paper,coffe --confidence-threshold 0.3 --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

#### 6. Download pretrained skills
```
mkdir -p data/checkpoints
cd data/checkpoints
wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ovmm_baseline_home_robot_challenge_2023_v0.2.zip
unzip ovmm_baseline_home_robot_challenge_2023_v0.2.zip -d ovmm_baseline_home_robot_challenge_2023_v0.2
cd ../../
```

