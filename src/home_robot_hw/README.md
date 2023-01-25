# Hardware backend for the Hello Stretch

- Wrapper around [stretch_body](https://github.com/hello-robot/stretch_body) and [stretch_ros](https://github.com/hello-robot/stretch_ros)
- Same interface as home_robot_sim

## Installation
1. Install firmware from Hello Robot
    ```sh
    # Copy robot factory data into your user workspace
    cp -r /etc/hello-robot/stretch-re* ~

    # Clone the official setup scripts
    cd ~
    git clone https://github.com/hello-robot/stretch_install
    cd stretch_install

    # Run setup script (DO NOT RUN BOTH)
    ./stretch_new_robot_install.sh  # if installing into a new robot
    ./stretch_new_user_install.sh  # if installing into a new user account on a already-set-up robot
    ```
1. Open `~/.bashrc`. You will see a block of commands that initializes Stretch, and another block that initializes Conda. If needed, move the stretch setup block BEFORE the conda initialization.
1. Launch a new bash shell. Activate an conda env with Python 3.8 installed.
1. Link `home_robot` and install ROS stack
    ```sh
    # Create symlink in catkin workspace
    ln -s /abs/path/to/home-robot/src/home_robot_hw $HOME/catkin_ws/src/home_robot

    # Install dependencies for catkin
    pip install empy catkin_pkg rospkg

    # Build catkin workspace
    cd ~/catkin_ws
    catkin_make

    # Add newly built setup.bash to .bashrc
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
    ```
1. Calibrate robot following instructions [here](https://github.com/hello-robot/stretch_ros/tree/master/stretch_calibration#checking-the-current-calibration-with-new-observations).
1. Generate URDF from calibration data: `rosrun stretch_calibration update_urdf_after_xacro_change.sh`.
1. Run `stretch_robot_system_check.py` to make sure that things are normal.

#### Additional hardware stack dependencies
1. Hector SLAM: `sudo apt install ros-noetic-hector-*`
1. (For grasping only) Detectron 2: `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

## Usage

### Launching the Hello Stretch hardware drivers
```sh
roslaunch home_robot startup_stretch_hector_slam.launch
```