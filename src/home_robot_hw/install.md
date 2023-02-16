
# Home Robot ROS Package

## Installation for Hello Robot Stretch

These assume a Hello Robot Stretch v2 on Ubuntu 20.04.

```
# Make sure ROS can find python properly
sudo apt install python-is-python3 pybind11-dev

# Install the core home_robot package
cd $HOME_ROBOT_ROOT/src/home_robot
pip install -e .

# Install SLAM dependency on the robot
sudo apt install ros-noetic-hector-slam 

# Set up the python package for ROS
ln -s $HOME_ROBOT_ROOT/src/home_robot_hw $HOME/catkin_ws/src/home_robot_hw

# Rebuild ROS to make sure paths are correct
cd $HOME/catkin_ws
catkin_make
```
