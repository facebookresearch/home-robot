
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

### Setting up ROS Network

We use a client-server setup for controlling the robots, where low-level control runs on the robot and large neural nets are evaluated on a "server" -- which here refers to a local workstation with a GPU. For best performance, your server should be on the same wireless network as the robot, preferrably with an ethernet connection to the router for lowest latency.

We recommend adding to your `~/.bashrc` file:

```
# Whatever your robot's IP address is
export HELLO_ROBOT_IP=10.0.0.6
# Whatever your server IP address is
export SERVER_IP=10.0.0.2
```

On the robot, add:
```
export ROS_IP=$RHELLO_ROBOT_IP
export ROS_MASTER_URI=http://$HELLO_ROBOT_IP:11311

# Optionally - make it clear to avoid issues
echo "Setting ROS_MASTER_URI to $ROS_MASTER_URI"
echo "Setting ROS IP to $ROS_IP"
```

On the server side, add:
```
export ROS_IP=$SERVER_IP
export ROS_MASTER_URI=http://$HELLO_ROBOT_IP:11311

# Optionally - make it clear to avoid issues
echo "Setting ROS_MASTER_URI to $ROS_MASTER_URI"
echo "Setting ROS IP to $ROS_IP"

# Helpful alias - connect to the robot
alias ssh-robot="ssh hello-robot@$HELLO_ROBOT_IP"
```

Setting the server/hello robot IP addresses explicitly will help reduce some potential errors with Ros multi-node communication.
