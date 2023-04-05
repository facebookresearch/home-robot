
# Home Robot ROS Package

## Installation for Hello Robot Stretch

The following instructions assume a functional Hello Stretch Robot that has already been set up. 
This means that you should be able to run `stretch_robot_system_check.py` and that the results are all green.

For instructions on how to set up the hardware, see the official documentation here: https://docs.hello-robot.com/0.2/

### Installing the Home Robot stack on the robot

```sh
# Make sure ROS can find python properly
sudo apt install python-is-python3 pybind11-dev

# Clone the repository
git clone https://github.com/facebookresearch/home-robot
HOME_ROBOT_ROOT=$(realpath home-robot)

# Install the core home_robot package
cd $HOME_ROBOT_ROOT/src/home_robot
pip install -e .

# Install SLAM dependency on the robot
sudo apt install ros-noetic-hector-slam 

# Install pinocchio kinematics library on the robot -
# this is easier than installing via pip since it needs to be in the base environment
sudo apt install ros-noetic-pinocchio

# Set up the python package for ROS
ln -s $HOME_ROBOT_ROOT/src/home_robot_hw $HOME/catkin_ws/src/home_robot_hw

#Due to the organization of the code we want to symlink `assets` folder into `home_robot_hw` package in `catkin_ws`. Run the following:
ln -s $HOME_ROBOT_ROOT/assets $HOME/catkin_ws/src/home_robot_hw/assets

# Rebuild ROS to make sure paths are correct
cd $HOME/catkin_ws
catkin_make
```

We recommend you add this to your robot's `~/.bashrc` or equivalent:
```sh
source $HOME/catkin_ws/devel/setup.bash
```

If using the `zsh` shell there is an equivalent:
```sh
source $HOME/catkin_ws/devel/setup.zsh
```

These will ensure you can run the ROS-side launch files easily, and focus development on the server side.


### Setting up ROS Network

We use a multi-system setup for controlling the robots, where low-level control runs on the robot and large neural nets are evaluated on a "Workstation" -- which here refers to a local computer with a GPU. For best performance, your Workstation should be on the same wireless network as the robot, preferrably with an ethernet connection to the router for lowest latency.

We recommend adding this to your `~/.bashrc` file on the Robot:
```sh
# Whatever your robot's IP address is
export HELLO_ROBOT_IP=10.0.0.6

export ROS_IP=$HELLO_ROBOT_IP
export ROS_MASTER_URI=http://$HELLO_ROBOT_IP:11311

# Optionally - make it clear to avoid issues
echo "Setting ROS_MASTER_URI to $ROS_MASTER_URI"
echo "Setting ROS IP to $ROS_IP"
```

On the Workstation side, add:
```sh
# Whatever your server IP address is
export WORKSTATION_IP=10.0.0.2
# Whatever your robot's IP address is
export HELLO_ROBOT_IP=10.0.0.6

export ROS_IP=$WORKSTATION_IP
export ROS_MASTER_URI=http://$HELLO_ROBOT_IP:11311

# Optionally - make it clear to avoid issues
echo "Setting ROS_MASTER_URI to $ROS_MASTER_URI"
echo "Setting ROS IP to $ROS_IP"

# Helpful alias - connect to the robot
alias ssh-robot="ssh hello-robot@$HELLO_ROBOT_IP"
```

Setting the Workstation/Robot IP addresses explicitly will help reduce some potential errors with ROS multi-node communication.

## Debugging ROS network issues

Make sure you can do this on both your Workstation and on the Robot itself:

On the server:
```
rostopic pub /test std_msgs/String "data: 'test'"  -r 10  -v
```

On the robot:
```
rostopic echo /test
```

And then try the reverse (run server command on robot and vice versa).

If you do not see the word "test" appearing when you run the `rostopic echo` command, then one of your machines cannot see the other! Either you did not configure the network properly (see above) or your network administrator is blocking the connection. One thing you can try is manually configuring your IPv4 connection and putting both machines on the same subnet (e.g. 10.0.0.1 and 10.0.0.2).


