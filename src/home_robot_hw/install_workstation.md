# Home Robot Remote ROS Client

We use a multi-system setup for controlling the Hello Stretch, where low-level control runs on the "Robot" and large neural nets are evaluated on a "Workstation" -- which here refers to a local computer with a GPU. 

On the Workstation, we install `home_robot_hw` as a pip package containing a client that is capable of connecting to ROS master running on the robot.

## Installation

```sh
# Create a conda env
mamba create -n <YOUR_ENV_NAME> -f src/home_robot_hw/environment.yml
conda activate <YOUR_ENV_NAME>

# Install the core home_robot package
pip install -e src/home_robot

# Install home_robot_hw
pip install -e src/home_robot_hw
```

### Setting up ROS Network

We recommend adding this to your `~/.bashrc` file on the Workstation:
```sh
# Whatever your workstation's IP address is
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

Setting the IP addresses explicitly will help reduce some potential errors with ROS multi-node communication.
