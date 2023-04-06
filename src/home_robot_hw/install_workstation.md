# Home Robot Remote ROS Client

We use a multi-system setup for controlling the Hello Stretch, where low-level control runs on the "Robot" and large neural nets are evaluated on a "Workstation" -- which here refers to a local computer with a GPU. 

On the Workstation, we install `home_robot_hw` as a pip package containing a client that is capable of connecting to ROS master running on the robot.

## Installation

After creating a conda env and [installing home_robot](../home_robot),

```sh
# Install dependencies
mamba env update -f src/home_robot_hw/environment.yml

# Install home_robot_hw
pip install -e src/home_robot_hw
```
