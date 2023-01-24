# Home Robot Core Package

## Installation
```sh
mamba env create -f environment.yml
conda activate home_robot
cd src/home_robot
pip install -e .
```

## Usage

### Launching helper nodes

```sh
python -m home_robot.nodes.goto_controller
python -m home_robot.nodes.state_estimator
```

### Running the client

```sh
python -m home_robot.client.local_hello_robot
```

Available commands:
```py
robot.get_base_state()  # returns base location in the form of [x, y, rz]
robot.set_nav_mode()  # enables continuous navigation
robot.set_pos_mode()  # enables position control
robot.set_yaw_tracking(value: bool)  # turns yaw tracking on/off (robot only tries to reach the xy position of goal if off)
robot.set_goal(xyt: list)  # sets the goal for the goto controller
robot.set_velocity(v: float, w: float)  # directly sets the linear and angular velocity of robot base (command gets overwritten immediately if goto controller is on)
```

Basic example:
```py
robot.set_nav_mode()  # Enables continuous control
robot.set_goal(]1.0, 0.0, 0.0])  # Sets XYZ target
robot.get_base_state()  # Shows the robot's XYZ coordinates
```