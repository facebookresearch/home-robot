# Home Robot Core Package

## Installation
```sh
mamba env create -n home_robot -f environment.yml
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
# Query states
robot.get_base_state()  # returns base location in the form of [x, y, rz]

# Mode switching
robot.switch_to_velocity_mode()  # enables base velocity control
robot.switch_to_navigation_mode()  # enables continuous navigation
robot.switch_to_manipulation_mode()  # enables gripper control

# Velocity mode
robot.set_velocity(v: float, w: float)  # directly sets the linear and angular velocity of robot base

# Navigation mode
robot.navigate_to(xyt: list, relative: bool = False, position_only: bool = False)

# Manipulation mode
robot.set_arm_joint_positions(joint_positions: list)  # joint positions: [BASE_TRANSLATION, ARM_LIFT, ARM_EXTENTION, WRIST_YAW, WRIST_PITCH, WRIST_ROLL]
robot.set_ee_pose(pos: list, quat: list, relative: bool = False)
```

