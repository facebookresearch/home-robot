# Hardware backend for the Hello Stretch

We use a multi-system setup for controlling the Hello Stretch, where low-level control runs on the "Robot" and large neural nets are evaluated on a "Workstation" -- which here refers to a local computer with a GPU. 
For best performance, your Workstation should be on the same wireless network as the Robot, preferrably with an ethernet connection to the router for lowest latency.

# Installation

- [Setup instructions on Robot](./install_robot.md)
- [Setup instructions on Workstation](./install_robot.md)

## Usage

### On Robot: Launching the Hello Stretch hardware drivers
```sh
roslaunch home_robot_hw startup_stretch_hector_slam.launch
```

### On Workstation: Interacting with the robot via the client

See [examples](../../examples/) for common API usage.

You can also start an interactive command line interface (CLI) with:
```sh
python -m home_robot_hw.remote.interactive_cli
```