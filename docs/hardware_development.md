# Developing on the Hello Robot Stretch

We use a multi-system setup for controlling the Hello Stretch, where low-level control runs on the "Robot" and large neural nets are evaluated on a "Workstation" -- which here refers to a local computer with a GPU. 
For best performance, your Workstation should be on the same wireless network as the Robot, preferrably with an ethernet connection to the router for lowest latency.

## Installation

- [Setup instructions on Robot](./install_robot.md)
- [Setup instructions on Workstation](./install_workstation.md)

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

## Appendix

### Syncing code between Robot and Workstation

Let `ROBOT_IP` store the robot's IP and let `WORKSTATION_IP` store the workstation's IP. If your local network doesn't have access to internet we recommend using `rsync` with `--update` flag to sync your code changes across the machines. Usage:
```
rsync -rvu /abs/path/to/local/home-robot $ROBOT_USER@$ROBOT_IP:/abs/path/to/remote/home-robot
```

The above command will do a *r*ecursive *u*pdating of changed files while echoing a *v*erbose output.

### Collecting data on the real robot
We provide scripts to collect data in H5 format using `Recorder` class. Follow the instructions for recording these files. If your application needs more/fewer data-sources, sub-class `Recorder` and over-ride the `save_frame` method
Collect the data through teleoperation.
  ```sh
  python collect_h5.py --task_name TASK_NAME  --dir_path DIR_PATH
  ```
  This will save the teleoperation files at `DIR_PATH/TASK_NAME-{iteration}/{datetime}.h5`.

  Turn on the controller and make sure it is connected to the robot (top two blue lights should be on). To give trajectory commands through the Xbox controller, open a separate terminal and:
  ```sh
  rosrun joy joy_node
  ```
Refer to [official hello robot keybindings](https://docs.hello-robot.com/0.2/stretch-tutorials/getting_started/images/xbox.png) to learn how to operate Stretch with the controller. We also provide a `Dataloaders` to load data into torch (WIP). 
