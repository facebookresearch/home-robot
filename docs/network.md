# Real Robot Network Setup

Proper network setup is crucial to getting good performance with HomeRobot. Low-cost mobile robots often do not have sufficient GPU to run state-of-the-art perception models. Instead, we rely on a client-server architecture, where ROS and low-level controllers run on the robot, and CPU- and GPU-intensive AI code runs on a workstation.

### Network Hardware

We used a Netgear Nighthawk router for experiments. One example that works is the [Nightawk AX4300](https://www.amazon.com/Netgear-Nighthawk-6-Stream-AX4300-Router/dp/B086ZNJ1J2/), available from Amazon. A better router will allow you to support more wireless devices.

After network setup is finished and [ROS](http://wiki.ros.org/noetic) is installed on your workstation, you can run the following command and should see the robot appear:
```
rviz -d $HOME_ROBOT_ROOT/src/home_robot_hw/launch/mapping_demo.rviz
```

Note that ROS will be installed in following steps as a part of our conda environment.

*Network access and code development.* Internet access is crucial to software development. We expect that either this router will have internet access, or that (on a corporate or academic network), the workstation will have two network connections, and you will use the [hardware code sync instructions](docs/hardware.md) to push code changes to the robot if necessary. In general, though, you should not need to make changes to code on the stretch. If on a corporate or academic network you may find it useful to buy a USB ethernet adapter to allow your workstation to have two network connections (for example, [this adapter](https://www.amazon.com/USB-Ethernet-Adapter-Gigabit-Switch/dp/B09GRL3VCN/) on amazon).

### Terminal Setup

After following the installation instructions, we recommend setting up your `~/.bashrc` on the robot workstation:

```
# Whatever your workstation's IP address is
export WORKSTATION_IP=10.0.0.2
# Whatever your robot's IP address is
export HELLO_ROBOT_IP=10.0.0.6

# Path to the codebase
export HOME_ROBOT_ROOT=/path/to/home-robot

export ROS_IP=$WORKSTATION_IP
export ROS_MASTER_URI=http://$HELLO_ROBOT_IP:11311

# Optionally - make it clear to avoid issues
echo "Setting ROS_MASTER_URI to $ROS_MASTER_URI"
echo "Setting ROS IP to $ROS_IP"

# Helpful alias - connect to the robot
alias ssh-robot="ssh hello-robot@$HELLO_ROBOT_IP"
```

### Testing

From your workstation, run this:
```
ssh-robot
roslaunch home_robot_hw startup_stretch_hector_slam.launch
```

On the workstation:
```
rviz -d $HOME_ROBOT_ROOT/src/home_robot_hw/launch/mapping_demo.rviz
```

You should see RVIZ come up, display images and video, and be able to wave your hand in front of the robot's camera and see minimal lag.
