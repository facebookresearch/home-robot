## Calibrated URDF for manipulation planning

Calibration is crucial for good grasping performance on the real robot, and will also improve performance of heuristic methods for navigation.

  1. Copy the DexWrist URDF and xacro onto your robot from the [Stretch tool share](https://github.com/hello-robot/stretch_tool_share/tree/master/tool_share/stretch_dex_wrist).
  1. Follow the [calibration instructions](https://github.com/hello-robot/stretch_ros/tree/master/stretch_calibration) provided by Hello Robot.
  1. Copy your `stretch.urdf` from the robot and update it as below, in "Loading the New URDF in HomeRobot", *if necessary*. This is an advanced tutorial; you should be able to get everything working without it, but might see worse grasping performance on your hardware.

Once you follow the [calibration instructions](https://github.com/hello-robot/stretch_ros/tree/master/stretch_calibration), you will have an URDF on your robot in `$HOME/catkin_ws/src/stretch_description/urdf`. You can check it with this command:
```
ls `rospack find stretch_description`/urdf | grep stretch.urdf
```
You should see the file listed. Running the installation guide, you should also see a visualization of your robot, having loaded this model.

### Explanation of Terms

*urdf*: Unified Robot Description Format. Basically, an XML file that describes a robot's geometry. Check out the [ROS documentation](https://docs.ros.org/en/foxy/Tutorials/Intermediate/URDF/URDF-Main.html) for some details if you are curious.

*Calibration*: your robot's geometry may vary slightly from ours; while we provide a calibrated URDF together with [our habitat models](https://github.com/cpaxton/hab_stretch), it may not work for you. Calibration will come up with a new URDF, which uniquely represents your robot. See the [Hello Robot instructions](https://github.com/hello-robot/stretch_ros/tree/master/stretch_calibration) for some examples of good and bad calibrations.

*DexWrist*: The particular wrist for stretch we use in HomeRobot; it adds two extra degrees of freedom.

## Loading the New URDF in HomeRobot

Following the Hello Robot calibration instructions will give you a `stretch.urdf` on your robot. We use this URDF to publish the robot's head pose, and for SLAM - meaning that navigation will work fine if this is all you do. However, manipulation is run on the server, and does not use ROS to load robot parameters - which makes things a bit more complex.

By default we load an URDF from `assets/hab_stretch/urdf/stretch_manip_mode.urdf` for manipulation. You need to replace this with the appropriate URDF for your robot, copied from `$HOME/catkin_ws/src/stretch_description/urdf/stretch.urdf`.

*What this does:* We add a couple dummy joints to make 6dof planning easier, but the URDF we provide isn't going to be calibrated for your specific robot. If you follow the calibration instructions from Hello, you can copy your `stretch.urdf` to your desktop though, and modify it to add these dummy joints. See the section below for more details.

### Explanation of Dummy Joints

Specifically, we add dummy joints for base x, y, and theta. These are useful for base motion planning; in particular, the addition of a prismatic x joint allows us to model motion in the direction of the robot's heading as an extra degree of freedom, bringing the Stretch with Dex Wrist up to 6dof, allowing for very flexible manipulation. We add two fixed joints, which are used when the robot is not in "manipulation mode" for full-body planning, but are currently unused in this example.


### Specific Changes

Copy your calibrated `stretch.urdf` to `stretch_manip_mode.urdf` and put it in `assets/hab_stretch/urdf/stretch_manip_mode.urdf` - you may want to create a new branch via `git` and commit the changes.

Then we add these links to the urdf file:
```
  <joint name="base_y_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0.0 0.0 0.0"/>
    <axis xyz="1.0 0.0 0.0"/>
    <parent link="base_xy_link"/>
    <child link="base_theta_link"/>
  </joint>
  <joint name="base_theta_joint" type="fixed">
    <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
    <axis xyz="0.0 0.0 1.0"/>
    <parent link="base_theta_link"/>
    <child link="base_link"/>
  </joint>
  <joint name="base_x_joint" type="prismatic">
    <origin xyz="0 0 0" rpy="0.0 0.0 0.0"/>
    <axis xyz="1.0 0.0 0.0"/>
    <parent link="world"/>
    <child link="base_xy_link"/>
    <limit effort="100.0" lower="-50.0" upper="50.0" velocity="3.0"/>
  </joint>
  <link name="base_xy_link">
    <collision>
      <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
      <geometry>
        <box size="0.01 0.01 0.01"/>
      </geometry>
    </collision>
  </link>
  <link name="base_theta_link">
    <collision>
      <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
      <geometry>
        <box size="0.01 0.01 0.01"/>
      </geometry>
    </collision>
  </link>
  <link name="world"/>
```

Just copy and paste that text into the file, and save. It should appear at the top level, inside the `<robot></robot>` tags.

### Search and Replace

We don't assume ROS on the server side, so there's one other piece of cleanup: replacing all meshes in your new modified urdf `stretch_manip_mode.urdf` with relative paths, instead of using ROS package management to find them.

Use search and replace in your favorite editor to replace:
```
package://stretch_description
```
with
```
..
```

Then you should be able to run an example! For example, set the robot up in front of a cup (arm facing the object) and run this:
```
python projects/real_world_ovmm/eval_episode.py --test-pick --pick-object cup
```
