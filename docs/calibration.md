## Calibrated URDF for manipulation planning

Calibration is crucial for good performance. 

  1. Copy the DexWrist URDF and xacro onto your robot from the [Stretch tool share](https://github.com/hello-robot/stretch_tool_share/tree/master/tool_share/stretch_dex_wrist).
  1. Follow the [calibration instructions](https://github.com/hello-robot/stretch_ros/tree/master/stretch_calibration) provided by Hello Robot.
  1. Copy your `stretch.urdf` from the robot and update it as below *if necessary*. This is an advanced tutorial; you should be able to get everything working without it, but might see some errors.


## Loading the URDF in HomeRobot

By default we load an URDF from `assets/hab_stretch/urdf/stretch_manip_mode.urdf` for manipulation. You need to replace this with the appropriate URDF for your robot.

This adds a couple dummy joints to make 6dof planning easier, but isn't going to be calibrated for your specific robot. If you follow the calibration instructions from Hello, you can copy your `stretch.urdf` to your desktop though, and modify it to add these dummy joints.

### Explanation of Dummy Joints

Specifically, we add dummy joints for base x, y, and theta. These are useful for base motion planning; in particular, the addition of a prismatic x joint allows us to model motion in the direction of the robot's heading as an extra degree of freedom, bringing the Stretch with Dex Wrist up to 6dof, allowing for very flexible manipulation. We add two fixed joints, which are used when the robot is not in "manipulation mode" for full-body planning, but are currently unused in this example.


### Specific Changes

Manip mode urdf changes like so:
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

### Search and Replace

We don't assume ROS on the server side, so there's one other piece of cleanup.

To use locally you also need to replace this:
```
package://stretch_description
```
with
```
..
```
