## Calibrated URDF for manipulation planning

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
