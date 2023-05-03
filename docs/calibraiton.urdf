## Calibrated URDF for manipulation planning

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
