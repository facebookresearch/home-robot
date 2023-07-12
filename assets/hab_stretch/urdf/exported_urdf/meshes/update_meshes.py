#!/usr/bin/env python
import os
import stretch_body.hello_utils as hu
batch_name = hu.read_fleet_yaml('stretch_re1_factory_params.yaml')['robot']['batch_name'].lower()
cmd='cp ~/catkin_ws/src/stretch_ros/stretch_description/meshes/'+batch_name+'/*.STL ~/catkin_ws/src/stretch_ros/stretch_description/meshes'
print("Copying in mesh files from batch:"+batch_name)
print(cmd)
os.system(cmd)
