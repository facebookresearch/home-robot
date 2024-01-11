## Troubleshooting Common Errors

### Module genpy has no attribute message

Full trace:

```
Traceback (most recent call last):                                                                                         [20/30]
  File "/home/cpaxton/src/home-robot/src/home_robot_hw/home_robot_hw/nodes/simple_grasp_server.py", line 11, in <module>          
    import rospy                                                                                                                  
  File "/home/cpaxton/miniconda3/envs/home_robot/envs/home-robot-ovmm/lib/python3.9/site-packages/rospy/__init__.py", line 47, in 
<module>                                                                                                                          
    from std_msgs.msg import Header                                                                                               
  File "/home/cpaxton/miniconda3/envs/home_robot/envs/home-robot-ovmm/lib/python3.9/site-packages/std_msgs/msg/__init__.py", line 
1, in <module>                                                                                                                    
    from ._Bool import *                                                                                                          
  File "/home/cpaxton/miniconda3/envs/home_robot/envs/home-robot-ovmm/lib/python3.9/site-packages/std_msgs/msg/_Bool.py", line 9, 
in <module>
    class Bool(genpy.Message):
AttributeError: module 'genpy' has no attribute 'Message'
```

This means ROS was not found. Usually you just need to do this:
```
source ~/catkin_ws/devel/setup.bash
conda activate home-robot
# run your code
python src/real_world_ovmm/tests/test_heuristic.py
```
