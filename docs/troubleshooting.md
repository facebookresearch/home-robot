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

### Torch related errors

```
opt/conda/envs/home-robot/lib/python3.9/site-packages/torch_geometric/typing.py:31: UserWarning: An issue occurred while importing 'torch-scatter'. Disabling its usage. Stacktrace: /opt/conda/envs/home-robot/lib/python3.9/site-packages/torch_scatter/_scatter_cpu.so: undefined symbol: _ZN2at4_ops6narrow4callERKNS_6TensorElll
  warnings.warn(f"An issue occurred while importing 'torch-scatter'. "
Traceback (most recent call last):
  File "/home/jaydv/Documents/home-robot/projects/real_world_ovmm/eval_episode.py", line 14, in <module>
    from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/agent/ovmm_agent/__init__.py", line 7, in <module>
    from .ovmm_agent import OpenVocabManipAgent, Skill, get_skill_as_one_hot_dict
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/agent/ovmm_agent/ovmm_agent.py", line 13, in <module>
    from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/agent/objectnav_agent/__init__.py", line 7, in <module>
    from .objectnav_agent import ObjectNavAgent
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/agent/objectnav_agent/objectnav_agent.py", line 16, in <module>
    from home_robot.mapping.instance import InstanceMemory
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/mapping/__init__.py", line 5, in <module>
    from .voxel import SparseVoxelMap, SparseVoxelMapNavigationSpace
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/mapping/voxel/__init__.py", line 5, in <module>
    from .planners import plan_to_frontier
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/mapping/voxel/planners.py", line 8, in <module>
    from home_robot.mapping.voxel.voxel import SparseVoxelMap
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/mapping/voxel/voxel.py", line 21, in <module>
    from home_robot.mapping.instance import Instance, InstanceMemory, InstanceView
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/mapping/instance/__init__.py", line 5, in <module>
    from .core import Instance, InstanceView
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/mapping/instance/core.py", line 15, in <module>
    from home_robot.utils.point_cloud_torch import get_bounds
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/utils/point_cloud_torch.py", line 19, in <module>
    from torch_geometric.nn.pool.voxel_grid import voxel_grid
  File "/opt/conda/envs/home-robot/lib/python3.9/site-packages/torch_geometric/__init__.py", line 1, in <module>
    import torch_geometric.utils
  File "/opt/conda/envs/home-robot/lib/python3.9/site-packages/torch_geometric/utils/__init__.py", line 8, in <module>
    from .dropout import dropout_adj, dropout_node, dropout_edge, dropout_path
  File "/opt/conda/envs/home-robot/lib/python3.9/site-packages/torch_geometric/utils/dropout.py", line 6, in <module>
    import torch_cluster  # noqa
  File "/opt/conda/envs/home-robot/lib/python3.9/site-packages/torch_cluster/__init__.py", line 18, in <module>
    torch.ops.load_library(spec.origin)
  File "/opt/conda/envs/home-robot/lib/python3.9/site-packages/torch/_ops.py", line 643, in load_library
    ctypes.CDLL(path)
  File "/opt/conda/envs/home-robot/lib/python3.9/ctypes/__init__.py", line 382, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: /opt/conda/envs/home-robot/lib/python3.9/site-packages/torch_cluster/_grid_cuda.so: undefined symbol: _ZN2at4_ops6narrow4callERKNS_6TensorElll

```

Error installing `torch-cluster`
```
pip uninstall torch-cluster
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```
```
/opt/conda/envs/home-robot/lib/python3.9/site-packages/torch_geometric/typing.py:31: UserWarning: An issue occurred while importing 'torch-scatter'. Disabling its usage. Stacktrace: /opt/conda/envs/home-robot/lib/python3.9/site-packages/torch_scatter/_scatter_cpu.so: undefined symbol: _ZN2at4_ops6narrow4callERKNS_6TensorElll
  warnings.warn(f"An issue occurred while importing 'torch-scatter'. "
Traceback (most recent call last):
  File "/home/jaydv/Documents/home-robot/projects/real_world_ovmm/eval_episode.py", line 14, in <module>
    from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/agent/ovmm_agent/__init__.py", line 7, in <module>
    from .ovmm_agent import OpenVocabManipAgent, Skill, get_skill_as_one_hot_dict
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/agent/ovmm_agent/ovmm_agent.py", line 13, in <module>
    from home_robot.agent.objectnav_agent.objectnav_agent import ObjectNavAgent
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/agent/objectnav_agent/__init__.py", line 7, in <module>
    from .objectnav_agent import ObjectNavAgent
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/agent/objectnav_agent/objectnav_agent.py", line 16, in <module>
    from home_robot.mapping.instance import InstanceMemory
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/mapping/__init__.py", line 5, in <module>
    from .voxel import SparseVoxelMap, SparseVoxelMapNavigationSpace
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/mapping/voxel/__init__.py", line 5, in <module>
    from .planners import plan_to_frontier
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/mapping/voxel/planners.py", line 8, in <module>
    from home_robot.mapping.voxel.voxel import SparseVoxelMap
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/mapping/voxel/voxel.py", line 21, in <module>
    from home_robot.mapping.instance import Instance, InstanceMemory, InstanceView
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/mapping/instance/__init__.py", line 6, in <module>
    from .instance_map import InstanceMemory
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/mapping/instance/instance_map.py", line 22, in <module>
    from home_robot.mapping.instance.matching import (
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/mapping/instance/matching.py", line 7, in <module>
    from home_robot.utils.bboxes_3d import (
  File "/home/jaydv/Documents/home-robot/src/home_robot/home_robot/utils/bboxes_3d.py", line 45, in <module>
    from pytorch3d.ops import box3d_overlap
  File "/opt/conda/envs/home-robot/lib/python3.9/site-packages/pytorch3d/ops/__init__.py", line 7, in <module>
    from .ball_query import ball_query
  File "/opt/conda/envs/home-robot/lib/python3.9/site-packages/pytorch3d/ops/ball_query.py", line 10, in <module>
    from pytorch3d import _C
ImportError: libtorch_cuda_cu.so: cannot open shared object file: No such file or directory
```

This means there was an error installing `pytorch3d`
```
pip uninstall pytorch3d
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html
```
