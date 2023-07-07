# Spatial-Language Attention Policies for Efficient Robot Learning

## Installation Instructions

1. Use conda to `conda create env -f requirements.yaml` 
  - IMPORTANT NOTE: If using mamba double check the pytorch, cudatoolkit, torchaudio and torchvision versions; it has trouble finding a solution for this combination.
2. Install prerequisite `torch-geometric` library and dependencies by running:
```bash
python -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install --no-index pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
python -m pip install torch_geometric
```

Note above installation will uninstall torch and torch-deps installed as part of OVMM. This is okay and intended behavior for running SLAP.

## Configuration Parameters

SLAP and PerAct agents are inherited from OVMMAgent. They take the base configuration 
that the base agent expects, but add following configurable parameters to control their behavior.

### SLAP 
Under `config.SLAP` one can find the following parameters:

- `dry_run (bool)`: Dry-run the agent so it makes predictions but does not move the robot 
- `min_depth (float)`: Minimum depth below which all point-cloud observations are cut-off
- `max_depth (float)`: Maximum depth above which all point-cloud observations are cut-off
- `z_min (float)`: Minimum height of the input point-cloud, observations lower than this are cut-off
- `voxel_size_1 (float)`: Voxelization resolution for removing duplicate observations when combining multiple views
- `voxel_size_2 (float)`: Voxelization resolution for final input to SLAP
- `visualize (bool)`: Whether to visualize results from SLAP during RT inference
- `save_logs (bool)`: Whether to save IPM + APM output to disk (saves as numpy multi-array)

Under `config.SLAP.IPM` one can find the following model-specific parameters (these should remain the same b/w training and inference): 
- `path (string)`: Weights to load for inference

Under `config.SLAP.APM` you have the following parameters (ensure consistency b/w training and inference): 
- `max_actions (int)`: maximum number of action prediction supported by APM per skill
- `path (string)`: path to model checkpoint
- `num_pts (int)`: Total number of points in the point-cloud
- `orientation_type (string)`: choice of `quaternion/rpy`
- `query_radius (float)`: Cropping radius around predicted interaction point
- `skill_to_action_file (string)`: File with pre-coded language description of each action

Extra params under `SLAP.IPM` and `SLAP.APM` used for training, but needed here for setting up defaults during model construction. 

Task-specific evaluation information for per-skill experiments is taken from `config.EVAL`: 
- `task_name`: Language description used during training
- `object_list`: List of objects for Detic to detect
- `num_keypoints`: Number of actions to predict for each skill
