# Spatial-Language Attention Policies for Efficient Robot Learning


Code has been tested on Ubuntu 18.04 with CUDA 11.6, python 3.9 and pytorch 1.12.1 (with associated torchaudio and
torchvision packages).

## Installation

### Instructions

1. Use conda to `conda env create -f requirements.yaml` 
2. Conda and mamba usually have trouble finding the right pytorch build for above mentioned specs.
   Next we will install prerequisites using pip within the environment. This includes fixing
   pytorch, installing prerequisites for (PyG)[https://pyg.org/] and finally the `torch-geometric` library and dependencies by running:
```bash
conda activate slap_base

python -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install --no-index pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
python -m pip install torch_geometric
```
3. Install home-robot and home-robot-hw
```bash
# return to HOME_ROBOT_ROOT
cd ../..

python -m pip install -e src/home_robot
python -m pip install -e src/home_robot_hw

# install Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Make sure to follow home-robot's [instructions](../../README.md#5-install-detic) to download checkpoint for Detectron2.

### Troubleshooting

- Encounter an error like following when you run any job:
  ```bash
  ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.XX' not found (required by /path/to/slap_base/lib/python3.9/site-packages/pinocchio/pinocchio_pywrap.cpython-39-x86_64-linux-gnu.so)
  ```  
  This is a known bug with `conda` which does not load environment variables as intended unless specified.
  Following is a quick way to fix this:
  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/env/slap-base/lib
  ```
  To do this more systematically refer to `conda`'s documentation on [managing environment variables](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#setting-environment-variables).

## Training SLAP

## Configuration Parameters for Training and Execution

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

## SLAP Configuration parameters for training

### IPM

Under  `config.SLAP.IPM` for training:
- `validate (bool)`: Whether to run IPM in validation mode to visualize ground-truth as well
- `datadir (bool)`: Placeholder so user can input `--datadir <path/to/local/data` while running training or validation
- `template (bool)`: Collects all H5 files matching template `<path/to/local/data/temlate>`
- `load (string)`: Placeholder so user can pass which checkpoint to load for validation or resuming training
- `resume (bool)`: Load weights at `load` path before starting training
- `learning_rate (float)`: Learning rate for LAMB optimizer
- `task_name (string)`: Name of the task/experiment being trained, useful for wandb and directory tracking
- `source (string)`: Loads data using `stretch/franka' configuration
- `max_iter (int)`: Total number of training iterations
- `wandb (bool)`: Store run in wandb project
- `split (string)`: Placeholder, user can input `--split <path/to/train-test-split.yaml>` for consistent experiments (if not given anything the model trains on all samples collected from H5s found in `datadir`)
- `data_augmentation (bool)`: Train the model with positional, orientational, and cropping data-augmentation ON
- `color_jitter (bool)`: Use color-jitter for additional data-augmentation
- `loss_fn (string)`: Loss-function to use for training IPM classifier (supports xent and bce; xent
  gave a better result in experiments)

### APM

fill-me