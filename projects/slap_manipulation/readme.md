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

`num_keypoints`: (int) Number of keypoints; currently corresponds to total number of action prediction modules trained for your task
`local_problem_size`: (float) Radius in meter around predicted interaction point which is cropped out as input for action prediction module
`num_pts`: (int) Cardinality of input point-cloud after removing duplicates
`execution.predict_action`: (True/False) Whether to predict action based on predicted interaction point (for debug purposes)

## Task list

- [x] Create a local environment consistent w/slap + home_robot
- [x] Port over IPM, APM, Components they depend upon
- [x] Code StretchManipulationEnv
- [x] IPM
  - [x] Finish data-pipeline
  - [x] Load pre-trained weights and run initial tests
  - [x] Edit to respect new simple API
- [x] APM
  - [x] Finish data-pipeline
  - [ ] Load pre-trained weights and run initial tests (need new weights for these; queued for after trainig)
  - [ ] Edit to respect new simple API
- [ ] --> Collect new data from stretch
- [ ] Edit ported dataloaders to visualize and train using collected H5s

