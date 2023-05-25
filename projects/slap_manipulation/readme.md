# Spatial-Language Attention Policies for Efficient Robot Learning

## Configuration Parameters

`num_keypoints`: (int) Number of keypoints; currently corresponds to total number of action prediction modules trained for your task
`local_problem_size`: (float) Radius in meter around predicted interaction point which is cropped out as input for action prediction module
`num_pts`: (int) Cardinality of input point-cloud after removing duplicates
`execution.predict_action`: (True/False) Whether to predict action based on predicted interaction point (for debug purposes)

## Task list

- [ ] Proper documentation
- [ ] Upload data and add data download instructions for users to tinker with
- [x] Create a local environment consistent w/slap + home_robot
- [x] Port over IPM, APM, Components they depend upon
- [x] Code StretchManipulationEnv
- [x] IPM
  - [x] Finish data-pipeline
  - [x] Load pre-trained weights and run initial tests
  - [x] Edit to respect new simple API
- [x] APM
  - [x] Finish data-pipeline
  - [x] Load pre-trained weights and run initial tests (need new weights for these; queued for after trainig)
  - [x] Edit to respect new simple API
- [x] Collect new data from stretch
- [x] Edit ported dataloaders to visualize and train using collected H5s

## Natural Language Commands to Steps
```
cd scripts
python get_cmd_to_steps_data.py
```
This will connect to wandb and download the data from the artifact `bring_x_from_y:latest`, containing 6 json files.
