# Spatial-Language Attention Policies for Efficient Robot Learning

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

