## Collect H5 Files

1. Make sure the main hector-slam launch file is up and running.
1. On a new terminal run `rosrun joy joy_node`
1. Turn on the controller and make sure it is connect to the PC
1. Run the data collection script: `python projects/collect_h5.py --task-name <name of task>`

Note that the script collect keyframe based demonstrations at the moment, i.e. user needs to communicate episode start, moments of interest during episode and episode end to the script.

### Controls for Data Collection

- To start an episode press the `Start` button on the controller (make sure the script starts writing to a file)
- To communicate keyframes, press `Back` button
- To end the episode press `Start` button again (we collect another frame at the end of the episode, it does not need explicit marking as a keyframe)
