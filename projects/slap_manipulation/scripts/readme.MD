## Test Detic Detections on Observations coming from camera-stream

- Launch bootup script on the robot
- Run `test_detic_on_obs.py` from $HOME_ROBOT_ROOT:
  `python projects/slap_manipulation/scripts/test_detic_on_obs.py --category object1 --category object2`
- If no objects are given the script will use default category vocab
  `MY_CATEGORIES = ["cup", "bottle", "drawer", "basket", "bowl", "computer", "mug"]`
