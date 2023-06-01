## Test Detic Detections on Observations coming from camera-stream

- Launch bootup script on the robot
- Run `test_detic_on_obs.py` from $HOME_ROBOT_ROOT:
  `python projects/slap_manipulation/scripts/test_detic_on_obs.py --category object1 --category object2`
- If no objects are given the script will use default category vocab
  `MY_CATEGORIES = ["cup", "bottle", "drawer", "basket", "bowl", "computer", "mug"]`



## Evaluating `GeneralLanguageAgent`

- `eval_lang_agent.py` is the entry-point for tinkering with `GeneralLanguageAgent` and `GeneralLanguageEnv`
- Recommend starting a run by entering given command from `$HOME_ROBOT_ROOT`: `python projects/slap_manipulation/scripts/eval_lang_agent.py --testing --task-id 4 --dry-run`
- `--dry-run` flag will make sure the environment doesn't actually make the robot move
- `--testing` flag is for running one of the predefined test-tasks rather than fetching plans from oracle
- `--task-id` is just the index of predefined test-task you wish to run
