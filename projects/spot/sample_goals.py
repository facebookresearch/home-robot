from configs.goals_abnb1 import GOALS
import numpy as np

exp_name = "spot_abnb1"
keys = GOALS.keys()
np.array(list(keys))
num_trials = 10
for traj_numb in range(num_trials):
    num_goals = np.random.randint(5,10)
    goals = np.random.choice(list(keys),size=(num_goals,),replace=False)
    goal_string = ",".join(goals)
    out = f"DISPLAY=:1 python projects/spot/goat.py --trajectory={exp_name}_trial{traj_numb} --goals={goal_string}"
    print(out)

