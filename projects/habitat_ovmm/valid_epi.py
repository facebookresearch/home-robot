import json
import os

import numpy as np

dataset_dir = "/private/home/xiaohanzhang/data/eplan"
with open(
    "/private/home/xiaohanzhang/accel-cortex/experiments/pmcvay/eplan_epi.json"
) as f:
    valid_epi = json.load(f)
start = os.listdir(dataset_dir).index(valid_epi[-1])
print(f"continue from index: {start}")
for epi in os.listdir(dataset_dir)[start + 1 :]:
    import pickle

    if os.path.exists(
        os.path.join(dataset_dir, epi, "obs_data.pkl")
    ) and os.path.exists(os.path.join(dataset_dir, epi, "bounds.json")):
        with open(os.path.join(dataset_dir, epi, "obs_data.pkl"), "rb") as f:
            obs_data = pickle.load(f)
        with open(os.path.join(dataset_dir, epi, "bounds.json"), "r") as f:
            annotation = json.load(f)

        found_obj = False
        found_recep = False
        for obs in obs_data:
            perceived_ids = np.unique(obs.task_observations["gt_instance_ids"])
            for target_id in annotation["object_ids"]:
                if target_id in perceived_ids:
                    found_obj = True
            for target_id in annotation["goal_recep_ids"]:
                if target_id in perceived_ids:
                    found_recep = True
        if found_obj and found_recep:
            valid_epi.append(epi)
            print(valid_epi)

print(len(valid_epi))
with open(
    "/private/home/xiaohanzhang/accel-cortex/experiments/pmcvay/eplan_epi.json", "w"
) as f:
    json.dump(valid_epi, f, indent=4)
