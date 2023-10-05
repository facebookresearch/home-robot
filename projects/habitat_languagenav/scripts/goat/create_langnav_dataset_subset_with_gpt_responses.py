import gzip
import json

# import openai
import os
import random

from gpt.run_target_landmark_extraction import generate_gpt_response
from tqdm import tqdm

N = 100  # number of episodes to make subset from
random.seed(100)

langnav_episodes_file = (
    "data/datasets/languagenav/hm3d/v0/val/content/vLpv2VX547B.json.gz"
)

# open gzip file and load json
with gzip.open(langnav_episodes_file, "rt") as f:
    langnav_episodes_dset = json.load(f)

langnav_episodes = langnav_episodes_dset["episodes"]
random_episodes = random.sample(langnav_episodes, N)
for ep in tqdm(random_episodes):
    # instruction = ep["instructions"][0].replace("Instruction: ", "# ")
    instruction = ep["instructions"][0]

    try:
        gpt_response = generate_gpt_response(instruction)
        target, landmarks = gpt_response.split("\n")
        target = target.replace("Target: ", "")
        landmarks = landmarks.replace("Landmarks: ", "")
        landmarks = landmarks.split(", ")
    except Exception as e:
        # to capture occassional RateLimitError
        print(e)
        print("Saving episodes with responses generated so far...")
        break

    if len(landmarks) == 1 and landmarks[0] == "None":
        landmarks = []

    ep["llm_response"] = {
        "target": target,
        "landmark": landmarks,
        "room_name": None,
        "full_response": ep["llm_response"]["full_response"],
    }

# create a copy of langnav_episode_dset
langnav_episodes_dset_subset = langnav_episodes_dset.copy()
langnav_episodes_dset_subset["episodes"] = [
    ep for ep in random_episodes if "llm_response" in ep.keys()
]

# save to file
langnav_episodes_file_subset = (
    "data/datasets/languagenav/hm3d/v0/val/content/vLpv2VX547B.json.gz"
)
os.makedirs(os.path.dirname(langnav_episodes_file_subset), exist_ok=True)
with gzip.open(langnav_episodes_file_subset, "wt") as f:
    json.dump(langnav_episodes_dset_subset, f)
