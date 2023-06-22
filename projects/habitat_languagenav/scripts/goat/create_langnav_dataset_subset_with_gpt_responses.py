import gzip
import json
# import openai
import os
import random
from tqdm import tqdm

from gpt.run import generate_gpt_response

N = 100 # number of episodes to make subset from
random.seed(100)

langnav_episodes_file = "data/datasets/languagenav/hm3d/v2/val/content/vLpv2VX547B.json.gz"

# open gzip file and load json
with gzip.open(langnav_episodes_file, "rt") as f:
    langnav_episodes_dset = json.load(f)

langnav_episodes = langnav_episodes_dset['episodes']
random_episodes = random.sample(langnav_episodes, N)

for ep in tqdm(random_episodes):
    instruction = ep['instructions'][0].replace("Instruction: ", "# ")

    try:
        gpt_response = generate_gpt_response(ep['instructions'][0])
    except Exception as e:
        # to capture occassional RateLimitError
        print(e)
        print("Saving episodes with responses generated so far...")
        break

    ep['llm_response'] = {
        'landmark': None,
        'room_name': None,
        'full_response': gpt_response
    }

# create a copy of langnav_episode_dset
langnav_episodes_dset_subset = langnav_episodes_dset.copy()
langnav_episodes_dset_subset['episodes'] = [ep for ep in random_episodes if 'llm_response' in ep.keys()]

# save to file
langnav_episodes_file_subset = "data/datasets/languagenav/hm3d/v2/minival/content/vLpv2VX547B.json.gz"
os.makedirs(os.path.dirname(langnav_episodes_file_subset), exist_ok=True)
with gzip.open(langnav_episodes_file_subset, "wt") as f:
    json.dump(langnav_episodes_dset_subset, f)

