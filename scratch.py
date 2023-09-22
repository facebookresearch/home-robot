import numpy as np
from matplotlib import pyplot as plt 
import os
from util.pyutil import sorted_file_match
from einops import rearrange, reduce, repeat
import pickle as pkl
import pathlib
from tqdm import tqdm

from glob import glob
folder = "/Volumes/Extreme Pro/airbnb_data/spot_abnb2_video3 => only for apartment view"
pathlib.Path(f'{folder}/depth_vis').mkdir(parents=True, exist_ok=True)
files,nums = sorted_file_match(f'{folder}/obs','(\d+).pkl')
for fil,num in tqdm(zip(files,nums)):
    fil = files[223]
    obs = pkl.load(open(f'{folder}/obs/{fil}','rb'))
    obs.task_observations.keys()
    de = obs.depth
    od = obs.task_observations['orig_depth']
    sf = obs.task_observations['semantic_frame']
    de/=de.max()
    od/=od.max()
    de = repeat(de,'h w -> h w c',c=3)
    od = repeat(od,'h w -> h w c',c=3)
    de.shape
    od.shape
    plt.imshow(np.hstack([de,od,sf[:,:,::-1]/255]))
    plt.savefig(f'{folder}/depth_vis/{num}.pdf')
    plt.clf()

files[223]
from util.pyutil import write_images
write_images('vis/test.png',sf[:,:,::-1])
write_images('vis/test.png',od)


# import sys
# sys.path.append('src/home_robot')
# import pickle
# data = pickle.load(open('locals.pkl', 'rb'))
# data.keys()
# for k,v in data.items():
    # locals()[k] = v




