#TODO Add data/spot utils here

import pickle
import skvideo.io
import numpy as np
import os
path = "/home/jaydv/Documents/home-robot/data/hw_exps/spot/2023-10-05-18-16-14/spot_output_2023-10-05-18-16-14.pkl"
o = open(path, 'rb+')
obj = pickle.load(o)
obj.keys()
img = []
for i in obj['rgb']:
    img.append(np.asarray(i))
skvideo.io.vwrite(f"video.gif", np.asarray(img))