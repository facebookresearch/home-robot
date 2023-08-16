import numpy as np
from matplotlib import pyplot as plt 

import sys
sys.path.append('src/home_robot')
import pickle
data = pickle.load(open('locals.pkl', 'rb'))
data.keys()
for k,v in data.items():
    locals()[k] = v



