import os
import numpy as np
import torch

from home_robot.utils.data_tools.writer import DataWriter
from home_robot.utils.data_tools.loader import DatasetBase

class SimpleDataset(DatasetBase):
    """Simple example data loader."""

    def get_datum(self, trial, idx):
        """Get a single training example given the index."""
        datum = {
            'temporal': {},
            'image': {},
        }
        for key in trial.temporal_keys:
            datum['temporal'][key] = torch.tensor(trial[key][idx])
        # for key in ['rgb']:
        #     datum['image'][key] = torch.tensor(trial[key][idx])

        return datum

if __name__=="__main__":
    dir_path = '~/H5s'
    task_name = 'test'
    dataset = SimpleDataset(os.path.expanduser(os.path.join(dir_path, task_name)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)    
    entry = next(iter(dataloader))
    print(entry)