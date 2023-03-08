import numpy as np
import torch
import tqdm

from home_robot.utils.data_tools.writer import DataWriter
from home_robot.utils.data_tools.loader import DatasetBase

class SimpleDataset(DatasetBase):
    """Simple example data loader."""

    def get_datum(self, trial, idx):
        """Get a single training example given the index."""
        breakpoint()
        datum = {
                'pos': torch.FloatTensor(trial['pos'][idx]),
                'res': torch.FloatTensor(trial['res'][idx] / np.pi),
                }
        return datum

if __name__=="__main__":
    dataset = SimpleDataset('/home/vidhij/H5s/test/')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)    
    entry = next(iter(dataloader))