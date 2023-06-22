# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

import torch

from home_robot.utils.data_tools.loader import DatasetBase, Trial


class SimpleDataset(DatasetBase):
    """Simple example data loader."""

    def get_datum(self, trial: Trial, idx: int):
        """Get a single training example given the index."""
        datum = {
            "temporal": {},
            "image": {},
        }
        for key in trial.temporal_keys:
            datum["temporal"][key] = torch.tensor(trial[key][idx])
        for key in trial.image_keys:
            datum["image"][key] = torch.tensor(trial.get_img(key, idx))

        return datum


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dir_path = "projects/data_collection/sample_data"
    task_name = "test"
    dataset = SimpleDataset(os.path.join(os.getcwd(), dir_path, task_name))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False
    )
    for idx, entry in enumerate(dataloader):
        print(f"Entry {idx}")
        # entry = next(iter(dataloader))
        for k, v in entry["temporal"].items():
            print("\t", k, ": ", v)

        plt.clf()
        plt.subplot(121)
        plt.imshow(entry["image"]["rgb"].squeeze().numpy())
        plt.subplot(122)
        plt.imshow(entry["image"]["depth"].squeeze().numpy())
        plt.show()
