# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Given a folder of h5 files, create a train/test/val split and dump it out to a yaml file
"""
import datetime
import glob
import os
from pprint import pprint

import click
import h5py
import numpy as np
import yaml


@click.command()
@click.option("-d", "--data-path")
@click.option("-t", "--template", default="*.h5")
@click.option("--train-num", type=int, help="number of trials for training")
@click.option("--val-num", type=int, help="number of trials for validation")
@click.option("-n", "--task-name", help="name to associate with this split")
def main(data_path, template, train_num, val_num, task_name):
    """function to dump out train and trial dicts associated with h5s in a folder"""
    files = sorted(glob.glob(os.path.join(data_path, template)))
    print("Found these files:", files)
    split = {"train": [], "test": [], "val": []}
    trial_names = np.array([])
    for filename in files:
        # Check each file to see how many entires it has
        with h5py.File(filename, "r") as h5:
            trial_names = np.concatenate((trial_names, (list(h5.keys()))))
    total_num = trial_names.shape[0]
    test_num = total_num - train_num - val_num
    random_idx = np.arange(trial_names.shape[0])
    np.random.shuffle(random_idx)
    train_names = trial_names[random_idx[:train_num]]
    test_names = trial_names[random_idx[train_num : train_num + test_num]]
    val_names = trial_names[random_idx[train_num + test_num :]]
    split["train"].extend(train_names.tolist())
    split["test"].extend(test_names.tolist())
    split["val"].extend(val_names.tolist())
    pprint(split)
    today = datetime.date.today()
    filename = f"./assets/train_test_val_split_{task_name}_{today}.yaml"
    with open(filename, "w") as yaml_file:
        yaml.dump(split, yaml_file, default_flow_style=False)
    print(f"Written to {filename}")


main()
