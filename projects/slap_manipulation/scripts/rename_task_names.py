# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to rename task-names consistently across episodes and files
Looks for a config file that maps old task-names to new task-names
"""
import glob
import os

import click
import h5py
import yaml


@click.command()
@click.option("--data-dir", default="./", help="Path where the files are stored")
@click.option(
    "--template",
    default="*.h5",
    help="Files will be looked up using regex: data-dir/template",
)
@click.option("--config-file", type=str, default="configs/std_task_name_mapping.yaml")
def full_rename(data_dir: str, template: str, config_file: str):
    """reads a config at path :config_file: that maps old task-names to new
    task-names and changes names consistently everywhere"""
    with open(config_file, "r") as f:
        task_mapping = yaml.load(f, Loader=yaml.FullLoader)
    files = glob.glob(os.path.join(data_dir, template))
    for file in files:
        h5 = h5py.File(file, "a")
        for g_name in h5:
            task_name = h5[g_name]["task_name"][()].decode("utf-8")
            h5[g_name]["task_name"][...] = task_mapping[task_name]
        h5.close()


@click.command()
@click.option("--data-dir", default="./", help="Path where the files are stored")
@click.option(
    "--template",
    default="*.h5",
    help="Files will be looked up using regex: data-dir/template",
)
@click.option("--from-key", type=str)
@click.option("--to-key", type=str)
def rename_edits(data_dir: str, template: str, from_key: str, to_key: str):
    """instead of config file, user can more precisely rename task-names from
    :from_key: to :to_key:"""
    files = glob.glob(os.path.join(data_dir, template))
    for file in files:
        h5 = h5py.File(file, "a")
        for g_name in h5:
            task_name = h5[g_name]["task_name"][()].decode("utf-8")
            if task_name == from_key:
                h5[g_name]["task_name"][...] = to_key
        h5.close()


if __name__ == "__main__":
    rename_edits()
