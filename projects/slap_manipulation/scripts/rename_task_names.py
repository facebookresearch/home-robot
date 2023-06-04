"""Script to rename task-names consistently across episodes and files"""
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
def full_rename(data_dir, template, config_file):
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
def rename_edits(data_dir, template, from_key, to_key):
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
