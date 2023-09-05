"""Script to edit the demonstrations so that the opening action
is farther back for open-object-drawer task.
Needed for putting/picking things from drawers."""

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
def edit_opening_drawer(data_dir, template):
    """moves the opening action farther back by 10 cm along y-axis"""
    action_index = 3
    files = glob.glob(os.path.join(data_dir, template))
    for file in files:
        h5 = h5py.File(file, "a")
        for g_name in h5:
            if "edited" in h5[g_name].keys():
                print(f"ALREADY PROCESSED. Skipping {g_name} in {file=}")
                continue
            action_position = h5[g_name]["ee_pose"][()]
            # we want to move the action forward by 0.1 along y-axis
            action_position[action_index][1] += 0.1
            h5[g_name]["ee_pose"][...] = action_position
            h5[g_name]["edited"] = 1.0
        h5.close()


if __name__ == "__main__":
    edit_opening_drawer()
