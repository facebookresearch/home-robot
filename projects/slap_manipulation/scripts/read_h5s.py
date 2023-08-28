# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Shows input image, name of the task, name of the file, name of the episode, the
labeled `FAIL/SUCCESS` status and names of all keys present for each episode.
"""
import glob
import os

import click
import h5py
import matplotlib.pyplot as plt

import home_robot.utils.data_tools.image as image


@click.command()
@click.option("--data-dir", default="./", help="Path where the files are stored")
@click.option(
    "--template",
    default="*.h5",
    help="Files will be looked up using regex: data-dir/template",
)
def main(data_dir, template):
    files = glob.glob(os.path.join(data_dir, template))
    for file in files:
        filename = file.split("/")[-1][:-3]
        dir_name = file.split("/")[-2]
        h5 = h5py.File(file, "r")
        for g_name in h5:
            print(f"\n{dir_name=}, {filename=}, {g_name=}")
            print(f"task-name = {h5[g_name]['task_name'][()].decode('utf-8')}")
            rgb = image.img_from_bytes(h5[g_name]["head_rgb/0"][()])
            print(h5[g_name]["demo_status"][()])
            plt.imshow(rgb)
            plt.show()
            print(h5[g_name].keys())
        h5.close()


if __name__ == "__main__":
    main()
