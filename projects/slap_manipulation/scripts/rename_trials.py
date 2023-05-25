"""Script to fix names of trials s.t. they are unique
Only required for the few files collected using prev version of collect_h5.py
"""
import glob
import os

import click
import h5py


@click.command()
@click.option("--data-dir", default="./", help="Path where the files are stored")
@click.option(
    "--template",
    default="*.h5",
    help="Files will be looked up using regex: data-dir/template",
)
@click.option("--mode", type=click.Choice(["read", "write"]), default="read")
def main(data_dir, template, mode):
    files = glob.glob(os.path.join(data_dir, template))
    for file in files:
        filename = file.split("/")[-1][:-3]
        if mode == "write":
            h5 = h5py.File(file, "r+")
            start_g_names = []
            for g_name in h5:
                start_g_names.append(g_name)
            for g_name in start_g_names:
                h5[f"{filename}_{g_name}"] = h5[g_name]
                del h5[g_name]
        else:
            h5 = h5py.File(file, "r")
            for g_name in h5:
                print(f"{filename},{g_name}")
                print(h5[g_name].keys())
        h5.close()


if __name__ == "__main__":
    main()
