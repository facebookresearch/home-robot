"""A script to create a new h5 file consisting of only the 1st episode from a list of h5 files"""
import glob
import os

import click
import h5py


@click.command()
@click.option("--new_file_name", type=click.Path())
def main(new_file_name):
    data_dir = "./joint_h5s/"
    # Get the list of h5 files
    h5_files = glob.glob(os.path.join(data_dir, "*.h5"))

    # Create a new h5 file
    new_h5_file = h5py.File(new_file_name, "w")

    # For each h5 file in the list
    for i, h5_file in enumerate(h5_files):
        print(f"{h5_file=}")
        new_h5_file[f"/0_{i}"] = h5py.ExternalLink(h5_file, "/0")

    # Close the new h5 file
    new_h5_file.close()


if __name__ == "__main__":
    main()
