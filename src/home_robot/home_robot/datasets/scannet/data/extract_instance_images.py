# This script is supposed to process downloaded scannet files into a format usable with ScanNet dataset.
# Code is as-is and hasn't been tested with edge cases/other data.
# # Modified from
# https://github.com/open-mmlab/mmdetection3d/blob/1.0/data/scannet/batch_load_scannet_data.py # noqa
# which was modified from https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py
#
# MIT License

# Copyright (c) Meta Platforms, Inc. and affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations.

Usage example: python ./batch_load_scannet_data.py
"""
import argparse
import datetime
import multiprocessing as mp
import os
import subprocess
from functools import partial
from os import path as osp

import numpy as np

# from load_scannet_data import export

DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
)


def export_one_scan(
    scan_name,
    output_folder,
    scannet_dir,
    test_mode=False,
):
    print("-" * 20 + f"begin {scan_name} {datetime.datetime.now()}")
    output_filename_prefix = osp.join(output_folder, scan_name)
    os.makedirs(output_filename_prefix, exist_ok=True)
    instance_filt_file = osp.join(
        scannet_dir, scan_name, scan_name + "_2d-instance-filt.zip"
    )
    instance_raw_file = osp.join(scannet_dir, scan_name, scan_name + "_2d-instance.zip")
    label_filt_file = osp.join(scannet_dir, scan_name, scan_name + "_2d-label-filt.zip")
    label_raw_file = osp.join(scannet_dir, scan_name, scan_name + "_2d-label.zip")
    subprocess.run(
        f"unzip -q -o {instance_filt_file} -d {output_filename_prefix}",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"unzip -q -o {instance_raw_file} -d {output_filename_prefix}",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"unzip -q -o {label_filt_file} -d {output_filename_prefix}",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"unzip -q -o {label_raw_file} -d {output_filename_prefix}",
        shell=True,
        check=True,
    )
    print("-" * 20 + f"END {scan_name} {datetime.datetime.now()}")


def batch_export(output_folder, scan_names_file, scannet_dir, test_mode=False, nproc=1):
    if test_mode and not os.path.exists(scannet_dir):
        # test data preparation is optional
        return
    if not os.path.exists(output_folder):
        print(f"Creating new data folder: {output_folder}")
        os.mkdir(output_folder)

    scan_names = [line.rstrip() for line in open(scan_names_file)]
    f = partial(
        export_one_scan,
        # scan_name=scan_name,
        output_folder=output_folder,
        scannet_dir=scannet_dir,
        test_mode=test_mode,
    )
    with mp.Pool(nproc) as p:
        p.map(f, scan_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_folder",
        default="./scannet_2d_instance_data",
        help="output folder of the result.",
    )
    parser.add_argument(
        "--train_scannet_dir", default="scans", help="scannet data directory."
    )
    parser.add_argument(
        "--test_scannet_dir", default="scans_test", help="scannet data directory."
    )
    parser.add_argument(
        "--train_scan_names_file",
        default="meta_data/scannet_train.txt",
        help="The path of the file that stores the scan names.",
    )
    parser.add_argument(
        "--test_scan_names_file",
        default="meta_data/scannetv2_test.txt",
        help="The path of the file that stores the scan names.",
    )
    parser.add_argument(
        "--nproc",
        default=1,
        type=int,
        help="How many processes to do in parallel",
    )
    args = parser.parse_args()
    batch_export(
        args.output_folder,
        args.train_scan_names_file,
        args.train_scannet_dir,
        test_mode=False,
        nproc=args.nproc,
    )
    # batch_export(
    #     args.output_folder,
    #     args.test_scan_names_file,
    #     args.test_scannet_dir,
    #     test_mode=True,
    #     nproc=args.nproc,
    # )


if __name__ == "__main__":
    main()
