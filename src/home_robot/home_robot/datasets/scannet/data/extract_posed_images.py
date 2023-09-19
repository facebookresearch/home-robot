# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
    This script is supposed to process downloaded scannet files into a format usable with ScanNet dataset.
    Code is as-is and hasn't been tested with edge cases/other data.

    # Modified from
    #   https://github.com/open-mmlab/mmdetection3d/blob/1.0/data/scannet/extract_posed_images.py # noqa
    #   from https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py # noqa
    
    Which has:
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
"""
import os
import struct
import zlib
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import imageio
import mmengine
import numpy as np

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}

COMPRESSION_TYPE_DEPTH = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


class RGBDFrame:
    """Class for single ScanNet RGB-D image processing."""

    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = b"".join(
            struct.unpack(
                "c" * self.color_size_bytes, file_handle.read(self.color_size_bytes)
            )
        )
        self.depth_data = b"".join(
            struct.unpack(
                "c" * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)
            )
        )

    def decompress_depth(self, compression_type):
        assert compression_type == "zlib_ushort"
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        assert compression_type == "jpeg"
        return imageio.imread(self.color_data)


class SensorData:
    """Class for single ScanNet scene processing.

    Single scene file contains multiple RGB-D images.
    """

    def __init__(self, filename, limit):
        self.version = 4
        self.load(filename, limit)

    def load(self, filename, limit):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = b"".join(struct.unpack("c" * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[
                struct.unpack("i", f.read(4))[0]
            ]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[
                struct.unpack("i", f.read(4))[0]
            ]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = []
            if limit > 0 and limit < num_frames:
                index = np.random.choice(
                    np.arange(num_frames), limit, replace=False
                ).tolist()
            else:
                index = list(range(num_frames))
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                if i in index:
                    self.frames.append(frame)

    def export_depth_images(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.fromstring(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )
            imageio.imwrite(
                os.path.join(output_path, self.index_to_str(f) + ".png"), depth
            )

    def export_color_images(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            color = self.frames[f].decompress_color(self.color_compression_type)
            imageio.imwrite(
                os.path.join(output_path, self.index_to_str(f) + ".jpg"), color
            )

    @staticmethod
    def index_to_str(index):
        return str(index).zfill(5)

    @staticmethod
    def save_mat_to_file(matrix, filename):
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for f in range(len(self.frames)):
            self.save_mat_to_file(
                self.frames[f].camera_to_world,
                os.path.join(output_path, self.index_to_str(f) + ".txt"),
            )

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.save_mat_to_file(
            self.intrinsic_color, os.path.join(output_path, "intrinsic.txt")
        )


def process_scene(scene_path, out_path, limit, idx):
    """Process single ScanNet scene.

    Extract RGB images, poses and camera intrinsics.
    """
    data = SensorData(os.path.join(scene_path, idx, f"{idx}.sens"), limit)
    output_path = os.path.join(out_path, "posed_images", idx)
    data.export_depth_images(output_path)
    data.export_color_images(output_path)
    data.export_intrinsics(output_path)
    data.export_poses(output_path)


def process_directory(path, out_path, limit, nproc):
    print(f"processing {path}")
    mmengine.track_parallel_progress(
        func=partial(process_scene, path, out_path, limit),
        tasks=os.listdir(path),
        nproc=nproc,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max-images-per-scene", type=int, default=300)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument(
        "--scans-dir", type=str, default="/datasets01/scannet/082518/scans"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/private/home/ssax/home-robot/projects/eval_scannet/scans",
    )
    parser.add_argument("--scans-test-dir", type=str, default=None)
    parser.add_argument("--output-test-dir", type=str, default=None)
    args = parser.parse_args()

    scans_path = Path(args.scans_dir)
    scans_test_path = (
        scans_path.parent / "scans_test"
        if args.scans_test_dir is None
        else args.scans_test_dir
    )
    scans_path = str(scans_path)
    out_path = Path(args.output_dir)
    out_test_path = (
        out_path.parent / "scans_test"
        if args.scans_test_dir is None
        else args.scans_test_dir
    )
    out_path = str(out_path)
    print(os.path.exists(scans_path))
    # process train and val scenes
    if os.path.exists(scans_path):
        process_directory(scans_path, out_path, args.max_images_per_scene, args.nproc)
    # process test scenes
    if os.path.exists(scans_test_path):
        process_directory(
            scans_test_path, out_test_path, args.max_images_per_scene, args.nproc
        )
