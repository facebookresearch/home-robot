# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Any, Dict

import h5py
import numpy as np

import home_robot.utils.data_tools.base as base
import home_robot.utils.data_tools.image as image


class DataWriter(object):
    """This class contains tools to write out data into an hdf5 file containing many different
    trials. This is a replacement for using numpy or pickle objects since hdf5 is slightly
    more suitable for training.

    The idea is that each trial is listed as a top-level hdf5 "group," which contains state,
    observation, action spaces, among other things.
    """

    def __init__(self, filename="data.h5", dirname=None):
        """
        Optionally initialize with a directory.
        """
        self.filename = filename
        self.dirname = dirname
        if dirname is not None:
            try:
                os.mkdir(dirname)
            except OSError as e:
                print(e)
            self.filename = os.path.join(self.dirname, self.filename)
        else:
            self.filename = self.filename
        self.num_trials = 0
        self.reset()

    def reset(self):
        """Reset all information about the trial here"""
        self.temporal_data = {}
        self.config_data = {}
        self.img_data = {}

    def add_img_frame(self, **data):
        data = self.fix_data(data)
        for k, v in data.items():
            if k in self.config_data:
                raise RuntimeError(
                    "duplicate key: " + str(k) + " was in config data already."
                )
            if k in self.temporal_data:
                raise RuntimeError(
                    "duplicate key: " + str(k) + " was in temporal data already."
                )
            if k not in self.img_data:
                self.img_data[k] = []
            data = image.img_to_bytes(v)
            self.img_data[k].append(data)

    def add_frame(self, **data):
        """Add data fields to tracked temporal data"""
        data = self.fix_data(data)
        for k, v in data.items():
            # TODO check data types here
            if k in self.config_data:
                raise RuntimeError(
                    "duplicate key: " + str(k) + " was in config data already."
                )
            if k in self.img_data:
                raise RuntimeError(
                    "duplicate key: " + str(k) + " was in image data already."
                )
            if k not in self.temporal_data:
                self.temporal_data[k] = []
            self.temporal_data[k].append(v)
        return True

    def fix_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten dictionaries"""
        new_data = {}
        for k, v in data.items():
            if isinstance(v, dict):
                # Compute a dictionary with flattened keys
                # This will then be added to the new data snapshot
                flattened_dict = self.flatten_dict(k, v)
                new_data.update(flattened_dict)
            else:
                new_data[k] = v
        return new_data

    def flatten_dict(self, key: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Take a dictionary (data) and turn it into a flat dict, with keys separated by
        slashes as per os.path.join"""
        new_data = {}
        for k, v in data.items():
            # Get the path correctly
            k2 = os.path.join(key, k)
            if isinstance(v, dict):
                # Call this function recursively
                # Open to slightly better ideas?
                new_data.update(self.flatten_dict(k2, v))
            else:
                new_data[k2] = v
        return new_data

    def add_config(self, **data):
        """Add data fields to tracked config (global) data"""
        data = self.fix_data(data)
        for k, v in data.items():
            # TODO check data types here
            if k in self.config_data:
                raise RuntimeError(
                    "duplicate key: " + str(k) + " was in config data already."
                )
            if k in self.temporal_data:
                raise RuntimeError(
                    "duplicate key: " + str(k) + " was in temporal data already."
                )
            if k in self.img_data:
                raise RuntimeError(
                    "duplicate key: " + str(k) + " was in image data already."
                )
            self.config_data[k] = v
        return True

    def write_trial(self, trial_id=None):
        """
        Finish adding data and write to hdf5 archive
        """
        if trial_id is None:
            trial_id = str(self.num_trials)
        else:
            trial_id = str(trial_id)
        self.num_trials += 1
        with h5py.File(self.filename, "a") as h5_file:
            trial = h5_file.create_group(trial_id)

            # Now write the example out here
            temporal_keys = ",".join(list(self.temporal_data.keys()))
            img_keys = ",".join(list(self.img_data.keys()))
            config_keys = ",".join(list(self.config_data.keys()))
            data = self.temporal_data
            data.update(self.config_data)
            for k, v in data.items():
                # Add this to the hdf5 file
                if k[-1] == "_":
                    raise RuntimeError(
                        "invalid name for dataset key: "
                        + str(k)
                        + " cannot end with _; this is reserved."
                    )
                try:
                    trial[k] = v
                except TypeError as e:
                    print(e)
                    print("Cannot use type: ", k)
                    import pdb

                    pdb.set_trace()
            for k, v in self.img_data.items():
                for i, bindata in enumerate(v):
                    ki = os.path.join(k, str(i))
                    trial[ki] = np.void(bindata)
            trial[base.TEMPORAL_KEYS] = temporal_keys
            trial[base.CONFIG_KEYS] = config_keys
            trial[base.IMAGE_KEYS] = img_keys

        # At the end, clear current stored data + configs
        self.reset()
        return True
