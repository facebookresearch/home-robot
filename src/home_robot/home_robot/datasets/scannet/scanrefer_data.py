# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/daveredrum/ScanRefer
# ScanRefer is licensed under a
# Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import pandas as pd


class ScanReferDataConfig:
    json_dir: Union[Path, str] = Path("scanrefer")


def load_scanrefer_data(json_fpath: Union[Path, str]):
    df = pd.read_json(json_fpath)

    # Rename to have some columns the same format as referit3d
    df = df.rename(
        columns={
            "scene_id": "scan_id",
            "object_name": "instance_type",
            "token": "tokens",
            "object_id": "target_id",
            "description": "utterance",
        }
    )
    df["stimulus_id"] = (
        df["scan_id"]
        + "_"
        + df["target_id"].astype(str)
        + "_"
        + df["ann_id"].astype(str)
    )
    df["dataset"] = "scanrefer"
    # We could load annotated viewpoints, too.
    # ID key is "<scene_id>-<object_id>_<ann_id>"
    # However, the viewpoints have "position" (3,) and "rotation" (3,).
    # Is rotation euler angles? If so, opencv camera convention?
    # I'm not opening that can of worms until we need it.
    return df


def get_num_distractor(
    scanrefer_df: pd.DataFrame,
    raw_to_label: Dict[str, int],  # Mapping of string to semantic label class
):
    # Placeholder code
    raise NotImplementedError
    # all_sem_labels = {}
    # cache = {}
    # for data in self.scanrefer:
    #     scene_id = data["scan_id"]
    #     object_id = data["target_id"]
    #     object_name = " ".join(data["instance_type"].split("_"))
    #     ann_id = data["ann_id"]

    #     if scene_id not in all_sem_labels:
    #         all_sem_labels[scene_id] = []

    #     if scene_id not in cache:
    #         cache[scene_id] = {}

    #     if object_id not in cache[scene_id]:
    #         cache[scene_id][object_id] = {}
    #         try:
    #             all_sem_labels[scene_id].append(self.raw2label[object_name])
    #         except KeyError:
    #             all_sem_labels[scene_id].append(17)

    # # convert to numpy array
    # all_sem_labels = {
    #     scene_id: np.array(all_sem_labels[scene_id])
    #     for scene_id in all_sem_labels.keys()
    # }

    # unique_multiple_lookup = {}
    # for data in self.scanrefer:
    #     scene_id = data["scene_id"]
    #     object_id = data["object_id"]
    #     object_name = " ".join(data["object_name"].split("_"))
    #     ann_id = data["ann_id"]

    #     try:
    #         sem_label = self.raw2label[object_name]
    #     except KeyError:
    #         sem_label = 17

    #     unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

    #     # store
    #     if scene_id not in unique_multiple_lookup:
    #         unique_multiple_lookup[scene_id] = {}

    #     if object_id not in unique_multiple_lookup[scene_id]:
    #         unique_multiple_lookup[scene_id][object_id] = {}

    #     if ann_id not in unique_multiple_lookup[scene_id][object_id]:
    #         unique_multiple_lookup[scene_id][object_id][ann_id] = None

    #     unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

    # return unique_multiple_lookup
