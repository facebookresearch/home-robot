# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/referit3d/referit3d
# ReferIt3D license:
# MIT License

# Copyright (c) 2020 referit3d

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

import logging
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReferIt3dDataConfig:
    nr3d_csv_fpath: Union[Path, str] = Path("referit3d") / "nr3d.csv"
    """ Absolute path to NR3D CSV, or else is interpreted relative to scannet root dir """

    sr3d_csv_fpath: Optional[Union[Path, str]] = Path("referit3d") / "sr3d.csv"
    """ Absolute path to SR3D CSV, or else is interpreted relative to scannet root dir """

    mentions_target_class_only: bool = True
    """ Only include utterances that mention the target class"""

    max_seq_len: int = 24
    """ Discard utterances longer than max_seq_len tokens"""


def decode_stimulus_string(s: str):
    """
    Split into scene_id, instance_label, # objects, target object id,
    distractors object id.

    :param s: the stimulus string
    """
    if len(s.split("-", maxsplit=4)) == 4:
        scene_id, instance_label, n_objects, target_id = s.split("-", maxsplit=4)
        distractors_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractors_ids = s.split(
            "-", maxsplit=4
        )

    instance_label = instance_label.replace("_", " ")
    n_objects = int(n_objects)
    target_id = int(target_id)
    distractors_ids = [int(i) for i in distractors_ids.split("-") if i != ""]
    assert len(distractors_ids) == n_objects - 1

    return scene_id, instance_label, n_objects, target_id, distractors_ids


def load_referit3d_data(
    nr3d_csv_fpath: Union[Path, str],
    scans_split: Sequence,
    mentions_target_class_only: bool = True,
    max_seq_len: int = 24,
    sr3d_csv_fpath: Optional[Union[Path, str]] = None,
):
    """
    Args:

    :param args:
        mentions_target_class_only: bool = True,
        max_seq_len: int = 24,
        refernr3d_csv_fpathit_csv:
    :param scans_split:
    :return:
    """
    assert nr3d_csv_fpath is not None, "Cannot use sr3d without nr3d"

    referit_data = pd.read_csv(nr3d_csv_fpath)

    if mentions_target_class_only:
        n_original = len(referit_data)
        referit_data = referit_data[referit_data["mentions_target_class"]]
        referit_data.reset_index(drop=True, inplace=True)
        logger.info(
            "Dropping utterances without explicit "
            "mention to the target class {}->{}".format(n_original, len(referit_data))
        )

    referit_data = referit_data[
        [
            "tokens",
            "instance_type",
            "scan_id",
            "dataset",
            "target_id",
            "utterance",
            "stimulus_id",
        ]
    ]
    referit_data.tokens = referit_data["tokens"].apply(literal_eval)

    # Add the is_train data to the pandas data frame (needed in creating data loaders for the train and test)
    is_train = referit_data.scan_id.apply(lambda x: x in scans_split["train"])
    referit_data["is_train"] = is_train
    referit_columns = (
        referit_data.columns
    )  # Get columns here because if DF has 0 rows, pandas also deletes the columns
    referit_data = referit_data[is_train]

    # Trim data based on token length
    # train_token_lens = referit_data.tokens[is_train].apply(lambda x: len(x))
    train_token_lens = referit_data.tokens.apply(lambda x: len(x))
    if len(train_token_lens) == 0:
        logger.info(f"No NR3D expressions found for scenes {scans_split['train']}")
    else:
        pctile = 95
        logger.info(
            f"{pctile}-th percentile of token length for remaining (training) data"
            + f" is: {np.percentile(train_token_lens, 95):.1f}"
        )
    n_original = len(referit_data)
    referit_data = referit_data[
        referit_data.tokens.apply(lambda x: len(x) <= max_seq_len)
    ]
    referit_data.reset_index(drop=True, inplace=True)
    logger.info(
        f"Dropping utterances with more than {max_seq_len} tokens, {n_original}->{len(referit_data)}"
    )

    # do this last, so that all the previous actions remain unchanged
    if sr3d_csv_fpath is not None:
        logger.info("Adding Sr3D as augmentation.")
        sr3d = pd.read_csv(sr3d_csv_fpath)
        sr3d.tokens = sr3d["tokens"].apply(literal_eval)
        is_train = sr3d.scan_id.apply(lambda x: x in scans_split["train"])
        sr3d["is_train"] = is_train
        sr3d = sr3d[is_train]
        sr3d = sr3d[referit_columns]
        logger.info(f"Dataset-size before augmentation: {len(referit_data)}")
        referit_data = pd.concat([referit_data, sr3d], axis=0)
        referit_data.reset_index(inplace=True, drop=True)
        logger.info(f"Dataset-size after augmentation: {len(referit_data)}")

    context_size = referit_data.stimulus_id.apply(
        lambda x: decode_stimulus_string(x)[2]
    )
    logger.info(
        "(mean) Random guessing among target-class test objects {:.4f}".format(
            (1 / context_size).mean()
        )
    )

    return referit_data
