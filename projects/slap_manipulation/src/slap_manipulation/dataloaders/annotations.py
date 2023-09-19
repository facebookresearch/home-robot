# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import yaml


def load_annotations_dict(filename):
    """load annotations from a file"""
    with open(filename) as f:
        data = yaml.safe_load(f)
    labels = {}
    for entry in data.values():
        key = entry["name"]
        skill_labels = []
        alts = entry["alt_name"]
        labels[key] = [key]
        for alt in alts:
            labels[key].append(alt)
    return labels


if __name__ == "__main__":
    print(load_annotations_dict("assets/language_variations/v0.yml"))
