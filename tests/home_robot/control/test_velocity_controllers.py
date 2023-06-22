# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import os

import numpy as np
import pytest
from utils import generate_controller_input, get_controller_output

from home_robot.control.goto_controller import GotoVelocityController

NUM_ENTRIES = 10
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = "test_xyt_data.json"


@pytest.fixture
def controller():
    return GotoVelocityController()


@pytest.fixture
def dataset(controller):
    dataset_path = os.path.join(DATASET_DIR, DATASET_FILE)

    # If dataset exists: load dataset
    if os.path.exists(dataset_path):
        assert os.path.isfile(dataset_path)
        with open(dataset_path, "r") as f:
            data = json.load(f)

    # If dataset doesn't exist: Generate dataset and save
    else:
        # Generate inputs
        data = []
        for _ in range(NUM_ENTRIES):
            controller_input = generate_controller_input()
            controller_output = get_controller_output(controller, controller_input)
            data.append((controller_input, controller_output))

        with open(dataset_path, "w") as f:
            json.dump(data, f)

    return data


def test_controller_input_output(controller, dataset):
    for x, y_ref in dataset:
        y = get_controller_output(controller, x)
        assert np.allclose(y, y_ref)
