# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Callable, Union

import torch
from atomicwrites import atomic_write

from home_robot.utils.threading import Interval

logger = logging.getLogger(__name__)


class DataDirectoryConsumer:
    """
    Watches a directory for new observations and loads them subject to rate-limiting.
    The directory should contain one file per timestep and be structured as:
    path_to_dir/
      1.pkl
      2.pkl
      ...
    As new .pkl files are added to the directory, this class loads the results and appends them
    to self.observations.

    The option `rate_limit` also sets a maxmimum number of observations to load each second. For example,
    setting rate_limit=5 would load a maximum of 5 observations a second, even if there are more currently
    in the directory.
    """

    def __init__(self, dir_path: Union[Path, str], rate_limit: int = 30):
        self.dir_path = Path(dir_path)
        if not self.dir_path.is_dir():
            raise ValueError(f"{dir_path} is not a valid directory path")

        self.observations = []
        self.current_obs_number = 0
        self.rate_limit = rate_limit
        self.sleep_time = 1.0 / rate_limit
        self._timer = Interval(self._consume_data, sleep_time=self.sleep_time)

    def start(self):
        self._timer.start()

    def _consume_data(self):
        file_path = self.dir_path / f"{self.current_obs_number + 1}.pkl"
        logger.info(f"reading {file_path}")
        if file_path.exists():
            with open(file_path, "rb") as f:
                self.observations.append(pickle.load(f))
            self.current_obs_number += 1

    def stop(self):
        self._timer.cancel()
        self._timer.join()


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    consumer = DataDirectoryConsumer("published_trajectory", rate_limit=1)
    logger.info("Starting consumer")
    consumer.start()
    time.sleep(10)  # let it run for 10 seconds
    consumer.stop()
    for o in consumer.observations:  # Prints the observations loaded in 10 seconds
        logger.info(o["camera_pose"])
