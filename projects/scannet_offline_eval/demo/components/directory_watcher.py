# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from loguru import logger

from home_robot.utils.threading import Interval


def get_most_recent_viz_directory() -> Optional[str]:
    """
    Get the path of the most recently created 'viz_data' directory under the '/data/hw_exps/spot' path.

    Returns:
        str: The path of the most recently created 'viz_data' directory, or None if no such directory exists.
    """
    search_path = f"{os.environ['HOME_ROBOT_ROOT']}/data/hw_exps/spot/*/viz_data/"
    directories = sorted(glob.glob(search_path), key=os.path.getctime, reverse=True)
    if not directories:
        logger.warning(f"No viz_data directories found in {search_path}")
    return directories[0] if directories else None


class DirectoryWatcher:
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

    def __init__(
        self,
        dir_path: Union[Path, str],
        rate_limit: int = 30,
        on_new_obs_callback: Optional[Callable] = None,
        obs_lookahead: int = 10,
    ):
        self.dir_path = Path(dir_path)
        if not self.dir_path.is_dir():
            raise ValueError(f"{dir_path} is not a valid directory path")

        self.observations = []
        self.current_obs_number = 0
        self.rate_limit = rate_limit
        self.sleep_time = 1.0 / rate_limit
        self.on_new_obs_callback = on_new_obs_callback
        self.obs_lookahead = obs_lookahead
        self._timer = Interval(self._consume_data, sleep_time=self.sleep_time)

    def _consume_data(self):
        file_path = (self.dir_path / f"{self.current_obs_number}.pkl").resolve()
        if file_path.exists():
            logger.info(f"[LOADING] from {self.current_obs_number}.pkl")
            with open(file_path, "rb") as f:
                current_obs = pickle.load(f)
            # self.observations.append(current_obs)
            self.current_obs_number += 1
            if self.on_new_obs_callback is not None:
                self.on_new_obs_callback(current_obs)
        else:
            for i in range(self.obs_lookahead):
                file_path = (
                    self.dir_path / f"{self.current_obs_number + i + 1}.pkl"
                ).resolve()
                if file_path.exists():
                    logger.debug(
                        f"No obs found for timestep {self.current_obs_number}.pkl, but found for timestep {self.current_obs_number + i + 1}.pkl."
                    )
                    self.current_obs_number += i + 1
                    return self._consume_data()

            logger.trace(
                f"[WAITING] No obs found for timestep {self.current_obs_number}.pkl"
            )
            if self.on_new_obs_callback is not None:
                self.on_new_obs_callback(None)
        return True

    def pause(self):
        self._timer.pause()

    def unpause(self):
        self._timer.unpause()

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.cancel()
        self._timer.join()


# Example Usage
if __name__ == "__main__":
    consumer = DirectoryWatcher("published_trajectory/obs", rate_limit=1)
    logger.info("Starting consumer")
    consumer.start()
    time.sleep(10)  # let it run for 10 seconds
    consumer.stop()
    for o in consumer.observations:  # Prints the observations loaded in 10 seconds
        logger.info(o["camera_pose"])
