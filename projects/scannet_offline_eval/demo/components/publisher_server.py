# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import base64
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from loguru import logger
from quart import Quart, websocket

from .app import app_config, svm_watcher
from .directory_watcher import DirectoryWatcher

# Setup small Quart server for streaming via websocket, one for each stream.
socket_server = Quart(__name__)
n_streams = 2

FPS = 5.0
WAIT_TIME = 1.0 / FPS
#########################################
# Observation consumer
#########################################
class ObservationDirectoryWatcher(DirectoryWatcher):
    def __init__(
        self,
        watch_dir: Path,
        fps: int = 1,
        loop: bool = False,
    ):
        super().__init__(watch_dir, rate_limit=fps, on_new_obs_callback=self.add_obs)
        self.observations = []
        self.rgb_jpeg = None
        self.loop = loop

    def add_obs(self, obs: Dict[str, Any]):
        self.observations = []
        if obs is None:
            logger.debug("No obs -- resetting")
            if self.loop:
                self.current_obs_number = 0
            return True

        logger.debug(f"Adding obs {self.current_obs_number}")
        self.rgb_jpeg = cv2.imencode(
            ".jpg", (obs["rgb"].cpu().numpy() * 255).astype(np.uint8)
        )[1].tobytes()
        return True


watcher = ObservationDirectoryWatcher(
    watch_dir=app_config.directory_watch_path,
    fps=FPS,
    loop=True,
)


@socket_server.websocket("/gripper-feed-ws")
async def random_data():
    last_obs = -1
    while True:
        frame = watcher.rgb_jpeg
        current_obs = watcher.current_obs_number
        if frame is not None and current_obs != last_obs:
            logger.warning("Sending frame")
            await websocket.send(
                f"data:image/jpeg;base64, {base64.b64encode(frame).decode()}"
            )
            last_obs = current_obs
        await asyncio.sleep(WAIT_TIME)


if __name__ == "__main__":
    watcher.start()
    socket_server.run(port=5000, debug=True)
