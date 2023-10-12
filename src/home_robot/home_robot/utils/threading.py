# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


class Interval(threading.Thread):
    """
    A Timer class that executes a given function at a specified interval.

    :param fn: Callable function to be executed.
    :type fn: Callable
    :param sleep_time: The time interval (in seconds) between each execution of the function.
    :type sleep_time: float
    :param count: The number of times the function will be executed.
                  If None, the function will be executed indefinitely until canceled.
    :type count: int, optional

    :ivar event: Threading Event used to stop the thread execution.
    :ivar fn: The function that will be executed at each interval.
    :ivar count: The remaining number of times the function will be executed.
    :ivar sleep_time: The time interval between each execution of the function.

    :example:
    >>> def print_hello():
    ...     print("Hello")
    ...
    >>> timer = TimerClass(print_hello, 1, 5)
    >>> timer.start()
    >>> # "Hello" will be printed 5 times at 1-second intervals.
    >>> timer.cancel()  # Cancels the timer

    """

    def __init__(
        self, fn: Callable, sleep_time: float, count: int = None, daemon=False
    ):
        threading.Thread.__init__(self, daemon=daemon)
        self.event = threading.Event()
        self.fn = fn
        self.count = count
        self.sleep_time = sleep_time
        self.unpause_event = threading.Event()
        self.unpause_event.set()

    def run(self):
        """
        Overrides the run method of threading.Thread.
        Executes the specified function at the given interval.
        """
        while (self.count is None or self.count > 0) and not self.event.is_set():
            start_time = time.time()
            self.unpause_event.wait()
            return_val = self.fn()

            if return_val not in [True, False]:
                raise ValueError(
                    "Function run in Interval must return a bool (is_not_finished)"
                )
            if not return_val:
                self.cancel()
            if self.sleep_time is None:
                self.unpause_event.clear()
            else:
                wait_time = self.sleep_time - (time.time() - start_time)
                self.event.wait(wait_time)
            if self.count is not None:
                self.count -= 1

    def cancel(self):
        """
        Cancels the Timer and stops the execution of the function at intervals.

        Note: you still need to call .join() for the program to exit!
        """
        self.event.set()
        self.unpause_event.set()

    def unpause(self):
        self.unpause_event.set()

    def pause(self):
        self.unpause_event.clear()
