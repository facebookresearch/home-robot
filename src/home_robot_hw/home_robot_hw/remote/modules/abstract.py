# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import abc
import threading
from typing import Callable

import rospy


def enforce_enabled(func):
    """Decorator for checking if a control method is executed while module is enabled."""

    def wrapper(self, *args, **kwargs):
        if self.is_enabled:
            return func(self, *args, **kwargs)
        else:
            err_str = f"{type(self).__name__}.{func.__name__} is only available in when the corresponding control mode is active."
            rospy.logerr(err_str)
            raise TypeError(err_str)

    return wrapper


class AbstractControlModule(abc.ABC):
    """
    Abstract control module that implements the following functionalities
    - Enabling / disabling the module to control access of methods
    - Concurrency management
    """

    _is_enabled: bool = False

    def __init__(self):
        self._wait_lock = threading.Lock()
        self._wait_threads = []

    def _register_wait(self, func):
        """
        Add a wait function that returns when actions have ended.
        The function will be run in a separate thread to not block users.
        is_busy() and wait() methods are available to interact with the thread.
        """
        with self._wait_lock:
            thr = threading.Thread(target=func)
            self._wait_threads.append(thr)
            thr.start()

    def _update_wait_threads(self):
        for i in range(len(self._wait_threads))[::-1]:
            thr = self._wait_threads[i]
            if not thr.is_alive():
                self._wait_threads.pop(i)

    def is_busy(self):
        """Check if any action is undergoing"""
        with self._wait_lock:
            self._update_wait_threads()
            return bool(self._busy_threads)

    def wait(self, timeout=None):
        """Wait for all action threads to complete"""
        with self._wait_lock:
            for i in range(len(self._wait_threads))[::-1]:
                thr = self._wait_threads[i]
                if thr.is_alive():
                    thr.join(timeout=timeout)
                if not thr.is_alive():  # check if thread is alive in case of a timeout
                    self._wait_threads.pop(i)

    @property
    def is_enabled(self):
        return self._is_enabled

    @abc.abstractmethod
    def _enable_hook(self) -> bool:
        """Called when interface is enabled."""
        pass

    @abc.abstractmethod
    def _disable_hook(self) -> bool:
        """Called when interface is disabled."""
        pass

    def enable(self) -> bool:
        """Allows methods decorated with 'enforce_enabled' to be run."""
        self._is_enabled = True
        return self._enable_hook()

    def disable(self) -> bool:
        """Causes methods decorated with 'enforce_enabled' to raise an error."""
        self._is_enabled = False
        return self._disable_hook()
