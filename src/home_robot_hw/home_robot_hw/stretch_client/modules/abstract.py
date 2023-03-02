import abc

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
    _is_enabled: bool = False

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
