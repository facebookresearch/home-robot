import abc
import rospy


def decorator(func):
    """Decorator for checking if a control method is executed while module is enabled."""
    def wrapper(self, *args, **kwargs):
        if self.is_enabled:
            return func(self, *args, **kwargs)
        else:
            rospy.logerr(
                f"{type(self).__name__} methods are only available in when the corresponding mode is active."
            )
            return None

    return wrapper


class AbstractControlModule(abc.ABC):
    _is_enabled: bool = False

    @property
    def is_enabled(self):
        return self._is_enabled

    @abc.abstractmethod
    def _enable_hook(self):
        """Called when interface is enabled."""
        pass

    @abc.abstractmethod
    def _disable_hook(self):
        """Called when interface is disabled."""
        pass

    def enable(self):
        """Allows methods decorated with 'enforce_enabled' to be run."""
        self._is_enabled = True
        return self._enable_hook()

    def disable(self):
        """Causes methods decorated with 'enforce_enabled' to raise an error."""
        self._is_enabled = False
        return self._disable_hook()