from typing import Any
from abc import ABC, abstractmethod

from .interfaces import Observations


class PerceptionModule(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, obs: Observations) -> Any:
        pass
