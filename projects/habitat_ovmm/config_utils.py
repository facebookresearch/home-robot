from typing import Optional

from habitat_baselines.config.default import get_config as get_habitat_config


def get_config(path: str, opts: Optional[list] = None):
    config = get_habitat_config(path)
    return config, ""
