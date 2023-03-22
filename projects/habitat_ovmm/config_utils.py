from typing import Optional

from habitat_baselines.config.default import _BASELINES_CFG_DIR
from habitat_baselines.config.default import get_config as get_habitat_config


def get_config(
    path: str, opts: Optional[list] = None, configs_dir: str = _BASELINES_CFG_DIR
):
    config = get_habitat_config(path, overrides=opts, configs_dir=configs_dir)
    return config, ""
