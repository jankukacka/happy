# ------------------------------------------------------------------------------
#  File: local_config.py
#  Author: Jan Kukacka
#  Date: 7/2018
# ------------------------------------------------------------------------------
#  Module for local configurations.
#
#  Reads constants from file %HOME%/.work_config
# ------------------------------------------------------------------------------

from pathlib import Path
import json

_local_config_cache__ = None


def local(key, default=None):
    global _local_config_cache__

    if _local_config_cache__ is None:
        config_path = Path.home() / '.work_config'
        if config_path.exists():
            with config_path.open() as file:
                _local_config_cache__ = json.load(file)
        else:
            _local_config_cache__ = {}

    if key in _local_config_cache__:
        return _local_config_cache__[key]

    if default is not None:
        _local_config_cache__[key] = default
        return default

    return None
