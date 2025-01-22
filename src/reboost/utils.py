from __future__ import annotations

import logging
from collections import namedtuple

logging.basicConfig(level=logging.INFO)


def dict2tuple(dictionary: dict) -> namedtuple:
    return namedtuple("parameters", dictionary.keys())(**dictionary)
