from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="reboost",
        description="%(prog)s command line interface",
    )
    parser.parse_args()
