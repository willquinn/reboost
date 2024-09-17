from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import colorlog

log = logging.getLogger(__name__)


def optical_cli() -> None:
    parser = argparse.ArgumentParser(
        prog="reboost-optical",
        description="%(prog)s command line interface",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""Increase the program verbosity""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # STEP 1: build evt file from hit tier
    evt_parser = subparsers.add_parser(
        "evt", help="build evt file from remage hit file"
    )
    evt_parser.add_argument("input", help="input hit LH5 file", metavar="INPUT_HIT")
    evt_parser.add_argument("output", help="output evt LH5 file", metavar="OUTPUT_EVT")

    # STEP 2a: build map file from evt tier
    map_parser = subparsers.add_parser(
        "createmap", help="build optical map from evt file(s)"
    )
    map_parser.add_argument(
        "--settings",
        action="store",
        help="""Select a config file for binning.""",
        required=True,
    )
    map_parser.add_argument(
        "input", help="input evt LH5 file", metavar="INPUT_EVT", nargs="+"
    )
    map_parser.add_argument("output", help="output map LH5 file", metavar="OUTPUT_MAP")

    args = parser.parse_args()

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(name)s [%(levelname)s] %(message)s")
    )
    logger = logging.getLogger("reboost.optical")
    logger.addHandler(handler)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # STEP 1: build evt file from hit tier
    if args.command == "evt":
        from reboost.optical.evt import build_optmap_evt

        build_optmap_evt(args.input, args.output)

    # STEP 2: build map file from evt tier
    if args.command == "createmap":
        from reboost.optical.create import create_optical_maps
        from reboost.optical.evt import read_optmap_evt

        # load settings for binning from config file.
        with Path.open(args.settings) as settings_f:
            settings = json.load(settings_f)

        optmap_events = read_optmap_evt(args.input)
        create_optical_maps(
            optmap_events,
            settings,
            chfilter=(),
            output_lh5_fn=args.output,
        )
