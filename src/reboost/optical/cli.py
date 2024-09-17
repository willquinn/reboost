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

    # STEP 2b: view maps
    mapview_parser = subparsers.add_parser("viewmap", help="view optical map")
    mapview_parser.add_argument("input", help="input evt LH5 file", metavar="INPUT_MAP")

    # STEP 2c: merge maps
    mapmerge_parser = subparsers.add_parser("mergemap", help="merge optical maps")
    mapmerge_parser.add_argument(
        "input", help="input map LH5 files", metavar="INPUT_MAP", nargs="+"
    )
    mapmerge_parser.add_argument(
        "output", help="output map LH5 file", metavar="OUTPUT_MAP"
    )
    mapmerge_parser.add_argument(
        "--settings",
        action="store",
        help="""Select a config file for binning.""",
        required=True,
    )

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

    # STEP 2a: build map file from evt tier
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

    # STEP 2b: view maps
    if args.command == "viewmap":
        from reboost.optical.mapview import view_optmap

        view_optmap(args.input)

    # STEP 2c: merge maps
    if args.command == "mergemap":
        from reboost.optical.create import merge_optical_maps

        # load settings for binning from config file.
        with Path.open(args.settings) as settings_f:
            settings = json.load(settings_f)

        merge_optical_maps(args.input, args.output, settings)
