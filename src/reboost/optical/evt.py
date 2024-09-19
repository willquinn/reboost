from __future__ import annotations

import logging

import numpy as np
from lgdo import lh5
from lgdo.types import Table

log = logging.getLogger(__name__)


def build_optmap_evt(lh5_in_file: str, lh5_out_file: str) -> None:
    log.info("reading file %s", lh5_in_file)

    vert_df = lh5.read("hit/vertices", lh5_file=lh5_in_file).view_as("pd")
    opti_df = lh5.read("hit/optical", lh5_file=lh5_in_file).view_as("pd")

    log.info("prepare evt table")
    _, hit_counts = np.unique(opti_df["evtid"], return_counts=True)
    max_hit_counts = hit_counts.max()
    # use smaller integer types uint8/16 to spare RAM when storing types
    hit_count_type = np.uint8 if max_hit_counts < 255 else np.uint16

    # prepare the original vertex coordinates.
    vert_df = vert_df.set_index("evtid", drop=True).drop(["n_part", "time"], axis=1)

    # add columns for all detectors.
    detectors = np.unique(opti_df["det_uid"])
    for d in detectors:
        if not str(d).isnumeric():
            log.warning("Detector ID %s is not numeric.", str(d))
        vert_df[str(d)] = hit_count_type(0)

    log.info("build evt table")
    # create a (fast) map for lookup of the hits in each detector.
    for t in opti_df[["evtid", "det_uid"]].itertuples(name=None, index=False):
        vert_df.loc[t[0], str(t[1])] += 1

    # sanity check that we did process all hits.
    hits_sum = 0
    for d in detectors:
        hits_sum += np.sum(vert_df[str(d)])
    assert hits_sum == len(opti_df)

    log.info("store evt file %s", lh5_out_file)
    lh5.write(Table(vert_df), name="optmap_evt", lh5_file=lh5_out_file, wo_mode="overwrite_file")


def read_optmap_evt(lh5_file: str, buffer_len: int = int(5e6)) -> lh5.LH5Iterator:
    return lh5.LH5Iterator(lh5_file, "optmap_evt", buffer_len=buffer_len)
