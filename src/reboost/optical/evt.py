from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
from lgdo import lh5
from lgdo.lh5 import LH5Iterator
from lgdo.types import Table

log = logging.getLogger(__name__)

EVT_TABLE_NAME = "optmap_evt"


def build_optmap_evt(
    lh5_in_file: str, lh5_out_file: str, detectors: Iterable[str | int], buffer_len: int = int(5e6)
) -> None:
    """Create a faster map for lookup of the hits in each detector, for each primary event."""
    log.info("reading file %s", lh5_in_file)

    vert_it = LH5Iterator(lh5_in_file, "hit/vertices", buffer_len=buffer_len)
    opti_it = LH5Iterator(lh5_in_file, "hit/optical", buffer_len=buffer_len)

    detectors = [str(d) for d in detectors]
    for d in detectors:
        if not d.isnumeric():
            log.warning("Detector ID %s is not numeric.", d)

    vert_df = None
    vert_df_bounds = None
    vert_it_count = 0
    hist_expected = 0

    # helper function for "windowed join". while iterating the optical hits, we have to
    # make sure that we always have the correct combined vertex/hit output table available.
    #
    # This function follows the assumption, that the output event ids are at least "somewhat"
    # monotonic, i.e. later chunks do not contain lower evtids than the previous chunk(s).
    # Going back is not implemented.
    def _ensure_vert_df(vert_it: LH5Iterator, evtid: int):
        nonlocal vert_df, vert_df_bounds, vert_it_count, hist_expected

        if vert_df_bounds is not None and vert_df is not None:
            if evtid < vert_df_bounds[0]:
                msg = "non-monotonic evtid encountered, but cannot go back"
                raise KeyError(msg)
            if evtid >= vert_df_bounds[0] and evtid <= vert_df_bounds[1]:
                return  # vert_df already contains the given evtid.

        # here, evtid > vert_df_bounds[1] (or vert_df_bounds is still None). We need to fetch
        # the next event table chunk.

        vert_it_count += 1
        # we have filled a dataframe, save it to disk.
        if vert_df is not None:
            # sanity check that we did process all hits.
            hits_sum = 0
            for d in detectors:
                hits_sum += np.sum(vert_df[d])
            assert hits_sum == hist_expected

            log.info("store evt file %s (%d)", lh5_out_file, vert_it_count - 1)
            lh5.write(Table(vert_df), name=EVT_TABLE_NAME, lh5_file=lh5_out_file, wo_mode="append")
            vert_df = None

        # read the next vertex chunk into memory.
        (vert_lgdo, vert_entry, vert_n_rows) = next(vert_it)
        vert_df = vert_lgdo.view_as("pd")

        # prepare vertex coordinates.
        vert_df = vert_df.set_index("evtid", drop=True).drop(["n_part", "time"], axis=1)
        vert_df_bounds = [vert_df.index.min(), vert_df.index.max()]
        hist_expected = 0
        # add columns for all detectors.
        for d in detectors:
            vert_df[d] = hit_count_type(0)

    log.info("prepare evt table")
    # use smaller integer type uint16 to spare RAM when storing types.
    hit_count_type = np.uint16
    for opti_it_count, (opti_lgdo, opti_entry, opti_n_rows) in enumerate(opti_it):
        assert (opti_it_count == 0) == (opti_entry == 0)
        opti_df = opti_lgdo.view_as("pd").iloc[0:opti_n_rows]

        log.info("build evt table (%d)", opti_it_count)
        for t in opti_df[["evtid", "det_uid"]].itertuples(name=None, index=False):
            _ensure_vert_df(vert_it, t[0])
            vert_df.loc[t[0], str(t[1])] += 1
            hist_expected += 1


def read_optmap_evt(lh5_file: str, buffer_len: int = int(5e6)) -> LH5Iterator:
    return LH5Iterator(lh5_file, EVT_TABLE_NAME, buffer_len=buffer_len)
