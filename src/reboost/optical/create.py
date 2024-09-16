from __future__ import annotations

import logging
from typing import Callable, Literal

import numpy as np
import pandas as pd
import scipy.optimize
from lgdo import lh5
from lgdo.types import Array
from numba import njit
from numpy.typing import NDArray

from .optmap import OpticalMap

log = logging.getLogger(__name__)


def get_channel_efficiency(rawid: int) -> float:  # noqa: ARG001
    return 0.99


def _optmaps_for_channels(
    optmap_events: pd.DataFrame,
    settings,
    chfilter: tuple[str] | Literal["*"] = (),
):
    all_det_ids = [ch_id for ch_id in list(optmap_events.columns) if ch_id.isnumeric()]
    eff = np.array([get_channel_efficiency(int(ch_id)) for ch_id in all_det_ids])

    if chfilter != "*":
        optmap_det_ids = [det for det in all_det_ids if det in chfilter]
    else:
        optmap_det_ids = all_det_ids

    log.info("creating empty optmaps")
    optmap_count = len(optmap_det_ids) + 1
    optmaps = [
        OpticalMap("all" if i == 0 else optmap_det_ids[i], settings)
        for i in range(optmap_count)
    ]

    return all_det_ids, eff, optmaps, optmap_det_ids


@njit(cache=True)
def _compute_hit_maps(hitcounts, eff, rng, optmap_count):
    mask = np.zeros((hitcounts.shape[0], optmap_count), dtype=np.bool)
    counts = hitcounts.sum(axis=1)
    for idx in range(hitcounts.shape[0]):
        if counts[idx] == 0:
            continue

        for ch_idx in range(hitcounts.shape[1]):
            c = rng.binomial(hitcounts[idx, ch_idx], eff[ch_idx])
            if c > 0:  # detected
                mask[idx, 0] = True
                mask[idx, ch_idx + 1] = True
    return mask


def _fill_hit_maps(
    optmaps: list[OpticalMap],
    loc,
    hitcounts: NDArray,
    eff: NDArray,
    rng,
):
    masks = _compute_hit_maps(hitcounts, eff, rng, len(optmaps))

    for i in range(len(optmaps)):
        locm = loc[masks[:, i]]
        optmaps[i].h_hits.fill(locm[:, 0], locm[:, 1], locm[:, 2])


def _count_multi_ph_detection(hitcounts) -> NDArray:
    hits_per_primary = hitcounts.sum(axis=1)
    bins = np.arange(0, hits_per_primary.max() + 1.5) - 0.5
    return np.histogram(hits_per_primary, bins)[0]


def _fit_multi_ph_detection(hits_per_primary) -> float:
    x = np.arange(0, hits_per_primary.max())
    popt, pcov = scipy.optimize.curve_fit(
        lambda x, p0, k: p0 * np.exp(-k * x), x[1:], hits_per_primary[1:]
    )
    best_fit_exponent = popt[1]

    log.info(
        "p(> 1 detected photon)/p(1 detected photon) = %f",
        sum(hits_per_primary[2:]) / hits_per_primary[1],
    )
    log.info(
        "p(> 1 detected photon)/p(<=1 detected photon) = %f",
        sum(hits_per_primary[2:]) / (hits_per_primary[0:2]),
    )

    return best_fit_exponent


def create_optical_maps(
    optmap_events: pd.DataFrame,
    settings,
    chfilter=(),
    output_lh5_fn=None,
    after_save: Callable[[int, str, OpticalMap]] | None = None,
) -> None:
    """
    Parameters
    ----------
    optmap_events
        :class:`pd.DataFrame` with columns ``{x,y,z}loc`` and one column (with numeric header) for
        each SiPM channel.
    chfilter
        tuple of detector ids that will be included in the resulting optmap. Those have to match
        the column names in ``optmap_events``.
    """
    all_det_ids, eff, optmaps, optmap_det_ids = _optmaps_for_channels(
        optmap_events, settings, chfilter=chfilter
    )

    hitcounts = optmap_events[all_det_ids].to_numpy()
    loc = optmap_events[["xloc", "yloc", "zloc"]].to_numpy()

    log.info("creating vertex histogram")
    optmaps[0].fill_vertex(loc)
    for i in range(1, len(optmaps)):
        optmaps[i].h_vertex = optmaps[0].h_vertex

    log.info("computing map")
    rng = np.random.default_rng()

    _fill_hit_maps(optmaps, loc, hitcounts, eff, rng)
    hits_per_primary = _count_multi_ph_detection(hitcounts)
    # hits_per_primary_exponent = _fit_multi_ph_detection(hits_per_primary)

    log.info("computing probability and storing")
    for i in range(len(optmaps)):
        optmaps[i].create_probability()
        optmaps[i].check_histograms()
        group = "all" if i == 0 else "_" + optmap_det_ids[i - 1]
        if output_lh5_fn is not None:
            optmaps[i].write_lh5(lh5_file=output_lh5_fn, group=group)

        if after_save is not None:
            after_save(i, group, optmaps[i])

        optmaps[i] = None

    if output_lh5_fn is not None:
        lh5.write(Array(hits_per_primary), "_hitcounts", lh5_file=output_lh5_fn)
