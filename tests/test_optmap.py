from __future__ import annotations

import numpy as np
import pytest
from lgdo import Array, Table, lh5

from reboost.optical.create import create_optical_maps, merge_optical_maps
from reboost.optical.evt import build_optmap_evt, read_optmap_evt


@pytest.fixture
def tbl_hits(tmp_path):
    evt_count = 100
    rng = np.random.default_rng(1234)
    loc = rng.uniform(size=(evt_count, 3))
    evtids = np.arange(1, evt_count + 1)

    tbl_vertices = Table(
        {
            "evtid": Array(evtids),
            "xloc": Array(loc[:, 0]),
            "yloc": Array(loc[:, 1]),
            "zloc": Array(loc[:, 2]),
            "n_part": Array(np.ones(evt_count)),
            "time": Array(np.ones(evt_count)),
        }
    )

    mask = rng.uniform(size=evt_count) < 0.2
    hit_count = np.sum(mask)

    tbl_optical = Table(
        {
            "evtid": Array(evtids[mask]),
            "det_uid": Array(np.ones(hit_count, dtype=np.int_)),
            "wavelength": Array(rng.normal(loc=400, scale=30, size=hit_count)),
            "time": Array(2 * np.ones(hit_count)),
        }
    )

    hit_file = tmp_path / "hit.lh5"
    lh5.write(tbl_vertices, name="hit/vertices", lh5_file=hit_file, wo_mode="overwrite_file")
    lh5.write(tbl_optical, name="hit/optical", lh5_file=hit_file, wo_mode="overwrite")
    return hit_file


def test_optmap_evt(tbl_hits, tmp_path):
    evt_out_file = tmp_path / "evt-out.lh5"
    build_optmap_evt(
        str(tbl_hits),
        str(evt_out_file),
        detectors=("1", "002", "003"),
        buffer_len=20,  # note: shorter window sizes (e.g. 10) do not work.
    )


@pytest.fixture
def tbl_evt(tmp_path):
    evt_count = 100
    rng = np.random.default_rng(1234)
    loc = rng.uniform(size=(evt_count, 3))
    hits = rng.geometric(p=0.9, size=(evt_count, 3)) - 1

    tbl_evt = Table(
        {
            "xloc": Array(loc[:, 0]),
            "yloc": Array(loc[:, 1]),
            "zloc": Array(loc[:, 2]),
            "001": Array(hits[:, 0]),
            "002": Array(hits[:, 1]),
            "003": Array(hits[:, 2]),
        }
    )

    evt_file = tmp_path / "evt.lh5"
    lh5.write(tbl_evt, name="optmap_evt", lh5_file=evt_file, wo_mode="overwrite_file")
    return evt_file


@pytest.mark.filterwarnings("ignore::scipy.optimize._optimize.OptimizeWarning")
def test_optmap_create(tbl_evt):
    settings = {
        "range_in_m": [[0, 1], [0, 1], [0, 1]],
        "bins": [10, 10, 10],
    }

    # test creation only with the summary map.
    evt_it = read_optmap_evt(str(tbl_evt), buffer_len=10)
    create_optical_maps(
        evt_it,
        settings,
        chfilter=(),
        output_lh5_fn=None,
    )

    # test creation with all detectors.
    evt_it = read_optmap_evt(str(tbl_evt), buffer_len=10)
    create_optical_maps(
        evt_it,
        settings,
        chfilter=("001", "002", "003"),
        output_lh5_fn=None,
    )

    # test creation with some detectors.
    evt_it = read_optmap_evt(str(tbl_evt), buffer_len=10)
    create_optical_maps(
        evt_it,
        settings,
        chfilter=("001"),
        output_lh5_fn=None,
    )


@pytest.mark.filterwarnings("ignore::scipy.optimize._optimize.OptimizeWarning")
def test_optmap_merge(tbl_evt, tmp_path):
    settings = {
        "range_in_m": [[0, 1], [0, 1], [0, 1]],
        "bins": [10, 10, 10],
    }

    map1_fn = str(tmp_path / "map1.lh5")
    evt_it = read_optmap_evt(str(tbl_evt), buffer_len=10)
    create_optical_maps(
        evt_it,
        settings,
        chfilter=("001"),
        output_lh5_fn=map1_fn,
    )
    map2_fn = str(tmp_path / "map2.lh5")
    evt_it = read_optmap_evt(str(tbl_evt), buffer_len=10)
    create_optical_maps(
        evt_it,
        settings,
        chfilter=("001"),
        output_lh5_fn=map2_fn,
    )

    map_merged_fn = str(tmp_path / "map-merged.lh5")
    merge_optical_maps([map1_fn, map2_fn], map_merged_fn, settings)
