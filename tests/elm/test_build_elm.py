from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from lgdo import Table, lh5

from reboost.build_elm import build_elm, get_elm_rows, get_stp_evtids


# test the basic (awkward operations) to get the elm rows
def test_get_elm_rows():
    # some basic cases
    stp_evtids = [0, 0, 0, 1, 1, 1, 6, 6, 7]
    vert = [0, 1, 6, 7]
    elm = get_elm_rows(stp_evtids, vert, start_row=0)

    assert ak.all(elm.evtid == [0, 1, 6, 7])
    assert ak.all(elm.n_rows == [3, 3, 2, 1])
    assert ak.all(elm.start_row == [0, 3, 6, 8])

    # test with a different start row

    elm = get_elm_rows(stp_evtids, vert, start_row=999)
    assert ak.all(elm.evtid == [0, 1, 6, 7])
    assert ak.all(elm.n_rows == [3, 3, 2, 1])
    assert ak.all(elm.start_row == [999, 1002, 1005, 1007])

    # test discard steps

    vert = [1, 6]
    elm = get_elm_rows(stp_evtids, vert, start_row=0)
    assert ak.all(elm.evtid == [1, 6])
    assert ak.all(elm.n_rows == [3, 2])
    assert ak.all(elm.start_row == [3, 6])

    # test gracefully fails

    # steps not in the vert table will cause it to fail
    with pytest.raises(ValueError):
        get_elm_rows(stp_evtids=[1, 3], vert=[1, 2, 4])

    # steps must be sorted
    with pytest.raises(ValueError):
        get_elm_rows(stp_evtids=[1, 3, 2], vert=[1, 2])

    # vertex evtids must be sorted
    with pytest.raises(ValueError):
        get_elm_rows(stp_evtids=[1, 3, 2], vert=[1, 2, 0])


# create some example inputs
@pytest.fixture
def test_data_files(tmp_path):
    rng = np.random.default_rng()

    # simple every evtid in vertices
    vertex_evtid = ak.Array({"evtid": np.arange(10000)})
    lh5.write(Table(vertex_evtid), "stp/vertices", tmp_path / "simple_test.lh5", wo_mode="of")

    # make some simple stp file
    steps_1 = ak.Array({"evtid": np.sort(rng.integers(0, 10000, size=21082))})
    steps_2 = ak.Array({"evtid": np.sort(rng.integers(0, 1000, size=1069))})

    lh5.write(Table(steps_1), "stp/det1", tmp_path / "simple_test.lh5", wo_mode="append")
    lh5.write(Table(steps_2), "stp/det2", tmp_path / "simple_test.lh5", wo_mode="append")

    # file with some gaps (multithreaded mode)

    vertex_evtid = ak.Array({"evtid": np.sort(np.unique(rng.integers(0, 200000, size=1000)))})
    lh5.write(Table(vertex_evtid), "stp/vertices", tmp_path / "gaps_test.lh5", wo_mode="of")

    # make some simple stp file
    steps_1 = ak.Array({"evtid": np.sort(rng.choice(vertex_evtid.evtid, size=21082))})
    steps_2 = ak.Array({"evtid": np.sort(rng.choice(vertex_evtid.evtid, size=1069))})

    lh5.write(Table(steps_1), "stp/det1", tmp_path / "gaps_test.lh5", wo_mode="append")
    lh5.write(Table(steps_2), "stp/det2", tmp_path / "gaps_test.lh5", wo_mode="append")
    return tmp_path


def test_read_stp_rows(test_data_files):
    # check reading from the start everything
    start_row, chunk_start, evtids = get_stp_evtids(
        "stp/det1",
        str(test_data_files / "simple_test.lh5"),
        "evtid",
        start_row=0,
        last_vertex_evtid=10000,
        stp_buffer=1000,
    )
    # read the evtid directly to compare
    evtids_read = lh5.read_as("stp/det1/evtid", str(test_data_files / "simple_test.lh5"), "np")
    assert chunk_start == 0
    assert np.all(evtids == evtids_read)

    # check the breaking
    # the number of evtids less than 1200 is index
    index = sum(evtids_read < 1200)
    start_row, chunk_start, evtids = get_stp_evtids(
        "stp/det1",
        str(test_data_files / "simple_test.lh5"),
        "evtid",
        start_row=0,
        last_vertex_evtid=1200,
        stp_buffer=1000,
    )
    # check we read far enough
    assert len(evtids) > index
    assert chunk_start == 0
    assert start_row == np.floor(index / 1000) * 1000

    # check updated start row

    start_row, chunk_start, evtids = get_stp_evtids(
        "stp/det1",
        str(test_data_files / "simple_test.lh5"),
        "evtid",
        start_row=300,
        last_vertex_evtid=10000,
        stp_buffer=1000,
    )
    # first chunk should be 300
    assert chunk_start == 300
    assert len(evtids) == len(evtids_read) - 300

    # check reading of last chunk

    start_row, chunk_start, evtids = get_stp_evtids(
        "stp/det1",
        str(test_data_files / "simple_test.lh5"),
        "evtid",
        start_row=21050,
        last_vertex_evtid=10000,
        stp_buffer=1000,
    )
    assert len(evtids) == len(evtids_read) - 21050
    assert start_row == 21050


def test_build_elm(test_data_files):
    # produce directly elm without iteration
    for test in ["simple", "gaps"]:
        evtids = lh5.read_as("stp/vertices/evtid", str(test_data_files / f"{test}_test.lh5"), "np")

        evtids1_read = lh5.read_as(
            "stp/det1/evtid", str(test_data_files / f"{test}_test.lh5"), "np"
        )
        evtids2_read = lh5.read_as(
            "stp/det2/evtid", str(test_data_files / f"{test}_test.lh5"), "np"
        )

        elm = build_elm(
            str(test_data_files / f"{test}_test.lh5"),
            None,
            id_name="evtid",
            evtid_buffer=1000,
            stp_buffer=100,
        )
        # elm should have the right evtid
        assert ak.all(elm.det1.evtid == evtids)
        assert ak.all(elm.det2.evtid == evtids)

        # total number of rows should be correct
        assert np.sum(elm.det1.n_rows) == len(evtids1_read)
        assert np.sum(elm.det2.n_rows) == len(evtids2_read)
