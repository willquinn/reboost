from __future__ import annotations

import awkward as ak
import numpy as np
from lgdo import Table

import reboost
from reboost.shape import cluster, group


def test_evtid_group():
    in_arr_evtid = ak.Array(
        {"evtid": [1, 1, 1, 2, 2, 10, 10, 11, 12, 12, 12], "time": np.zeros(11)}
    )

    # test with table input

    in_tab = Table(in_arr_evtid)
    out = group.group_by_evtid(in_tab)
    out_ak = out.view_as("ak")
    assert ak.all(out_ak.evtid == [[1, 1, 1], [2, 2], [10, 10], [11], [12, 12, 12]])
    assert ak.all(out_ak.time == [[0, 0, 0], [0, 0], [0, 0], [0], [0, 0, 0]])

    # test with ak.Array input
    out = group.group_by_evtid(in_tab.view_as("ak"))
    out_ak = out.view_as("ak")
    assert ak.all(out_ak.evtid == [[1, 1, 1], [2, 2], [10, 10], [11], [12, 12, 12]])
    assert ak.all(out_ak.time == [[0, 0, 0], [0, 0], [0, 0], [0], [0, 0, 0]])

    # test the eval in build hit also
    out_eval = reboost.core.evaluate_hit_table_layout(
        in_tab,
        "reboost.shape.group.group_by_evtid(STEPS)",
    )

    out_eval_ak = out_eval.view_as("ak")

    assert ak.all(out_ak.evtid == out_eval_ak.evtid)
    assert ak.all(out_ak.time == out_eval_ak.time)


def test_time_group():
    # time units are ns
    in_arr_evtid = ak.Array(
        {
            "evtid": [1, 1, 1, 2, 2, 2, 2, 2, 11, 12, 12, 12, 15, 15, 15, 15, 15],
            "time": [
                0,
                -2000,
                3000,
                0,
                100,
                1200,
                17000,
                17010,
                0,
                0,
                0,
                -5000,
                150,
                151,
                152,
                3000,
                3100,
            ],
        }
    )

    in_tab = Table(in_arr_evtid)

    # 1us =1000ns
    out = group.group_by_time(in_tab, window=1)
    out_ak = out.view_as("ak")
    assert ak.all(
        out_ak.evtid
        == [[1], [1], [1], [2, 2], [2], [2, 2], [11], [12], [12, 12], [15, 15, 15], [15, 15]]
    )
    assert ak.all(
        out_ak.time
        == [
            [-2000],
            [0],
            [3000],
            [0, 100],
            [1200],
            [17000, 17010],
            [0],
            [-5000],
            [0, 0],
            [150, 151, 152],
            [3000, 3100],
        ]
    )


def test_cluster_basic():
    trackid = ak.Array([[1, 1, 1, 2, 2, 3, 3, 7], [2, 2, 2, 3, 3, 3], [1]])

    assert ak.all(
        cluster.cluster_by_sorted_field(trackid, trackid).view_as("ak")
        == ak.Array([[[1, 1, 1], [2, 2], [3, 3], [7]], [[2, 2, 2], [3, 3, 3]], [[1]]])
    )


def test_step_length():
    trackid = ak.Array([[1, 1, 1, 2, 2, 3, 3, 7], [2, 2, 2, 3, 3, 3], [1]])
    x = ak.Array([[0, 0, 1, 1, 2, 2, 3, 4], [0, 1, 4, 5, 5, 6], [0]])
    y = ak.Array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0]])
    z = ak.Array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0]])

    x_cluster = cluster.cluster_by_sorted_field(trackid, x).view_as("ak")
    y_cluster = cluster.cluster_by_sorted_field(trackid, y).view_as("ak")
    z_cluster = cluster.cluster_by_sorted_field(trackid, z).view_as("ak")

    steps = cluster.step_lengths(x_cluster, y_cluster, z_cluster).view_as("ak")

    for idx, sub_array in enumerate(steps):
        assert ak.all(sub_array == ak.Array([[[0, 1], [1], [1], []], [[1, 3], [0, 1.0]], []])[idx])
