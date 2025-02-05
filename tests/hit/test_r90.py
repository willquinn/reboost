from __future__ import annotations

from pathlib import Path

from lgdo import Array, Table, lh5

import reboost
from reboost.build_glm import build_glm


def test_basic(test_gen_lh5):
    hits, time_dict = reboost.build_hit.build_hit(
        f"{Path(__file__).parent}/configs/r90_test.yaml",
        args={},
        stp_files=f"{test_gen_lh5}/basic.lh5",
        glm_files=f"{test_gen_lh5}/basic_glm.lh5",
        hit_files=None,
    )

    print(hits)


def test_gen_lh5(tmp_path):
    # write a basic lh5 file

    evtid = [0, 0, 1, 1, 1]
    edep = [100, 200, 10, 20, 300]  # keV
    xloc = [0.1, 0.2, 0.001, 0.0001, 0.3]
    yloc = [0.4, 0.002, 0.002, 0.001, 0.0003]
    zloc = [0.0001, 0.02, 0.4, 0.006, 0.03]
    time = [0, 1.5, 0.1, 2.1, 3.7]  # ns
    vertices = [0, 1]
    tab = Table(
        {
            "evtid": Array(evtid),
            "edep": Array(edep),
            "xloc": Array(xloc),
            "yloc": Array(yloc),
            "zloc": Array(zloc),
            "time": Array(time),
        }
    )
    lh5.write(tab, "stp/det1", f"{tmp_path}/basic.lh5", wo_mode="of")
    lh5.write(
        Table({"evtid": Array(vertices)}),
        "stp/vertices",
        f"{tmp_path}/basic.lh5",
        wo_mode="append",
    )

    build_glm(f"{tmp_path}/basic.lh5", f"{tmp_path}/basic_glm.lh5", id_name="evtid")


if __name__ == "__main__":
    test_gen_lh5("test_files")
    test_basic("test_files")
