from __future__ import annotations

from pathlib import Path

from lgdo import lh5

from reboost.cli import cli


def test_cli(tmp_path):
    # cli for build_glm

    cli(
        [
            "build-glm",
            "--id-name",
            "evtid",
            "-w",
            "--glm-file",
            f"{tmp_path}/glm.lh5",
            "--stp-file",
            f"{Path(__file__).parent}/hit/test_files/beta_small.lh5",
        ]
    )

    glm = lh5.read("glm/det001", f"{tmp_path}/glm.lh5").view_as("ak")
    assert glm.fields == ["evtid", "n_rows", "start_row"]

    cli(
        [
            "build-hit",
            "--config",
            f"{Path(__file__).parent}/hit/configs/hit_config.yaml",
            "-w",
            "--glm-file",
            f"{tmp_path}/glm.lh5",
            "--stp-file",
            f"{Path(__file__).parent}/hit/test_files/beta_small.lh5",
            "--hit-file",
            f"{tmp_path}/hit.lh5",
            "--args",
            f"{Path(__file__).parent}/hit/configs/args.yaml",
        ]
    )

    hit1 = lh5.read("det001/hit", f"{tmp_path}/hit.lh5").view_as("ak")
    assert hit1.fields == [
        "evtid",
        "t0",
        "evtid",
        "truth_energy",
        "active_energy",
        "smeared_energy",
    ]
