from __future__ import annotations

import uuid
from getpass import getuser
from pathlib import Path
from tempfile import gettempdir

import pytest
from lgdo import lh5

from reboost.cli import cli

_tmptestdir = Path(gettempdir()) / Path(f"reboost-tests-{getuser()}-{uuid.uuid4()!s}")


@pytest.fixture(scope="session")
def tmptestdir():
    Path.mkdir(_tmptestdir)
    return _tmptestdir


@pytest.fixture(scope="session")
def test_cli(tmptestdir):
    # cli for build_glm

    cli(
        [
            "build-glm",
            "--id-name",
            "evtid",
            "-w",
            "--glm-file",
            f"{tmptestdir}/glm.lh5",
            "--stp-file",
            f"{Path(__file__).parent}/hit/test_files/beta_small.lh5",
        ]
    )

    glm = lh5.read("glm/det001", f"{tmptestdir}/glm.lh5").view_as("ak")
    assert glm.fields == ["evtid", "n_rows", "start_row"]

    cli(
        [
            "build-hit",
            "--config",
            f"{Path(__file__).parent}/hit/configs/hit_config.yaml",
            "-w",
            "--glm-file",
            f"{tmptestdir}/glm.lh5",
            "--stp-file",
            f"{Path(__file__).parent}/hit/test_files/beta_small.lh5",
            "--hit-file",
            f"{tmptestdir}/hit.lh5",
            "--args",
            f"{Path(__file__).parent}/hit/configs/args.yaml",
        ]
    )

    hit1 = lh5.read("det001/hit", f"{tmptestdir}/hit.lh5").view_as("ak")
    assert hit1.fields == ["evtid", "t0", "truth_energy", "active_energy", "smeared_energy"]
