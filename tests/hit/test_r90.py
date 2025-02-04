from __future__ import annotations

from pathlib import Path

import reboost


def test_basic(test_gen_lh5):
    hits, time_dict = reboost.build_hit.build_hit(
        f"{Path(__file__).parent}/configs/r90_test.yaml",
        args={},
        stp_files=f"{test_gen_lh5}/basic.lh5",
        glm_files=f"{test_gen_lh5}/basic_glm.lh5",
        hit_files=None,
    )

    print(hits)
