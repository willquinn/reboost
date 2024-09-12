from __future__ import annotations

import reboost


def test_core():
    assert reboost.hello_world() == "Hello World!"
