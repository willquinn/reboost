from __future__ import annotations

import importlib.util
import types

from numba import njit


def numba_pdgid_funcs():
    """Load a numby-optimized copy of the scikit-hep/particle package."""
    spec_pdg = importlib.util.find_spec("particle.pdgid.functions")
    pdg_func = importlib.util.module_from_spec(spec_pdg)
    spec_pdg.loader.exec_module(pdg_func)

    def _digit2(pdgid, loc: int) -> int:
        e = 10 ** (loc - 1)
        return (pdgid // e % 10) if pdgid >= e else 0

    pdg_func._digit = _digit2

    for fname, f in pdg_func.__dict__.items():
        if not callable(f) or not isinstance(f, types.FunctionType):
            continue
        setattr(pdg_func, fname, njit(f, cache=True))

    return pdg_func
