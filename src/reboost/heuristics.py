from __future__ import annotations

import logging

import awkward as ak
import numpy as np
from lgdo import Array

log = logging.getLogger(__name__)


def cumsum(layout, **kwargs):
    _ = kwargs  # Ignore unused kwargs

    if layout.is_numpy:
        return ak.contents.NumpyArray(np.cumsum(layout.data))

    return None


def calculate_R90(edep: np.ndarray, xloc: np.ndarray, yloc: np.ndarray, zloc: np.ndarray):
    """Computes R90 for each hit in a ged.

    Parameters
    ----------
    edep
        awkward array of energy
    xloc
        awkward array of x coordinate position
    yloc
        awkward array of y coordinate position
    zloc
        awkward array of z coordinate position

    Returns
    -------
    r90
    """
    # Total energy per event
    tot_energy = ak.sum(edep, axis=-1, keepdims=True)

    # Compute energy-weighted mean positions
    mean_x = ak.sum(edep * xloc, axis=-1, keepdims=True) / tot_energy
    mean_y = ak.sum(edep * yloc, axis=-1, keepdims=True) / tot_energy
    mean_z = ak.sum(edep * zloc, axis=-1, keepdims=True) / tot_energy

    xdiff = xloc - mean_x
    ydiff = yloc - mean_y
    zdiff = zloc - mean_z

    xdiff = xdiff * xdiff
    ydiff = ydiff * ydiff
    zdiff = zdiff * zdiff

    dist = np.sqrt(xdiff + ydiff + zdiff)

    # Sort distances and corresponding edep within each event
    sorted_indices = ak.argsort(dist, axis=-1)
    sorted_dist = dist[sorted_indices]
    sorted_edep = edep[sorted_indices]

    b = ak.transform(cumsum, sorted_edep)
    c = b[1:] - b[:-1, -1]
    cumsum_edep = ak.concatenate([b[:1], c])
    threshold = 0.9 * tot_energy
    r90_indices = ak.argmax(cumsum_edep >= threshold, axis=-1, keepdims=True)

    # Extract R90 distances
    r90 = sorted_dist[r90_indices]

    return Array(ak.flatten(r90))
