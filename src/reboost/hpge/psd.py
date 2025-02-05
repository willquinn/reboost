from __future__ import annotations

import logging

import awkward as ak
import numpy as np
from lgdo import Array

log = logging.getLogger(__name__)


def r90(edep: ak.Array, xloc: ak.Array, yloc: ak.Array, zloc: ak.Array) -> Array:
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

    # Compute distance of each edep to the weighted mean
    dist = np.sqrt(xdiff + ydiff + zdiff)

    # Sort distances and corresponding edep within each event
    sorted_indices = ak.argsort(dist, axis=-1)
    sorted_dist = dist[sorted_indices]
    sorted_edep = edep[sorted_indices]

    def cumsum(layout, **kwargs):  # noqa: ARG001
        if layout.is_numpy:
            return ak.contents.NumpyArray(np.cumsum(layout.data))  # noqa: disallow-caps

        return None

    # Calculate the cumulative sum of energies for each event
    cumsum_edep = ak.transform(
        cumsum, sorted_edep
    )  # Implement cumulative sum over whole jagged array
    if len(edep) == 1:
        cumsum_edep_corrected = cumsum_edep
    else:
        cumsum_edep_corrected = (
            cumsum_edep[1:] - cumsum_edep[:-1, -1]
        )  # correct to get cumsum of each lower level array
        cumsum_edep_corrected = ak.concatenate(
            [
                cumsum_edep[:1],  # The first element of the original cumsum is correct
                cumsum_edep_corrected,
            ]
        )

    threshold = 0.9 * tot_energy
    r90_indices = ak.argmax(cumsum_edep_corrected >= threshold, axis=-1, keepdims=True)
    r90 = sorted_dist[r90_indices]

    return Array(ak.flatten(r90).to_numpy())
