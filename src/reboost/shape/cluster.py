from __future__ import annotations

import logging

import awkward as ak
import numpy as np
from lgdo import VectorOfVectors

log = logging.getLogger(__name__)


def cluster_by_sorted_field(
    cluster_variable: ak.Array | VectorOfVectors, field: ak.Array | VectorOfVectors
) -> VectorOfVectors:
    """Extract the cumulative lengths of the clusters per event.

    Example processing block:

    .. code-block:: yaml

        trackid_cluster_edep: reboost.shape.cluster.cluster_by_sorted_field(HITS.trackid,HITS.edep)

    Parameters
    ----------
    trackid
        the id of the tracks.
    field
        another field to cluster.

    Returns
    -------
    A VectorOfVectors with the clustered field, an additional axis is present due to the clustering
    """
    if isinstance(cluster_variable, VectorOfVectors):
        cluster_variable = cluster_variable.view_as("ak")

    if isinstance(field, VectorOfVectors):
        field = cluster_variable.view_as("ak")

    # run length of each cluster
    counts = ak.run_lengths(cluster_variable)

    n_cluster = ak.num(counts, axis=-1)
    clusters = ak.unflatten(ak.flatten(field), ak.flatten(counts))

    return VectorOfVectors(ak.unflatten(clusters, n_cluster))


def step_lengths(
    x_cluster: ak.Array | VectorOfVectors,
    y_cluster: ak.Array | VectorOfVectors,
    z_cluster: ak.Array | VectorOfVectors,
) -> VectorOfVectors:
    """Compute the distance between consecutive steps.

    This is based on calculating the distance between consecutive steps in the same track,
    thus the input arrays should already be clustered (have dimension 3). The output
    will have a similar shape to the input with one less entry in the outermost dimension.

    Example config (assuming that the clustered positions are obtained already):

    .. code-block:: yaml

        step_lengths: reboost.shape.cluster.step_lengths(HITS.cluster_x,HITS.cluster_y,HITS.cluster_z))

    Parameters
    ----------
    x_cluster
        The x location of each step in each cluster and event.
    y_cluster
        The y location of each step in each cluster and event.
    z_cluster
        The z location of each step in each cluster and event.

    Returns
    -------
    a `VectorOfVectors` of the step lengths in each cluster.
    """
    data = [x_cluster, y_cluster, z_cluster]

    for idx, var in enumerate(data):
        if isinstance(var, VectorOfVectors):
            data[idx] = var.view_as("ak")
        # check shape
        if data[idx].ndim != 3:
            msg = f"The input array for step lengths must be 3 dimensional not {data[idx.dim]}"
            raise ValueError(msg)

    counts = ak.num(data[0], axis=-1)
    data = np.vstack([ak.flatten(ak.flatten(var)).to_numpy() for var in data])
    dist = np.append(np.sqrt(np.sum(np.diff(data, axis=1) ** 2, axis=0)), 0)

    n_cluster = ak.num(counts, axis=-1)
    clusters = ak.unflatten(ak.Array(dist), ak.flatten(counts))

    out = ak.unflatten(clusters, n_cluster)
    return VectorOfVectors(out[:, :, :-1])
