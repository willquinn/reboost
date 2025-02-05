from __future__ import annotations

import logging

import awkward as ak
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

    counts = ak.run_lengths(cluster_variable)

    n_cluster = ak.num(counts, axis=-1)
    clusters = ak.unflatten(ak.flatten(field), ak.flatten(counts))

    return ak.unflatten(clusters, n_cluster)
