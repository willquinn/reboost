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
    counts = ak.run_lengths(cluster_variable)

    n_cluster = ak.num(counts, axis=-1)
    clusters = ak.unflatten(ak.flatten(field), ak.flatten(counts))

    return VectorOfVectors(ak.unflatten(clusters, n_cluster))
