from __future__ import annotations

import awkward as ak
import numpy as np
from lgdo import lh5

from reboost.hpge.drift_time import ReadDTFile
from reboost.hpge.psd import calculate_drift_times, psa_heuristic_numba
from reboost.math.functions import piecewise_linear_activeness
from reboost.shape.cluster import apply_cluster, cluster_by_step_length
from reboost.shape.group import group_by_time


def do_cluster(grouped_data, cluster_size_mm, dt_file):
    cluster_indices = cluster_by_step_length(
        grouped_data["evtid"],
        grouped_data["xloc"] * 1000,  # Convert to mm
        grouped_data["yloc"] * 1000,  # Convert to mm
        grouped_data["zloc"] * 1000,  # Convert to mm
        grouped_data["dist_to_surf"] * 1000,  # Convert to mm
        threshold=cluster_size_mm,
    )
    clustered_data = {
        f"{field}": apply_cluster(cluster_indices, grouped_data[f"{field}"])
        for field in ["edep", "xloc", "yloc", "zloc", "time", "dist_to_surf"]
    }
    clustered_data["activeness"] = piecewise_linear_activeness(
        clustered_data["dist_to_surf"], 0.5, 0.5
    )

    cluster_energy = ak.sum(clustered_data["edep"] * clustered_data["activeness"], axis=-1)
    cluster_energy_shaped = ak.broadcast_arrays(clustered_data["edep"], cluster_energy)[1]
    weights = ak.where(cluster_energy_shaped > 0, clustered_data["edep"] / cluster_energy_shaped, 0)

    clusters = {
        f"{field}": ak.sum(clustered_data[f"{field}"] * weights, axis=-1)
        for field in ["xloc", "yloc", "zloc", "time", "dist_to_surf", "activeness"]
    }
    clusters["edep"] = cluster_energy

    clusters["cluster_size"] = (
        ak.sum(
            weights
            * (
                (
                    clustered_data["xloc"]
                    - ak.broadcast_arrays(clustered_data["xloc"], clusters["xloc"])[1]
                )
                ** 2
                + (
                    clustered_data["yloc"]
                    - ak.broadcast_arrays(clustered_data["yloc"], clusters["yloc"])[1]
                )
                ** 2
                + (
                    clustered_data["zloc"]
                    - ak.broadcast_arrays(clustered_data["zloc"], clusters["zloc"])[1]
                )
                ** 2
            ),
            axis=-1,
        )
        ** 0.5
    )

    clusters["drift_time"] = calculate_drift_times(
        clusters["xloc"], clusters["yloc"], clusters["zloc"], dt_file
    )

    return ak.Array(clusters)


def test_dt_heuristic():
    det_pos = [50, 0, -30]
    dt_file = ReadDTFile("/root/reboost/tests/hpge/test_files/B99000A_drift_time_map.dat", *det_pos)
    cluster_size_mm = 0.1

    data = lh5.read_as("stp/det001/", "test_files/internal_electron.lh5", "ak")
    grouped_data = group_by_time(data, evtid_name="evtid").view_as("ak")

    # First calculate the dt heur on the none clustered data
    drift_times = calculate_drift_times(
        grouped_data["xloc"], grouped_data["yloc"], grouped_data["zloc"], grouped_data
    )
    drift_times_flat = ak.flatten(drift_times).to_numpy()
    energies_flat = ak.flatten(grouped_data["edep"]).to_numpy()
    event_offsets = np.append(0, np.cumsum(ak.num(drift_times)))
    _non_clustered_dth = psa_heuristic_numba(drift_times_flat, energies_flat, event_offsets)

    # Now cluster the data, then do dt heur
    cluster_data = do_cluster(grouped_data, cluster_size_mm, dt_file)
    drift_times_flat = ak.flatten(cluster_data["drift_time"]).to_numpy()
    energies_flat = ak.flatten(cluster_data["edep"]).to_numpy()
    event_offsets = np.append(0, np.cumsum(ak.num(cluster_data["drift_time"])))
    _clustered_dth = psa_heuristic_numba(drift_times_flat, energies_flat, event_offsets)


if __name__ == "__main__":
    test_dt_heuristic()
