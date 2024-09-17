from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from lgdo.lh5 import LH5Store
from matplotlib import colors

log = logging.getLogger(__name__)


def _get_weights(viewdata: dict):
    rolled = np.roll([0, 1, 2], -viewdata["axis"])
    weights = viewdata["weights"].transpose(*rolled)

    extent = [viewdata["edges"][i] for i in np.flip(rolled) if i != viewdata["axis"]]
    extent_d = [(e[1] - e[0]) / 2 for e in extent]
    extent = [(e[0] - extent_d[i], e[-1] + extent_d[i]) for i, e in enumerate(extent)]

    return weights, np.array(extent).flatten()


def _slice_text(viewdata: dict) -> str:
    axis_name = ["x", "y", "z"][viewdata["axis"]]
    axis_pos = viewdata["edges"][viewdata["axis"]][viewdata["idx"]]
    return f"slice {axis_name} = {axis_pos:.2f}"


def _process_key(event):
    fig = event.canvas.figure
    viewdata = fig.__reboost

    if event.key == "up":
        viewdata["axis"] = min(viewdata["axis"] + 1, 2)
    elif event.key == "down":
        viewdata["axis"] = max(viewdata["axis"] - 1, 0)
    elif event.key == "right":
        viewdata["idx"] += 1
    elif event.key == "left":
        viewdata["idx"] -= 1

    w, extent = _get_weights(viewdata)
    viewdata["idx"] = max(min(viewdata["idx"], w.shape[0] - 1), 0)

    # update figure
    ax = fig.axes[0]
    ax.texts[0].set_text(_slice_text(viewdata))
    ax.images[0].set_array(w[viewdata["idx"]])
    ax.images[0].set_extent(extent)
    fig.canvas.draw()


def view_optmap(optmap_fn: str, detid: str = "all", start_axis: int = 2) -> None:
    store = LH5Store(keep_open=True)

    optmap_all = store.read(f"/{detid}/p_det", optmap_fn)[0]
    optmap_edges = tuple([b.edges for b in optmap_all.binning])
    optmap_weights = optmap_all.weights.nda

    fig = plt.figure()
    fig.canvas.mpl_connect("key_press_event", _process_key)
    fig.__reboost = {
        "axis": start_axis,
        "weights": optmap_weights,
        "edges": optmap_edges,
        "idx": int(optmap_edges[start_axis].shape[0] / 2),
    }
    cmap = plt.cm.inferno.with_extremes(bad="w", under="w")
    weights, extent = _get_weights(fig.__reboost)
    plt.imshow(
        weights[fig.__reboost["idx"]],
        # norm=colors.Normalize(vmin=1e-100),
        norm=colors.LogNorm(vmin=1e-4),
        aspect=1,
        interpolation="none",
        cmap=cmap,
        extent=extent,
    )

    plt.text(0, 1.02, _slice_text(fig.__reboost), transform=fig.axes[0].transAxes)

    plt.colorbar()
    plt.show()
