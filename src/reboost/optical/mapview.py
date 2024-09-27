from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lgdo.lh5 import LH5Store
from matplotlib import colors

log = logging.getLogger(__name__)


def _get_weights(viewdata: dict):
    rolled = np.roll([0, 1, 2], -viewdata["axis"])
    weights = viewdata["weights"].transpose(*rolled)

    extent_axes = [i for i in np.flip(rolled) if i != viewdata["axis"]]
    extent = [viewdata["edges"][i] for i in extent_axes]
    extent_d = [(e[1] - e[0]) / 2 for e in extent]
    extent = [(e[0] - extent_d[i], e[-1] + extent_d[i]) for i, e in enumerate(extent)]

    labels = [["x", "y", "z"][i] + " [m]" for i in extent_axes]

    return weights, np.array(extent).flatten(), labels


def _slice_text(viewdata: dict) -> str:
    axis_name = ["x", "y", "z"][viewdata["axis"]]
    axis_pos = viewdata["edges"][viewdata["axis"]][viewdata["idx"]]
    return f"{viewdata['detid']} | slice {axis_name} = {axis_pos:.2f} m"


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

    w, extent, labels = _get_weights(viewdata)
    viewdata["idx"] = max(min(viewdata["idx"], w.shape[0] - 1), 0)

    # update figure
    ax = fig.axes[0]
    ax.texts[0].set_text(_slice_text(viewdata))
    ax.images[0].set_array(w[viewdata["idx"]])
    ax.images[0].set_extent(extent)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    fig.canvas.draw()


def view_optmap(
    optmap_fn: str,
    detid: str = "all",
    start_axis: int = 2,
    cmap_min: float = 1e-3,
    cmap_max: float = 1e-2,
    title: str | None = None,
) -> None:
    store = LH5Store(keep_open=True)

    optmap_all = store.read(f"/{detid}/p_det", optmap_fn)[0]
    optmap_edges = tuple([b.edges for b in optmap_all.binning])
    optmap_weights = optmap_all.weights.nda.copy()

    lower_count = np.sum((optmap_weights > 0) & (optmap_weights < cmap_min))
    if lower_count > 0:
        log.warning(
            "%d cells are non-zero and lower than the current colorbar minimum %.2e",
            lower_count,
            cmap_min,
        )
    higher_count = np.sum(optmap_weights > cmap_max)
    if higher_count > 0:
        log.warning(
            "%d cells are non-zero and higher than the current colorbar maximum %.2e",
            higher_count,
            cmap_max,
        )

    # set zero/close-to-zero values to a very small, but nonzero value. This means we can
    # style those cells using the `under` style and, can style `bad` (i.e. everything < 0
    # after this re-assignment) in a different way.
    optmap_weights[(optmap_weights >= 0) & (optmap_weights < 1e-50)] = min(1e-10, cmap_min)

    fig = plt.figure()
    fig.canvas.mpl_connect("key_press_event", _process_key)
    start_axis_len = optmap_edges[start_axis].shape[0] - 1
    fig.__reboost = {
        "axis": start_axis,
        "weights": optmap_weights,
        "detid": detid,
        "edges": optmap_edges,
        "idx": min(int(start_axis_len / 2), start_axis_len - 1),
    }

    cmap = plt.cm.plasma.with_extremes(bad="w", under="gray", over="red")
    weights, extent, labels = _get_weights(fig.__reboost)
    plt.imshow(
        weights[fig.__reboost["idx"]],
        norm=colors.LogNorm(vmin=cmap_min, vmax=cmap_max),
        aspect=1,
        interpolation="none",
        cmap=cmap,
        extent=extent,
    )

    plt.suptitle(Path(optmap_fn).stem if title is None else title)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    plt.text(0, 1.02, _slice_text(fig.__reboost), transform=fig.axes[0].transAxes)
    plt.colorbar()
    plt.show()
