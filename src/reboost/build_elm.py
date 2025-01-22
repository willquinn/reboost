from __future__ import annotations

import logging

import awkward as ak
import numpy as np
from lgdo import Array, Table, lh5
from lgdo.lh5 import LH5Iterator, LH5Store
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)


def get_elm_rows(stp_evtids: ArrayLike, vert: ArrayLike, *, start_row: int = 0) -> ak.Array:
    """Get the rows of the event lookup map (elm).

    Parameters
    ----------
    stp_evtids
        Array of evtids for the steps
    vert
        Array of simulated evtid for the vertices.
    start_row
        The index of the first element of stp_evtids.

    Returns
    -------
    an awkward array of the `elm`.
    """
    # convert inputs
    if not isinstance(stp_evtids, np.ndarray):
        stp_evtids = np.array(stp_evtids)
    if not isinstance(vert, np.ndarray):
        vert = np.array(vert)

    # check that the steps and vertices are sorted or the algorithm will fail

    if not np.all(vert[:-1] <= vert[1:]):
        msg = "The vertices must be sorted"
        raise ValueError(msg)

    if not np.all(stp_evtids[:-1] <= stp_evtids[1:]):
        msg = "The steps must be sorted"
        raise ValueError(msg)

    # convert to numpy
    stp_indices = np.arange(len(stp_evtids)) + start_row

    # cut the arrays
    sel = (stp_evtids >= vert[0]) & (stp_evtids <= vert[-1])
    stp_evtids = stp_evtids[sel]
    stp_indices = stp_indices[sel]

    # build output
    output = ak.Array({"evtid": vert})

    # restructure to jagged array
    counts = ak.run_lengths(stp_evtids)
    steps = ak.unflatten(stp_evtids, counts)
    indices = ak.unflatten(stp_indices, counts)

    ak_tmp = ak.Array({"evtid": ak.fill_none(ak.firsts(steps), np.nan), "indices": indices})

    # find indices to insert the new entries
    positions = np.searchsorted(output.evtid, ak_tmp.evtid)

    # add a check that every ak_tmp evtid is in output.evtid ?
    if not np.all(np.isin(ak_tmp.evtid, output.evtid)):
        bad_evtid = ak_tmp.evtid[~np.isin(ak_tmp.evtid, output.evtid)]
        msg = f"Error not every evtid in the stp table is present in the vertex table {bad_evtid} are not"
        raise ValueError(msg)

    # get the start row
    start_row = np.array(ak.Array([np.nan] * len(vert)), dtype=float)
    start_row[positions] = ak.fill_none(ak.firsts(ak_tmp.indices), np.nan)

    n_row = np.array(ak.Array([0] * len(vert)), dtype=float)
    n_row[positions] = ak.num(ak_tmp.indices)

    # add to the  output
    output["n_rows"] = n_row
    output["start_row"] = start_row

    return output


def get_stp_evtids(
    lh5_table: str,
    stp_file: str,
    id_name: str,
    start_row: int,
    last_vertex_evtid: int,
    stp_buffer: int,
) -> tuple[int, int, ak.Array]:
    """Extracts the rows of a stp file corresponding to a particular range of `evtid`.
    The reading starts at `start_row` to allow for iterating through the file. The iteration
    stops when the `evtid` being read is larger than `last_vertex_evtid`.

    Parameters
    ----------
    lh5_table
        the table name to read.
    stp_file
        the file name path.
    id_name
        the name of the `evtid` field.
    start_row
        the row to begin reading.
    last_vertex_evtid
        the last evtid to read up to.
    stp_buffer
        the number of rows to read at once.

    Returns
    -------
         a tuple of the updated `start_row`, the first row for the chunk and an awkward Array of the steps.
    """

    # make a LH5Store
    store = LH5Store()

    # some outputs
    evtids_proc = None
    last_evtid = 0
    chunk_start = 0

    # get the total number of rows
    n_rows_tot = store.read_n_rows(f"{lh5_table}/{id_name}", stp_file)

    # iterate over the file
    # stop when the entire file is read

    while start_row < n_rows_tot:
        # read the file
        lh5_obj, n_read = store.read(
            f"{lh5_table}/{id_name}",
            stp_file,
            start_row=start_row,
            n_rows=stp_buffer,
        )
        evtids = lh5_obj.view_as("ak")

        # if the evtids_proc is not set then this is the first valid chunk
        if evtids_proc is None:
            evtids_proc = evtids
            chunk_start = start_row
        elif evtids[0] <= last_vertex_evtid:
            evtids_proc = ak.concatenate((evtids_proc, evtids))

        # get the last evtid
        last_evtid = evtids[-1]

        # if the last evtid is greater than the last vertex we can stop reading
        if last_evtid > last_vertex_evtid or (start_row + n_read >= n_rows_tot):
            break

        # increase rhe start row for the next read

        if start_row + n_read <= n_rows_tot:
            start_row += n_read

    return start_row, chunk_start, evtids_proc


def build_elm(
    stp_file: str,
    elm_file: str | None,
    *,
    id_name: str = "g4_evtid",
    evtid_buffer: int = 10000,
    stp_buffer: int = 1000,
) -> ak.Array | None:
    """Builds a g4_evtid look up (elm) from the stp data.

    This object is used by `reboost` to efficiency iterate through the data.
    It consists of a :class:`LGDO.VectorOfVectors` for each lh5_table in the input files.
    The rows of this :class:`LGDO.VectorOfVectors` correspond to the `id_name` while the data
    are the `stp` indices for this event.

    Parameters
    ----------
    stp_file
        path to the stp (input) file.
    elm_file
        path to the elm data, can also be `None` in which case an `ak.Array` is returned in memory.
    id_name
        name of the evtid file, default `g4_evtid`.
    stp_buffer
        the number of rows of the step file to read at a time
    evtid_buffer
        the number of evtids to read at a time

    Returns
    -------
    either `None` or an `ak.Array`
    """

    store = LH5Store()

    # loop over the lh5_tables
    lh5_table_list = [table for table in lh5.ls(stp_file, "stp/") if "vertices" not in table]

    # get rows in the table
    elm_sum = {lh5_table.replace("stp/", ""): None for lh5_table in lh5_table_list}

    # start row for each table
    start_row = {lh5_tab: 0 for lh5_tab in lh5_table_list}

    vfield = f"stp/vertices/{id_name}"

    # iterate over the vertex table
    for vert_obj, vidx, n_evtid in LH5Iterator(stp_file, vfield, buffer_len=evtid_buffer):
        # range of vertices
        vert_ak = vert_obj.view_as("ak")[:n_evtid]

        for idx, lh5_table in enumerate(lh5_table_list):
            # create the output table
            out_tab = Table(size=len(vert_ak))

            # read the stp rows starting from `start_row` until the
            # evtid is larger than that in the vertices

            start_row_tmp, chunk_row, evtids = get_stp_evtids(
                lh5_table,
                stp_file,
                id_name,
                start_row[lh5_table],
                last_vertex_evtid=vert_ak[-1],
                stp_buffer=stp_buffer,
            )

            start_row[lh5_table] = start_row_tmp

            # now get the elm rows

            elm = get_elm_rows(evtids, vert_ak, start_row=chunk_row)

            for field in ["evtid", "n_rows", "start_row"]:
                out_tab.add_field(field, Array(elm[field]))

            # write the output file
            mode = "of" if (vidx == 0 and idx == 0) else "append"

            lh5_subgroup = lh5_table.replace("stp/", "")

            if elm_file is not None:
                store.write(out_tab, f"elm/{lh5_subgroup}", elm_file, wo_mode=mode)
            else:
                elm_sum[lh5_subgroup] = (
                    elm
                    if elm_sum[lh5_subgroup] is None
                    else ak.concatenate((elm_sum[lh5_subgroup], elm))
                )

    return ak.Array(elm_sum)
