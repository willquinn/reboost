from __future__ import annotations

import io
import logging
import typing
from collections.abc import Mapping

import awkward as ak
from legendmeta import AttrsDict
from lgdo import lh5
from lgdo.lh5 import LH5Store
from lgdo.types import LGDO

from . import core, shape, utils

log = logging.getLogger(__name__)


class ELMIterator:
    """A class to iterate over the rows of an event lookup map"""

    def __init__(
        self,
        elm_file: str,
        stp_file: str,
        lh5_group: str,
        start_row: int,
        n_rows: int | None,
        *,
        stp_field: str = "stp",
        read_vertices: bool = False,
        buffer: int = 10000,
    ):
        """Constructor for the ELMIterator.

        Parameters
        ----------
        elm_file
            the file containing the event lookup map.
        stp_file
            the file containing the steps to read.
        lh5_subgroup
            the name of the lh5 subgroup to read.
        start_row
            the first row to read.
        n_rows
            the number of rows to read, if `None` read them all.
        read_vertices
            whether to read also the vertices table.
        buffer
            the number of rows to read at once.

        """

        # initialise
        self.elm_file = elm_file
        self.stp_file = stp_file
        self.lh5_group = lh5_group
        self.start_row = start_row
        self.start_row_tmp = start_row
        self.n_rows = n_rows
        self.buffer = buffer
        self.current_i_entry = 0
        self.read_vertices = read_vertices
        self.stp_field = stp_field

        # would be good to replace with an iterator
        self.sto = LH5Store()
        self.n_rows_read = 0

    def __iter__(self) -> typing.Iterator:
        self.current_i_entry = 0
        self.n_rows_read = 0
        self.start_row_tmp = self.start_row
        return self

    def __next__(self) -> tuple[LGDO, LGDO | None, int, int]:
        # get the number of rows to read
        rows_left = self.n_rows - self.n_rows_read
        n_rows = self.buffer if (self.buffer > rows_left) else rows_left

        # read the elm rows
        elm_rows, n_rows_read = self.sto.read(
            f"elm/{self.lh5_group}", self.elm_file, start_row=self.start_row_tmp, n_rows=n_rows
        )

        self.n_rows_read += n_rows_read
        self.start_row_tmp += n_rows_read

        if n_rows_read == 0:
            raise StopIteration

        # view our elm as an awkward array
        elm_ak = elm_rows.view_as("ak")

        # remove empty rows
        elm_ak = elm_ak[elm_ak.n_rows > 0]

        # extract range of stp rows to read
        start = elm_ak.start_row[0]
        n = sum(elm_ak.n_rows)

        stp_rows, n_steps = self.sto.read(
            f"{self.stp_field}/{self.lh5_group}", self.stp_file, start_row=start, n_rows=n
        )

        self.current_i_entry += 1

        if self.read_vertices:
            vert_rows, _ = self.sto.read(
                f"{self.stp_field}/vertices",
                self.stp_file,
                start_row=self.start_row,
                n_rows=n_rows,
            )
        else:
            vert_rows = None
        # vertex table should have same structure as elm

        return (stp_rows, vert_rows, self.current_i_entry, n_steps)


def build_hit(
    config: Mapping | str,
    args: Mapping | AttrsDict,
    stp_files: list | str,
    elm_files: list | str,
    file_out: str | None = None,
    *,
    start_evtid: int = 0,
    n_evtid: int | None = None,
    in_field: str = "stp",
    out_field: str = "hit",
    merge_input_files: bool = True,
    buffer: int = int(5e6),
) -> None | ak.Array:
    """Build the hit tier from the remage step files.

    Parameters
    ----------
    config
        dictionary or path to YAML file containing the processing chain. For example:

        .. code-block:: yaml

            # dictionary of objects useful for later computation. they are constructed with
            # auxiliary data (e.g. metadata). They can be accessed later as OBJECTS (all caps)
            objects:
             lmeta: LegendMetadata(ARGS.legendmetadata)
             geometry: pyg4ometry.load(ARGS.gdml)
             user_pars: legendmeta.TextDB(ARGS.par)
             dataprod_pars: legendmeta.TextDB(ARGS.dataprod_cycle)

            # processing chain is defined to act on a group of detectors
            processing_groups:

                # start with HPGe stuff, give it an optional name
                - name: geds

                  # this is a list of included detectors (part of the processing group)
                  output_detectors: OBJECTS.lmeta.channelmap(on=ARGS.timestamp)
                    .group('system').geds
                    .group('analysis.status').on
                    .map('name').keys()

                  # which columns we actually want to see in the output table
                  outputs:
                     - t0
                     - evtid
                     - energy
                     - r90
                     - drift_time

                  # in this section we define objects that will be instantiated at each
                  # iteration of the for loop over input tables (i.e. detectors)
                  detector_objects:
                     # The following assumes that the detector metadata is stored in the GDML file
                     pyobj: legendhpges.make_hpge(pygeomtools.get_sensvol_metadata(OBJECTS.geometry, DETECTOR))
                     phyvol: OBJECTS.geometry.physical_volume_dict[DETECTOR]
                     drift_time_map: lgdo.lh5.read(DETECTOR, ARGS.dtmap_file)

                  # this defines "hits", i.e. layout of the output hit table
                  # group steps by time and evtid with 10us window
                  hit_table_layout: reboost.shape.group_by_time(STEPS, window=10)

                  # finally, the processing chain
                  operations:

                    t0: ak.fill_none(ak.firsts(HITS.time, axis=-1), np.nan)

                    evtid: ak.fill_none(ak.firsts(HITS.__evtid, axis=-1), np.nan)

                    # distance to the nplus surface in mm
                    distance_to_nplus_surface_mm: reboost.hpge.distance_to_surface(
                        HITS.__xloc, HITS.__yloc, HITS.__zloc,
                        DETECTOR_OBJECTS.pyobj,
                        DETECTOR_OBJECTS.phyvol.position.eval(),
                        surface_type='nplus')

                    # activness based on FCCD (no TL)
                    activeness: ak.where(
                        HITS.distance_to_nplus_surface_mm <
                            lmeta.hardware.detectors.germanium.diodes[DETECTOR].characterization.combined_0vbb_fccd_in_mm.value,
                        0,
                        1
                        )

                    activeness2: reboost.math.piecewise_linear(
                        HITS.distance_to_nplus_surface_mm,
                        PARS.tlayer[DETECTOR].start_in_mm,
                        PARS.fccd_in_mm,
                        )

                    # summed energy of the hit accounting for activeness
                    energy_raw: ak.sum(HITS.__edep * HITS.activeness, axis=-1)

                    # energy with smearing
                    energy: reboost.math.sample_convolve(
                        scipy.stats.norm, # resolution distribution
                        loc=HITS.energy_raw, # parameters of the distribution (observable to convolve)
                        scale=np.sqrt(PARS.a + PARS.b * HITS.energy_raw) # another parameter
                        )

                    # this is going to return "run lengths" (awkward jargon)
                    clusters_lengths: reboost.shape.cluster.naive(
                        HITS, # can also pass the exact fields (x, y, z)
                        size=1,
                        units="mm"
                        )

                    # example of low level reduction on clusters
                    energy_clustered: ak.sum(ak.unflatten(HITS.__edep, HITS.clusters_lengths), axis=-1)

                    # example of using a reboost helper
                    steps_clustered: reboost.shape.reduction.energy_weighted_average(HITS, HITS.clusters_lengths)

                    r90: reboost.hpge.psd.r90(HITS.steps_clustered)

                    drift_time: reboost.hpge.psd.drift_time(
                        HITS.steps_clustered,
                        DETECTOR_OBJECTS.drift_time_map
                        )

                # example basic processing of steps in scintillators
                - name: lar
                  output_detectors: scintillators
                  outputs:
                    - evtid
                    - tot_edep_wlsr

                  operations:
                    tot_edep_wlsr: ak.sum(HITS[(HITS.__detuid == 0) & (HITS.__zloc < 3000)].__edep, axis=-1)

                - name: spms

                  output_detectors: OBJECTS.lmeta.channelmap(on=ARGS.timestamp)
                    .group("system").spms
                    .group("analysis.status").on
                    .map("name").keys()

                  # by default, reboost looks in the steps input table for a table with the
                  # same name as the current detector. This can be overridden for special processors
                  input_detectors_mapping:
                    scintillators: OUTPUT_DETECTORS

                  outputs:
                    - t0
                    - evtid
                    - pe_times

                  detector_objects:
                     meta: pygeomtools.get_sensvol_metadata(OBJECTS.geometry, DETECTOR)
                     optmap_lar: lgdo.lh5.read(DETECTOR, "optmaps/pen", ARGS.optmap_path)
                     optmap_pen: lgdo.lh5.read(DETECTOR, "optmaps/lar", ARGS.optmap_path)

                  hit_table_layout: reboost.shape.group_by_time(STEPS, window=10)

                  operations:
                    pe_times_lar: reboost.spms.detected_photoelectrons(
                        STEPS,
                        DETECTOR_OBJECTS.optmap_lar,
                        0
                     )

                    pe_times_pen: reboost.spms.detected_photoelectrons(
                        STEPS,
                        DETECTOR_OBJECTS.optmap_pen,
                        1
                     )

                    pe_times: ak.concatenate([HITS.pe_times_lar, HITS.pe_times_pen], axis=-1)

        args
            dictionary or :class:`legendmeta.AttrsDict` of the global arguments.
        start_evtid
            first evtid to read.
        n_evtid
            number of evtid to read, if `None` read all.
        file_out
            name of the output file, if `None` return an `ak.Array` in memory of the post-processed hits.
        stp_files
            list of string of the path to the stp files to process.
        elm_files
            list of string of the path to the elm files to process.
        in_field
            name of the input field in the remage output.
        out_field
            name of the output field
        merge_input_files
            whether input files should be merged into one outpit
        buffer
            buffer for use in the `LH5Iterator`.

    """

    # extract the config file
    if isinstance(config, str):
        config = io.load_dict(config)

    # get the arguments
    if not isinstance(args, AttrsDict):
        args = AttrsDict(args)

    # get the global objects
    global_objects_dict = {
        obj_name: core.eval_object(expression, local_dict={"ARGS": args})
        for obj_name, expression in config.get("objects", {}).items()
    }

    # convert to a tuple
    global_objects = utils.dict2tuple(global_objects_dict)

    # get the input files
    if isinstance(stp_files, str):
        stp_files = list(stp_files)

    if isinstance(elm_files, str):
        elm_files = list(elm_files)

    output_table = None

    # iterate over files
    for file_idx, (stp_file, elm_file) in enumerate(zip(stp_files, elm_files)):
        # loop over processing groups
        for group_idx, proc_group in enumerate(config["processing_groups"]):
            # get the list of detectors
            output_detectors = core.get_detector_list(proc_group["output_detectors"])

            # get the mapping from output to input
            in_detectors_mapping = core.get_detector_mapping(
                proc_group.get("input_detector_mapping", None), output_detectors=output_detectors
            )

            # loop over detectors
            for in_det_idx, (in_detector, out_detectors) in enumerate(in_detectors_mapping.items()):
                # get detector objects
                det_objects = core.get_detector_objects(
                    output_detectors=output_detectors,
                    args=args,
                    global_objects=global_objects,
                    proc_group=proc_group,
                )

                # begin iterating over the elm
                for out_det_idx, out_detector in enumerate(out_detectors):
                    # loop over the rows
                    elm_it = ELMIterator(
                        elm_file,
                        stp_file,
                        lh5_group=in_detector,
                        start_row=start_evtid,
                        stp_field=in_field,
                        n_rows=n_evtid,
                        read_vertices=True,
                        buffer=buffer,
                    )
                    for stps, _, chunk_idx, _ in elm_it:
                        # converting to awwkard
                        ak_obj = stps.view_as("ak")

                        hit_table = shape.group.eval_hit_table_layout(
                            ak_obj, expression=proc_group["hit_table_layout"]
                        )

                        local_dict = {
                            "DETECTOR_OBJECTS": det_objects[out_detector],
                            "OBJECTS": global_objects,
                        }

                        for field, expression in proc_group["operations"].items():
                            # evaluate the expression
                            col = core.eval_expression(
                                hit_table,
                                table_name="STEPS",
                                expression=expression,
                                local_dict=local_dict,
                            )
                            hit_table.add_field(field, col)

                        # remove unwanted fields
                        existing_cols = list(hit_table.keys())
                        for col in existing_cols:
                            if col not in proc_group["outputs"]:
                                hit_table.remove_column(col, delete=True)

                        # write the file
                        file_out_tmp = (
                            f"{file_out.split('.')[0]}_{file_idx}.lh5"
                            if (merge_input_files is False)
                            else file_out
                        )

                        # get the IO mode
                        wo_mode = io.get_wo_mode(
                            file_idx=file_idx,
                            group_idx=group_idx,
                            in_det_idx=in_det_idx,
                            out_det_idx=out_det_idx,
                            chunk_idx=chunk_idx,
                        )

                        if file_out is not None:
                            file_out_tmp = (
                                f"{file_out.split('.')[0]}_{file_idx}.lh5"
                                if (merge_input_files is False)
                                else file_out
                            )
                            lh5.write(
                                hit_table,
                                f"{out_detector}/{out_field}",
                                file_out_tmp,
                                wo_mode=wo_mode,
                            )

                        else:
                            output_table = (
                                hit_table.view_as("ak")
                                if output_table is None
                                else ak.concatenate((output_table, hit_table.view_as("ak")))
                            )

    if output_table is not None:
        return output_table
    return None
