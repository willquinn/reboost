from __future__ import annotations

import io
import logging
from collections.abc import Mapping

import awkward as ak
from legendmeta import AttrsDict
from lgdo import lh5

import utils

from . import core, shape

log = logging.getLogger(__name__)


def build_hit(
    config: Mapping | str,
    args: Mapping | AttrsDict,
    input_files: list | str,
    file_out: str | None = None,
    *,
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
        file_out
            name of the output file, if `None` return an `ak.Array` in memory of the post-processed hits.
        input_files
            list of string of the path to the files to process.
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
    if isinstance(input_files, str):
        input_files = list(input_files)

    output_table = None

    # iterate over files
    for file_idx, file_in in enumerate(input_files):
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

                # begin iterating over the input file / files
                it, entries, max_idx = io.get_iterator(
                    file=file_in, lh5_table=f"{in_field}/{in_detector}", buffer=buffer
                )
                buffer_rows = None

                # iterate over the LH5 file
                for idx, (lh5_obj, _, n_rows) in enumerate(it):
                    # converting to awwkard
                    ak_obj = lh5.view_as(lh5_obj, "ak")

                    if idx == max_idx:
                        ak_obj = ak_obj[:n_rows]

                    ak_obj, buffer_rows, wo_mode = core.merge_arrays(
                        ak_obj, buffer_rows=buffer_rows
                    )

                    # loop over output detectors
                    for out_det_idx, out_detector in enumerate(out_detectors):
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
                            chunk_idx=idx,
                            max_chunk_idx=max_idx,
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
