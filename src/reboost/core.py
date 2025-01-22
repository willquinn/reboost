from __future__ import annotations

import logging

from . import utils

logging.basicConfig(level=logging.INFO)


def get_detector_objects(output_detectors, proc_group, args, global_objects):
    """Get the detector objects"""
    det_objects_dict = {}
    for output_detector in output_detectors:
        det_objects_dict[output_detector] = utils.dict2tuple(
            {
                obj_name: eval_object(
                    obj_expression,
                    local_dict={
                        "ARGS": args,
                        "DETECTOR": output_detector,
                        "OBJECTS": global_objects,
                    },
                )
                for obj_name, obj_expression in proc_group.get("detector_objects", {}).items()
            }
        )
    return utils.dict2tuple(det_objects_dict)


def eval_object(obj_expression, local_dict=None):
    raise NotImplementedError


def merge_arrays(ak_obj, buffer_rows=None):
    raise NotImplementedError


def eval_expression(hit_table, table_name, expression, local_dict):
    raise NotImplementedError
