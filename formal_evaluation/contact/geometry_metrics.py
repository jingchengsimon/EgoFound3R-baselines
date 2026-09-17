"""Formal unsigned surface distances, using the existing GT contact kernel."""
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
from formal_evaluation.common.mano_sampling import marker_vertex_ids_195, upsample_mano_markers
from formal_evaluation.contact.metrics import compute_contact_metrics


@lru_cache(maxsize=1)
def surface_kernel():
    # Load the unchanged numerical kernel without importing training/model builders.
    path = Path(__file__).parents[1] / 'vendor/egofound3r_dataloader/egohandmetric_prompt/data/contact.py'
    spec = importlib.util.spec_from_file_location('formal_contact_surface_kernel', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def query_geometry(prediction):
    from formal_evaluation.evaluate_six_dataset import _prediction_geometry_camera
    result = {}
    for granularity in ('joint','marker','vertex'):
        points, _ = _prediction_geometry_camera(prediction,granularity)
        if points is not None:result[granularity]=points
    return result


def frame_geometry_metrics(query, query_valid, scene, mano_faces, device='cpu'):
    """Query predicted/GT points against GT object and opposite-hand surfaces.

    Match official object_and_interhand semantics: unknown object invalidates
    the frame; an absent opposite hand is a known absent source. Never use the
    query hand itself as a surface. Input coordinates and distances are metres.
    """
    import torch
    kernel = surface_kernel()
    tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=device)
    obj = kernel._valid_mesh((tensor(scene['object_vertices']), tensor(scene['object_faces'], torch.long)), device=torch.device(device))
    faces = tensor(mano_faces, torch.long)
    output = {}
    for prefix, points in query.items():
        points = np.asarray(points); shape = points.shape[:-1]
        target = np.full(shape, np.nan, np.float32); distance = target.copy()
        mask = np.zeros(shape, bool); distance_mask = mask.copy()
        for side in range(2):
            if obj is None or not query_valid[side]: continue
            pts = tensor(points[side]); good = torch.isfinite(pts).all(-1)
            thresholds = kernel._joint_thresholds(torch.device(device)) if prefix == 'joint' else torch.full(good.shape, kernel.MARKER_CONTACT_THRESHOLD_METERS, device=device)
            first = kernel._source_targets(pts, good, obj, thresholds, point_to_mesh_compute_device=device)
            other = 1 - side
            if bool(scene['hand_valid'][other]):
                other_mesh = kernel._valid_mesh((tensor(scene['hand_vertices'][other]), faces), device=torch.device(device))
                if other_mesh is None: continue
                second = kernel._source_targets(pts, good, other_mesh, thresholds, point_to_mesh_compute_device=device)
            else:
                second = kernel._known_absent_source(good)
            values, valid = kernel._union_sources(first[0], first[1], second[0], second[1])
            dist, dvalid = kernel._combine_distances(first[2], first[1], second[2], second[1])
            v = valid.cpu().numpy(); dv = dvalid.cpu().numpy()
            target[side] = np.where(v, values.cpu().numpy(), np.nan)
            distance[side] = np.where(dv, dist.cpu().numpy(), np.nan)
            mask[side] = v; distance_mask[side] = dv
        output.update({prefix+'_contact_target': target, prefix+'_contact_mask': mask,
                       prefix+'_contact_distance': distance, prefix+'_contact_distance_mask': distance_mask})
    return output


def aggregate_geometry_contact(prediction, target, keep):
    """All valid distances and predicted-contact distances are both retained."""
    result = {}
    for prefix in ('joint', 'marker', 'vertex'):
        probability = prediction.get(prefix+'_contact_probability')
        mask = target.get(prefix+'_contact_mask')
        if probability is not None and mask is not None:
            valid = np.asarray(mask, bool) & np.asarray(keep)[:, None, None]
            valid &= np.asarray(prediction['hand_valid'], bool)[..., None]
            result.update({prefix+'_contact_'+k: v for k, v in compute_contact_metrics(probability, target[prefix+'_contact_target'], valid).items()})
        distance = prediction.get(prefix+'_contact_distance')
        distance_mask = prediction.get(prefix+'_contact_distance_mask')
        if distance is None or distance_mask is None: continue
        valid = np.asarray(distance_mask, bool) & np.asarray(keep)[:, None, None] & np.isfinite(distance)
        for name, selected in [('all_valid', valid), ('predicted_contact', valid & (np.asarray(probability) >= .5) if probability is not None else np.zeros_like(valid))]:
            result[prefix+'_contact_distance_'+name+'_mm'] = float(np.asarray(distance)[selected].mean()*1000) if selected.any() else float('nan')
            result[prefix+'_contact_distance_'+name+'_count'] = int(selected.sum())
    return result
