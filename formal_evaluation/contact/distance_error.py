"""Paired geometric distance MAE; both query sets use identical GT surfaces."""
import numpy as np


def window_distance_errors(prediction, target, keep):
    """Arrays use metres and canonical [frame, hand, point] correspondence.

    Callers must verify frame/hand/point identities and shared GT reference
    surfaces. Never pass native distance-head outputs as geometric distances.
    Dataset aggregation uses the existing finite-window mean.
    """
    result = {}
    for prefix, points in (("joint", 21), ("marker", 195), ("vertex", 778)):
        stem = prefix + "_contact_"
        pd, gd = (np.asarray(a[stem + "distance"]) for a in (prediction, target))
        pm, gm = (np.asarray(a[stem + "distance_mask"], bool) for a in (prediction, target))
        probability = np.asarray(prediction[stem + "probability"])
        label = np.asarray(target[stem + "target"])
        label_mask = np.asarray(target[stem + "mask"], bool)
        shape = (len(keep), 2, points)
        if any(a.shape != shape for a in (pd, gd, pm, gm, probability, label, label_mask)):
            raise ValueError("distance correspondence shape mismatch: " + prefix)
        valid = pm & gm & np.isfinite(pd) & np.isfinite(gd) & np.asarray(keep, bool)[:, None, None]
        error_mm = np.abs(pd - gd) * 1000
        subsets = {
            "all": valid,
            "pred_contact": valid & np.isfinite(probability) & (probability >= .5),
            "gt_contact": valid & label_mask & np.isfinite(label) & (label > .5),
        }
        for name, selected in subsets.items():
            key = prefix + "_contact_distance_mae_" + name
            result[key + "_mm"] = float(error_mm[selected].mean()) if selected.any() else float("nan")
            result[key + "_count"] = int(selected.sum())
    return result
