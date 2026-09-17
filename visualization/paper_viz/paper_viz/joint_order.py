"""Per-method 21-joint orderings, mapped onto the GT/MANO-21 skeleton convention.

GT, WiLoR, EgoForce, HaWoR and EgoFound3R emit the interleaved 21-joint ordering
used by ``scene.SKELETON_EDGES`` (wrist, then four joints per finger:
thumb, index, middle, ring, pinky).  PAD-Hand and ReViV4D instead emit the native
MANO ordering (wrist, 15 finger joints in MANO finger order, then the five tips
appended), so drawing them with the default edge list produces crossed bones.

The permutations below were recovered from data, not guessed: for every frame of
the 5-window ARCTIC smoke segment (100 samples per side) the per-frame
GT-to-method nearest assignment was aggregated and solved once with a Hungarian
assignment; the result is identical on both hands and every frame.
"""
from __future__ import annotations

import numpy as np

JOINT_REORDER: dict[str, tuple[int, ...]] = {
    # GT index -> PAD-Hand / ReViV4D index
    "pad_hand": (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 20, 7, 8, 9, 19),
    "reviv4d": (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20),
}


def joints_in_gt_order(method: str, joints: np.ndarray) -> np.ndarray:
    """Reorder ``(..., 21, 3)`` joints so SKELETON_EDGES connects the right pairs."""
    order = JOINT_REORDER.get(method)
    if order is None:
        return joints
    return joints[..., np.asarray(order), :]
