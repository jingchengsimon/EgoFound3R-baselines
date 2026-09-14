"""Select and serialize the per-side WiLoR inputs consumed by PAD-Hand."""

import numpy as np


SHAPES = {
    "vertices": (778, 3),
    "cam_t": (3,),
    "global_orient": (1, 3, 3),
    "hand_pose": (15, 3, 3),
    "betas": (10,),
    "is_right": (),
    "img_size": (2,),
    "scaled_focal": (),
}


def select_hands(frame_hands, both_hands=False):
    """Return left/right slots, or the demo's original single preferred hand."""
    right = next((hand for hand in frame_hands if hand["is_right"] > 0.5), None)
    if both_hands:
        left = next((hand for hand in frame_hands if hand["is_right"] <= 0.5), None)
        return (left, right)
    return (right if right is not None else next(iter(frame_hands), None),)


def pack_frames(frames, both_hands=False):
    """Store missing detections as NaN, preserving one array slot per side."""
    sides = 2 if both_hands else 1
    arrays = {key: np.full((len(frames), sides, *shape), np.nan, dtype=np.float32)
              for key, shape in SHAPES.items()}
    for frame, hands in enumerate(frames):
        if len(hands) != sides:
            raise ValueError("hand slot count does not match output format")
        for side, hand in enumerate(hands):
            if hand is None:
                continue
            if both_hands and bool(hand["is_right"] > 0.5) != bool(side):
                raise ValueError("hand side does not match its output slot")
            for key, shape in SHAPES.items():
                value = np.asarray(hand[key], dtype=np.float32)
                if value.shape != shape:
                    raise ValueError(f"{key} has shape {value.shape}, expected {shape}")
                arrays[key][frame, side] = value
    if not both_hands:
        arrays = {key: value[:, 0] for key, value in arrays.items()}
    return arrays
