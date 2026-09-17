"""Shared visual style: palettes, fonts, legends, unavailable styling."""
from __future__ import annotations

import colorsys
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent / "assets"
HAND_FONT = ASSETS / "fonts" / "PatrickHand-Regular.ttf"

METHODS = ("ego", "wilor", "hawor", "reviv4d", "pad_hand", "egoforce", "dyn_hamr", "gt")
METHOD_LABELS = {
    "ego": "EgoFound3R (ours)",
    "wilor": "WiLoR",
    "hawor": "HaWoR",
    "reviv4d": "ReViV4D",
    "pad_hand": "PAD-Hand",
    "egoforce": "EgoForce",
    "dyn_hamr": "Dyn-HaMR",
    "gt": "Ground Truth",
}
METHOD_SHORT = {
    "ego": "EgoFound3R", "wilor": "WiLoR", "hawor": "HaWoR", "reviv4d": "ReViV4D",
    "pad_hand": "PAD-Hand", "egoforce": "EgoForce", "dyn_hamr": "Dyn-HaMR", "gt": "GT",
    "ego_contact": "Ego contact head", "s2contact": "S2Contact",
    "contactopt": "ContactOpt", "interactvlm": "InteractVLM",
}
METHOD_COLORS = {
    "ego": (140, 74, 91),
    "wilor": (94, 129, 172),
    "hawor": (122, 158, 126),
    "reviv4d": (201, 124, 93),
    "pad_hand": (154, 140, 184),
    "egoforce": (184, 160, 74),
    "dyn_hamr": (110, 123, 139),
    "gt": (58, 58, 62),
}
BASELINES = ("ego_contact", "s2contact", "contactopt", "interactvlm")
BASELINE_LABELS = {
    "ego_contact": "Ego contact head (pred)",
    "s2contact": "S2Contact",
    "contactopt": "ContactOpt",
    "interactvlm": "InteractVLM",
}
LIGHT = np.array([[0.91, 0.72, 0.77], [0.70, 0.82, 0.92]])
DARK = np.array([[0.70, 0.32, 0.43], [0.25, 0.46, 0.68]])

# ---------------------------------------------------------------------------
# Signal palette.
#
# "reference" reproduces the frozen inference visualisation of commit 8fc061a
# (`egohandmetric_prompt/inference_visualization.py` and the training monitor
# `training_visualization.py`): per-side marker / joint colours, binary
# visibility and contact class at p = 0.5, and a TURBO scale clipped at 50 mm for
# contact distance with edge colours averaged from their two endpoints.
#
# "soft" keeps the previous continuous ramps (red->yellow->green for visibility,
# cyan->yellow->red for contact, near-red->far-blue for distance) which read more
# gently on photographs but are not the reference convention.
# ---------------------------------------------------------------------------
PALETTE_MODE = "reference"
# Signal columns: "wireframe" = points + face edges (mmpose / monitor style);
# "face" = a faint per-face wash (mean of the three vertex values, nearest surface
# only) that makes the front/back occlusion readable, with the points + edges
# drawn on top at full strength so they stay the visual focus.
SIGNAL_STYLE = "wireframe"
SIGNAL_FACE_ALPHA = 0.30
# Occluded faces are painted first at SIGNAL_FACE_ALPHA * SIGNAL_FACE_OCCLUDED_FACTOR
# so a face's transparency follows its occlusion state rather than its depth.
SIGNAL_FACE_OCCLUDED_FACTOR = 0.35
# Points, edges and joints keep their signal colour and only change opacity:
# visible geometry is opaque, occluded geometry is drawn with this alpha so the
# value stays readable while the front/back ordering stays obvious.
OCCLUDED_ALPHA = 0.45

GEOMETRY_MARKER_RGB = ((47, 109, 176), (217, 120, 44))     # left blue, right orange
GEOMETRY_JOINT_RGB = ((255, 245, 0), (255, 0, 255))        # left yellow, right magenta
VISIBILITY_POSITIVE_RGB = (0, 255, 0)                      # visible (p >= 0.5)
VISIBILITY_NEGATIVE_RGB = (255, 0, 0)                      # hidden (p < 0.5)
CONTACT_POSITIVE_RGB = (255, 31, 31)                       # contact (p >= 0.5)
CONTACT_NEGATIVE_RGB = (89, 182, 217)                      # non-contact (p < 0.5)
SEMANTIC_UNKNOWN_RGB = (190, 190, 190)                     # non-finite value
CONTACT_DISTANCE_DISPLAY_MAX_M = 0.05                      # TURBO scale 0-50 mm

# Joint markers: bright fill plus a dark rim so they stay legible on both the dark
# tabletop and the bright paper, and on top of the mesh wireframe.
JOINT_OUTLINE = (34, 30, 26)

HAND_RGB = (np.array(GEOMETRY_MARKER_RGB[0], np.uint8), np.array(GEOMETRY_MARKER_RGB[1], np.uint8))
JOINT_RGB_SIDES = (np.array(GEOMETRY_JOINT_RGB[0], np.uint8), np.array(GEOMETRY_JOINT_RGB[1], np.uint8))
JOINT_RGB = tuple(int(c) for c in GEOMETRY_JOINT_RGB[0])

# Continuous ramps kept for PALETTE_MODE == "soft".
CONTACT_STOPS = np.array([[35, 199, 216], [251, 192, 45], [240, 59, 59]], np.float32)
VISIBILITY_STOPS = np.array([[229, 57, 53], [251, 192, 45], [105, 190, 85], [0, 166, 81]], np.float32)
DISTANCE_STOPS = np.array([[230, 45, 38], [250, 194, 46], [45, 205, 220], [37, 99, 235]], np.float32)

# Brightness knob. The reference palette is deliberately muted (it was tuned for
# the training monitor), so the figure lifts value/saturation: geometry the most,
# signal colours mildly. Hue is preserved, so the colour meaning is unchanged.
GEOMETRY_BRIGHTNESS = 1.0
SIGNAL_BRIGHTNESS = 1.0
# Geometry columns: mesh edges and vertex markers are drawn thicker than the signal
# wireframe so the hand outline and the individual vertices stay readable.
# Geometry columns reuse the signal rendering (radius-1 nodes, 1 px edges), so
# they default to the same stroke weight as the visibility / contact / distance
# columns and can be thickened through these two knobs if ever needed.
GEOMETRY_LINE_WIDTH = 1
GEOMETRY_DOT_RADIUS = 1
# Opacity of the occluded (back-side) dots and edges in the geometry columns.
# They stay visible - occlusion is shown by a lower opacity, never by hiding the
# point - but geometry gets its own knob so the front surface can dominate.
GEOMETRY_OCCLUDED_ALPHA = 0.45
# The dots and edges are mixed toward white so they read brighter than the face
# wash underneath them (the side colours already sit at value = 1, so a plain HSV
# lift cannot brighten them any further).
GEOMETRY_STROKE_LIGHTEN = 0.65
# Width of the outer silhouette contour drawn over each hand wireframe (0 = off).
GEOMETRY_CONTOUR_WIDTH = 0
# Same faint wash as the signal columns.
GEOMETRY_FACE_WASH = 0.30
BACKGROUND = (247, 247, 247)
INK = (38, 49, 63)
MUTED = (96, 107, 120)
UNAVAILABLE_FILL = (232, 232, 230)
UNAVAILABLE_LINE = (190, 190, 188)


def interpolate_palette(values: np.ndarray, stops: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    position = values * (len(stops) - 1)
    low = np.minimum(position.astype(np.int32), len(stops) - 2)
    fraction = (position - low)[..., None]
    colors = (1 - fraction) * stops[low] + fraction * stops[low + 1]
    return _boost_array(colors.astype(np.uint8), SIGNAL_BRIGHTNESS)


def set_palette(mode: str) -> None:
    global PALETTE_MODE
    if mode not in ("reference", "soft"):
        raise ValueError(f"unknown palette mode: {mode}")
    PALETTE_MODE = mode


def set_signal_style(style_name: str) -> None:
    global SIGNAL_STYLE
    if style_name not in ("wireframe", "face"):
        raise ValueError(f"unknown signal style: {style_name}")
    SIGNAL_STYLE = style_name


def set_face_alpha(alpha: float) -> None:
    global SIGNAL_FACE_ALPHA
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"face alpha must be within [0, 1], got {alpha}")
    SIGNAL_FACE_ALPHA = float(alpha)


def set_face_occluded_factor(factor: float) -> None:
    global SIGNAL_FACE_OCCLUDED_FACTOR
    if not 0.0 <= factor <= 1.0:
        raise ValueError(f"occluded factor must be within [0, 1], got {factor}")
    SIGNAL_FACE_OCCLUDED_FACTOR = float(factor)


def set_occluded_alpha(alpha: float) -> None:
    global OCCLUDED_ALPHA
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"occluded alpha must be within [0, 1], got {alpha}")
    OCCLUDED_ALPHA = float(alpha)


def _boost(rgb, saturation: float = 1.0, value: float = 1.0):
    red, green, blue = (c / 255.0 for c in rgb)
    hue, sat, val = colorsys.rgb_to_hsv(red, green, blue)
    sat = min(1.0, sat * saturation)
    val = min(1.0, val * value)
    red, green, blue = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(round(red * 255)), int(round(green * 255)), int(round(blue * 255)))


def _boost_array(colors: np.ndarray, value: float) -> np.ndarray:
    if value == 1.0:
        return colors
    return np.array([_boost(tuple(int(c) for c in row), 1.0, value) for row in colors],
                    dtype=colors.dtype)


def set_brightness(geometry: float = 1.0, signal: float = 1.0) -> None:
    """Lift the palette brightness while keeping every hue identical."""
    global GEOMETRY_BRIGHTNESS, SIGNAL_BRIGHTNESS, HAND_RGB
    if geometry < 1.0 or signal < 1.0:
        raise ValueError("brightness factors must be >= 1.0")
    GEOMETRY_BRIGHTNESS, SIGNAL_BRIGHTNESS = float(geometry), float(signal)
    HAND_RGB = tuple(
        np.array(_boost(base, saturation=1.0 + 0.40 * (geometry - 1.0),
                        value=geometry), np.uint8)
        for base in GEOMETRY_MARKER_RGB
    )


def set_geometry_stroke(line_width: int, dot_radius: int) -> None:
    global GEOMETRY_LINE_WIDTH, GEOMETRY_DOT_RADIUS
    if line_width < 1 or dot_radius < 1:
        raise ValueError("geometry line width and dot radius must be >= 1")
    GEOMETRY_LINE_WIDTH, GEOMETRY_DOT_RADIUS = int(line_width), int(dot_radius)


def set_geometry_face_alpha(alpha: float) -> None:
    global GEOMETRY_FACE_WASH
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"geometry face alpha must be within [0, 1], got {alpha}")
    GEOMETRY_FACE_WASH = float(alpha)


def set_geometry_occluded_alpha(alpha: float) -> None:
    global GEOMETRY_OCCLUDED_ALPHA
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"geometry occluded alpha must be within [0, 1], got {alpha}")
    GEOMETRY_OCCLUDED_ALPHA = float(alpha)


def set_geometry_lighten(lighten: float) -> None:
    global GEOMETRY_STROKE_LIGHTEN
    if not 0.0 <= lighten <= 1.0:
        raise ValueError(f"geometry stroke lighten must be within [0, 1], got {lighten}")
    GEOMETRY_STROKE_LIGHTEN = float(lighten)


def set_geometry_contour(width: int) -> None:
    global GEOMETRY_CONTOUR_WIDTH
    if width < 0:
        raise ValueError("geometry contour width must be >= 0")
    GEOMETRY_CONTOUR_WIDTH = int(width)


def _binary_colors(values: np.ndarray, positive, negative) -> np.ndarray:
    values = np.asarray(values, np.float32)
    colors = np.empty(values.shape + (3,), np.uint8)
    colors[...] = np.asarray(_boost(SEMANTIC_UNKNOWN_RGB, 1.0, SIGNAL_BRIGHTNESS), np.uint8)
    finite = np.isfinite(values)
    colors[finite & (values >= 0.5)] = np.asarray(_boost(positive, 1.0, SIGNAL_BRIGHTNESS), np.uint8)
    colors[finite & (values < 0.5)] = np.asarray(_boost(negative, 1.0, SIGNAL_BRIGHTNESS), np.uint8)
    return colors


def contact_palette(values):
    if PALETTE_MODE == "reference":
        return _binary_colors(values, CONTACT_POSITIVE_RGB, CONTACT_NEGATIVE_RGB)
    return interpolate_palette(values, CONTACT_STOPS)


def visibility_palette(values):
    if PALETTE_MODE == "reference":
        return _binary_colors(values, VISIBILITY_POSITIVE_RGB, VISIBILITY_NEGATIVE_RGB)
    return interpolate_palette(values, VISIBILITY_STOPS)


def distance_palette(values_m):
    values_m = np.asarray(values_m, np.float32)
    if PALETTE_MODE == "soft":
        return interpolate_palette(values_m / CONTACT_DISTANCE_DISPLAY_MAX_M, DISTANCE_STOPS)
    import cv2

    colors = np.empty(values_m.shape + (3,), np.uint8)
    colors[...] = np.asarray(_boost(SEMANTIC_UNKNOWN_RGB, 1.0, SIGNAL_BRIGHTNESS), np.uint8)
    finite = np.isfinite(values_m)
    normalized = np.clip(values_m[finite] / CONTACT_DISTANCE_DISPLAY_MAX_M, 0.0, 1.0)
    indices = np.rint(normalized * 255.0).astype(np.uint8).reshape(-1, 1)
    turbo = cv2.applyColorMap(indices, cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1]
    colors[finite] = _boost_array(turbo, SIGNAL_BRIGHTNESS)
    return colors


def font(size: int, bold: bool = False, hand: bool = True) -> ImageFont.FreeTypeFont:
    if hand and HAND_FONT.is_file():
        return ImageFont.truetype(str(HAND_FONT), size)
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def unavailable_tile(width: int, height: int, label: str = "unavailable") -> Image.Image:
    tile = Image.new("RGB", (width, height), UNAVAILABLE_FILL)
    draw = ImageDraw.Draw(tile)
    step = 18
    for offset in range(-height, width, step):
        draw.line((offset, 0, offset + height, height), fill=UNAVAILABLE_LINE, width=1)
    text_font = font(max(16, height // 12), hand=True)
    box = draw.textbbox((0, 0), label, font=text_font)
    tw, th = box[2] - box[0], box[3] - box[1]
    pad = 10
    x = (width - tw) // 2
    y = (height - th) // 2
    draw.rectangle((x - pad, y - pad, x + tw + pad, y + th + pad), fill=(250, 250, 248))
    draw.text((x, y), label, fill=MUTED, font=text_font)
    return tile


def gradient_bar(width: int, height: int, stops: np.ndarray) -> Image.Image:
    values = np.linspace(0.0, 1.0, width, dtype=np.float32)
    colors = interpolate_palette(values, stops)
    bar = np.repeat(colors[None, :, :], height, axis=0)
    return Image.fromarray(bar, "RGB")
