"""2D camera-space overlay rendering with the fixed per-frame column contract.

Columns (left to right): raw RGB, baseline geometry (WiLoR, PAD-Hand, EgoForce,
Dyn-HaMR, HaWoR, ReViV4D), EgoFound3R geometry with predicted K and GT K,
GT geometry, then Ego/GT pairs
for visibility, contact and contact distance. Hands are drawn in the marker /
mesh-wireframe style used by the training monitor and mmpose: vertices as dots,
MANO face edges as lines, colored by the signal value. In figures one frame is
one row and selected frames stack vertically; each video frame is a semantic
4-column x 5-row grid and time plays along the video axis.
"""
from __future__ import annotations

from PIL import Image, ImageDraw

import numpy as np
from numba import njit

from . import style
from .inputs import WindowSources, arrays, geometry_path, method_vertices_camera, rgb_path
from .joint_order import joints_in_gt_order
from .raster import derived_distance, derived_visibility, rasterize
from .scene import SKELETON_EDGES

COL_W = 480

COLUMNS = (
    ("rgb", "geometry"),
    ("wilor", "geometry"), ("pad_hand", "geometry"), ("egoforce", "geometry"),
    ("dyn_hamr", "geometry"), ("hawor", "geometry"), ("reviv4d", "geometry"),
    ("ego", "geometry"), ("ego_gt_k", "geometry"), ("gt", "geometry"),
    ("ego", "visibility"), ("gt", "visibility"),
    ("ego", "contact"), ("gt", "contact"),
    ("ego", "distance"), ("gt", "distance"),
)
COLUMN_LABELS = ("RGB", "WiLoR", "PAD-Hand", "EgoForce", "Dyn-HaMR", "HaWoR", "ReViV4D",
                 "EgoFound3R", "EgoFound3R (GT K)", "GT", "Ego visibility", "GT visibility",
                 "Ego contact", "GT contact", "Ego distance", "GT distance")


class Frame2D:
    """Per-frame camera-space context shared by every column cell."""

    def __init__(self, window: WindowSources, index: int, mano, cell_w: int = COL_W):
        self.window = window
        self.index = index
        self.mano = mano
        rgb = Image.open(rgb_path(window, index))
        if rgb.mode != "RGB":          # convert() would copy an already-RGB frame
            rgb = rgb.convert("RGB")
        self.scale = cell_w / rgb.width
        self.width = cell_w
        self.height = int(round(rgb.height * self.scale))
        self.rgb = rgb.resize((self.width, self.height), Image.Resampling.LANCZOS)
        record = window.record
        self.K = (np.array(record["intrinsics"][index], float) * self.scale).astype(float)
        self.K[2, 2] = 1.0
        self.ego_K = self._resolve_ego_K(window, index)
        geometry = arrays(geometry_path(window, index))
        self.object_vertices = geometry["object_vertices"].astype(np.float32)
        self.object_faces = geometry["object_faces"].astype(np.int64)
        self.gt_vertices = geometry["hand_vertices"].astype(float)
        self.gt_valid = geometry["hand_valid"].astype(bool)
        zbuf = np.full((self.height, self.width), np.inf, np.float32)
        owner = np.full((self.height, self.width), -1, np.int16)
        face_id = np.full((self.height, self.width), -1, np.int32)
        if len(self.object_vertices):
            rasterize(self.object_vertices, self.object_faces, self.K, 1.0, self.height,
                      self.width, 0, zbuf, owner, face_id)
        self.object_zbuf = zbuf
        self.object_owner = owner
        self._edges = mano.edges
        self._scene_cache = {}

    def _resolve_ego_K(self, window: WindowSources, index: int):
        """Cell-space intrinsics for the Ego column: predicted K mapped through keep_aspect.

        The model predicts intrinsics in its own resized input frame (for the smoke
        segment 256x176). The renderer draws in source pixels, so the pred K has to be
        undone through the recorded source->model affine before scaling to the cell.
        GT and baseline columns keep using the calibrated K.
        """
        ego = window.methods.get("ego")
        if ego is None or "intrinsics_pred" not in ego:
            return None
        K = np.array(ego["intrinsics_pred"][index], float)
        if "input_affine" in ego:
            affine = np.array(ego["input_affine"], float)
            K = np.diag([1.0 / affine[0, 0], 1.0 / affine[1, 1], 1.0]) @ K
        K = (K * self.scale).astype(float)
        K[2, 2] = 1.0
        return K

    def hand_scene(self, vertices: np.ndarray, valid: np.ndarray, K: np.ndarray | None = None,
                   with_object: bool = True, key=None):
        K = self.K if K is None else K
        if key is None:
            key = id(vertices)
        cache_key = (key, bool(with_object), tuple(np.asarray(valid, bool).ravel()),
                     float(K[0, 0]), float(K[0, 2]), float(K[1, 1]), float(K[1, 2]))
        cached = self._scene_cache.get(cache_key)
        if cached is not None:
            return cached
        if with_object:
            zbuf = self.object_zbuf.copy()
            owner = self.object_owner.copy()
        else:
            zbuf = np.full((self.height, self.width), np.inf, np.float32)
            owner = np.full((self.height, self.width), -1, np.int16)
        face_id = np.full((self.height, self.width), -1, np.int32)
        for side in range(2):
            if valid[side] and np.isfinite(vertices[side]).all():
                rasterize(vertices[side].astype(np.float32), self.mano.faces, K, 1.0,
                          self.height, self.width, side + 1, zbuf, owner, face_id)
        self._scene_cache[cache_key] = (zbuf, owner, face_id)
        return zbuf, owner, face_id


def project_side(frame: Frame2D, vertices: np.ndarray, K: np.ndarray | None = None):
    projected = vertices @ (frame.K if K is None else K).T
    depth = vertices[:, 2]
    uv = projected[:, :2] / np.where(depth[:, None] > 1e-6, projected[:, 2:3], np.nan)
    px = uv[:, 0]
    py = uv[:, 1]
    ok = np.isfinite(px) & np.isfinite(py) & (depth > 1e-6)
    return px, py, depth, ok


def side_visibility(frame: Frame2D, vertices: np.ndarray, valid: np.ndarray, side: int,
                    scene=None, K: np.ndarray | None = None, with_object: bool = True):
    scene = scene if scene is not None else frame.hand_scene(vertices, valid, K=K,
                                                             with_object=with_object)
    return depth_visibility(frame, vertices[side], scene, side, K)


def depth_visibility(frame: Frame2D, points: np.ndarray, scene, side: int,
                     K: np.ndarray | None = None):
    """Per-point visibility from the scene z-buffer (works for vertices and joints)."""
    zbuf, owner = scene[0], scene[1]
    px, py, depth, ok = project_side(frame, points, K)
    height, width = owner.shape
    ix = np.rint(px).astype(np.int64)
    iy = np.rint(py).astype(np.int64)
    inside = ok & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    visible = np.zeros(len(depth), bool)
    ids = np.flatnonzero(inside)
    tolerance = np.maximum(0.002, 0.004 * depth[ids])
    visible[ids] = ((owner[iy[ids], ix[ids]] == side + 1)
                    & (np.abs(zbuf[iy[ids], ix[ids]] - depth[ids]) <= tolerance))
    return px, py, ok, visible


def joint_visibility(frame: Frame2D, points: np.ndarray, scene, side: int,
                     K: np.ndarray | None = None):
    """Per-joint visibility: dimmed only when *other* geometry sits in front.

    A skeletal joint lives inside the hand volume, so the nearest rendered surface
    at its pixel is normally the hand's own front face.  Applying the mesh-vertex
    test (``depth_visibility``) here marks most joints "occluded" - and since the
    hand is drawn as a *transparent wireframe*, the viewer does not see anything
    covering them; the only visible effect is that bones and joints lose opacity
    and start blending into the mesh underneath.  Joints are therefore dimmed only
    when a different surface (the object or the other hand) is in front of them.
    """
    zbuf, owner = scene[0], scene[1]
    px, py, depth, ok = project_side(frame, points, K)
    height, width = owner.shape
    ix = np.rint(px).astype(np.int64)
    iy = np.rint(py).astype(np.int64)
    inside = ok & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    visible = np.ones(len(depth), bool)
    ids = np.flatnonzero(inside)
    owner_ids = owner[iy[ids], ix[ids]]
    tolerance = np.maximum(0.002, 0.004 * depth[ids])
    covered = ((owner_ids >= 0) & (owner_ids != side + 1)
               & (zbuf[iy[ids], ix[ids]] + tolerance < depth[ids]))
    visible[ids] = ~covered
    return px, py, ok, visible


def wireframe_cell(frame: Frame2D, vertices: np.ndarray, valid: np.ndarray, vertex_colors,
                   occlusion: bool = True,
                   dot: int = 1, base: Image.Image | None = None,
                   K: np.ndarray | None = None,
                   with_object: bool = True, width: int = 1,
                   key=None,
                   occluded_alpha: int | None = None) -> Image.Image:
    """Vertices as dots plus MANO face edges as lines, coloured per vertex.

    Colour always carries the signal value; opacity always carries occlusion.
    Everything is painted back-to-front (painter's algorithm) into one RGBA layer,
    visible geometry opaque and occluded geometry at ``style.OCCLUDED_ALPHA``, then
    composited once, so the dots and lines themselves show the front/back ordering
    without changing any hue.

    An edge whose two ends carry different values is split at its midpoint so each
    half keeps its own endpoint's colour.  Blending the two endpoint colours into a
    single stroke turns a binary boundary (visibility, contact: green/red,
    red/blue) into a muddy average that no longer reads as a class change; the
    split keeps the boundary crisp, which is also how the mmpose reference draws
    "edges take their endpoint colour".
    """
    # alpha_composite below builds a new image, so neither branch has to copy:
    # ``base`` is only read, and the RGB frame is never written to.
    image = frame.rgb if base is None else (base if base.mode == "RGB" else base.convert("RGB"))
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scene = (frame.hand_scene(vertices, valid, K=K, with_object=with_object, key=key)
             if occlusion else None)
    if occluded_alpha is None:
        occluded_alpha = int(round(255 * style.OCCLUDED_ALPHA))
    tables = [[tuple(int(c) for c in row) for row in vertex_colors[side]] for side in range(2)]
    segments, dots, projected = [], [], {}
    for side in range(2):
        if not valid[side]:
            continue
        px, py, depth, ok = project_side(frame, vertices[side], K)
        projected[side] = (px, py)
        if occlusion:
            visible = depth_visibility(frame, vertices[side], scene, side, K)[3]
        else:
            visible = np.ones(len(px), bool)
        for a, b in frame._edges:
            if not (ok[a] and ok[b]):
                continue
            is_visible = bool(visible[a] and visible[b])
            segments.append((0.5 * (depth[a] + depth[b]), side, a, b, is_visible))
        for j in range(len(px)):
            if not ok[j]:
                continue
            dots.append((depth[j], side, j, bool(visible[j])))
    # Painter's algorithm: farthest first, nearest last.
    segments.sort(key=lambda item: -item[0])
    dots.sort(key=lambda item: -item[0])
    for _, side, a, b, is_visible in segments:
        px, py = projected[side]
        alpha = 255 if is_visible else occluded_alpha
        color_a = tables[side][a]
        color_b = tables[side][b]
        if color_a == color_b:
            draw.line((px[a], py[a], px[b], py[b]), fill=(*color_a, alpha), width=width)
        else:
            midx = 0.5 * (px[a] + px[b])
            midy = 0.5 * (py[a] + py[b])
            draw.line((px[a], py[a], midx, midy), fill=(*color_a, alpha), width=width)
            draw.line((midx, midy, px[b], py[b]), fill=(*color_b, alpha), width=width)
    for _, side, j, is_visible in dots:
        px, py = projected[side]
        color = tables[side][j]
        fill = (*color, 255) if is_visible else (*color, occluded_alpha)
        draw.ellipse((px[j] - dot, py[j] - dot, px[j] + dot, py[j] + dot), fill=fill)
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def geometry_cell(frame: Frame2D, vertices: np.ndarray, valid: np.ndarray,
                  joints: np.ndarray | None = None, K: np.ndarray | None = None,
                  key=None) -> Image.Image:
    """Mesh cell: the same point-and-edge rendering as the signal columns.

    The geometry columns run through ``signal_cell`` with a constant per-side
    colour instead of a per-vertex signal value, so the mesh is drawn exactly like
    the visibility / contact / distance columns (radius-1 nodes, 1 px edges, faint
    face wash, occlusion only changes opacity).  The joint skeleton is then laid on
    top.  The object is ignored - a hand cell shows the hand mesh, so the only
    depth cue is the hand's own z-buffer (``with_object=False``).
    """
    values = np.zeros((2, vertices.shape[1]), np.float32)
    image = signal_cell(frame, vertices, valid, values, geometry_palette,
                        occlusion=True, K=K, with_object=False, key=key,
                        dot=style.GEOMETRY_DOT_RADIUS, width=style.GEOMETRY_LINE_WIDTH,
                        face_alpha=style.GEOMETRY_FACE_WASH,
                        occluded_alpha=int(round(255 * style.GEOMETRY_OCCLUDED_ALPHA)),
                        stroke_palette=geometry_stroke_palette)
    if style.GEOMETRY_CONTOUR_WIDTH > 0:
        # The contour is a new element, so (unlike the tuned line width and dot
        # radius) it is scaled with the cell: --geometry-contour is the width at
        # the 900 px figure cell, and scales down for the 360 px video cells.
        contour = max(1, int(round(style.GEOMETRY_CONTOUR_WIDTH * frame.width / 900.0)))
        image = draw_contour(frame, image,
                             frame.hand_scene(vertices, valid, K=K, with_object=False, key=key),
                             valid, contour)
    if joints is not None:
        image = draw_joints(frame, image, joints, valid, K=K,
                            scene=frame.hand_scene(vertices, valid, K=K, with_object=False, key=key))
    return image


def geometry_palette(values: np.ndarray) -> np.ndarray:
    """Constant per-side hand colour, following the signal palette contract.

    Used for the face wash, so the surface keeps the plain side colour.
    """
    colors = np.empty(np.shape(values) + (3,), np.uint8)
    for side in range(2):
        colors[side] = style.HAND_RGB[side]
    return colors


def geometry_stroke_palette(values: np.ndarray) -> np.ndarray:
    """Same constant colours, mixed toward white for the dots and edges.

    Keeps the hue while lifting luminance, so the strokes stay brighter than the
    face wash of the same hand (the marker colours are already at value = 1).
    """
    colors = np.empty(np.shape(values) + (3,), np.uint8)
    lighten = style.GEOMETRY_STROKE_LIGHTEN
    for side in range(2):
        base = np.asarray(style.HAND_RGB[side], np.float32)
        colors[side] = np.rint(base * (1.0 - lighten) + 255.0 * lighten).astype(np.uint8)
    return colors


def draw_contour(frame: Frame2D, base: Image.Image, scene, valid: np.ndarray,
                 width: int) -> Image.Image:
    """Trace the outer silhouette of each drawn hand: dark rim + bright side colour.

    The contour is taken from the hand's own z-buffer mask, so it follows the
    projected mesh exactly (no boundary edges exist on a closed MANO mesh).
    """
    import cv2

    image = base.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    owner = scene[1]
    kernel = np.ones((3, 3), np.uint8)
    for side in range(2):
        if not valid[side]:
            continue
        mask = np.ascontiguousarray((owner == side + 1).astype(np.uint8))
        if not mask.any():
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        color = tuple(int(c) for c in style.HAND_RGB[side])
        for contour in contours:
            if len(contour) < 8:
                continue
            points = [tuple(int(v) for v in point[0]) for point in contour]
            points.append(points[0])
            draw.line(points, fill=style.JOINT_OUTLINE, width=width + 2, joint="curve")
            draw.line(points, fill=color, width=width, joint="curve")
    return image


def draw_joints(frame: Frame2D, base: Image.Image, joints: np.ndarray,
                valid: np.ndarray, K: np.ndarray | None = None,
                scene=None) -> Image.Image:
    """Overlay the 21-joint skeleton on top of a hand cell (mmpose / monitor style).

    Bones and joints use the reference per-side colour (yellow left, magenta right)
    with a dark rim, drawn on a layer above the mesh.  The skeleton is an
    annotation layer: it stays opaque over the hand's own wireframe and is dimmed
    only where the object or the other hand really covers it (see
    ``joint_visibility``), painted back-to-front by depth.
    """
    image = base.convert("RGB").copy()
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    joint_outline = style.JOINT_OUTLINE
    occluded_alpha = int(round(255 * style.OCCLUDED_ALPHA))
    for side in range(2):
        if not valid[side]:
            continue
        if scene is not None:
            px, py, ok, visible = joint_visibility(frame, joints[side], scene, side, K)
        else:
            px, py, _, ok = project_side(frame, joints[side], K)
            visible = np.ones(len(px), bool)
        depth = np.asarray(joints[side], float)[:, 2]
        joint_rgb = tuple(int(c) for c in style.JOINT_RGB_SIDES[side])
        bones = sorted(((0.5 * (depth[a] + depth[b]), a, b) for a, b in SKELETON_EDGES
                        if ok[a] and ok[b]), key=lambda item: -item[0])
        for _, a, b in bones:
            alpha = 255 if (visible[a] and visible[b]) else occluded_alpha
            # The skeleton sits on top of the mesh: dark rim + per-side reference
            # joint colour (yellow left, magenta right) so it stays legible over
            # the brightened wireframe.
            draw.line((px[a], py[a], px[b], py[b]), fill=(*joint_outline, alpha), width=4)
            draw.line((px[a], py[a], px[b], py[b]), fill=(*joint_rgb, alpha), width=2)
        nodes = sorted(((depth[j], j) for j in range(len(px)) if ok[j]), key=lambda item: -item[0])
        for _, j in nodes:
            alpha = 255 if visible[j] else occluded_alpha
            draw.ellipse((px[j] - 3, py[j] - 3, px[j] + 3, py[j] + 3),
                         fill=(*joint_outline, alpha))
            draw.ellipse((px[j] - 2, py[j] - 2, px[j] + 2, py[j] + 2), fill=(*joint_rgb, alpha))
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def signal_wireframe(frame: Frame2D, vertices: np.ndarray, valid: np.ndarray,
                     values: np.ndarray, palette, occlusion: bool = True,
                     K: np.ndarray | None = None,
                     with_object: bool = True,
                     base: Image.Image | None = None, key=None,
                     dot: int = 1, width: int = 1,
                     occluded_alpha: int | None = None) -> Image.Image:
    """Colour every vertex by its signal value; occlusion only changes opacity."""
    return wireframe_cell(frame, vertices, valid, palette(values), occlusion=occlusion,
                          K=K, with_object=with_object, base=base, key=key,
                          dot=dot, width=width, occluded_alpha=occluded_alpha)


@njit(cache=True)
def fill_occluded_faces(canvas, xy, faces, face_ids, colors, alpha, one_minus, mask):
    """Alpha-composite the listed triangles into a float32 RGB canvas.

    Same maths as the previous per-face numpy version (edge function on pixel
    centres, clipped to the triangle's own bounding box and to ``mask``), but the
    barycentric loop runs compiled: this pass used to dominate the 2D render.
    Everything is float32 so the result matches the numpy path bit for bit.
    """
    height = canvas.shape[0]
    width = canvas.shape[1]
    for k in range(face_ids.shape[0]):
        fi = face_ids[k]
        i0 = faces[fi, 0]
        i1 = faces[fi, 1]
        i2 = faces[fi, 2]
        xa = xy[i0, 0]
        ya = xy[i0, 1]
        xb = xy[i1, 0]
        yb = xy[i1, 1]
        xc = xy[i2, 0]
        yc = xy[i2, 1]
        x0 = int(np.floor(min(min(xa, xb), xc)))
        x1 = int(np.ceil(max(max(xa, xb), xc)))
        y0 = int(np.floor(min(min(ya, yb), yc)))
        y1 = int(np.ceil(max(max(ya, yb), yc)))
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > width - 1:
            x1 = width - 1
        if y1 > height - 1:
            y1 = height - 1
        if x1 < x0 or y1 < y0:
            continue
        denominator = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
        if abs(denominator) < 1e-9:
            continue
        for py in range(y0, y1 + 1):
            yy = py + 0.5
            for px in range(x0, x1 + 1):
                if not mask[py, px]:
                    continue
                xx = px + 0.5
                w0 = ((yb - yc) * (xx - xc) + (xc - xb) * (yy - yc)) / denominator
                w1 = ((yc - ya) * (xx - xc) + (xa - xc) * (yy - yc)) / denominator
                w2 = 1.0 - w0 - w1
                if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                    continue
                canvas[py, px, 0] = canvas[py, px, 0] * one_minus + colors[fi, 0] * alpha
                canvas[py, px, 1] = canvas[py, px, 1] * one_minus + colors[fi, 1] * alpha
                canvas[py, px, 2] = canvas[py, px, 2] * one_minus + colors[fi, 2] * alpha


def face_wash(frame: Frame2D, vertices: np.ndarray, valid: np.ndarray, rgb: np.ndarray,
              K: np.ndarray | None = None, with_object: bool = True,
              alpha: float = 0.30, occluded_factor: float = 0.35,
              key=None) -> Image.Image:
    """Faint per-face wash from per-vertex colours (2, 778, 3).

    Alpha follows the **occlusion state**, not the raw depth.  A triangle is
    "visible" when it owns at least one pixel of the scene z-buffer, i.e. it is the
    nearest surface somewhere; every other triangle of that hand is occluded.

    1. all occluded triangles are laid down first at ``alpha * occluded_factor``,
       clipped to this hand's own silhouette, so the hidden side reads as a faint
       layer behind the surface;
    2. the visible nearest surface is then composited on top at the full ``alpha``.

    Painting in that back-to-front order keeps the front signal colours crisp while
    still showing which parts of the hand sit behind the surface.
    """
    base = np.asarray(frame.rgb.convert("RGB")).astype(np.float32)
    zbuf, owner, face_id = frame.hand_scene(vertices, valid, K=K, with_object=with_object, key=key)
    rgb = np.asarray(rgb, np.float32)
    faces = frame.mano.faces
    face_count = len(faces)
    occluded_alpha = np.float32(alpha * occluded_factor)
    occluded_one_minus = np.float32(1.0 - alpha * occluded_factor)
    for side in range(2):
        if not valid[side]:
            continue
        mask = owner == side + 1
        if not mask.any():
            continue
        side_vertices = np.asarray(vertices[side], np.float32)
        px, py, _, ok = project_side(frame, side_vertices, K)
        xy = np.stack([px, py], axis=1)
        colors = rgb[side][faces].mean(axis=1)
        owned = np.bincount(face_id[mask], minlength=face_count)
        hidden = np.flatnonzero(owned == 0)
        if len(hidden):
            projectable = ok[faces[hidden, 0]] & ok[faces[hidden, 1]] & ok[faces[hidden, 2]]
            hidden = hidden[projectable]
        if len(hidden):
            fill_occluded_faces(base, xy, faces, hidden, colors, occluded_alpha,
                                occluded_one_minus, mask)
        ys, xs = np.nonzero(mask)
        ids = face_id[ys, xs]
        base[ys, xs] = base[ys, xs] * (1.0 - alpha) + colors[ids] * alpha
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")


def signal_face_fill(frame: Frame2D, vertices: np.ndarray, valid: np.ndarray,
                     values: np.ndarray, palette, K: np.ndarray | None = None,
                     with_object: bool = True, alpha: float = 0.30,
                     occluded_factor: float = 0.35, key=None) -> Image.Image:
    """Faint signal wash: per-face colour = mean of the three vertex signal colours."""
    return face_wash(frame, vertices, valid, palette(values), K=K,
                     with_object=with_object, alpha=alpha, occluded_factor=occluded_factor,
                     key=key)


def skeleton_cell(frame: Frame2D, joints: np.ndarray, valid: np.ndarray,
                  method: str, K: np.ndarray | None = None) -> Image.Image:
    """Joint-only geometry cell (ReViV4D): outlined bones plus highlighted joints.

    Bones are drawn as a dark rim with a saturated method-colour core, matching the
    yellow joint markers, so the skeleton keeps the same visual weight as the mesh
    columns on both the dark tabletop and the bright paper.
    """
    base = frame.rgb.copy()
    draw = ImageDraw.Draw(base)
    rgb = style.METHOD_COLORS.get(method, (120, 120, 120))
    for side in range(2):
        if not valid[side]:
            continue
        px, py, depth, ok = project_side(frame, joints[side], K)
        joint_rgb = tuple(int(c) for c in style.JOINT_RGB_SIDES[side])
        bones = sorted(((0.5 * (depth[a] + depth[b]), a, b) for a, b in SKELETON_EDGES
                        if ok[a] and ok[b]), key=lambda item: -item[0])
        for _, a, b in bones:
            draw.line((px[a], py[a], px[b], py[b]), fill=style.JOINT_OUTLINE, width=4)
            draw.line((px[a], py[a], px[b], py[b]), fill=rgb, width=2)
        nodes = sorted(((depth[j], j) for j in range(len(px)) if ok[j]), key=lambda item: -item[0])
        for _, j in nodes:
            draw.ellipse((px[j] - 3.5, py[j] - 3.5, px[j] + 3.5, py[j] + 3.5),
                         fill=style.JOINT_OUTLINE)
            draw.ellipse((px[j] - 2.5, py[j] - 2.5, px[j] + 2.5, py[j] + 2.5),
                         fill=joint_rgb)
    return base


def joints_camera(data: dict) -> np.ndarray:
    if "hand_joints_camera" in data:
        return data["hand_joints_camera"].astype(float)
    pose = data["camera_c2w"].astype(float)
    return np.einsum("tji,tsvj->tsvi", pose[:, :3, :3],
                     data["hand_joints_world"].astype(float) - pose[:, None, None, :3, 3])


def window_vertices(window: WindowSources, method: str, mano) -> np.ndarray:
    """(T, 2, 778, 3) camera-space vertices for one method, memoised per window.

    ``method_vertices_camera`` converts and copies the whole clip, so calling it
    once per column re-does the same work up to six times per frame; the cached
    array also keeps ``id(vertices)`` stable for the scene cache.
    """
    store = window.cache.setdefault("vertices", {})
    if method not in store:
        store[method] = method_vertices_camera(window.methods[method], mano)
    return store[method]


def window_joints(window: WindowSources, method: str):
    """(T, 2, 21, 3) camera-space joints for one method, or None when absent."""
    store = window.cache.setdefault("joints", {})
    if method not in store:
        data = window.methods[method]
        store[method] = (joints_camera(data)
                         if ("hand_joints_camera" in data or "hand_joints_world" in data)
                         else None)
    return store[method]


def method_geometry(frame: Frame2D, method: str, index: int):
    window = frame.window
    source_method = "ego" if method == "ego_gt_k" else method
    data = window.methods.get(source_method)
    if data is None:
        return None
    if method == "gt":
        joints = window_joints(window, "gt")
        joints = joints[index] if joints is not None else None
        return frame.gt_vertices, frame.gt_valid, joints
    try:
        vertices = window_vertices(window, source_method, frame.mano)[index]
    except KeyError:
        joints = window_joints(window, source_method)
        if joints is None:
            return None
        joints = joints_in_gt_order(method, joints[index])
        return ("skeleton", joints, data["hand_valid"][index].astype(bool))
    valid = data["hand_valid"][index].astype(bool) & np.isfinite(vertices).all(axis=(1, 2))
    joints = window_joints(window, source_method)
    joints = joints[index] if joints is not None else None
    if joints is not None:
        joints = joints_in_gt_order(method, joints)
    return vertices, valid, joints


def signal_cell(frame: Frame2D, vertices: np.ndarray, valid: np.ndarray,
                values: np.ndarray, palette, occlusion: bool = True,
                K: np.ndarray | None = None,
                with_object: bool = True, key=None, dot: int = 1, width: int = 1,
                face_alpha: float | None = None,
                occluded_alpha: int | None = None,
                stroke_palette=None) -> Image.Image:
    """Signal column: points + edges, optionally over a faint face wash.

    ``wireframe`` draws only points and edges.  ``face`` adds a low-alpha per-face
    wash underneath so the front/back occlusion is readable, but keeps the points
    and edges on top at full strength as the visual focus.

    ``stroke_palette`` overrides the palette used for the dots and edges only, so
    a column can draw strokes that are brighter than its own face wash.
    """
    face_alpha = style.SIGNAL_FACE_ALPHA if face_alpha is None else face_alpha
    strokes = palette if stroke_palette is None else stroke_palette
    if style.SIGNAL_STYLE == "wireframe":
        return signal_wireframe(frame, vertices, valid, values, strokes,
                                occlusion=occlusion, K=K, with_object=with_object, key=key,
                                dot=dot, width=width, occluded_alpha=occluded_alpha)
    base = signal_face_fill(frame, vertices, valid, values, palette, K=K,
                            with_object=with_object, alpha=face_alpha,
                            occluded_factor=style.SIGNAL_FACE_OCCLUDED_FACTOR, key=key)
    return signal_wireframe(frame, vertices, valid, values, strokes, occlusion=occlusion,
                            K=K, with_object=with_object, base=base, key=key,
                            dot=dot, width=width, occluded_alpha=occluded_alpha)


def column_cell(frame: Frame2D, method: str, signal: str, index: int) -> Image.Image:
    window = frame.window
    if method == "rgb":
        return frame.rgb.copy()
    K, with_object = frame.K, True
    if method in {"ego", "ego_gt_k"}:
        # The Ego hand lives in the model's predicted camera frame: project with the
        # requested K, and let only the hand itself act as a depth occluder (object
        # geometry is in the calibrated frame).
        if method == "ego":
            K = frame.ego_K if frame.ego_K is not None else frame.K
        with_object = False
    key = (method, index)
    if signal == "geometry":
        packed = method_geometry(frame, method, index)
        if packed is None:
            return style.unavailable_tile(frame.width, frame.height)
        if isinstance(packed[0], str):
            _, joints, valid = packed
            return skeleton_cell(frame, joints, valid, method, K=K)
        vertices, valid, joints = packed
        return geometry_cell(frame, vertices, valid, joints, K=K, key=key)
    if method == "gt":
        vertices, valid = frame.gt_vertices, frame.gt_valid
        zbuf, owner = frame.hand_scene(vertices, valid, key=key)[:2]
        if signal == "visibility":
            per_vertex = derived_visibility(vertices, frame.K, 1.0, zbuf, owner)
            return signal_cell(frame, vertices, valid, per_vertex,
                               style.visibility_palette, occlusion=False, key=key)
        if signal == "contact":
            gt = window.methods.get("gt")
            if gt is None or "marker_contact_target" not in gt:
                return style.unavailable_tile(frame.width, frame.height)
            target = gt["marker_contact_target"][index].astype(np.float32)
            mask = gt["marker_contact_mask"][index].astype(bool)
            per_vertex = frame.mano.interpolate_scalar(target * mask)
            return signal_cell(frame, vertices, valid, per_vertex, style.contact_palette,
                               key=key)
        distance = derived_distance(vertices, frame.object_vertices)
        per_vertex = np.where(np.isfinite(distance), distance,
                              style.CONTACT_DISTANCE_DISPLAY_MAX_M)
        return signal_cell(frame, vertices, valid, per_vertex, style.distance_palette,
                           key=key)
    ego = window.methods.get("ego")
    if ego is None:
        return style.unavailable_tile(frame.width, frame.height)
    vertices = window_vertices(window, "ego", frame.mano)[index]
    valid = ego["hand_valid"][index].astype(bool) & np.isfinite(vertices).all(axis=(1, 2))
    if signal == "visibility":
        if "vertex_visibility_probability" in ego:
            values = ego["vertex_visibility_probability"][index].astype(np.float32)
            return signal_cell(frame, vertices, valid, values, style.visibility_palette,
                               occlusion=False, K=K, with_object=False, key=key)
        ego_full = window.methods.get("ego_full") or ego
        if "marker_visibility" not in ego_full:
            return style.unavailable_tile(frame.width, frame.height)
        per_vertex = frame.mano.interpolate_scalar(ego_full["marker_visibility"][index].astype(np.float32))
        return signal_cell(frame, vertices, valid, per_vertex,
                           style.visibility_palette, occlusion=False, K=K, with_object=False,
                           key=key)
    if signal == "contact":
        if "vertex_contact_probability" in ego:
            values = ego["vertex_contact_probability"][index].astype(np.float32)
            return signal_cell(frame, vertices, valid, values, style.contact_palette,
                               K=K, with_object=False, key=key)
        contact_npz = window.baselines.get("ego_contact")
        if contact_npz is None:
            return style.unavailable_tile(frame.width, frame.height)
        per_vertex = contact_npz["vertex_contact_probability"][index].astype(np.float32)
        return signal_cell(frame, vertices, valid, per_vertex, style.contact_palette,
                           K=K, with_object=False, key=key)
    if "vertex_contact_distance" in ego:
        distance = ego["vertex_contact_distance"][index].astype(np.float32)
        per_vertex = np.where(np.isfinite(distance), distance,
                              style.CONTACT_DISTANCE_DISPLAY_MAX_M)
        return signal_cell(frame, vertices, valid, per_vertex, style.distance_palette,
                           K=K, with_object=False, key=key)
    contact_npz = window.baselines.get("ego_contact")
    if contact_npz is None:
        return style.unavailable_tile(frame.width, frame.height)
    distance = contact_npz["vertex_contact_distance"][index].astype(np.float32)
    mask = contact_npz.get("vertex_contact_distance_mask")
    if mask is not None:
        distance = np.where(mask[index].astype(bool), distance, style.CONTACT_DISTANCE_DISPLAY_MAX_M)
    per_vertex = np.where(np.isfinite(distance), distance, style.CONTACT_DISTANCE_DISPLAY_MAX_M)
    return signal_cell(frame, vertices, valid, per_vertex, style.distance_palette,
                       K=K, with_object=False, key=key)
