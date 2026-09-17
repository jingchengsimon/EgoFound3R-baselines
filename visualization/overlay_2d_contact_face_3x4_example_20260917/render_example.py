#!/usr/bin/env python3
"""Render one verified 3x4 face-level 2D overlay example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFont


DATASET = "h2o"
DATASET_RANK = 17
CACHE_ID = "abc07caace9823faa3ffe7ce"
FRAME_INDEX = 52
FRAME_ID = "000208"
COL_W, ROW_H = 480, 500
HEADER_H, FOOTER_H = 82, 112
LABEL_H, STATUS_H = 38, 34
IMAGE_H = ROW_H - LABEL_H - STATUS_H
ALPHA = 122


REMOTE_SOURCES = {
    "record": f"/mnt/workspace/sjc/eval_artifacts/prep_h2o_60f_strict_20260822T030835Z_5001_g1/window_inputs/h2o/{CACHE_ID}/window_input.json",
    "rgb": f"/mnt/workspace/sjc/eval_artifacts/prep_h2o_60f_strict_20260822T030835Z_5001_g1/window_inputs/h2o/{CACHE_ID}/rgb/052_000208.png",
    "geometry": f"/mnt/workspace/sjc/eval_artifacts/prep_h2o_60f_strict_20260822T030835Z_5001_g1/window_inputs/h2o/{CACHE_ID}/geometry/052_000208.npz",
    "gt": f"oss://quic-pre-train/ego/eval_artifacts/formal_standard_reports_10methods_20260824/h2o/gt_cache/h2o/{CACHE_ID}.npz",
    "gt_contact": f"/mnt/cpfs/sjc/eval_artifacts/result3_completion_20260910_v1/vertex_contact_distance_gt/h2o/{CACHE_ID}.npz",
    "ego_prediction": f"oss://quic-pre-train/ego/eval_artifacts/p95_table_freeze_20260915/recompute_dependency/base3e_traincrop_512_stride5_1078f15_20260908/h2o/h2o/egofound3r/formal/{CACHE_ID}/predictions.npz",
    "ego_contact": f"/mnt/cpfs/sjc/eval_artifacts/result3_completion_20260910_v1/ego_vertex_contact_distance_8095_retry3_20260911/egofound3r_stride5/shards/h2o/h2o/{CACHE_ID}.npz",
    "s2contact": f"/mnt/cpfs/sjc/eval_artifacts/result3_completion_20260910_v1/s2contact_vertex_8095_retry7_20260911/predictions/h2o/0/s2contact/formal/{CACHE_ID}/predictions.npz",
    "contactopt": f"/mnt/cpfs/sjc/eval_artifacts/result3_completion_20260910_v1/contactopt_vertex_8095_retry3_20260911/predictions/h2o/0/contactopt/formal/{CACHE_ID}/predictions.npz",
    "interactvlm": f"oss://quic-pre-train/ego/eval_artifacts/p95_table_freeze_20260915/result3_direct/result3_contact_distance_20260912_v1/interactvlm100/windows/h2o/{CACHE_ID}.npz",
    "mapping": "/mnt/workspace/sjc/DATA/runtime_worktrees/EgoFound3R_pair_new_3e533881_20260907/egohandmetric_prompt/data/mano_upsampling/mano_195_to_778.npz",
}


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


F20, F16, F14, F12 = (get_font(size) for size in (20, 16, 14, 12))


@njit(cache=True)
def rasterize(vertices, faces, K, scale, height, width, owner_code, zbuf, owner, face_id):
    for fi in range(faces.shape[0]):
        i0, i1, i2 = faces[fi]
        x0, y0, z0 = vertices[i0]
        x1, y1, z1 = vertices[i1]
        x2, y2, z2 = vertices[i2]
        if z0 <= 1e-6 or z1 <= 1e-6 or z2 <= 1e-6:
            continue
        u0 = (K[0, 0] * x0 / z0 + K[0, 2]) * scale
        v0 = (K[1, 1] * y0 / z0 + K[1, 2]) * scale
        u1 = (K[0, 0] * x1 / z1 + K[0, 2]) * scale
        v1 = (K[1, 1] * y1 / z1 + K[1, 2]) * scale
        u2 = (K[0, 0] * x2 / z2 + K[0, 2]) * scale
        v2 = (K[1, 1] * y2 / z2 + K[1, 2]) * scale
        xmin = max(0, int(np.floor(min(u0, u1, u2))))
        xmax = min(width - 1, int(np.ceil(max(u0, u1, u2))))
        ymin = max(0, int(np.floor(min(v0, v1, v2))))
        ymax = min(height - 1, int(np.ceil(max(v0, v1, v2))))
        if xmin > xmax or ymin > ymax:
            continue
        den = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        if abs(den) < 1e-8:
            continue
        for py in range(ymin, ymax + 1):
            yy = py + 0.5
            for px in range(xmin, xmax + 1):
                xx = px + 0.5
                w0 = ((v1 - v2) * (xx - u2) + (u2 - u1) * (yy - v2)) / den
                w1 = ((v2 - v0) * (xx - u2) + (u0 - u2) * (yy - v2)) / den
                w2 = 1.0 - w0 - w1
                if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                    continue
                invz = w0 / z0 + w1 / z1 + w2 / z2
                if invz <= 0:
                    continue
                depth = 1.0 / invz
                if depth < zbuf[py, px]:
                    zbuf[py, px] = depth
                    owner[py, px] = owner_code
                    face_id[py, px] = fi


def interpolate_palette(values: np.ndarray, stops: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0, 1)
    position = values * (len(stops) - 1)
    low = np.minimum(position.astype(np.int32), len(stops) - 2)
    fraction = (position - low)[..., None]
    return ((1 - fraction) * stops[low] + fraction * stops[low + 1]).astype(np.uint8)


CONTACT_STOPS = np.array([[35, 199, 216], [251, 192, 45], [240, 59, 59]], np.float32)
VISIBILITY_STOPS = np.array([[229, 57, 53], [251, 192, 45], [105, 190, 85], [0, 166, 81]], np.float32)
DISTANCE_STOPS = np.array([[230, 45, 38], [250, 194, 46], [45, 205, 220], [37, 99, 235]], np.float32)
HAND_COLORS = (np.array([226, 86, 150], np.uint8), np.array([55, 132, 235], np.uint8))


def contact_palette(values: np.ndarray) -> np.ndarray:
    return interpolate_palette(values, CONTACT_STOPS)


def visibility_palette(values: np.ndarray) -> np.ndarray:
    return interpolate_palette(values, VISIBILITY_STOPS)


def distance_palette(values_m: np.ndarray) -> np.ndarray:
    return interpolate_palette(np.asarray(values_m) / 0.05, DISTANCE_STOPS)


def reconstruct_ego_vertices(markers: np.ndarray, mapping: dict[str, np.ndarray]) -> np.ndarray:
    neighbors = mapping["neighbor_indices"]
    weights = mapping["geometry_weights"]
    anchors = mapping["source_vertex_ids"]
    center = markers.mean(axis=1, keepdims=True)
    vertices = center + (
        (markers[:, neighbors, :] - center[:, :, None, :]) * weights[None, :, :, None]
    ).sum(axis=2)
    vertices[:, anchors] = markers
    return vertices.astype(np.float32)


def interpolate_marker_scalar(values: np.ndarray, mapping: dict[str, np.ndarray]) -> np.ndarray:
    weights = np.maximum(mapping["geometry_weights"], 0)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    result = (values[:, mapping["neighbor_indices"]] * weights[None, :, :]).sum(axis=2)
    result[:, mapping["source_vertex_ids"]] = values
    return result.astype(np.float32)


def rgb_tile(rgb: Image.Image, title: str, status: str):
    tile = Image.new("RGB", (COL_W, ROW_H), (34, 39, 44))
    draw = ImageDraw.Draw(tile)
    draw.text((8, 9), title, fill=(239, 241, 243), font=F14)
    scale = min(COL_W / rgb.width, IMAGE_H / rgb.height)
    size = (round(rgb.width * scale), round(rgb.height * scale))
    resized = rgb.resize(size, Image.Resampling.LANCZOS)
    x0 = (COL_W - size[0]) // 2
    y0 = LABEL_H + (IMAGE_H - size[1]) // 2
    tile.paste(resized, (x0, y0))
    draw.rectangle((0, 0, COL_W - 1, ROW_H - 1), outline=(75, 82, 89))
    draw.text((8, ROW_H - 25), status, fill=(196, 205, 214), font=F12)
    return tile, (x0, y0, size[0], size[1]), scale


def make_scene(object_vertices, object_faces, hand_vertices, hand_valid, faces, K, height, width, scale):
    zbuf = np.full((height, width), np.inf, np.float32)
    owner = np.full((height, width), -1, np.int16)
    face_id = np.full((height, width), -1, np.int32)
    rasterize(object_vertices.astype(np.float32), object_faces.astype(np.int64), K, scale,
              height, width, 0, zbuf, owner, face_id)
    for side in range(2):
        if hand_valid[side]:
            rasterize(hand_vertices[side].astype(np.float32), faces, K, scale,
                      height, width, side + 1, zbuf, owner, face_id)
    return zbuf, owner, face_id


def derived_visibility(vertices, K, scale, zbuf, owner):
    height, width = zbuf.shape
    output = np.zeros((2, 778), np.float32)
    for side in range(2):
        q = vertices[side] @ K.T
        depth = vertices[side, :, 2]
        uv = q[:, :2] / np.where(depth[:, None] > 1e-6, q[:, 2:3], np.nan)
        px = np.rint(uv[:, 0] * scale).astype(np.int64)
        py = np.rint(uv[:, 1] * scale).astype(np.int64)
        inside = (depth > 1e-6) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        ids = np.flatnonzero(inside)
        tolerance = np.maximum(0.002, 0.003 * depth[ids])
        output[side, ids] = (
            (owner[py[ids], px[ids]] == side + 1)
            & (np.abs(zbuf[py[ids], px[ids]] - depth[ids]) <= tolerance)
        )
    return output


def face_tile(rgb, title, status, scene, faces, values, vertex_mask, hand_valid,
              palette, sides=(0, 1), geometry=False):
    tile, (x0, y0, width, height), _ = rgb_tile(rgb, title, status)
    _, owner, face_id = scene
    rgba = np.zeros((height, width, 4), np.uint8)
    face_pixels = {}
    for side in sides:
        if not hand_valid[side]:
            face_pixels[str(side)] = 0
            continue
        mask = owner == side + 1
        ids = face_id[mask]
        if geometry:
            colors = np.repeat(HAND_COLORS[side][None], ids.size, axis=0)
            valid_pixels = np.ones(ids.size, bool)
        else:
            valid_faces = vertex_mask[side][faces].all(axis=1)
            face_scores = values[side][faces].mean(axis=1)
            valid_pixels = valid_faces[ids]
            colors = palette(face_scores[ids])
        ys, xs = np.nonzero(mask)
        ys, xs, colors = ys[valid_pixels], xs[valid_pixels], colors[valid_pixels]
        rgba[ys, xs, :3] = colors
        rgba[ys, xs, 3] = ALPHA
        face_pixels[str(side)] = int(len(xs))
    layer = Image.fromarray(rgba, "RGBA")
    result = tile.convert("RGBA")
    result.alpha_composite(layer, (x0, y0))
    return result.convert("RGB"), face_pixels


def legend_block(draw: ImageDraw.ImageDraw, y: int) -> None:
    draw.text((14, y), "Face score = mean of 3 vertex values; nearest surface wins the scene z-buffer; alpha=0.48.",
              fill=(232, 235, 238), font=F14)
    x = 990
    for label, palette, points in (
        ("contact", contact_palette, ((0.0, "0"), (0.5, ".5"), (1.0, "1"))),
        ("visible", visibility_palette, ((0.0, "0"), (0.5, ".5"), (1.0, "1"))),
    ):
        draw.text((x, y), label, fill=(232, 235, 238), font=F12)
        x += 55
        for value, text in points:
            color = tuple(int(c) for c in palette(np.array([value]))[0])
            draw.rectangle((x, y, x + 20, y + 15), fill=color)
            draw.text((x + 23, y), text, fill=(220, 225, 230), font=F12)
            x += 54
        x += 12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    record = json.loads((args.input_root / "window_input.json").read_text())
    if record["dataset"] != DATASET or record["cache_id"] != CACHE_ID:
        raise ValueError("window identity mismatch")
    if str(record["frame_ids"][FRAME_INDEX]) != FRAME_ID:
        raise ValueError("frame identity mismatch")

    rgb = Image.open(args.input_root / "rgb.png").convert("RGB")
    gt = load(args.input_root / "gt.npz")
    geometry = load(args.input_root / "geometry.npz")
    mapping = load(args.input_root / "mapping.npz")
    ego = load(args.input_root / "ego_prediction.npz")
    arrays = {
        "gt": load(args.input_root / "gt_contact.npz"),
        "ego": load(args.input_root / "ego_contact.npz"),
        "s2contact": load(args.input_root / "s2contact.npz"),
        "contactopt": load(args.input_root / "contactopt.npz"),
        "interactvlm": load(args.input_root / "interactvlm.npz"),
    }
    faces = mapping["faces"].astype(np.int64)

    gt_vertices = gt["hand_vertices_camera"][FRAME_INDEX].astype(np.float32)
    gt_hand_valid = gt["hand_valid"][FRAME_INDEX].astype(bool)
    ego_vertices = reconstruct_ego_vertices(ego["hand_markers_camera"][FRAME_INDEX], mapping)
    ego_hand_valid = ego["hand_valid"][FRAME_INDEX].astype(bool)

    src_w, src_h = rgb.size
    display_scale = min(COL_W / src_w, IMAGE_H / src_h)
    display_w, display_h = round(src_w * display_scale), round(src_h * display_scale)
    gt_k = gt["intrinsics"][FRAME_INDEX].astype(np.float32)
    valid_k = ego["intrinsics_valid"].astype(bool) & np.isfinite(ego["intrinsics"]).all(axis=(1, 2))
    anchors = np.flatnonzero(valid_k)
    ego_k_index = int(anchors[np.abs(anchors - FRAME_INDEX).argmin()])
    side = min(src_w, src_h)
    crop_x, crop_y = (src_w - side) // 2, (src_h - side) // 2
    crop_scale = side / 512
    crop_to_rgb = np.array(
        [[crop_scale, 0, crop_x + (crop_scale - 1) / 2],
         [0, crop_scale, crop_y + (crop_scale - 1) / 2],
         [0, 0, 1]], dtype=np.float32,
    )
    ego_k = crop_to_rgb @ ego["intrinsics"][ego_k_index]

    object_vertices = geometry["object_vertices"].astype(np.float32)
    object_faces = geometry["object_faces"].astype(np.int64)
    gt_scene = make_scene(object_vertices, object_faces, gt_vertices, gt_hand_valid, faces,
                          gt_k, display_h, display_w, display_scale)
    ego_scene = make_scene(object_vertices, object_faces, ego_vertices, ego_hand_valid, faces,
                           ego_k, display_h, display_w, display_scale)
    method_scenes = {"ego": ego_scene, "interactvlm": gt_scene}
    for method in ("s2contact", "contactopt"):
        method_vertices = arrays[method]["hand_vertices_camera"][FRAME_INDEX].astype(np.float32)
        method_valid = arrays[method]["hand_valid"][FRAME_INDEX].astype(bool)
        method_scenes[method] = make_scene(
            object_vertices, object_faces, method_vertices, method_valid, faces,
            gt_k, display_h, display_w, display_scale,
        )
    gt_visibility = derived_visibility(gt_vertices, gt_k, display_scale, *gt_scene[:2])
    ego_visibility = interpolate_marker_scalar(ego["marker_visibility"][FRAME_INDEX], mapping)

    all_valid = np.ones((2, 778), bool)
    tiles = []
    pixel_stats = {}

    def add(key, *tile_args, **tile_kwargs):
        tile, counts = face_tile(*tile_args, **tile_kwargs)
        tiles.append(tile)
        pixel_stats[key] = counts

    add("gt_geometry", rgb, "GT | geometry", "GT MANO 778 | scene z-buffer", gt_scene,
        faces, None, all_valid, gt_hand_valid, None, geometry=True)
    add("gt_distance", rgb, "GT | contact distance [0-50 mm]", "red=near | blue=>=50 mm", gt_scene,
        faces, arrays["gt"]["vertex_contact_distance"][FRAME_INDEX],
        arrays["gt"]["vertex_contact_distance_mask"][FRAME_INDEX], gt_hand_valid, distance_palette)
    add("gt_contact", rgb, "GT | contact", "face mean of 778 vertex targets", gt_scene,
        faces, arrays["gt"]["vertex_contact_target"][FRAME_INDEX],
        arrays["gt"]["vertex_contact_mask"][FRAME_INDEX], gt_hand_valid, contact_palette)
    add("gt_visibility", rgb, "GT | visibility [z-buffer derived]", "red hidden -> green visible", gt_scene,
        faces, gt_visibility, all_valid, gt_hand_valid, visibility_palette)

    add("ego_geometry", rgb, "Ego stride5 | geometry [195->778]", "nearest valid K anchor; scene z-buffer", ego_scene,
        faces, None, all_valid, ego_hand_valid, None, geometry=True)
    add("ego_distance", rgb, "Ego stride5 | contact distance [0-50 mm]", "red=near | blue=>=50 mm", ego_scene,
        faces, arrays["ego"]["vertex_contact_distance"][FRAME_INDEX],
        arrays["ego"]["vertex_contact_distance_mask"][FRAME_INDEX], ego_hand_valid, distance_palette)
    add("ego_contact", rgb, "Ego stride5 | contact [195->778]", "continuous predicted probability", ego_scene,
        faces, arrays["ego"]["vertex_contact_probability"][FRAME_INDEX],
        arrays["ego"]["vertex_contact_distance_mask"][FRAME_INDEX], ego_hand_valid, contact_palette)
    add("ego_visibility", rgb, "Ego stride5 | visibility [195->778]", "predicted probability; z-buffer rendered", ego_scene,
        faces, ego_visibility, all_valid, ego_hand_valid, visibility_palette)

    add("ego_contact_compare", rgb, "Ego stride5 | contact [195->778]", "right hand | predicted geometry", ego_scene,
        faces, arrays["ego"]["vertex_contact_probability"][FRAME_INDEX],
        arrays["ego"]["vertex_contact_distance_mask"][FRAME_INDEX], ego_hand_valid, contact_palette, sides=(1,))
    for method, label, suffix in (
        ("s2contact", "S2Contact | contact [778]", "predicted 778 mesh"),
        ("contactopt", "ContactOpt | contact [778]", "predicted 778 mesh"),
        ("interactvlm", "InteractVLM | contact [6890->778]", "GT MANO projection"),
    ):
        hand_valid = arrays[method]["hand_valid"][FRAME_INDEX].astype(bool)
        probability = arrays[method]["vertex_contact_probability"][FRAME_INDEX]
        valid_mask = (
            arrays[method]["vertex_contact_distance_mask"][FRAME_INDEX].astype(bool)
            if "vertex_contact_distance_mask" in arrays[method]
            else np.isfinite(probability)
        )
        add(method, rgb, label, f"right hand | {suffix}", method_scenes[method],
            faces, arrays[method]["vertex_contact_probability"][FRAME_INDEX],
            valid_mask, hand_valid, contact_palette, sides=(1,))

    canvas = Image.new("RGB", (COL_W * 4, HEADER_H + ROW_H * 3 + FOOTER_H), (22, 27, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 14), "3x4 face-level 2D overlay | H2O rank 17 | center 000208 | cache abc07caace9823faa3ffe7ce",
              fill=(244, 245, 247), font=F20)
    draw.text((14, 49), "All overlays are clipped to RGB pixels; object + two-hand nearest-surface z-buffer; bottom row compares right-hand contact.",
              fill=(201, 210, 218), font=F14)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, 4)
        canvas.paste(tile, (col * COL_W, HEADER_H + row * ROW_H))
    footer_y = HEADER_H + ROW_H * 3 + 10
    legend_block(draw, footer_y)
    draw.text((14, footer_y + 34),
              "S2Contact/ContactOpt use their predicted 778 meshes; InteractVLM uses GT MANO only as projection support for its mapped contact field.",
              fill=(196, 205, 214), font=F14)
    draw.text((14, footer_y + 62),
              "Contact: cyan=0, yellow=0.5, red=1 | Visibility: red=hidden, yellow=partial face score, green=visible.",
              fill=(226, 229, 232), font=F14)

    out_png = args.output_dir / "h2o_r017_000208_3x4_face_contact_example.png"
    out_json = args.output_dir / "h2o_r017_000208_3x4_face_contact_example.json"
    canvas.save(out_png, optimize=True)
    method_stats = {}
    for method in ("ego", "s2contact", "contactopt", "interactvlm"):
        probability = arrays[method]["vertex_contact_probability"][FRAME_INDEX, 1]
        mask = (
            arrays[method]["vertex_contact_distance_mask"][FRAME_INDEX, 1].astype(bool)
            if "vertex_contact_distance_mask" in arrays[method]
            else np.isfinite(probability)
        )
        method_stats[method] = {
            "shape": list(arrays[method]["vertex_contact_probability"].shape),
            "right_valid_vertices": int(mask.sum()),
            "right_contact_vertices_p_ge_0_5": int((mask & (probability >= 0.5)).sum()),
            "right_probability_min": float(np.nanmin(probability[mask])),
            "right_probability_max": float(np.nanmax(probability[mask])),
        }
    metadata = {
        "dataset": DATASET,
        "dataset_rank": DATASET_RANK,
        "cache_id": CACHE_ID,
        "sequence_id": record["sequence_id"],
        "window_id": record["window_id"],
        "frame_index": FRAME_INDEX,
        "frame_id": FRAME_ID,
        "rgb_size": [src_w, src_h],
        "output_size": list(canvas.size),
        "layout": "3x4",
        "rendering": {
            "granularity": "MANO 778 triangular faces",
            "face_score": "mean of three vertex values",
            "alpha": ALPHA / 255,
            "occlusion": "object plus both hands, nearest-surface z-buffer",
            "rgb_clip": True,
            "gt_visibility": "derived from GT hand/object scene z-buffer",
            "ego_visibility": "marker_visibility interpolated 195-to-778, then face mean",
            "comparison_hand": "right",
            "comparison_geometry": {
                "ego": "predicted 195-to-778 mesh",
                "s2contact": "predicted 778 mesh",
                "contactopt": "predicted 778 mesh",
                "interactvlm": "GT MANO 778 projection support",
            },
        },
        "ego_intrinsics_source_frame_index": ego_k_index,
        "ego_intrinsics_source_frame_id": str(record["frame_ids"][ego_k_index]),
        "method_stats": method_stats,
        "overlay_pixels": pixel_stats,
        "sources": REMOTE_SOURCES,
        "output": str(out_png),
    }
    out_json.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"png": str(out_png), "json": str(out_json), "method_stats": method_stats}, indent=2))


if __name__ == "__main__":
    main()
