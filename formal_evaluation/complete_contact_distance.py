#!/usr/bin/env python3
"""Export three-granularity contact distances and aggregate registered windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Mapping

import numpy as np

from formal_evaluation.aggregate_contact_distance import window_metrics
from formal_evaluation.common.aggregation import aggregate_windows
from formal_evaluation.common.mano_sampling import marker_vertex_ids_195
from formal_evaluation.contact.geometry_metrics import frame_geometry_metrics


GRANULARITIES = {"joint": 21, "marker": 195, "vertex": 778}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def remap_direct_gt_row(row: Mapping[str, object], index_parent: Path) -> dict[str, object]:
    result = dict(row)
    for name in ("array_path", "metadata_path"):
        if name in result and "/gt_cache/" in str(result[name]):
            result[name] = str(index_parent / str(result[name]).split("/gt_cache/", 1)[1])
    return result


def load_spec(path: Path) -> dict[str, object]:
    """Expand a compact method spec from the reviewed sibling dataset contracts."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    if "dataset_contract" not in spec:
        return spec
    contract = json.loads((path.parent / spec.pop("dataset_contract")).read_text(encoding="utf-8"))
    masks = json.loads((path.parent / spec.pop("mask_contract")).read_text(encoding="utf-8"))
    prediction_index = spec.pop("prediction_index")
    distance_index = spec.pop("distance_index")
    spec["datasets"] = {}
    for dataset, source in contract["datasets"].items():
        spec["datasets"][dataset] = {
            "expected_windows": source["expected_windows"],
            "gt_index": source["gt_index"],
            "gt_index_sha256": source["gt_index_sha256"],
            "input_index": source["input_index"],
            "prediction_indices": [prediction_index],
            "distance_index": distance_index,
            "mask": masks["datasets"][dataset]["mask"],
        }
    return spec


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def prediction_path(row: Mapping[str, object]) -> Path:
    if "array_path" in row:
        return Path(str(row["array_path"]))
    directory = Path(str(row["prediction_dir"]))
    return directory / "predictions.npz"


def prediction_metadata(row: Mapping[str, object]) -> dict[str, object]:
    if "prediction_dir" not in row:
        return {}
    path = Path(str(row["prediction_dir"])) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_key(row: Mapping[str, object], aliases: Mapping[str, str]) -> str:
    raw = str(row.get("window_id", row.get("cache_id", "")))
    if not raw:
        raise ValueError("index row has no window_id/cache_id")
    return aliases.get(raw, raw)


def indexed_rows(paths: list[str], dataset: str, aliases: Mapping[str, str]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for raw_path in paths:
        for row in read_jsonl(Path(raw_path)):
            if row.get("dataset") not in (None, dataset):
                continue
            key = canonical_key(row, aliases)
            if key in result:
                raise ValueError(f"duplicate prediction window: {dataset}/{key}")
            result[key] = row
    return result


def validate_probability(name: str, value: np.ndarray, frames: int, points: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (frames, 2, points):
        raise ValueError(f"{name} shape mismatch: {array.shape}")
    finite = np.isfinite(array)
    if np.any((array[finite] < 0) | (array[finite] > 1)):
        raise ValueError(f"{name} is outside [0,1]")
    return array


def validate_mapping(mapping: Mapping[str, object]) -> dict[str, np.ndarray]:
    result = {}
    for side in ("left", "right"):
        ids = np.asarray(mapping[side], dtype=np.int64)
        if ids.shape != (778,) or len(np.unique(ids)) != 778 or np.any((ids < 0) | (ids >= 6890)):
            raise ValueError(f"invalid {side} MANO-to-SMPL-H mapping")
        result[side] = ids
    return result


def interact_probabilities(
    prediction_dir: Path,
    canonical: Mapping[str, np.ndarray],
    mapping: Mapping[str, object],
    frames: int,
) -> dict[str, np.ndarray]:
    """Add native vertex scores while retaining canonical joint/marker scores."""
    ids = validate_mapping(mapping)
    hand_valid = np.asarray(canonical["hand_valid"], dtype=bool)
    if hand_valid.shape != (frames, 2):
        raise ValueError(f"hand_valid shape mismatch: {hand_valid.shape}")
    scores = []
    for index in range(frames):
        path = prediction_dir / "native" / f"{index:03d}.npz"
        native = load_npz(path)
        score = np.asarray(native["pred_contact_3d_smplh"], dtype=np.float32).reshape(-1)
        if score.shape != (6890,) or not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
            raise ValueError(f"invalid native InteractVLM vector: {path}")
        scores.append(score)
    native_scores = np.stack(scores)
    vertex = np.full((frames, 2, 778), np.nan, dtype=np.float32)
    for side_index, side in enumerate(("left", "right")):
        vertex[:, side_index] = native_scores[:, ids[side]]
        vertex[~hand_valid[:, side_index], side_index] = np.nan
    result = {
        "joint_contact_probability": validate_probability(
            "joint_contact_probability", canonical["joint_contact_probability"], frames, 21
        ),
        "marker_contact_probability": validate_probability(
            "marker_contact_probability", canonical["marker_contact_probability"], frames, 195
        ),
        "vertex_contact_probability": vertex,
        "hand_valid": hand_valid,
    }
    marker_from_native = vertex[:, :, marker_vertex_ids_195()]
    valid = hand_valid[:, :, None] & np.isfinite(result["marker_contact_probability"])
    if not np.allclose(result["marker_contact_probability"][valid], marker_from_native[valid], atol=1e-6):
        raise ValueError("canonical InteractVLM marker scores differ from native topology projection")
    return result


def geometry_matches(
    query: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray], query_valid: np.ndarray
) -> bool:
    valid = np.asarray(query_valid, dtype=bool)
    for prefix, target_name in (("joint", "hand_joints_camera"), ("vertex", "hand_vertices_camera")):
        if prefix not in query or target_name not in target:
            return False
        actual = np.asarray(query[prefix])
        expected = np.asarray(target[target_name])
        if actual.shape != expected.shape or valid.shape != actual.shape[:2]:
            return False
        selected = np.broadcast_to(valid[..., None, None], actual.shape)
        if not np.allclose(actual[selected], expected[selected], rtol=1e-5, atol=1e-7, equal_nan=True):
            return False
    return True


def prepared_geometry(input_index: Path, dataset: str, aliases: Mapping[str, str]) -> dict[str, dict[str, object]]:
    result = {}
    for item in read_jsonl(input_index):
        record = json.loads(Path(str(item["window_input"])).read_text(encoding="utf-8"))
        if record.get("dataset") != dataset:
            continue
        key = canonical_key(record, aliases)
        if key in result:
            raise ValueError(f"duplicate prepared window: {dataset}/{key}")
        paths = [Path(str(path)) for path in record["geometry_paths"]]
        if len(paths) != len(record["frame_ids"]):
            raise ValueError(f"prepared geometry/frame mismatch: {dataset}/{key}")
        result[key] = {**record, "geometry_paths": paths}
    return result


def verify_prepared_input_provenance(
    spec: Mapping[str, object], dataset: str, prepared: Mapping[str, Mapping[str, object]],
    predictions: Mapping[str, Mapping[str, object]], aliases: Mapping[str, str]
) -> None:
    """Prove that each predicted hand came from the registered neutral-geometry cache."""
    builder = Path(str(spec["cache_builder"]))
    if sha256(builder) != spec["cache_builder_sha256"]:
        raise ValueError("contact cache builder checksum mismatch")
    forward = json.loads(Path(str(spec["forward_spec"])).read_text(encoding="utf-8"))
    if forward["method"] != spec["method"]:
        raise ValueError("forward/cache method mismatch")
    restored = read_jsonl(Path(str(forward["restored_index"])))
    rows = {}
    for pair in forward["datasets"][dataset]["cache_pairs"]:
        matches = [
            row for row in restored
            if Path(str(row["source"])).name.removesuffix(".parts") == pair["index"]
        ]
        if len(matches) != 1 or not matches[0].get("sha256"):
            raise ValueError(f"missing unique restored cache-index provenance: {dataset}/{pair['index']}")
        index_path = Path(str(matches[0]["destination"]))
        if not index_path.is_file() or index_path.stat().st_size != matches[0]["bytes"]:
            raise ValueError(f"restored cache-index size mismatch: {index_path}")
        for row in read_jsonl(index_path):
            key = (canonical_key(row, aliases), int(row["time_index"]))
            if key in rows:
                raise ValueError(f"duplicate cache input frame: {dataset}/{key}")
            rows[key] = row
    for window_id, prepared_row in prepared.items():
        prediction = load_npz(prediction_path(predictions[window_id]))
        valid = np.asarray(prediction["hand_valid"], dtype=bool)
        if valid.shape != (60, 2) or valid[:, 0].any():
            raise ValueError(f"contact cache must be right-hand-only: {dataset}/{window_id}")
        cached_times = {time for (key, time) in rows if key == window_id}
        if cached_times != set(np.flatnonzero(valid[:, 1])):
            raise ValueError(f"cache/prediction hand coverage mismatch: {dataset}/{window_id}")
        for time in cached_times:
            cache_row = rows[(window_id, time)]
            if str(cache_row["frame_id"]) != str(prepared_row["frame_ids"][time]):
                raise ValueError(f"cache frame identity mismatch: {dataset}/{window_id}/{time}")
            if Path(str(cache_row["geometry_path"])) != prepared_row["geometry_paths"][time]:
                raise ValueError(f"cache geometry path mismatch: {dataset}/{window_id}/{time}")


def load_faces(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        faces = np.load(path, allow_pickle=False)
    else:
        with path.open("rb") as stream:
            faces = pickle.load(stream, encoding="latin1")["f"]
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= 778:
        raise ValueError("invalid MANO faces")
    return faces


def query_from_prediction(
    prediction: Mapping[str, np.ndarray], prepared: Mapping[str, object] | None, frames: int
) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]] | None, str]:
    vertices = prediction.get("hand_vertices_camera")
    joints = prediction.get("hand_joints_camera")
    scenes = None
    source = "canonical_prediction"
    if vertices is None or joints is None:
        if prepared is None:
            raise ValueError("prediction has no hand geometry and no explicit prepared geometry")
        scenes = [load_npz(path) for path in prepared["geometry_paths"]]
        if len(scenes) != frames:
            raise ValueError("prepared scene count mismatch")
        vertices = np.stack([scene["hand_vertices"] for scene in scenes])
        joints = np.stack([scene["hand_joints"] for scene in scenes])
        source = "prepared_model_input"
    vertices = np.asarray(vertices, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    if vertices.shape != (frames, 2, 778, 3) or joints.shape != (frames, 2, 21, 3):
        raise ValueError(f"hand geometry shape mismatch: {vertices.shape}, {joints.shape}")
    return {
        "joint": joints,
        "marker": vertices[:, :, marker_vertex_ids_195()],
        "vertex": vertices,
    }, scenes, source


def distance_arrays(source: Mapping[str, np.ndarray], frames: int) -> dict[str, np.ndarray]:
    result = {}
    for prefix, points in GRANULARITIES.items():
        for suffix in ("distance", "distance_mask"):
            name = f"{prefix}_contact_{suffix}"
            array = np.asarray(source[name])
            if array.shape != (frames, 2, points):
                raise ValueError(f"{name} shape mismatch: {array.shape}")
            result[name] = array.astype(bool if suffix.endswith("mask") else np.float32, copy=False)
        mask = result[f"{prefix}_contact_distance_mask"]
        distance = result[f"{prefix}_contact_distance"]
        if not np.isfinite(distance[mask]).all() or not np.isnan(distance[~mask]).all():
            raise ValueError(f"invalid undefined-distance encoding: {prefix}")
    return result


def restrict_distances_to_predicted_hands(
    distances: Mapping[str, np.ndarray], hand_valid: np.ndarray
) -> dict[str, np.ndarray]:
    """Exclude hands for which the method emitted no prediction."""
    result = {key: np.asarray(value).copy() for key, value in distances.items()}
    for prefix in GRANULARITIES:
        mask_name = f"{prefix}_contact_distance_mask"
        distance_name = f"{prefix}_contact_distance"
        result[mask_name] &= np.asarray(hand_valid, dtype=bool)[..., None]
        result[distance_name][~result[mask_name]] = np.nan
    return result


def recompute_distances(
    query: Mapping[str, np.ndarray], query_valid: np.ndarray, scenes: list[dict[str, np.ndarray]], faces: np.ndarray
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {}
    for frame_index, scene in enumerate(scenes):
        values = frame_geometry_metrics(
            {key: value[frame_index] for key, value in query.items()},
            query_valid[frame_index],
            scene,
            faces,
            "cpu",
        )
        for key, value in values.items():
            if key.endswith("_contact_distance") or key.endswith("_contact_distance_mask"):
                collected.setdefault(key, []).append(value)
    return distance_arrays({key: np.stack(value) for key, value in collected.items()}, len(scenes))


def load_mask(source: Mapping[str, object], identities: set[str]) -> tuple[dict[str, list[bool]] | None, str | None]:
    mask_spec = source.get("mask")
    if mask_spec is None:
        return None, None
    path = Path(str(mask_spec["path"]))
    digest = sha256(path)
    if digest != mask_spec["sha256"]:
        raise ValueError("mask checksum mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    masks = dict(zip(payload["window_ids"], payload["excluded"], strict=True))
    if set(masks) != identities:
        raise ValueError("mask window identities mismatch")
    return masks, digest


def run(spec: Mapping[str, object], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=False)
    mode = str(spec["mode"])
    faces = load_faces(Path(str(spec["mano_faces"]))) if mode == "canonical" else None
    mapping = json.loads(Path(str(spec["mano_to_smplh"])).read_text()) if mode == "interactvlm" else None
    report: dict[str, object] = {
        "status": "complete",
        "method": spec["method"],
        "windows": 0,
        "schemes": {},
        "datasets": {},
        "geometry_audit": {},
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
    }
    output_rows = []
    for dataset, source in spec["datasets"].items():
        gt_path = Path(str(source["gt_index"]))
        if source.get("gt_index_sha256") and sha256(gt_path) != source["gt_index_sha256"]:
            raise ValueError(f"GT index checksum mismatch: {dataset}")
        gt_rows = [
            remap_direct_gt_row(row, gt_path.parent)
            for row in read_jsonl(gt_path)
        ]
        aliases = {str(row["cache_id"]): str(row["window_id"]) for row in gt_rows}
        full_gt = {str(row["window_id"]): row for row in gt_rows}
        predictions = indexed_rows(list(source["prediction_indices"]), dataset, aliases)
        expected = int(source["expected_windows"])
        if len(predictions) != expected or not set(predictions).issubset(full_gt):
            raise ValueError(
                f"GT/prediction coverage mismatch: {dataset} "
                f"({len(full_gt)}/{len(predictions)}/{expected})"
            )
        gt = {key: full_gt[key] for key in predictions}
        distance_rows = indexed_rows([str(source["distance_index"])], dataset, aliases)
        if not set(gt).issubset(distance_rows):
            raise ValueError(f"GT distance coverage mismatch: {dataset}")
        distance_rows = {key: distance_rows[key] for key in gt}
        prepared = prepared_geometry(Path(str(source["input_index"])), dataset, aliases) if mode == "canonical" else {}
        if mode == "canonical":
            contract = Path(spec["distance_contract"])
            if sha256(contract) != spec["distance_contract_sha256"]:
                raise ValueError("GT-distance source contract checksum mismatch")
            overlay_source = json.loads(contract.read_text())["datasets"][dataset]
            if overlay_source["input_index"] != source["input_index"] or overlay_source["gt_index"] != source["gt_index"]:
                raise ValueError("GT-distance overlay came from different prepared geometry")
        if mode == "canonical" and set(prepared) != set(gt):
            raise ValueError(f"prepared geometry coverage mismatch: {dataset}")
        masks, mask_digest = load_mask(source, set(gt))
        scheme_values: dict[str, list[dict[str, object]]] = {"unfiltered": []}
        if masks is not None:
            scheme_values["all8_p95"] = []
        reused = recomputed = 0
        prepared_provenance_verified = False
        dataset_dir = output_root / "windows" / dataset
        dataset_dir.mkdir(parents=True)
        for window_id, gt_row in gt.items():
            pred_row = predictions[window_id]
            metadata = prediction_metadata(pred_row)
            frames = len(gt_row["frame_ids"])
            if frames != 60 or [str(value) for value in gt_row["frame_ids"]] != [str(value) for value in metadata.get("frame_ids", gt_row["frame_ids"])]:
                raise ValueError(f"prediction frame identity mismatch: {dataset}/{window_id}")
            if [str(value) for value in distance_rows[window_id]["frame_ids"]] != [str(value) for value in gt_row["frame_ids"]]:
                raise ValueError(f"distance frame identity mismatch: {dataset}/{window_id}")
            prediction = load_npz(prediction_path(pred_row))
            if mode == "interactvlm":
                if "GT MANO" not in str(metadata.get("geometry_for_projection", "")):
                    raise ValueError(f"InteractVLM projection geometry is not declared GT: {dataset}/{window_id}")
                probabilities = interact_probabilities(Path(str(pred_row["prediction_dir"])), prediction, mapping, frames)
                distances = distance_arrays(load_npz(prediction_path(distance_rows[window_id])), frames)
                geometry_source = "registered_gt_projection"
                reused += 1
            else:
                probabilities = {"hand_valid": np.asarray(prediction["hand_valid"], dtype=bool)}
                for prefix, points in GRANULARITIES.items():
                    probabilities[f"{prefix}_contact_probability"] = validate_probability(
                        f"{prefix}_contact_probability", prediction[f"{prefix}_contact_probability"], frames, points
                    )
                if [str(value) for value in prepared[window_id]["frame_ids"]] != [str(value) for value in gt_row["frame_ids"]]:
                    raise ValueError(f"prepared frame identity mismatch: {dataset}/{window_id}")
                native_geometry = all(k in prediction for k in ("hand_vertices_camera", "hand_joints_camera"))
                if not native_geometry:
                    if not prepared_provenance_verified:
                        verify_prepared_input_provenance(spec, dataset, prepared, predictions, aliases)
                        prepared_provenance_verified = True
                    # The audited model input and the GT-distance producer use the same
                    # per-frame geometry paths. No second GT-cache fit is involved.
                    geometry_source = "cache_index_verified_prepared_model_input"
                    can_reuse = True
                else:
                    query, scenes, geometry_source = query_from_prediction(prediction, prepared[window_id], frames)
                    scenes = [load_npz(path) for path in prepared[window_id]["geometry_paths"]]
                    overlay_queries = {"hand_vertices_camera": np.stack([v["hand_vertices"] for v in scenes]),
                                       "hand_joints_camera": np.stack([v["hand_joints"] for v in scenes])}
                    can_reuse = geometry_matches(query, overlay_queries, probabilities["hand_valid"])
                if can_reuse:
                    distances = distance_arrays(load_npz(prediction_path(distance_rows[window_id])), frames)
                    reused += 1
                else:
                    distances = recompute_distances(query, probabilities["hand_valid"], scenes, faces)
                    recomputed += 1
            distances = restrict_distances_to_predicted_hands(distances, probabilities["hand_valid"])
            arrays = {**probabilities, **distances}
            output_path = dataset_dir / f"{gt_row['cache_id']}.npz"
            with output_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            schemes = {}
            for scheme in scheme_values:
                excluded = np.asarray(masks[window_id], dtype=bool) if masks is not None else np.zeros(frames, dtype=bool)
                keep = ~excluded if scheme == "all8_p95" else np.ones(frames, dtype=bool)
                values = {"window_id": window_id, **window_metrics(arrays, keep)}
                scheme_values[scheme].append(values)
                schemes[scheme] = values
            output_rows.append({
                "method": spec["method"], "dataset": dataset, "window_id": window_id,
                "cache_id": gt_row["cache_id"], "frame_ids": gt_row["frame_ids"],
                "array_path": str(output_path), "array_sha256": sha256(output_path),
                "geometry_source": geometry_source, "schemes": schemes,
            })
        for scheme, values in scheme_values.items():
            report["schemes"].setdefault(scheme, {})[dataset] = aggregate_windows(values, method=str(spec["method"]))
        report["datasets"][dataset] = {"windows": expected, "mask_sha256": mask_digest}
        report["geometry_audit"][dataset] = {"gt_distance_reused_windows": reused, "recomputed_windows": recomputed}
        report["windows"] += expected
        print(json.dumps({"dataset": dataset, "windows": expected, "reused": reused, "recomputed": recomputed}), flush=True)
    if report["windows"] != int(spec["expected_windows"]) or len(output_rows) != int(spec["expected_windows"]):
        raise ValueError("total window count mismatch")
    (output_root / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in output_rows), encoding="utf-8")
    (output_root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    summary = {"status": "complete", "method": spec["method"], "windows": report["windows"],
               "index_sha256": sha256(output_root / "index.jsonl"), "report_sha256": sha256(output_root / "report.json")}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_root / "COMPLETE").write_text("complete\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(load_spec(args.spec), args.output_root)


if __name__ == "__main__":
    main()
