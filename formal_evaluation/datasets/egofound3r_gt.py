"""Latest EgoFound3R dataloader bridge for six-dataset evaluation GT."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_LOADERS = {
    "h2o": "h2o",
    "taco": "taco",
    "hot3d": "hot3d_aria",
    "oakink_v2": "oakink_v2",
    "arctic": "egoforce_arctic",
    "hoi4d": "hoi4d",
}
CONTACT_MODES = {dataset: "object_and_interhand" for dataset in DATASET_LOADERS}
VENDORED_DATALOADER_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "egofound3r_dataloader"
TACO_DATA_ROOT = Path("/mnt/cpfs/sjc/DATA/TACO_resized")


def canonical_sequence_id(dataset: str, sequence_id: str) -> str:
    """Map the split manifest's H2O identity to its loader identity."""
    return f"{sequence_id}/cam4" if dataset == "h2o" else sequence_id


def validate_window_row(row: Mapping[str, object]) -> tuple[str, str, list[str]]:
    dataset = row.get("dataset")
    sequence_id = row.get("sequence_id")
    frame_ids = row.get("frame_ids")
    if dataset not in DATASET_LOADERS:
        raise ValueError(f"unsupported dataset: {dataset!r}")
    if not isinstance(sequence_id, str) or not sequence_id:
        raise ValueError("window sequence_id must be a non-empty string")
    if not isinstance(frame_ids, list) or not frame_ids or not all(isinstance(item, str) for item in frame_ids):
        raise ValueError("window frame_ids must be a non-empty string list")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError(f"{dataset}/{sequence_id}: duplicate frame IDs")
    if row.get("window_size") != len(frame_ids):
        raise ValueError(f"{dataset}/{sequence_id}: window_size disagrees with frame_ids")
    return dataset, sequence_id, frame_ids


def source_indices_for_window(frame_dataset: Any, raw_sequence_id: str, frame_ids: Sequence[str]) -> list[int]:
    """Resolve one manifest window without altering the loader's sequence order."""
    try:
        sequence_indices = frame_dataset.sequence_to_indices[raw_sequence_id]
    except KeyError as error:
        raise KeyError(f"loader has no sequence {raw_sequence_id!r}") from error

    index_entries = getattr(frame_dataset, "_record_index", None)
    if index_entries is not None:
        observed_ids = [str(index_entries[index]["frame_id"]) for index in sequence_indices]
    else:
        observed_ids = [str(frame_dataset._get_record(index).frame_id) for index in sequence_indices]
    positions = {frame_id: position for position, frame_id in enumerate(observed_ids)}
    missing = [frame_id for frame_id in frame_ids if frame_id not in positions]
    if missing:
        raise KeyError(f"{raw_sequence_id}: manifest frames missing from loader: {missing[:3]}")
    selected_positions = [positions[frame_id] for frame_id in frame_ids]
    first = selected_positions[0]
    if selected_positions != list(range(first, first + len(frame_ids))):
        raise ValueError(f"{raw_sequence_id}: manifest window is not contiguous in loader order")
    return [int(sequence_indices[position]) for position in selected_positions]


class _ExactWindowDataset:
    """Restrict a parent FrameDataset to one sequence; TemporalChunkDataset builds the chunk."""

    def __init__(self, parent: Any, raw_sequence_id: str, source_indices: Sequence[int]) -> None:
        self.parent = parent
        self.source_indices = [int(index) for index in source_indices]
        self.sequence_to_indices = {raw_sequence_id: list(range(len(self.source_indices)))}

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int) -> Any:
        return self.parent[self.source_indices[index]]


def _dataloader_runtime() -> tuple[Any, Any, Any, Any]:
    vendor = str(VENDORED_DATALOADER_ROOT)
    if not (VENDORED_DATALOADER_ROOT / "egohandmetric_prompt" / "data" / "stages.py").is_file():
        raise FileNotFoundError(f"missing vendored EgoFound3R dataloader: {VENDORED_DATALOADER_ROOT}")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from egohandmetric_prompt.data.base import TemporalChunkDataset
    from egohandmetric_prompt.data.marker_vertices import MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195
    from egohandmetric_prompt.data.stages import build_named_frame_dataset
    from egohandmetric_prompt.data.training import ManoVertexBuilder, MarkerBatchCollator

    return (
        build_named_frame_dataset,
        TemporalChunkDataset,
        (ManoVertexBuilder, MarkerBatchCollator),
        MANO_MESHGRAPHORMER_LEVEL1_MARKER_VERTEX_IDS_195,
    )


@dataclass
class SixDatasetGroundTruth:
    """Materialize exact split windows with the current dataloader and collator."""

    roots: Mapping[str, str | Path]
    mano_dir: str | Path
    scene_visibility_device: str | None = None
    interhand_contact_compute_device: str | None = None

    def __post_init__(self) -> None:
        missing = set(DATASET_LOADERS) - set(self.roots)
        extra = set(self.roots) - set(DATASET_LOADERS)
        if missing or extra:
            raise ValueError(f"dataset roots must match {sorted(DATASET_LOADERS)}; missing={sorted(missing)}, extra={sorted(extra)}")
        if Path(self.roots["taco"]) != TACO_DATA_ROOT:
            raise ValueError(f"taco must use {TACO_DATA_ROOT}, got {self.roots['taco']}")
        build_dataset, temporal_chunk, training, marker_ids = _dataloader_runtime()
        mano_builder, collator = training
        self._temporal_chunk = temporal_chunk
        self._mano_builder = mano_builder(self.mano_dir)
        self._datasets = {
            dataset: build_dataset(
                loader_name,
                root_override=self.roots[dataset],
                split="all",
                load_rgb=True,
                load_depth=True,
            )
            for dataset, loader_name in DATASET_LOADERS.items()
        }
        self._collator = collator(
            stage="posttrain",
            stream_name="marker",
            marker_vertex_ids=marker_ids,
            mano_vertex_builder=self._mano_builder,
            contact_supervision_by_dataset=CONTACT_MODES,
            scene_visibility_device=self.scene_visibility_device,
            interhand_contact_compute_device=self.interhand_contact_compute_device,
        )
        self._geometry_collator = collator(
            stage="posttrain",
            stream_name="marker",
            marker_vertex_ids=marker_ids,
            mano_vertex_builder=self._mano_builder,
            include_scene_occlusion_in_visibility=False,
            contact_supervision_by_dataset={dataset: "disabled" for dataset in DATASET_LOADERS},
        )

    def chunk_for_window(self, row: Mapping[str, object]) -> dict[str, Any]:
        """Return one exact, sequence-preserving chunk from the latest FrameDataset."""
        dataset, sequence_id, frame_ids = validate_window_row(row)
        parent = self._datasets[dataset]
        raw_sequence_id = canonical_sequence_id(dataset, sequence_id)
        source_indices = source_indices_for_window(parent, raw_sequence_id, frame_ids)
        exact_frames = _ExactWindowDataset(parent, raw_sequence_id, source_indices)
        chunks = self._temporal_chunk(
            exact_frames,
            num_frames=len(frame_ids),
            window_stride=len(frame_ids),
            drop_last=True,
        )
        if len(chunks) != 1:
            raise RuntimeError(f"{dataset}/{sequence_id}: exact window created {len(chunks)} chunks")
        chunk = chunks[0]
        if chunk["frame_ids"] != frame_ids:
            raise RuntimeError(f"{dataset}/{sequence_id}: loader frame IDs drift from split manifest")
        return chunk

    def geometry_for_window(self, row: Mapping[str, object]) -> list[dict[str, Any]]:
        """Canonical camera-space MANO/object geometry without contact labels or method features."""
        chunk = self.chunk_for_window(row)
        batch = self._geometry_collator([chunk])
        output: list[dict[str, Any]] = []
        for sample, hands in zip(chunk["samples"], batch["normalized_hands"][0], strict=True):
            hand_vertices = []
            hand_joints = []
            hand_valid = []
            for hand in hands:
                vertices, joints = (None, None) if hand is None else self._mano_builder.build_vertices_and_joints(hand)
                valid = vertices is not None and joints is not None and tuple(vertices.shape) == (778, 3) and tuple(joints.shape) == (21, 3)
                hand_vertices.append(None if not valid else vertices.detach().cpu().numpy())
                hand_joints.append(None if not valid else joints.detach().cpu().numpy())
                hand_valid.append(valid)
            if len(hand_vertices) != 2:
                raise ValueError(f"canonical geometry requires two hand slots, got {len(hand_vertices)}")
            object_mesh = self._geometry_collator._object_mesh_for_sample(sample)
            output.append({
                "hand_vertices": hand_vertices,
                "hand_joints": hand_joints,
                "hand_valid": hand_valid,
                "object_vertices": None if object_mesh is None else object_mesh[0].detach().cpu().numpy(),
                "object_faces": None if object_mesh is None else object_mesh[1].detach().cpu().numpy(),
                "rgb": sample["rgb"],
                "rgb_ref": sample.get("rgb_ref"),
                "scene_objects": sample.get("extras", {}).get("scene_objects", []),
            })
        return output

    def batch_for_window(self, row: Mapping[str, object]) -> dict[str, Any]:
        dataset, sequence_id, frame_ids = validate_window_row(row)
        chunk = self.chunk_for_window(row)
        batch = self._collator([chunk])
        source = batch["batch_sources"][0]
        if source["frame_ids"] != frame_ids:
            raise RuntimeError(f"{dataset}/{sequence_id}: collator frame IDs drift from split manifest")
        batch["formal_evaluation_depth_refs"] = [sample.get("depth_ref") for sample in chunk["samples"]]
        return batch

    def audit_window(self, row: Mapping[str, object]) -> dict[str, object]:
        """Return exact-frame and GT-mask evidence without caching GT outside the run."""
        dataset, sequence_id, frame_ids = validate_window_row(row)
        batch = self.batch_for_window(row)
        source = batch["batch_sources"][0]
        result: dict[str, object] = {
            "dataset": dataset,
            "sequence_id": sequence_id,
            "frame_ids": frame_ids,
            "loader_sequence_id": source["sequence_id"],
            "frame_count": len(frame_ids),
            "camera_pose_convention": "world_to_camera",
        }
        for key in (
            "three_r_supervision_mask",
            "depth_supervision_mask",
            "camera_pose_supervision_mask",
            "intrinsics_supervision_mask",
            "hand_valid_mask",
            "raw_joint_supervision_mask",
            "contact_supervision_mask",
            "marker_contact_supervision_mask",
        ):
            value = batch[key]
            result[f"{key}_count"] = int(value.sum().item())
        if dataset == "taco":
            refs = batch["formal_evaluation_depth_refs"]
            paths = [str(ref.get("path", "")) for ref in refs if isinstance(ref, Mapping)]
            official_root = TACO_DATA_ROOT / "Egocentric_Depth_Videos_official_uint16" / "Egocentric_Depth_Videos"
            if len(paths) != len(frame_ids) or not all(path.startswith(str(official_root)) for path in paths):
                raise RuntimeError(f"{dataset}/{sequence_id}: did not resolve official uint16 depth")
            result["official_uint16_depth_required"] = True
            result["depth_gt_root"] = str(official_root)
        return result


def camera_c2w_from_batch(batch: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Convert the loader's world-to-camera poses to the canonical evaluator's c2w convention."""
    import numpy as np

    poses = batch["camera_pose"][0]
    valid = batch["camera_pose_supervision_mask"][0]
    if hasattr(poses, "detach"):
        poses = poses.detach().cpu().numpy()
    if hasattr(valid, "detach"):
        valid = valid.detach().cpu().numpy()
    poses = np.asarray(poses, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    output = np.full_like(poses, np.nan)
    if np.any(valid):
        output[valid] = np.linalg.inv(poses[valid])
    return output, valid
