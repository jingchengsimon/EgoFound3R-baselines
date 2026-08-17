from __future__ import annotations

from pathlib import Path

from torch.utils.data import ConcatDataset

from egohandmetric_prompt.data.base import CombinedFrameDataset, TemporalChunkDataset
from egohandmetric_prompt.configs import default_dataset_root_overrides
from egohandmetric_prompt.data.sequence_splits import (
    MANAGED_SEQUENCE_SPLIT_DATASETS,
    SequenceSplitFrameDataset,
)
from egohandmetric_prompt.data.datasets import (
    ArcticFrameDataset,
    EgoForceArcticFrameDataset,
    EgoForceH2OFrameDataset,
    EgoForceHot3dFrameDataset,
    EgoTouchFrameDataset,
    ForeHoiFrameDataset,
    H2OFrameDataset,
    Hoi4dFrameDataset,
    HoloAssistFrameDataset,
    Hot3dAriaFrameDataset,
    OakInkV2FrameDataset,
    OakInkV1FrameDataset,
    ReInterHandFrameDataset,
    Stera10MFrameDataset,
    TacoFrameDataset,
    WhimFrameDataset,
    default_data_root,
)


MIDTRAIN_DATASET_NAMES = (
    "h2o",
    "hot3d_aria",
    "hoi4d",
    "oakink_v2",
    "whim",
    "forehoi",
    "reinterhand",
    "stera_10m",
    "egoforce_arctic",
    "taco",
    "oakink_v1",
    "holoassist",
)

POSTTRAIN_MARKER_DATASET_NAMES = (
    "h2o",
    "hot3d_aria",
    "oakink_v2",
    "egoforce_arctic",
)

POSTTRAIN_3R_DATASET_NAMES = (
    "h2o",
    "hot3d_aria",
    "hoi4d",
    "oakink_v2",
    "egoforce_arctic",
    "stera_10m",
)

ARCTIC_DATASET_NAMES = ("arctic", "egoforce_arctic")


def default_sequence_split_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "dataset_sequence_splits.json"


def _resolve_default_root(name: str, data_root: Path) -> Path:
    dataset_root_overrides = default_dataset_root_overrides()
    if name in dataset_root_overrides:
        return dataset_root_overrides[name]
    taco_candidates = (
        data_root / "TACO_resized",
        data_root.parent / "TACO_resized",
        data_root.parent.parent / "DATA" / "TACO_resized",
    )
    taco_root = next((candidate for candidate in taco_candidates if candidate.is_dir()), taco_candidates[0])
    mapping = {
        "h2o": data_root / "H2O" / "h2o_data",
        "hot3d_aria": data_root / "HOT3D" / "hot3d" / "hot3d" / "dataset",
        "hoi4d": data_root / "HOI4D",
        "oakink_v2": data_root / "OakInk-v2",
        "arctic": data_root / "EgoForce" / "ARCTIC",
        "whim": data_root / "whim",
        "forehoi": data_root / "ForeHOI",
        "reinterhand": data_root / "ReInterHand" / "InterWild" / "tool" / "ReInterHand" / "download",
        "egotouch": data_root / "EgoTouch",
        "stera_10m": data_root / "stera-10m",
        "egoforce_h2o": data_root / "EgoForce" / "H2O",
        "egoforce_hot3d": data_root / "EgoForce" / "HOT3D",
        "egoforce_arctic": data_root / "EgoForce" / "ARCTIC",
        # TACO_resized contains the verified RGB/annotation export and, when
        # present, the lossless official uint16 depth sidecar.
        "taco": taco_root if taco_root.is_dir() else data_root / "TACO",
        "oakink_v1": data_root / "OakInk-v1",
        "holoassist": data_root / "HoloAssist",
    }
    if name not in mapping:
        raise KeyError(f"未知数据集名称：{name}")
    return mapping[name]


def build_named_frame_dataset(
    name: str,
    *,
    data_root: str | Path | None = None,
    root_override: str | Path | None = None,
    **kwargs,
):
    resolved_data_root = default_data_root() if data_root is None else Path(data_root)
    root = Path(root_override) if root_override is not None else _resolve_default_root(name, resolved_data_root)

    requested_split = str(kwargs.get("split", "all"))
    manifest_key = MANAGED_SEQUENCE_SPLIT_DATASETS.get(name)
    should_filter_partition = manifest_key is not None and requested_split in {"train", "trainval", "test"}
    if should_filter_partition:
        kwargs = dict(kwargs)
        kwargs["split"] = "all"

    if name == "h2o":
        dataset = H2OFrameDataset(root, **kwargs)
    elif name == "hot3d_aria":
        manifest_path = kwargs.pop("manifest_path", root / "hot3d_aria_local_manifest.jsonl")
        dataset = Hot3dAriaFrameDataset(root, manifest_path=manifest_path, **kwargs)
    elif name == "hoi4d":
        dataset = Hoi4dFrameDataset(root, **kwargs)
    elif name == "oakink_v2":
        manifest_path = kwargs.pop("manifest_path", root / "oakink_preview_manifest.jsonl")
        dataset = OakInkV2FrameDataset(root, manifest_path=manifest_path, **kwargs)
    elif name == "arctic":
        dataset = ArcticFrameDataset(root, **kwargs)
    elif name == "whim":
        dataset = WhimFrameDataset(root, **kwargs)
    elif name == "forehoi":
        dataset = ForeHoiFrameDataset(root, **kwargs)
    elif name == "reinterhand":
        manifest_path = kwargs.pop("manifest_path", root / "reinterhand_ego_manifest.jsonl")
        dataset = ReInterHandFrameDataset(root, manifest_path=manifest_path, **kwargs)
    elif name == "egotouch":
        dataset = EgoTouchFrameDataset(root, **kwargs)
    elif name == "stera_10m":
        dataset = Stera10MFrameDataset(root, **kwargs)
    elif name == "egoforce_h2o":
        dataset = EgoForceH2OFrameDataset(root, **kwargs)
    elif name == "egoforce_hot3d":
        dataset = EgoForceHot3dFrameDataset(root, **kwargs)
    elif name == "egoforce_arctic":
        dataset = EgoForceArcticFrameDataset(root, **kwargs)
    elif name == "taco":
        dataset = TacoFrameDataset(root, **kwargs)
    elif name == "oakink_v1":
        dataset = OakInkV1FrameDataset(root, **kwargs)
    elif name == "holoassist":
        dataset = HoloAssistFrameDataset(root, **kwargs)
    else:
        raise KeyError(f"未知数据集名称：{name}")

    if should_filter_partition:
        return SequenceSplitFrameDataset(
            dataset,
            dataset_name=name,
            manifest_key=manifest_key,
            split=requested_split,
            manifest_path=default_sequence_split_manifest_path(),
        )
    return dataset


def build_concat_frame_dataset(
    dataset_names: tuple[str, ...] | list[str],
    *,
    data_root: str | Path | None = None,
    **kwargs,
):
    datasets = [build_named_frame_dataset(name, data_root=data_root, **kwargs) for name in dataset_names]
    return CombinedFrameDataset(datasets)


def build_concat_chunk_dataset(
    dataset_names: tuple[str, ...] | list[str],
    *,
    num_frames: int,
    window_stride: int = 1,
    data_root: str | Path | None = None,
    **kwargs,
):
    datasets = []
    for name in dataset_names:
        frame_dataset = build_named_frame_dataset(name, data_root=data_root, **kwargs)
        datasets.append(TemporalChunkDataset(frame_dataset, num_frames=num_frames, window_stride=window_stride))
    return ConcatDataset(datasets)
