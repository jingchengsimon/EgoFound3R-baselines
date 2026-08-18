# Dataset adapter index

This is the canonical lookup point for dataset integration. Method-specific
resize, normalization, intrinsics scaling, chunking and model input conversion
do not belong here; they remain in each baseline adapter.

| Dataset key | Implementation | Registered splits | RGB contract |
|---|---|---|---|
| `h2o` | `formal_evaluation/datasets/h2o.py:H2ODatasetAdapter` | `test -> subject4_ego` | Read-only original frame paths, frame IDs and per-frame `(width,height)` audit |

Registry and shared output type:

- `formal_evaluation/datasets/registry.py:DATASET_ADAPTERS`
- `formal_evaluation/datasets/registry.py:get_dataset_adapter`
- `formal_evaluation/datasets/base.py:FrameSequence`

To add a dataset: implement the same `sequence_ids()` and
`select_contiguous()` contract, register one key in `DATASET_ADAPTERS`, and add
one row above. Never modify or resize files inside a dataset root.

## Six-dataset split manifests

The frozen manifests consumed by future dataloaders and evaluation scripts are:

- `manifests/sequence_splits/{h2o,taco,hot3d,oakink_v2,arctic,hoi4d}.json`
- `manifests/dataset_sequence_splits.json`
- `manifests/evaluation_test_windows_seed0.jsonl`

`build_dataset_splits.py` creates the permanent sequence split and the sampled
evaluation-window manifest without loading or changing dataset files:

```bash
python -m formal_evaluation.datasets.build_dataset_splits \
  --manifest h2o=/path/h2o.json \
  --manifest taco=/path/taco.json \
  --manifest hot3d=/path/hot3d.json \
  --manifest oakink_v2=/path/oakink_v2.json \
  --manifest arctic=/path/arctic.json \
  --manifest hoi4d=/path/hoi4d.json \
  --output-dir /path/splits
```

Each input JSON has `sequences`, whose rows contain `sequence_id` and ordered
`frame_ids`. H2O and TACO additionally require `official_splits`. H2O accepts
the official frame paths and collapses them to canonical sequence IDs. For
every dataset, candidates start at the first frame of each selected evaluation
sequence, use 60 frames with stride 60, never overlap, and drop an incomplete
final tail.
Outputs are `dataset_sequence_splits.json` and
`evaluation_test_windows_seed0.jsonl`.

H2O split keys are `train`, `val`, and `test`; TACO preserves `train` and the
official `test_1`–`test_4` labels (S1–S4) for every sequence. Its evaluation
selection draws 50% of the available sequences independently from each of the
four official test subsets. HOT3D, OakInk-v2, ARCTIC, and HOI4D use
deterministic 10%/10%/25%/15% custom test partitions and select half of those
test sequences for evaluation. The command outputs every valid 60-frame clip;
H2O outputs `trainval/test`; the other five datasets output `train/test`.

`export_dataset_manifests.py` is the read-only bridge from the existing
EgoFound3R dataset indexes to the six normalized inputs used by the command
above. Pass the copied `configs/dataset_sequence_splits.json` with
`--sequence-splits`; this validates TACO's frozen train/test membership against
the vendored official S1–S4 list. It is not a training dataloader.
