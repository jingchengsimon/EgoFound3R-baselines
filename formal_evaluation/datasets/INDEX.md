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
