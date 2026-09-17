# Reusable methods

This document is a human-readable catalog of reusable project modules. It
describes what each module does; operational commands and environment details
remain in the runtime registry and setup documentation.

## Adapter architecture

### Dataset Adapter

Provides a common, read-only view of a dataset: splits, sequence and frame
identity, original RGB paths, and basic input metadata such as resolution. It
does not resize source files or prepare model-specific tensors.

### Method Adapter

Converts the common dataset view into each baseline's official input and turns
native predictions into the project's evaluation format. Model-specific
resize, normalization, camera conversion, temporal windows, and chunking live
here.

This separation lets new datasets reuse every existing baseline integration,
while each baseline keeps the preprocessing required by its official method.

## Dataset adapters

### H2O

Discovers the H2O test split, selects reproducible contiguous sequences, and
returns original frame paths, frame IDs, and per-frame resolution metadata. It
also exposes mixed-resolution inputs without modifying the dataset.

Index: [`formal_evaluation/datasets/INDEX.md`](formal_evaluation/datasets/INDEX.md)

## Maintenance

When a new reusable module is explicitly accepted as part of the project,
add a short entry here describing its main purpose. Keep implementation paths
in an index when precise lookup is useful; avoid copying detailed setup steps
into this document.
