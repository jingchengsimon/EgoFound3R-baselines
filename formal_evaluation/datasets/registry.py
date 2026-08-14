from __future__ import annotations

from pathlib import Path

from .h2o import H2ODatasetAdapter


DATASET_ADAPTERS = {"h2o": H2ODatasetAdapter}


def get_dataset_adapter(name: str, root: Path):
    try:
        adapter = DATASET_ADAPTERS[name.lower()]
    except KeyError as error:
        raise ValueError(f"unregistered dataset adapter: {name}") from error
    return adapter(root)
