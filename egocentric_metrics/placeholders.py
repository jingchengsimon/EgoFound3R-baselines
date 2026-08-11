"""Explicit results for metrics requiring an unresolved external protocol."""

from __future__ import annotations


def placeholder_result(reason: str) -> dict[str, object]:
    return {"status": "placeholder", "value": None, "reason": reason}


def implemented_result(value: object) -> dict[str, object]:
    return {"status": "implemented", "value": value}
