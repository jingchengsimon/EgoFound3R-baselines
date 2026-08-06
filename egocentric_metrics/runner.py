"""Python-only metric selection and evaluation entry point."""

from __future__ import annotations

from .placeholders import implemented_result, placeholder_result
from .registry import METRIC_REGISTRY


def available_metrics(*, include_placeholders: bool = True) -> tuple[str, ...]:
    """Return stable metric names available to ``evaluate``."""
    return tuple(
        name for name, spec in METRIC_REGISTRY.items()
        if include_placeholders or spec.compute is not None
    )


def evaluate(
    inputs: dict[str, object],
    metrics: list[str] | tuple[str, ...],
    config: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Evaluate selected named metrics using caller-provided in-memory inputs."""
    settings = {} if config is None else dict(config)
    results: dict[str, dict[str, object]] = {}
    for name in metrics:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"unknown metric: {name}")
        spec = METRIC_REGISTRY[name]
        if spec.compute is None:
            results[name] = placeholder_result(spec.placeholder_reason or "metric protocol is unresolved")
            continue
        missing = [key for key in spec.required_inputs if key not in inputs]
        if missing:
            raise ValueError(f"metric '{name}' requires missing input(s): {', '.join(missing)}")
        results[name] = implemented_result(spec.compute(inputs, settings))
    return results
