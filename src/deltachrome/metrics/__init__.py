"""Metric plugins.

Importing this package registers the built-in metrics. New metric families are
added by dropping a module here that calls ``register_metric``.
"""

from deltachrome.metrics import (  # noqa: F401  (imported for their side effect)
    conformance,
    difference,
    distortion,
    flatfield,
    gain,
    noise,
    registration,
    scale,
    sfr,
    tolerance,
    uniformity,
)
from deltachrome.metrics.base import (
    MetricContext,
    MetricResult,
    MetricSpec,
    PatchDatum,
    PatchValue,
    available_metrics,
    get_metric,
    register_metric,
)

__all__ = [
    "MetricContext",
    "MetricResult",
    "MetricSpec",
    "PatchDatum",
    "PatchValue",
    "available_metrics",
    "get_metric",
    "register_metric",
]
