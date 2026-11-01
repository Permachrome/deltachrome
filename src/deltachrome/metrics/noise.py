"""Noise, as the standard deviation of L* within a flat patch.

FADGI's tables give two limits: an upper one that tightens with star level (< 4 /
3 / 2 / 1), and a lower one fixed at 0.25 across every level, described as
*"a warning should be raised if the image doesn't meet this criteria"*.

The lower limit is not a quality floor -- it is a tell. A capture with almost no
measurable noise has usually had noise reduction applied, which destroys fine
detail the digitisation was meant to preserve. So it is reported as a warning
rather than a failure, exactly as the guidelines describe it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from deltachrome.metrics.base import (
    MetricContext,
    MetricResult,
    MetricSpec,
    PatchValue,
    aggregate,
    evaluate_condition,
    register_metric,
    select_patches,
)

DEFAULT_LOWER_LIMIT = 0.25


@register_metric("noise")
def noise(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    sampling = ctx.extras.get("sampling")
    if sampling is None:
        return MetricResult(
            metric_id=spec.metric_id,
            outcome="incomplete",
            units=spec.units or "std dev of L*",
            condition=spec.pass_condition,
            reason="No sampling result was supplied to the metric context.",
        )

    selection = spec.patch_selection
    if selection in (None, "all", "all patches"):
        selection = {"roles": ["achromatic_border", "achromatic_ramp", "achromatic_other"]}
    selected = select_patches(ctx.data, selection)

    per_patch: list[PatchValue] = []
    for datum in selected:
        sample = sampling.samples.get(datum.patch_id)
        if sample is None or sample.lab_std is None or not np.isfinite(sample.L_std):
            continue
        per_patch.append(
            PatchValue(
                patch_id=datum.patch_id,
                role=datum.role,
                value=sample.L_std,
                measured_lab=tuple(float(v) for v in datum.measured_lab),  # type: ignore[arg-type]
                reference_lab=(
                    tuple(float(v) for v in datum.reference_lab)
                    if datum.has_reference else None
                ),
                flags=list(sample.flags),
                position_relative=datum.position_relative,
            )
        )

    metadata: dict[str, Any] = {"patch_selection": selection, "n_selected": len(per_patch)}

    if not per_patch:
        return MetricResult(
            metric_id=spec.metric_id,
            outcome="incomplete",
            units=spec.units or "std dev of L*",
            condition=spec.pass_condition,
            reason=(
                "No flat neutral patch produced a usable per-pixel L* spread. Noise is "
                "measured inside uniform patches, so the target must declare some."
            ),
            metadata=metadata,
        )

    value, agg_meta = aggregate(
        [pv.value for pv in per_patch], spec.aggregation or "max"
    )
    metadata.update(agg_meta)

    worst = max(per_patch, key=lambda pv: pv.value)
    quietest = min(per_patch, key=lambda pv: pv.value)
    metadata["noisiest_patch"] = worst.patch_id
    metadata["quietest_patch"] = quietest.patch_id
    metadata["quietest_value"] = quietest.value

    notes = [
        "Standard deviation of L* inside each flat patch, computed on color-managed "
        "pixels rather than device values.",
    ]

    lower = spec.algorithm.get("lower_limit", DEFAULT_LOWER_LIMIT)
    if lower is not None and quietest.value < float(lower):
        metadata["below_lower_limit"] = True
        notes.append(
            "Noise below FADGI's lower limit; noise reduction likely applied."
        )
    else:
        metadata["below_lower_limit"] = False

    passed, threshold = evaluate_condition(value, spec.pass_condition)
    if threshold is not None:
        metadata["threshold"] = threshold

    return MetricResult(
        metric_id=spec.metric_id,
        outcome="evaluated",
        units=spec.units or "std dev of L*",
        value=value,
        passed=passed,
        condition=spec.pass_condition,
        aggregation=spec.aggregation or "max",
        per_patch=per_patch,
        notes=notes,
        metadata=metadata,
    )
