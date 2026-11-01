"""Reproduction scale accuracy: does the file's claimed resolution match the capture?

FADGI states this one as "% Difference from Header PPI", which makes it the only
metric in the set that compares a measurement against *metadata* rather than
against a reference measurement or a threshold. That is the point of it. A header
resolution can be set to anything -- defaulted by an export dialog, carried over
from an earlier step, or simply wrong -- and everything downstream that reasons
about physical size trusts it: print output, on-screen scale, any measurement made
from the file, and the catalogue record that says what was digitised at what
resolution.

So the comparison is between the rate measured off the chart's own printed ruler
and the rate the file claims. They should agree; the metric exists because they
sometimes do not, and nothing else in a workflow would notice.

Reported as an **absolute** percentage, because the criteria are written two-sided
("<+/- 2%") while the profile conditions that encode them are one-sided
(``value < 2``). The signed difference is kept in the metadata, since which way the
file is wrong is what tells you where the mistake was made.
"""

from __future__ import annotations

from typing import Any

from deltachrome.imaging.rulers import sampling_rate
from deltachrome.metrics.base import (
    MetricContext,
    MetricResult,
    MetricSpec,
    evaluate_condition,
    register_metric,
)

METRIC_ID = "reproduction_scale_accuracy"

#: Disagreement between a file's X and Y resolution above which the pixels are not
#: square and one number cannot describe the scale.
ANISOTROPY_TOLERANCE = 0.001


def _incomplete(spec: MetricSpec, reason: str, metadata: dict[str, Any]) -> MetricResult:
    return MetricResult(
        metric_id=spec.metric_id, outcome="incomplete", units=spec.units,
        condition=spec.pass_condition, reason=reason, metadata=metadata,
    )


@register_metric(METRIC_ID)
def reproduction_scale_accuracy(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    units = spec.units or "% difference from header ppi"
    image = ctx.extras.get("image")
    readings = ctx.extras.get("rulers") or []
    metadata: dict[str, Any] = {"rulers_read": len(readings)}

    if not readings:
        return _incomplete(
            spec,
            f"No printed scale was read from this capture. {ctx.target.display_name} "
            "must declare a scale region with bounds and a division size, and the "
            "ruler has to be findable in the image. Without a measured rate there is "
            "nothing to compare the header against -- and comparing the header "
            "against itself would pass everything.",
            metadata,
        )

    measured = sampling_rate(readings)
    metadata["measured"] = measured
    if measured.get("warning"):
        metadata["ruler_disagreement"] = measured["warning"]

    header = getattr(image, "header_ppi", None) if image is not None else None
    if not header:
        return _incomplete(
            spec,
            "The file records no resolution in its header, so there is nothing to "
            f"check the measured {measured['ppi']:.1f} ppi against. That is not the "
            "same as the scale being right: a master with no stated resolution "
            "leaves every downstream consumer to guess.",
            metadata,
        )

    header_x, header_y = float(header[0]), float(header[1])
    metadata["header_ppi"] = {"x": header_x, "y": header_y}
    notes: list[str] = []
    if abs(header_x - header_y) > ANISOTROPY_TOLERANCE * max(header_x, header_y):
        notes.append(
            f"The file describes non-square pixels: {header_x:.1f} × "
            f"{header_y:.1f} ppi."
        )
    claimed = (header_x + header_y) / 2.0
    signed = 100.0 * (measured["ppi"] - claimed) / claimed
    value = abs(signed)

    metadata.update({
        "measured_ppi": round(measured["ppi"], 3),
        "claimed_ppi": round(claimed, 3),
        "signed_difference_pct": round(signed, 3),
        "ruler_agreement": measured.get("agreement"),
    })

    notes.append(
        f"Measured {measured['ppi']:.1f} ppi from "
        + ", ".join(
            f"{name} ({entry['divisions_used']} divisions of "
            f"{entry['division_mm']:g} mm, fit residual {entry['fit_residual_px']:.2f} px)"
            for name, entry in measured["rulers"].items()
        )
        + f"; the file's header claims {claimed:.1f} ppi."
    )
    notes.append(
        "Reported as an absolute percentage, because the criterion is two-sided "
        f"(+/-) while the profile states it as a ceiling. Signed: {signed:+.2f}%, so "
        + ("the capture resolves finer than the file claims."
           if signed > 0 else "the capture resolves coarser than the file claims.")
    )
    if value > 50.0:
        notes.append(
            "File's stated resolution is far from the chart's scale. "
            "Check file metadata."
        )

    passed, threshold = evaluate_condition(value, spec.pass_condition)
    if threshold is not None:
        metadata["threshold"] = threshold

    return MetricResult(
        metric_id=spec.metric_id, outcome="evaluated", units=units, value=value,
        passed=passed, condition=spec.pass_condition,
        aggregation=spec.aggregation or "absolute difference",
        notes=notes, metadata=metadata,
    )
