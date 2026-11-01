"""Lightness uniformity, as FADGI defines it.

FADGI's tables state the units explicitly: *"Lightness Uniformity (Units
Colorimetric -- Standard Deviation Divided by Mean)"*, with thresholds of < 8% /
5% / 3% / 1% for 1- to 4-star. So the reported value is a **percentage**, the
standard deviation of measured L* across a set of nominally identical neutral
patches divided by their mean L*.

What this measures: the spread in measured L* across patches the reference file
says are the same neutral, distributed over the area the chart occupies.

What it does not measure: uniformity across the full camera frame. The chart
covers part of the frame, so the measurement covers that part. FADGI's own
GoldenThread scopes device-level target uniformity the same way, so this is
consistent with practice -- but the report says so, because "uniformity" sounds
like it should mean the whole frame.

**Which patches.** FADGI does not name them. A chart like the ColorChecker SG has
several repeated neutral families at different lightnesses, and the choice matters
enormously: standard deviation divided by mean is far harsher on a dark family than
a light one, because the denominator is small. A 0.5 L* spread is 0.5% at L*96 and
5% at L*10 -- the same physical non-uniformity, ten times the score. Defaulting to
the *lightest* family follows the convention for uniformity references (best
signal-to-noise, and what a dedicated uniformity target provides), and every
family's value is reported so nothing is hidden by the choice.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from deltachrome.metrics.base import (
    MetricContext,
    MetricResult,
    MetricSpec,
    PatchValue,
    evaluate_condition,
    register_metric,
    select_patches,
)

MIN_SAMPLES = 4
#: Reference-L* gap above which two neutrals are treated as different families.
FAMILY_GAP = 2.0


#: What one of this metric's readings is in. See MetricResult.reading_units.
READING_UNITS = "dL* from family mean"


@register_metric("lightness_uniformity")
def lightness_uniformity(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    selection = spec.patch_selection
    if selection in (None, "all", "all patches"):
        selection = {"roles": ["achromatic_border"]}
    selected = select_patches(ctx.data, selection)

    units = spec.units or "% (std dev / mean L*)"
    metadata: dict[str, Any] = {
        "patch_selection": selection,
        "sample_point_count": len(selected),
        "definition": "standard deviation of measured L* divided by mean measured L*",
    }

    if len(selected) < MIN_SAMPLES:
        return MetricResult(
            metric_id=spec.metric_id,
            outcome="incomplete",
            units=units,
            condition=spec.pass_condition,
            reason=(
                f"Only {len(selected)} uniformity patch(es) available; at least "
                f"{MIN_SAMPLES} are needed for a standard deviation to mean anything."
            ),
            metadata=metadata,
        )

    measured_L = np.array([d.measured_lab[0] for d in selected], dtype=np.float64)
    # A uniformity set does not need reference values: the metric is the spread of
    # the set about its own mean, so the set is its own reference. Some charts print
    # none for these patches -- the ISA Golden Thread's twenty registration squares
    # are the best areal-coverage sample here and carry no measured values at all.
    # What is lost without them is the split into families, which needs a nominal
    # lightness to split on, and the chart-corrected figure, which needs to know how
    # much of the spread the chart itself contributes.
    referenced = all(d.has_reference for d in selected)
    if referenced:
        reference_L = np.array([d.reference_lab[0] for d in selected], dtype=np.float64)
        families = _families(reference_L)
    else:
        reference_L = np.full(len(selected), np.nan)
        families = [np.arange(len(selected))]
    metadata["reference_values_available"] = referenced

    summaries = []
    for members in families:
        m, r = measured_L[members], reference_L[members]
        mean = float(m.mean())
        std = float(m.std(ddof=1)) if m.size > 1 else 0.0
        # The chart's own patches are not perfectly identical. Removing that known
        # spread isolates the camera's contribution; it is reported alongside the
        # literal FADGI figure rather than replacing it, because the standard asks
        # for the raw ratio.
        corrected = (m - m.mean()) - (r - r.mean())
        summaries.append(
            {
                "reference_L_mean": (
                    round(float(r.mean()), 2) if referenced else None
                ),
                "count": int(len(members)),
                "measured_L_mean": round(mean, 3),
                "measured_L_std": round(std, 4),
                "std_over_mean_pct": round(100.0 * std / mean, 4) if mean else None,
                "reference_L_std": (
                    round(float(r.std(ddof=1)) if r.size > 1 else 0.0, 4)
                    if referenced else None
                ),
                "chart_corrected_std_over_mean_pct": (
                    round(100.0 * float(corrected.std(ddof=1)) / mean, 4)
                    if referenced and mean and corrected.size > 1
                    else None
                ),
                "worst_patch_deviation_L": (
                    round(float(np.abs(corrected).max()), 3) if referenced
                    else round(float(np.abs(m - m.mean()).max()), 3)
                ),
            }
        )

    usable = [s for s in summaries if s["std_over_mean_pct"] is not None]
    if not usable:
        return MetricResult(
            metric_id=spec.metric_id,
            outcome="incomplete",
            units=units,
            condition=spec.pass_condition,
            reason="Every neutral family had a mean L* of zero; the ratio is undefined.",
            metadata=metadata,
        )

    choice = str(spec.algorithm.get("family_selection", "lightest")).lower()
    if not referenced:
        # One set, nothing to choose between. Recorded as the reason rather than
        # letting the default silently look like a decision that was made.
        choice = "single set (no reference values to split families on)"
        chosen = usable[0]
    elif choice == "worst":
        chosen = max(usable, key=lambda s: s["std_over_mean_pct"])
    elif choice == "darkest":
        chosen = min(usable, key=lambda s: s["reference_L_mean"])
    else:
        choice = "lightest"
        chosen = max(usable, key=lambda s: s["reference_L_mean"])

    value = float(chosen["std_over_mean_pct"])

    per_patch = []
    for members, summary in zip(families, summaries):
        for index in members:
            datum = selected[index]
            per_patch.append(
                PatchValue(
                    patch_id=datum.patch_id,
                    role=datum.role,
                    value=float(measured_L[index] - summary["measured_L_mean"]),
                    measured_lab=tuple(float(v) for v in datum.measured_lab),  # type: ignore[arg-type]
                    reference_lab=(
                        tuple(float(v) for v in datum.reference_lab)
                        if datum.has_reference else None
                    ),
                    flags=list(datum.flags),
                    position_relative=datum.position_relative,
                )
            )

    metadata.update(
        {
            "neutral_families": summaries,
            "family_selection": choice,
            "reported_family_reference_L": chosen["reference_L_mean"],
            "sample_distribution": _distribution(ctx, selected),
        }
    )

    notes = [
        "Reported as FADGI defines it: standard deviation of measured L* divided by "
        "mean measured L*, as a percentage.",
        "Measured over the target's areal coverage, not the full camera frame. This "
        "matches how FADGI's GoldenThread scopes device-level target uniformity.",
        f"Based on {len(selected)} sample points; see sample_distribution for where "
        "they sit on the chart.",
    ]

    passed, threshold = evaluate_condition(value, spec.pass_condition)
    if threshold is not None:
        metadata["threshold"] = threshold

    return MetricResult(
        metric_id=spec.metric_id,
        outcome="evaluated",
        units=units,
        # Each reading is a sample's signed difference from its family's mean L*,
        # not a coefficient of variation.
        reading_units=READING_UNITS,
        value=value,
        passed=passed,
        condition=spec.pass_condition,
        aggregation=spec.aggregation,
        per_patch=per_patch,
        notes=notes,
        metadata=metadata,
    )


def _families(reference_L: np.ndarray, tolerance: float = FAMILY_GAP) -> list[np.ndarray]:
    """Group sample indices by reference lightness.

    A chart's repeated neutrals cluster tightly around a few nominal values with
    wide gaps between them, so a simple gap split is sufficient and free of tuning
    parameters beyond the gap width.
    """
    order = np.argsort(reference_L)
    groups: list[list[int]] = [[int(order[0])]]
    for previous, current in zip(order, order[1:]):
        if reference_L[current] - reference_L[previous] > tolerance:
            groups.append([])
        groups[-1].append(int(current))
    return [np.array(g, dtype=int) for g in groups]


def _distribution(ctx: MetricContext, selected: list) -> dict[str, Any]:
    """Where the uniformity samples actually sit, in target-relative coordinates."""
    if not ctx.target.grid_regions:
        return {}
    points = np.array(
        [ctx.target.center_relative(d.patch) for d in selected], dtype=np.float64
    )
    hull_u = float(points[:, 0].max() - points[:, 0].min())
    hull_v = float(points[:, 1].max() - points[:, 1].min())
    distribution: dict[str, Any] = {
        "regions": sorted({d.patch.region for d in selected}),
        "spread_u": round(hull_u, 3),
        "spread_v": round(hull_v, 3),
        # Against the fiducial quad, not the card. It can exceed 1, and on the ISA
        # Golden Thread it does: the fiducials are the four corner modules' own
        # centers, so a sample set reaching past them covers more ground than the
        # quad does. Naming it "of the chart" implied a denominator that was never
        # measured and made a good result look like an impossible one.
        "covered_fraction_of_fiducial_quad": round(hull_u * hull_v, 3),
        "centroid": [round(float(points[:, 0].mean()), 3), round(float(points[:, 1].mean()), 3)],
    }

    # The perimeter/interior split describes where samples sit within their own
    # lattice, which only says anything if some lattice here has an interior. Four
    # squares at the corners of a module are a 2x2 grid, and every cell of a 2x2 is
    # on its edge -- reporting "20 on the perimeter, 0 interior" would read as a
    # finding about the chart rather than an artefact of the arithmetic.
    grids = [ctx.target.grid_for(d.patch) for d in selected]
    if any(g is not None and g.rows > 2 and g.columns > 2 for g in grids):
        perimeter = sum(1 for d in selected if ctx.target.is_perimeter(d.patch))
        distribution["on_perimeter"] = perimeter
        distribution["in_interior"] = len(selected) - perimeter
    else:
        distribution["perimeter_split"] = (
            "not meaningful: no lattice holding these samples is larger than 2x2, so "
            "every one of them is on its own grid's edge by construction"
        )
    return distribution
