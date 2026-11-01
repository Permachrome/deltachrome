"""Gain modulation â the local slope of the tone reproduction curve.

Defined in ISO/TS 19264-1:2017 Â§6.2:

    "The gain is a measure for the slope of the OECF. The gain modulation describes
    the variation of the slope in an L* original versus L* output curve. If the gain
    changes significantly it can happen that even with the tone reproduction being
    within tolerances the slope is too flat to differentiate tonal values that are
    close to each other."

That last sentence is the point of the metric, and why tone response alone is not
enough. A capture can sit inside its ÎL*2000 tolerance at every patch while the
curve flattens across part of the scale, so tones that differ in the original come
back nearly identical. Tone response measures where the curve *is*; gain modulation
measures how fast it is *moving*.

**Pairing.** ISO gives the rule: incremental gain is computed "for gray scale steps
with a tonal difference of at least 10 L* values", illustrated for a 5 L*-increment
scale as comparing patch 1 with 3, 2 with 4, and so on. Deltachrome generalises
that rather than hard-coding a step of two, because real ramps are not evenly
spaced -- a ColorChecker SG steps by roughly 8 L* and a LaserSoft ISO 12641-2 scale
by under 1 -- so each patch is paired with the nearest one at least
``min_delta_l`` away.

    gain_i = (L*measured_i - L*measured_j) / (L*reference_i - L*reference_j)

**Aim.** ISO: "gi is supposed to be 1 from L* 95 to â L* 5". FADGI turns that into
bands that tighten with star level, and splits them by where the pair sits on the
scale -- "highlight patches (average L* between 95 and 85)" get the tighter band.
"Average L*" is the mean of the pair's reference values, which is what selects a
pair into one band or the other here.

A gain below 1 means the curve is too flat there: tonal separation is being lost. A
gain above 1 means it is too steep, exaggerating differences and clipping sooner.
Both are failures, which is why the criterion is a band and not a ceiling.
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

#: ISO's minimum tonal separation between the two patches of a gain pair.
DEFAULT_MIN_DELTA_L = 10.0
#: ISO: the aim of 1 applies "from L* 95 to â L* 5".
DEFAULT_EVALUATION_RANGE = (5.0, 95.0)
#: FADGI's split: "highlight patches (average L* between 95 and 85)".
DEFAULT_HIGHLIGHT_BAND = (85.0, 95.0)


def _pairs(reference_L: np.ndarray, min_delta: float) -> list[tuple[int, int]]:
    """Pair each patch with the nearest one at least ``min_delta`` L* away.

    Taking the *nearest* qualifying patch rather than a fixed step keeps the gain
    local: pairing across a wider span would average out exactly the slope
    variation the metric exists to find.
    """
    order = np.argsort(-reference_L)
    out: list[tuple[int, int]] = []
    for a in range(len(order)):
        i = order[a]
        for b in range(a + 1, len(order)):
            j = order[b]
            if reference_L[i] - reference_L[j] >= min_delta:
                out.append((int(i), int(j)))
                break
    return out


def gain_modulation(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    band = tuple(spec.algorithm.get("l_star_range", DEFAULT_HIGHLIGHT_BAND))
    is_highlight = "highlight" in spec.metric_id
    min_delta = float(spec.algorithm.get("min_delta_l", DEFAULT_MIN_DELTA_L))
    lo, hi = spec.algorithm.get("evaluation_range", DEFAULT_EVALUATION_RANGE)

    selection = spec.patch_selection
    if selection in (None, "all", "all patches"):
        selection = {"roles": ["achromatic_ramp"]}
    selected = [d for d in select_patches(ctx.data, selection) if d.has_reference]

    metadata: dict[str, Any] = {
        "definition": "ISO/TS 19264-1:2017 6.2 incremental gain from the TRC",
        "min_delta_l": min_delta,
        "evaluation_range_l": [lo, hi],
        "band": "highlight" if is_highlight else "other",
        "highlight_band_l": list(band),
        "n_ramp_patches": len(selected),
    }

    if len(selected) < 2:
        return MetricResult(
            metric_id=spec.metric_id, outcome="incomplete", units=spec.units or "gain",
            condition=spec.pass_condition, metadata=metadata,
            reason="Gain needs at least two neutral ramp patches to form a pair.",
        )

    measured = np.array([d.measured_lab[0] for d in selected], dtype=np.float64)
    reference = np.array([d.reference_lab[0] for d in selected], dtype=np.float64)

    pairs = _pairs(reference, min_delta)
    if not pairs:
        return MetricResult(
            metric_id=spec.metric_id, outcome="incomplete", units=spec.units or "gain",
            condition=spec.pass_condition, metadata=metadata,
            reason=(
                f"No two neutral steps are {min_delta:g} L* apart on this target."
            ),
        )

    per_patch: list[PatchValue] = []
    out_of_range = 0
    for i, j in pairs:
        # Where a pair sits on the scale is judged by its mean reference L*. That is
        # FADGI's own convention -- it defines the tighter band as "highlight patches
        # (average L* between 95 and 85)" -- so ISO's "aim of 1 from L* 95 to = L* 5"
        # is applied the same way, for consistency. Testing each patch individually
        # instead would drop any pair containing the chart's brightest patch, which
        # on a ColorChecker SG sits just above L* 95 and is exactly the patch the
        # highlight band exists to exercise.
        mean_l = (reference[i] + reference[j]) / 2.0
        if not (lo <= mean_l <= hi):
            out_of_range += 1
            continue
        in_highlight = band[0] <= mean_l <= band[1]
        if in_highlight != is_highlight:
            continue
        denominator = reference[i] - reference[j]
        if abs(denominator) < 1e-9:
            continue
        gain = float((measured[i] - measured[j]) / denominator)
        per_patch.append(
            PatchValue(
                patch_id=f"{selected[i].patch_id}-{selected[j].patch_id}",
                role=selected[i].role,
                value=gain,
                measured_lab=tuple(float(v) for v in selected[i].measured_lab),  # type: ignore[arg-type]
                reference_lab=tuple(float(v) for v in selected[i].reference_lab),  # type: ignore[arg-type]
                flags=sorted(set(selected[i].flags) | set(selected[j].flags)),
            )
        )

    metadata["n_pairs_formed"] = len(pairs)
    metadata["n_pairs_outside_evaluation_range"] = out_of_range
    metadata["n_pairs_in_band"] = len(per_patch)

    if not per_patch:
        return MetricResult(
            metric_id=spec.metric_id, outcome="incomplete", units=spec.units or "gain",
            condition=spec.pass_condition, metadata=metadata,
            reason=(
                f"No neutral pair averages L* {band[0]:g}–{band[1]:g} on this target; "
                f"its ramp tops out at L* {reference.max():.1f}."
                if is_highlight else
                f"No neutral pair falls outside L* {band[0]:g}–{band[1]:g} "
                "on this target."
            ),
        )

    gains = np.array([pv.value for pv in per_patch])
    # A band criterion fails in either direction, so the representative value is the
    # gain furthest from the aim of 1.0 -- reporting a mean would hide a local
    # flat spot inside an otherwise well-behaved curve.
    worst = per_patch[int(np.argmax(np.abs(gains - 1.0)))]
    value = worst.value

    metadata.update({
        "worst_pair": worst.patch_id,
        "gain_min": float(gains.min()),
        "gain_max": float(gains.max()),
        "gain_mean": float(gains.mean()),
    })

    notes = [
        f"Incremental gain over {len(per_patch)} patch pair(s), each separated by at "
        f"least {min_delta} L* in the reference, per ISO/TS 19264-1 6.2. The aim is 1.0.",
        f"Reported value is the gain furthest from 1.0 ({worst.patch_id}); a mean would "
        "hide a local flat spot in an otherwise well-behaved curve.",
    ]

    passed, threshold = evaluate_condition(value, spec.pass_condition)
    if threshold is not None:
        metadata["nearest_bound"] = threshold

    return MetricResult(
        metric_id=spec.metric_id,
        outcome="evaluated",
        units=spec.units or "gain",
        value=value,
        passed=passed,
        condition=spec.pass_condition,
        aggregation=spec.aggregation or "worst_case",
        per_patch=per_patch,
        notes=notes,
        metadata=metadata,
    )


for _metric_id in ("gain_modulation_highlight", "gain_modulation_other"):
    register_metric(_metric_id)(gain_modulation)
