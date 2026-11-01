"""Per-patch difference metrics: color accuracy, white balance, tone response.

These four FADGI metrics share one shape -- compute a difference per patch, then
aggregate -- and differ only in which difference formula and which patches. They
are therefore one implementation registered under several ids, rather than four
near-identical plugins that would drift apart.

The distinction FADGI actually draws, which a generic "max and average" model
would lose:

* Average Color Accuracy is the mean Delta E2000 over all patches.
* Color Accuracy 90th Percentile is the 90th percentile, *not* the maximum.
* White Balance is Delta E(a*b*) with L* excluded, on the gray patches only, and
  FADGI's wording is per-patch ("for any given gray patch") -- so the worst patch
  must clear the threshold, not the average of them.
* Tone Response is stated in Delta L*2000, which includes CIEDE2000's S_L
  weighting, not plain |dL*|.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from deltachrome.errors import ProfileError
from deltachrome.imaging.color import (
    delta_e,
    delta_e_2000_ab,
    delta_e_ab,
    delta_L,
    delta_L_2000,
)
from deltachrome.metrics.base import (
    DEFAULT_PERCENTILE_METHOD,
    MetricContext,
    MetricResult,
    MetricSpec,
    PatchValue,
    aggregate,
    evaluate_condition,
    register_metric,
    select_patches,
)

MEASURES = {
    "delta_e_2000": lambda s, r: delta_e(s, r, "delta_e_2000"),
    "delta_e_76": lambda s, r: delta_e(s, r, "delta_e_76"),
    "delta_e_94": lambda s, r: delta_e(s, r, "delta_e_94"),
    "delta_e_ab": delta_e_ab,
    "delta_e_2000_ab": delta_e_2000_ab,
    "delta_l_2000": delta_L_2000,
    "delta_l": lambda s, r: np.abs(delta_L(s, r)),
}

MEASURE_UNITS = {
    "delta_e_2000": "dE2000",
    "delta_e_76": "dE76",
    "delta_e_94": "dE94",
    "delta_e_ab": "dE(a*b*)",
    "delta_e_2000_ab": "dE2000(a*b*)",
    "delta_l_2000": "dL*2000",
    "delta_l": "dL*",
}


def _measure_name(spec: MetricSpec) -> str:
    """Resolve which difference formula a profile entry asks for."""
    algorithm = spec.algorithm
    raw = algorithm.get("measure") or algorithm.get("delta_e_variant")
    if raw is None:
        raise ProfileError(
            f"Metric {spec.metric_id!r} does not name a difference formula. Set "
            '"delta_e_variant" or "measure" in its algorithm block; supported: '
            f"{sorted(MEASURES)}."
        )
    name = str(raw).strip().lower()
    if name not in MEASURES:
        raise ProfileError(
            f"Metric {spec.metric_id!r} asks for unknown difference formula {raw!r}. "
            f"Supported: {sorted(MEASURES)}."
        )
    return name


def _color_accuracy_regions(ctx: MetricContext, spec: MetricSpec) -> set[str]:
    """The regions a target says its color patches are, where "all patches" means them.

    FADGI's color accuracy is "the average deviation of all color patches measured".
    On most charts the neutral ramp sits inside the color array and is part of it.
    The UTT prints its four grey scales apart from its two color bars, and they are
    80 of its 134 patches, so counting them would make the figure mostly a grey-scale
    score. OpenDICE and iQ-Analyzer both judge its 54 color patches only. Which
    patches those are is a fact about the chart, so the target declares it under
    ``metric_defaults.color_accuracy.uses_regions``. It applies only to a guideline
    asking for all patches; one naming roles or ids gets exactly what it named.
    """
    if spec.requires_capability != "color_accuracy":
        return set()
    selection = spec.patch_selection
    if not (isinstance(selection, str) and selection.strip().lower() in {"all", "all patches", "*"}):
        return set()
    declared = (ctx.target.metric_defaults.get("color_accuracy") or {}).get("uses_regions")
    return set(declared or ())


def patch_difference_metric(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    """Compute a per-patch difference, aggregate it, and judge it."""
    measure_name = _measure_name(spec)
    measure = MEASURES[measure_name]

    selected = select_patches(ctx.data, spec.patch_selection)
    scope = _color_accuracy_regions(ctx, spec)
    if scope:
        selected = [d for d in selected if d.patch.region in scope]
    # A patch carried without a reference value cannot take part in a comparison
    # against one. Dropped here rather than upstream, because the same patch is a
    # legitimate uniformity sample.
    without_reference = [d.patch_id for d in selected if not d.has_reference]
    selected = [d for d in selected if d.has_reference]
    notes: list[str] = []
    metadata: dict[str, Any] = {
        "measure": measure_name,
        "patch_selection": spec.patch_selection,
        "excluded_without_reference": without_reference,
        "n_selected": len(selected),
    }
    if scope:
        metadata["regions"] = sorted(scope)

    if not selected:
        return MetricResult(
            metric_id=spec.metric_id,
            outcome="incomplete",
            units=spec.units or MEASURE_UNITS[measure_name],
            condition=spec.pass_condition,
            aggregation=spec.aggregation,
            reason=(
                f"No patches matched {spec.patch_selection!r}. Either the target has "
                "no patches in that role, or role resolution could not classify them "
                "from the reference data."
            ),
            metadata=metadata,
        )

    per_patch: list[PatchValue] = []
    for datum in selected:
        value = float(measure(datum.measured_lab, datum.reference_lab))
        per_patch.append(
            PatchValue(
                patch_id=datum.patch_id,
                role=datum.role,
                value=value,
                measured_lab=tuple(float(v) for v in datum.measured_lab),  # type: ignore[arg-type]
                reference_lab=tuple(float(v) for v in datum.reference_lab),  # type: ignore[arg-type]
                flags=list(datum.flags),
                position_relative=datum.position_relative,
            )
        )

    percentile_method = spec.algorithm.get("percentile_method", DEFAULT_PERCENTILE_METHOD)
    value, agg_meta = aggregate(
        [pv.value for pv in per_patch],
        spec.aggregation,
        percentile_method=percentile_method,
    )
    metadata.update(agg_meta)

    worst = max(per_patch, key=lambda pv: pv.value)
    metadata["worst_patch"] = worst.patch_id
    metadata["worst_value"] = worst.value

    flagged = [pv.patch_id for pv in per_patch if pv.flags]
    if flagged:
        metadata["flagged_patches"] = flagged
        notes.extend(_flag_notes(per_patch, flagged))


    passed, threshold = evaluate_condition(value, spec.pass_condition)
    if threshold is not None:
        metadata["threshold"] = threshold

    return MetricResult(
        metric_id=spec.metric_id,
        outcome="evaluated",
        units=spec.units or MEASURE_UNITS[measure_name],
        value=value,
        passed=passed,
        condition=spec.pass_condition,
        aggregation=spec.aggregation,
        per_patch=per_patch,
        notes=notes,
        metadata=metadata,
    )


for _metric_id in (
    "average_color_accuracy",
    # Metamorfoze states color accuracy as a mean *and a maximum*, where FADGI
    # states a mean and a 90th percentile. Its own id so a report names which is
    # being read rather than showing two rows called the same thing.
    "color_accuracy_max",
    "color_accuracy_90th_percentile",
    "white_balance",
    "tone_response_oecf",
    "patch_difference",
):
    register_metric(_metric_id)(patch_difference_metric)


#: What each sampling flag means for a colour comparison, and what to do about it.
#: Separated because the remedies are different and a note that lumps them together
#: sends the operator to the wrong control: overexposure and a gamut overflow both
#: arrive as "clipping", and one is fixed at the light and the other at the working
#: space.
FLAG_EXPLANATIONS: tuple[tuple[str, str], ...] = (
    (
        "gamut_clipped",
        "a channel is held at the encoding floor while the patch is bright in "
        "another -- the color is more saturated than the capture's working space "
        "can encode, so its value is truncated and its chroma understated. Not an "
        "exposure fault: re-shoot in a wider space.",
    ),
    (
        "channel_floor",
        "a channel is held at the floor while the patch is bright in another, in a "
        "capture still encoded in the camera's or scanner's own profile. There is no "
        "wider space to move to: that channel received almost no light, so its value "
        "is truncated. More exposure lifts it only if the brightest channel has room.",
    ),
    (
        "clipped_high",
        "pixels at the top of the encoding range. The highlight is not recorded, so "
        "the patch reads darker than it is. Reduce exposure.",
    ),
    (
        "clipped_low",
        "pixels at the bottom of the encoding range with nothing bright in the "
        "patch -- the shadow is not recorded. Increase exposure.",
    ),
    (
        "mean_median_divergence",
        "the ROI's mean and median disagree, which is dust, a scratch or a specular "
        "hit rather than the patch's own color. Clean and re-shoot.",
    ),
)


def _flag_notes(per_patch, flagged) -> list[str]:
    """One line per KIND of sampling flag, with what it means and what to do.

    Each patch is attributed to one kind only, and ``gamut_clipped`` and
    ``channel_floor`` are tested first: each always accompanies ``clipped_low`` --
    a refinement of it, not a separate event -- so counting both would
    double-count the same patch and report a shadow problem that is not there.
    """
    notes = [
        f"{len(flagged)} of {len(per_patch)} contributing patches carry sampling "
        "flags. Their values are INCLUDED in the figure above and should be treated "
        "as suspect; what each flag means:"
    ]
    seen: set[str] = set()
    for name, explanation in FLAG_EXPLANATIONS:
        ids = [
            pv.patch_id
            for pv in per_patch
            if pv.patch_id not in seen
            and any(f.split("(")[0] == name for f in pv.flags)
        ]
        if not ids:
            continue
        seen.update(ids)
        shown = ", ".join(sorted(ids)[:8]) + (" ..." if len(ids) > 8 else "")
        notes.append(f"  {len(ids)} {name} -- {explanation} ({shown})")
    return notes
