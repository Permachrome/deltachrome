"""Geometric distortion, read from the scales a chart prints across its field.

Metamorfoze names three failure modes -- *"pincushion, where the sides of a square
show some degree of concavity"*, barrel, and *"horizontal or vertical lines shifted
relative to each other, turning squares into diamonds and circles into ovals"* --
gives one tolerance for all of them, and says only that distortion is *"measured
using defined elements in technical test targets, for both the horizontal and
vertical dimensions."*

So the measurement is: **the magnification is supposed to be the same everywhere
and in every direction; how much is it not?** A chart that prints rules across its
field answers that directly, because each rule is a local measurement of scale at a
known place along a known axis. The ISA Golden Thread prints twenty -- a centimetre
and an inch rule on all four sides of each of its five modules -- which is ten
readings per axis at five field positions.

**Why the rules and not the chart's own geometry.** Fitting the fiducials or the
patch lattice and reporting the residual would be circular here: the Golden Thread's
declared geometry was itself measured from a capture, so any distortion in that
capture is already baked into the positions being compared against. A printed rule
has a physical pitch that no fit can absorb.

**What this does and does not see.** Scale varying across the field, and scale
differing between axes, are both measured directly. Curvature *within* one rule's
span is not: a rule is short, and a comb fitted through it absorbs mild bowing into
its pitch. On a target whose rules span the frame that limit would matter; on one
printing them at five separate positions, a pincushion shows up as the corners
reading a different scale from the center, which is exactly what is measured here.
The edge-line fit residual reported beside every slanted edge is the other half of
that picture, and ``edge_fit: quadratic`` exists for it.
"""

from __future__ import annotations


import numpy as np

from deltachrome.metrics.base import (
    MetricContext,
    MetricResult,
    MetricSpec,
    PatchValue,
    evaluate_condition,
    register_metric,
    resolve_condition,
)

#: Fewest rules that can say anything. Two readings give a difference with no idea
#: of how repeatable it is; four is the least that can show a field pattern.
MIN_RULES = 4


def _percent(values: np.ndarray) -> float:
    return float((values.max() - values.min()) / values.mean() * 100.0)


def _rule_center(reading) -> tuple[float, float] | None:
    """The middle of the band a rule was read from.

    A rule is a band rather than a point, and the scale it reports belongs to the
    whole of its span -- which is the metric's stated limit: curvature *within*
    one rule is absorbed into its pitch. The center is where the reading is
    plotted, not a claim that the measurement was made at a point.
    """
    bounds = getattr(reading, "bounds_relative", None)
    if not bounds or len(bounds) != 4:
        return None
    u0, v0, u1, v1 = (float(v) for v in bounds)
    return ((u0 + u1) / 2.0, (v0 + v1) / 2.0)


def _rule_quad(reading, placement) -> list[list[float]] | None:
    """The band the ticks were found in, in image pixels.

    Declared relative to the fiducials and mapped through the placement, so it is
    the band the reading was actually taken from rather than a rectangle drawn
    around where the ticks turned out to be. A rule whose band sits off the
    printed scale is the commonest way this measurement goes wrong, and that is
    only visible by looking at the band.
    """
    bounds = getattr(reading, "bounds_relative", None)
    if placement is None or not bounds or len(bounds) != 4:
        return None
    u0, v0, u1, v1 = (float(v) for v in bounds)
    corners = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=np.float64)
    return [[float(x), float(y)] for x, y in placement.to_pixel(corners)]


def _position(region_id: str) -> str:
    parts = region_id.split("_")
    return parts[1] if len(parts) > 2 else "chart"


def _unit_corrections(readings) -> tuple[dict[str, float], str | None]:
    """A factor per unit family bringing each printed scale onto the pooled mean.

    **A chart's metric and imperial engravings disagreeing is not distortion.** This
    metric is the *variation* of magnification with place and direction, and a
    systematic offset between two units moves every rule of one family together --
    it has no place and no direction. Pooling the raw rates puts it in the answer
    anyway: on the ISA FADGI - ISO 19264 the two combs differ by about 0.75%, which
    would arrive as three quarters of a percent of pincushion on a perfect capture.
    So each family is brought onto the common mean first, and the offset is reported
    separately as what it is.

    **Unless unit and axis are confounded**, in which case the correction cannot be
    told from a real anisotropy and is refused. A chart printing millimetres only
    across and inches only down gives no way to separate "the two engravings differ"
    from "the horizontal and vertical magnifications differ", and normalising would
    silently erase the second. Every target here prints both units on both axes,
    which is what makes the correction admissible on them; the check is made from
    the readings rather than assumed.
    """
    families: dict[str, list[float]] = {}
    axes_by_unit: dict[str, set[str]] = {}
    for reading in readings:
        families.setdefault(reading.unit, []).append(float(reading.ppi))
        axes_by_unit.setdefault(reading.unit, set()).add(getattr(reading, "axis", "u"))
    if len(families) < 2:
        return {}, None

    all_axes = set().union(*axes_by_unit.values())
    confounded = [unit for unit, axes in axes_by_unit.items() if axes != all_axes]
    if confounded:
        return {}, (
            "The chart's scales disagree between units, and the disagreement was "
            f"NOT corrected for: {', '.join(sorted(confounded))} is not printed on "
            "every axis this chart carries, so a systematic difference between the "
            "engravings cannot be told apart from a real difference between the "
            "axes. Whatever the offset is, it is inside the figure below."
        )

    means = {unit: float(np.mean(values)) for unit, values in families.items()}
    pooled = float(np.mean(list(means.values())))
    return {unit: pooled / mean for unit, mean in means.items()}, None


#: What one of this metric's readings is in. See MetricResult.reading_units.
READING_UNITS = "ppi"


@register_metric("geometric_distortion")
def geometric_distortion(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    """Spread of measured scale across the field and between axes, as a percentage."""
    units = spec.units or "%"
    condition, band_notes, metadata = resolve_condition(spec, ctx)
    readings = list(ctx.extras.get("rulers") or [])
    metadata = {**metadata, "rules_read": len(readings)}

    if len(readings) < MIN_RULES:
        return MetricResult(
            metric_id=spec.metric_id, outcome="incomplete", units=units,
            condition=condition, metadata=metadata,
            reason=(
                f"Only {len(readings)} printed scale(s) could be read, fewer than the "
                f"{MIN_RULES} this needs. Distortion is the *variation* in scale "
                "across the field, so it takes several rules at different places -- "
                "one rule measures the sampling rate and says nothing about "
                "distortion. The target must declare rules spanning its field, and "
                "they must be findable in the capture."
            ),
        )

    raw = np.array([r.ppi for r in readings], dtype=np.float64)
    corrections, confounded_note = _unit_corrections(readings)
    rates = np.array(
        [r.ppi * corrections.get(r.unit, 1.0) for r in readings], dtype=np.float64
    )
    axes = np.array([getattr(r, "axis", "u") for r in readings])
    horizontal, vertical = rates[axes == "u"], rates[axes == "v"]

    field_spread = _percent(rates)
    components: dict[str, float] = {"field_spread": round(field_spread, 4)}
    value = field_spread

    if horizontal.size and vertical.size:
        anisotropy = float(
            abs(horizontal.mean() - vertical.mean()) / rates.mean() * 100.0
        )
        components["axis_anisotropy"] = round(anisotropy, 4)
        components["horizontal_mean_ppi"] = round(float(horizontal.mean()), 3)
        components["vertical_mean_ppi"] = round(float(vertical.mean()), 3)

    # Per position, both axes together: a module whose two axes disagree is the
    # local shear that turns a square into a diamond, and pooling hides it.
    per_position: dict[str, dict[str, float]] = {}
    for position in sorted({_position(r.region_id) for r in readings}):
        here = [r for r in readings if _position(r.region_id) == position]
        h = [r.ppi * corrections.get(r.unit, 1.0)
             for r in here if getattr(r, "axis", "u") == "u"]
        v = [r.ppi * corrections.get(r.unit, 1.0)
             for r in here if getattr(r, "axis", "u") == "v"]
        entry: dict[str, float] = {"rules": len(here)}
        if h:
            entry["horizontal_ppi"] = round(float(np.mean(h)), 3)
        if v:
            entry["vertical_ppi"] = round(float(np.mean(v)), 3)
        if h and v:
            entry["anisotropy_percent"] = round(
                abs(np.mean(h) - np.mean(v)) / np.mean(h + v) * 100.0, 4
            )
        per_position[position] = entry

    local = [e["anisotropy_percent"] for e in per_position.values()
             if "anisotropy_percent" in e]
    if local:
        components["worst_local_anisotropy"] = round(max(local), 4)
        value = max(value, max(local))

    # The floor. Each rule's pitch carries a standard error from its own comb fit,
    # so a spread smaller than that is the measurement, not the lens.
    errors = np.array([
        getattr(r, "standard_error_px", 0.0) / r.pitch_px * 100.0
        for r in readings if r.pitch_px
    ])
    floor = float(np.sqrt(2) * errors.mean()) if errors.size else 0.0

    from deltachrome.imaging.rulers import unit_agreement

    agreement = unit_agreement(readings)
    if len(agreement.get("by_unit", {})) > 1:
        components["unit_disagreement"] = round(agreement["disagreement"] * 100.0, 4)

    passed, threshold = evaluate_condition(value, condition) if condition else (None, None)
    metadata.update({
        "components": components,
        "per_position": per_position,
        "mean_ppi": round(float(rates.mean()), 3),
        "measurement_floor_percent": round(floor, 4),
        "threshold": threshold,
        "unit_agreement": agreement,
        "unit_correction": {u: round(c, 6) for u, c in corrections.items()},
    })

    notes = list(band_notes)
    notes.append(
        "Worst of the field spread "
        f"({components['field_spread']:.3f}%)"
        + (f" and the worst single position's anisotropy between axes "
           f"({components['worst_local_anisotropy']:.3f}%)" if local else "")
        + ". Metamorfoze gives one tolerance for pincushion, barrel and the shear "
        "that turns squares into diamonds, so the worst of them governs."
    )
    if confounded_note:
        notes.append(confounded_note)
    elif corrections:
        offset = agreement["disagreement"] * 100.0
        named = ", ".join(
            f"{unit} {entry['ppi']:.2f} ppi" for unit, entry in
            sorted(agreement["by_unit"].items())
        )
        notes.append(
            f"The chart's own scales disagree by {offset:.3f}% between units "
            f"({named}), and that offset was REMOVED before measuring anything "
            "here. It moves every rule of one unit together, so it has no place and "
            "no direction and is not distortion -- it is the chart's printing, or a "
            "wrongly declared division. Left in, it would arrive as that much "
            "pincushion on a perfect capture."
            + ("" if agreement["real"] else
               " On this capture it is inside what the comb fits themselves could "
               f"account for ({agreement['floor'] * 100.0:.3f}%), so removing it changed "
               "little.")
        )

    placement = ctx.extras.get("placement")
    return MetricResult(
        metric_id=spec.metric_id, outcome="evaluated", units=units, value=value,
        # Each reading is one printed rule's sampling rate; the verdict is the
        # variation between them, as a percent.
        reading_units=READING_UNITS,
        passed=passed, condition=condition,
        # Not spec.aggregation: the profile's field defaults to "mean", and a mean
        # of a field spread and an anisotropy is not a quantity. This metric always
        # reduces by taking the worst, because the standard gives one tolerance for
        # every form of distortion it names.
        aggregation="worst component",
        per_patch=[
            PatchValue(
                patch_id=r.region_id,
                role=f"scale_{getattr(r, 'axis', 'u')}",
                value=float(r.ppi),
                measured_lab=(0.0, 0.0, 0.0),
                reference_lab=None,
                flags=list(getattr(r, "warnings", []) or [])[:1],
                position_relative=_rule_center(r),
                roi_px=_rule_quad(r, placement),
            )
            for r in readings
        ],
        notes=notes, metadata=metadata,
    )
