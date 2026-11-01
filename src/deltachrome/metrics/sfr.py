"""Spatial frequency response metrics, from slanted edges.

Four quantities are read off the same measured curve, and they answer different
questions:

``sfr10_sampling_efficiency``  where the response dies, relative to the sampling
                              grid's own limit. How much of the resolution the
                              pixel count promises is actually delivered.
``sfr50``                      where the response halves. Tracks perceived
                              sharpness better than the 10% point.
``sfr_response_at_nyquist``    how much survives *at* the limit. High values mean
                              detail above the limit is aliasing into the image.
``sharpening``                 the peak of the curve. Above 1.0 the system is
                              boosting contrast it did not capture.

Each is measured on every available edge and reduced to the **worst** one, because
a system that is sharp in the center and soft in the corners is soft. Which
direction "worst" lies in depends on the criterion -- higher is better for sampling
efficiency, lower for response at Nyquist -- so it is derived from the profile's
own pass condition rather than assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from deltachrome.errors import ProfileError
from deltachrome.imaging.sfr import (
    DEFAULT_PEAK_SEARCH_MIN_FREQUENCY,
    LEAKAGE_SUSPECT_FREQUENCY,
    SFRError,
    compute_sfr,
)
from deltachrome.metrics.resolution import resolve_sampling_ppi
from deltachrome.metrics.base import (
    MetricContext,
    MetricResult,
    MetricSpec,
    PatchValue,
    evaluate_condition,
    register_metric,
    rescale_condition,
)
from deltachrome.text import plural

_UPPER_BOUND = re.compile(r"value\s*<")
_LOWER_BOUND = re.compile(r"value\s*>")

READINGS: dict[str, Callable[[Any], float | None]] = {
    "sfr10_sampling_efficiency": lambda r: r.sampling_efficiency(0.10),
    "sfr50": lambda r: r.sfr50(),
    # Native cycles per pixel, unlike ``sfr50`` above, which is already divided by
    # Nyquist. Kept separate so the unit conversion has an unscaled quantity to
    # start from -- see FREQUENCY_UNITS.
    "mtf50": lambda r: r.frequency_at(0.50),
    "sfr_response_at_nyquist": lambda r: r.response_at_nyquist(),
    "sharpening": lambda r: r.peak_modulation(),
}

#: What MTF50 can be expressed in, and what each costs.
#:
#: The measurement is a frequency in cycles per pixel and nothing else; every other
#: unit is that number times a property of the capture. ``cycles/px`` and the
#: Nyquist percentage need nothing, so they can always be reported. Anything per
#: millimetre or per inch needs a **sampling rate at the target plane**, which means
#: a chart that carries a scale or an extent someone measured -- and a run that
#: cannot establish one reports incomplete rather than guessing.
#:
#: One line pair is one cycle, so lp/mm and cycles/mm are the same number under two
#: names. Both are accepted because standards and lens data sheets use different
#: ones for the same quantity.
FREQUENCY_UNITS: dict[str, str | float] = {
    "cycles/px": 1.0,
    "% of half-sampling freq": 200.0,   # f / 0.5 * 100
    "lp/mm": "per_mm",
    "cycles/mm": "per_mm",
    "lp/inch": "per_inch",
    "cycles/inch": "per_inch",
}

#: Metamorfoze states its resolution requirement in lp/mm, which is the reason this
#: metric exists, so that is what it reports unless asked otherwise.
DEFAULT_FREQUENCY_UNITS = "lp/mm"

#: How several edges are combined. Neither FADGI nor ISO 19264-1 says: both state
#: per-edge criteria and stop. The default follows OpenDICE, the open FADGI-aligned
#: implementation, which displays "the RGB average sampling efficiency values of all
#: the edge regions" -- so a Deltachrome figure lines up with a GoldenThread report
#: of the same capture rather than needing conversion.
#:
#: ``worst`` is the more conservative gate and is one setting away; whichever is
#: chosen, worst, mean, median and spread are all reported, because a mean can pass a
#: system with one badly soft corner and the spread is what shows it.
EDGE_AGGREGATIONS = ("worst", "mean", "median")
DEFAULT_EDGE_AGGREGATION = "mean"

#: Reported in the units the standard states them in, not in whatever the maths
#: produced. FADGI gives sampling efficiency as "Measurement is a Ratio %" with
#: thresholds like "> 90%", and SFR50 as a "Percentage of Half Sampling Frequency";
#: both come off the curve as fractions and are scaled here. Response at Nyquist
#: and sharpening are stated as modulation and stay as they are.
SCALE = {
    "sfr10_sampling_efficiency": 100.0,
    "sfr50": 100.0,
    "mtf50": 1.0,          # replaced per run by the chosen unit -- see _unit_scale
    "sfr_response_at_nyquist": 1.0,
    "sharpening": 1.0,
}

UNITS = {
    "sfr10_sampling_efficiency": "% of Nyquist",
    "sfr50": "% of half-sampling freq",
    "mtf50": DEFAULT_FREQUENCY_UNITS,
    "sfr_response_at_nyquist": "modulation",
    "sharpening": "max modulation",
}


def _worst(values: np.ndarray, condition: str | None) -> int:
    """Index of the least favourable edge, given what the criterion asks for."""
    if condition and _LOWER_BOUND.search(condition) and "<= value" not in condition:
        return int(np.argmin(values))          # higher is better
    if condition and _UPPER_BOUND.search(condition) and "value <" in condition:
        return int(np.argmax(values))          # lower is better
    if condition and "<= value <=" in condition:
        center = np.mean([float(x) for x in re.findall(r"-?\d+\.?\d*", condition)])
        return int(np.argmax(np.abs(values - center)))
    return int(np.argmax(values))


def _settings(spec: MetricSpec) -> dict[str, Any]:
    """Map a profile's algorithm block onto the analyzer's parameters."""
    algorithm = spec.algorithm or {}
    settings: dict[str, Any] = {}
    if algorithm.get("linearize") is False:
        settings["_skip_linearize"] = True
    if "edge_fit" in algorithm:
        settings["edge_fit"] = algorithm["edge_fit"]
    if "esf_supersample_bin_width" in algorithm:
        settings["supersample_bin_width"] = float(algorithm["esf_supersample_bin_width"])
    if "hamming_window" in algorithm:
        settings["window"] = "hamming" if algorithm["hamming_window"] else "none"
    if "window" in algorithm:
        settings["window"] = algorithm["window"]
    for key in ("derivative", "derivative_correction", "centroid_threshold", "detrend",
                "channel", "binning", "integer_phase_cycles"):
        if key in algorithm:
            settings[key] = algorithm[key]
    return settings

@dataclass(frozen=True)
class UnitScale:
    """How a frequency reading and its criterion are carried into one unit.

    Two factors, not one, and the difference is the whole point. ``scale`` takes
    the measurement from cycles per pixel into the unit being reported.
    ``condition_scale`` takes the *profile's threshold* from the unit the profile
    declared into that same unit. When nobody overrides anything the two units are
    the same and ``condition_scale`` is exactly 1.0; when a run asks for a
    different unit, both sides move together and the verdict cannot change.
    """

    scale: float
    units: str
    condition_scale: float
    notes: list[str]
    metadata: dict[str, Any]
    reason: str | None


def _resolve_unit(units: str) -> float | str:
    if units not in FREQUENCY_UNITS:
        raise ProfileError(
            f"Unknown frequency_units {units!r}; expected one of "
            f"{sorted(FREQUENCY_UNITS)}"
        )
    return FREQUENCY_UNITS[units]


def _unit_scale(ctx: MetricContext, spec: MetricSpec) -> UnitScale:
    """The factors taking cycles-per-pixel and the profile's threshold into the
    unit this run reports in.

    Only MTF50 has a choice to make; every other reading is already dimensionless.
    """
    if spec.metric_id != "mtf50":
        return UnitScale(
            SCALE[spec.metric_id], spec.units or UNITS[spec.metric_id],
            1.0, [], {}, None,
        )

    algorithm = spec.algorithm or {}
    #: What the profile itself states the criterion in. Never overwritten by a run
    #: override -- the threshold is written in this unit and has to be converted
    #: *from* it, so losing it is what made the override change verdicts.
    declared = str(
        algorithm.get("frequency_units") or spec.units or DEFAULT_FREQUENCY_UNITS
    )
    requested = str(algorithm.get("frequency_units_override") or declared)
    declared_factor = _resolve_unit(declared)
    requested_factor = _resolve_unit(requested)

    metadata: dict[str, Any] = {"frequency_units": requested}
    if requested != declared:
        metadata["frequency_units_declared"] = declared

    # The threshold only has to be converted when there is one to convert. A
    # profile that states no criterion can be re-expressed in any unit for free.
    value_needs_rate = isinstance(requested_factor, str)
    condition_needs_rate = (
        isinstance(declared_factor, str)
        and spec.pass_condition is not None
        and requested != declared
    )

    rate = None
    if value_needs_rate or condition_needs_rate:
        rate, reason = resolve_sampling_ppi(ctx)
        if rate is None:
            if value_needs_rate:
                why = (
                    f"MTF50 in {requested} is a frequency per unit of physical "
                    "distance, so it needs the capture's sampling rate at the target "
                    f"plane, and that could not be established. {reason} "
                    "Alternatively, set frequency_units to 'cycles/px', which needs "
                    "no scale -- but it is not the quantity the requirement is "
                    "written in."
                )
            else:
                why = (
                    f"This run asked for MTF50 in {requested}, but the profile states "
                    f"its criterion in {declared} -- a frequency per unit of physical "
                    "distance. Converting the threshold into the requested unit needs "
                    f"the capture's sampling rate, and that could not be established. "
                    f"{reason} The reading itself is available in {requested}; it is "
                    "the criterion that cannot be carried across, and scoring against "
                    "an unconverted threshold would compare two different quantities."
                )
            return UnitScale(0.0, requested, 1.0, [], metadata, why)

    def factor_of(unit_factor: float | str) -> float:
        if isinstance(unit_factor, float):
            return unit_factor
        assert rate is not None
        return rate.pixels_per_mm if unit_factor == "per_mm" else rate.ppi

    scale = factor_of(requested_factor)
    # Exactly 1.0 rather than a ratio that rounds to it: the common case must not
    # perturb a threshold by a float epsilon. A profile stating no criterion has
    # nothing to carry across, which is why it can be re-expressed without a scale.
    condition_scale = 1.0
    if requested != declared and spec.pass_condition is not None:
        condition_scale = scale / factor_of(declared_factor)

    notes: list[str] = []
    if rate is not None and value_needs_rate:
        notes.extend(rate.notes)
    if rate is not None:
        metadata.update({
            "sampling_ppi": round(rate.ppi, 2),
            "sampling_source": rate.source,
        })
    if value_needs_rate:
        metadata["nyquist_limit"] = round(
            rate.nyquist_lp_per_mm if requested_factor == "per_mm" else rate.ppi / 2.0,
            3,
        )

    return UnitScale(scale, requested, condition_scale, notes, metadata, None)


def sfr_metric(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    reading = READINGS[spec.metric_id]
    unit = _unit_scale(ctx, spec)
    scale, units, unit_notes, unit_metadata = (
        unit.scale, unit.units, unit.notes, unit.metadata
    )
    # The criterion is carried into the reported unit along with the reading.
    # Scaling one and not the other is a verdict change wearing a presentation
    # change's clothes, which is exactly what this used to do.
    condition = rescale_condition(spec.pass_condition, unit.condition_scale)
    edges = ctx.extras.get("edges") or []
    metadata: dict[str, Any] = {"edges_available": len(edges), **unit_metadata}

    if unit.reason is not None:
        return MetricResult(
            metric_id=spec.metric_id, outcome="incomplete", units=units,
            condition=condition, metadata=metadata, reason=unit.reason,
        )

    if not edges:
        return MetricResult(
            metric_id=spec.metric_id, outcome="incomplete", units=units,
            condition=condition, metadata=metadata,
            reason=(
                "No slanted edge was extracted from this capture. The target must "
                "declare at least one slanted_edge region with geometry, and the "
                "feature must be findable in the image."
            ),
        )

    settings = _settings(spec)
    skip_linearize = settings.pop("_skip_linearize", False)
    per_edge: list[PatchValue] = []
    values: list[float] = []
    diagnostics: list[tuple[float, float, float]] = []
    notes: list[str] = list(unit_notes)
    failures: list[str] = []

    for edge in edges:
        try:
            result = compute_sfr(
                edge.pixels,
                linearize=None if skip_linearize else getattr(edge, "linearize", None),
                **settings,
            )
        except SFRError as exc:
            failures.append(f"{edge.edge_id}: {exc}")
            continue
        raw = reading(result)
        value = None if raw is None else raw * scale
        if value is None:
            failures.append(
                f"{edge.edge_id}: the response never reaches the level this metric reads."
            )
            continue
        values.append(float(value))
        if spec.metric_id == "sharpening":
            diagnostics.append((
                result.peak_modulation(0.0),
                result.peak_frequency(0.0) or 0.0,
                result.overshoot(),
                result.peak_modulation(LEAKAGE_SUSPECT_FREQUENCY),
            ))
        per_edge.append(
            PatchValue(
                patch_id=edge.edge_id,
                role=edge.region_id,
                value=float(value),
                measured_lab=(0.0, 0.0, 0.0),
                reference_lab=(0.0, 0.0, 0.0),
                flags=[w.split(".")[0] for w in result.warnings],
            )
        )
        metadata.setdefault("edge_angles_deg", []).append(round(result.edge_angle_deg, 3))
        metadata.setdefault("edge_fit_residual_px", []).append(
            result.settings.get("edge_fit_residual_px")
        )

    if not values:
        return MetricResult(
            metric_id=spec.metric_id, outcome="incomplete", units=units,
            condition=condition, metadata=metadata,
            reason="No edge yielded a usable response. " + "; ".join(failures[:3]),
        )

    array = np.asarray(values)
    how = str(spec.algorithm.get("edge_aggregation", DEFAULT_EDGE_AGGREGATION)).lower()
    if how not in EDGE_AGGREGATIONS:
        raise ProfileError(
            f"Unknown edge_aggregation {how!r}; expected one of {list(EDGE_AGGREGATIONS)}"
        )
    index = _worst(array, spec.pass_condition)
    if how == "worst":
        value = float(array[index])
    elif how == "mean":
        value = float(array.mean())
    else:
        value = float(np.median(array))
    metadata.update({
        "edge_aggregation": how,
        "scale_applied": scale,
        "edges_measured": len(values),
        "worst_edge": per_edge[index].patch_id,
        "min": float(array.min()), "max": float(array.max()),
        "mean": float(array.mean()),
        "spread": float(array.max() - array.min()),
        "analysis_settings": settings or "ISO 12233 defaults",
        "linearized": not skip_linearize,
    })
    passed, threshold = evaluate_condition(value, condition)
    if threshold is not None:
        metadata["threshold"] = threshold

    if failures:
        metadata["failed_edges"] = failures
        notes.append(f"{plural(len(failures), 'slanted edge')} could not be analyzed.")

    if spec.metric_id == "sharpening" and diagnostics:
        diag = np.asarray(diagnostics)
        full_peak, peak_f, overshoot = diag[:, 0].max(), diag[:, 1], diag[:, 2]
        floored_peak = float(diag[:, 3].max())
        metadata.update({
            "peak_over_all_frequencies": float(full_peak),
            "peak_above_leakage_band": floored_peak,
            "peak_frequency_range_cyc_px": [float(peak_f.min()), float(peak_f.max())],
            "spatial_overshoot_pct": {
                "min": float(overshoot.min()), "mean": float(overshoot.mean()),
                "max": float(overshoot.max()),
            },
            "peak_search_min_frequency": DEFAULT_PEAK_SEARCH_MIN_FREQUENCY,
            "leakage_suspect_frequency": LEAKAGE_SUSPECT_FREQUENCY,
        })
        if full_peak > 1.0 and peak_f.max() < LEAKAGE_SUSPECT_FREQUENCY:
            notes.append(
                "Contrast peak likely an analysis artifact, not sharpening."
            )

    return MetricResult(
        metric_id=spec.metric_id, outcome="evaluated", units=units, value=value,
        passed=passed, condition=condition,
        aggregation=spec.aggregation or "worst_edge",
        per_patch=per_edge, notes=notes, metadata=metadata,
    )


for _metric_id in READINGS:
    register_metric(_metric_id)(sfr_metric)
