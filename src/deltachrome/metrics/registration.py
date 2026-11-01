"""Color-channel misregistration.

The measurement itself lives in :mod:`deltachrome.imaging.registration`; this is
the profile-facing plugin that reduces it to one number and a verdict.

Three reduction choices have to be made and **none of them is specified by FADGI
or ISO 19264-1**, which both define the quantity and stop:

``pairing``      whether the two edges a square feature presents on one axis are
                 combined into one reading (default) or reported separately
``component``    whether the criterion applies per axis, as OpenDICE plots it
                 (default), or to the displacement vector
``aggregation``  how a field of measured points becomes one figure

So all three are parameters, and worst, mean, median and spread are reported
whichever drives the verdict -- the same treatment ``edge_aggregation`` gets, and
for the same reason: a report has to be comparable against a tool that chose
differently, without being re-run.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from deltachrome.errors import ProfileError
from deltachrome.imaging.registration import (
    CHANNEL_PAIRS,
    FieldPoint,
    field_points,
    field_vectors,
    measure_edges,
    pair_name,
    rotation_between_records,
)
from deltachrome.metrics.base import (
    MetricContext,
    MetricResult,
    MetricSpec,
    PatchValue,
    VectorReading,
    evaluate_condition,
    register_metric,
)

METRIC_ID = "color_channel_misregistration"

AGGREGATIONS = ("worst", "mean", "median")
#: Unlike SFR, the conservative reduction is the default here. OpenDICE plots every
#: position against the limit lines rather than displaying one averaged figure, so
#: what a reader judges from that screen is the worst point; the profiles state the
#: aggregation as ``max``; and unlike sampling efficiency, misregistration is a
#: local defect -- one corner fringing badly is a fault whether or not the rest of
#: the field is clean.
DEFAULT_AGGREGATION = "worst"


def _settings(spec: MetricSpec) -> dict[str, Any]:
    algorithm = spec.algorithm or {}
    settings: dict[str, Any] = {}
    for key in ("centroid_threshold", "edge_fit"):
        if key in algorithm:
            settings[key] = algorithm[key]
    if algorithm.get("linearize") is False:
        settings["linearize"] = False
    return settings


def _vector_field(
    vectors: dict[str, dict[str, Any]], placement
) -> list[VectorReading]:
    """One arrow per position that gave both axes, for the worst channel pair.

    The worst pair, because that is the pair the reported number comes from --
    drawing all three would put three arrows at each position and invite the
    reader to compare a picture against a figure taken from only one of them.
    All three stay in the metadata.
    """
    if placement is None:
        return []
    out: list[VectorReading] = []
    for region, entry in sorted(vectors.items()):
        pair = max(entry["pairs"], key=lambda k: entry["pairs"][k]["magnitude_px"])
        relative = placement.to_relative(
            np.asarray(entry["center_px"], dtype=np.float64)
        )
        out.append(VectorReading(
            label=region,
            position_relative=(float(relative[0]), float(relative[1])),
            vector_px=(
                float(entry["pairs"][pair]["dx"]),
                float(entry["pairs"][pair]["dy"]),
            ),
            source=pair,
        ))
    return out


def _radial_decomposition(
    vectors: dict[str, dict[str, Any]], placement
) -> dict[str, Any]:
    """Split each displacement into radial and tangential parts, **per pair**.

    This is the whole reason direction is worth recording. Two faults produce
    identical magnitudes and are told apart only by where the arrows point:

    * A **magnification difference** between the records -- one record imaged
      very slightly larger than another -- is purely radial and grows in
      proportion to distance from the optical axis. Every arrow points out of, or
      into, the center.
    * A **plain translation** -- the records offset bodily, as a mis-clocked
      tri-linear array gives -- is the same size and direction everywhere, so the
      radial component changes sign across the field and averages to nothing.

    Decomposed about the center of the **frame**, not of the chart, because the
    optical axis is a property of the lens. Where the chart is not centerd in the
    frame those differ, and using the chart would rotate every arrow's
    decomposition by the offset.

    **One channel pair at a time, and that is a correction.** This used to take
    the worst-magnitude pair at each position independently, which silently mixes
    pairs across the field -- and the pairs do not share a sign convention, so
    ``G-B`` and ``B-R`` point in opposite directions for the same physical
    displacement. On the ISA Golden Thread capture the worst pair is ``B-R`` at
    four modules and ``G-B`` at the fifth, which are 180 degrees apart: pooled,
    the direction coherence read 0.598 and the run reported that neither pattern
    was clear. Taken per pair the same capture reads 0.986 for ``G-B`` and 0.979
    for ``B-R`` -- two clean bodily offsets. The radial growth flips sign between
    them too (-0.73 against +0.74), so averaging across pairs is not a weaker
    answer but a meaningless one.

    A pair is a physical quantity and the field is its own population, so each
    gets its own decomposition and the caller says which it means.

    Reported as numbers rather than as a verdict: with five positions on an ISA
    target these are weak statistics, and a chart whose own printing is
    chromatically offset would produce a radial-looking field with nothing wrong
    with the system at all.
    """
    if placement is None or len(vectors) < 3:
        return {}
    width, height = placement.image_size
    center = np.array([width / 2.0, height / 2.0])

    pairs: list[str] = []
    for entry in vectors.values():
        for name in entry.get("pairs", {}):
            if name not in pairs:
                pairs.append(name)

    by_pair: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        radial: list[float] = []
        tangential: list[float] = []
        radii: list[float] = []
        angles: list[float] = []
        for entry in vectors.values():
            reading = (entry.get("pairs") or {}).get(pair)
            if reading is None:
                continue
            offset = np.asarray(entry["center_px"], dtype=np.float64) - center
            radius = float(np.hypot(*offset))
            if radius < 1e-9:
                continue
            unit = offset / radius
            v = np.array([reading["dx"], reading["dy"]])
            radial.append(float(v @ unit))
            tangential.append(float(v @ np.array([-unit[1], unit[0]])))
            radii.append(radius)
            angles.append(float(np.arctan2(v[1], v[0])))

        if len(radial) < 3:
            continue

        radial_array = np.asarray(radial)
        growth = None
        if np.std(radii) > 1e-9 and np.std(radial_array) > 1e-9:
            growth = round(float(np.corrcoef(radii, radial_array)[0, 1]), 3)

        # Circular spread of the arrow directions: near one when every arrow
        # points the same way, which is what a bodily translation looks like.
        coherence = float(np.abs(np.exp(1j * np.asarray(angles)).mean()))

        by_pair[pair] = {
            "mean_radial_px": round(float(radial_array.mean()), 5),
            "mean_abs_radial_px": round(float(np.abs(radial_array).mean()), 5),
            "mean_abs_tangential_px": round(float(np.abs(tangential).mean()), 5),
            "radial_growth_with_radius": growth,
            "direction_coherence": round(coherence, 3),
            "n": len(radial),
        }

    if not by_pair:
        return {}
    return {"about": "frame center", "by_pair": by_pair}


def _pattern_of(reading: dict[str, Any]) -> str | None:
    """Which of the two faults one pair's arrows look like, or neither."""
    radial = reading["mean_abs_radial_px"]
    tangential = reading["mean_abs_tangential_px"]
    growth = reading["radial_growth_with_radius"]
    if reading["direction_coherence"] > 0.8 and radial <= tangential * 1.5:
        return "bodily offset"
    if growth is not None and growth > 0.7 and radial > tangential:
        return "different magnification"
    return None


def _decomposition_note(decomposition: dict[str, Any]) -> str:
    """One line naming what each pair's arrows look like.

    Deliberately short, and shorter than it was. Each pair's pattern is named in
    the same breath as its coherence, so the sentence that used to follow --
    "the arrows point the same way everywhere, which is a bodily offset" --
    restated a parenthetical two clauses earlier. What survives is the one
    qualification a reader cannot infer: a chart whose own printing is
    chromatically offset imitates a magnification difference exactly, so that
    finding alone carries a caveat.
    """
    by_pair = decomposition.get("by_pair") or {}
    if not by_pair:
        return ""
    n = max(reading["n"] for reading in by_pair.values())
    named = {pair: _pattern_of(reading) for pair, reading in by_pair.items()}
    base = (
        f"Displacement decomposed about the frame center over {n} position(s), "
        "per channel pair. Direction coherence: "
        + ", ".join(
            f"{pair} {reading['direction_coherence']:.2f}"
            + (f" ({named[pair]})" if named[pair] else "")
            for pair, reading in by_pair.items()
        )
        + "."
    )
    if any(kind == "different magnification" for kind in named.values()):
        return base + (
            " A chart whose own printed edges are chromatically offset would read "
            "as a different magnification too, so confirm against a second target."
        )
    if any(named.values()):
        return base
    return base + (
        " No pair shows a clear pattern, which with this few positions is the "
        "common answer -- the per-position figures are the measurement and the "
        "decomposition only a pointer."
    )


def _readings(
    points: list[FieldPoint], component: str, vectors: dict[str, dict[str, Any]],
    placement=None, bounds: dict[str, tuple[int, int, int, int]] | None = None,
) -> tuple[list[PatchValue], list[str]]:
    """One value per reported position, with the id a reader can locate it by.

    ``placement`` turns each point's pixel center into a place on the chart. It
    is optional because the reading stands without it -- a run that cannot supply
    one still reports every number, it just cannot draw them where they happened.
    """
    values: list[PatchValue] = []
    notes: list[str] = []
    bounds = bounds or {}

    def place(center) -> tuple[float, float] | None:
        if placement is None or center is None:
            return None
        relative = placement.to_relative(np.asarray(center, dtype=np.float64))
        return (float(relative[0]), float(relative[1]))

    def region_of(edge_ids) -> list[list[float]] | None:
        """The feature the records were differenced across.

        A reported position is a *pair* of opposite edges, so the region is the
        two ROIs together -- which for a square feature is very nearly the
        square. Shown as one rectangle rather than two, because the reading is
        one number formed from both and outlining them separately would suggest
        two readings.
        """
        boxes = [bounds[e] for e in edge_ids if e in bounds]
        if not boxes:
            return None
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        return [[float(x0), float(y0)], [float(x1), float(y0)],
                [float(x1), float(y1)], [float(x0), float(y1)]]

    centers = {p.region_id: p.center_px for p in points}
    by_region: dict[str, list[str]] = {}
    for point in points:
        by_region.setdefault(point.region_id, []).extend(point.edge_ids)

    if component == "vector":
        if vectors:
            for region, entry in vectors.items():
                worst = max(entry["pairs"], key=lambda k: entry["pairs"][k]["magnitude_px"])
                values.append(
                    PatchValue(
                        patch_id=region,
                        role=f"worst pair {worst}",
                        value=float(entry["worst_magnitude_px"]),
                        measured_lab=(0.0, 0.0, 0.0),
                        reference_lab=(0.0, 0.0, 0.0),
                        position_relative=place(centers.get(region)),
                        roi_px=region_of(by_region.get(region, [])),
                    )
                )
            return values, notes

    for point in points:
        worst = max(point.translation, key=lambda k: abs(point.translation[k]))
        flags = [] if point.paired else ["unpaired"]
        values.append(
            PatchValue(
                patch_id=f"{point.region_id}:{point.axis}",
                role=f"worst pair {worst}",
                value=float(point.spread_px),
                measured_lab=(0.0, 0.0, 0.0),
                reference_lab=(0.0, 0.0, 0.0),
                flags=flags,
                position_relative=place(point.center_px),
                roi_px=region_of(point.edge_ids),
            )
        )
    return values, notes


@register_metric(METRIC_ID)
def misregistration_metric(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    units = spec.units or "pixels"
    edges = ctx.extras.get("edges") or []
    metadata: dict[str, Any] = {"edges_available": len(edges)}

    if not edges:
        return MetricResult(
            metric_id=spec.metric_id, outcome="incomplete", units=units,
            condition=spec.pass_condition, metadata=metadata,
            reason=(
                "No slanted edge was extracted from this capture. Channel "
                "registration is measured on the same edges the SFR metrics use, so "
                "the target must declare at least one slanted_edge region and the "
                "feature must be findable in the image."
            ),
        )

    algorithm = spec.algorithm or {}
    pairing = str(algorithm.get("pairing", "opposite_edges")).lower()
    if pairing not in ("opposite_edges", "per_edge"):
        raise ProfileError(
            f"Unknown pairing {pairing!r}; expected 'opposite_edges' or 'per_edge'."
        )
    component = str(algorithm.get("component", "per_axis")).lower()
    if component not in ("per_axis", "vector"):
        raise ProfileError(
            f"Unknown component {component!r}; expected 'per_axis' or 'vector'."
        )
    how = str(algorithm.get("registration_aggregation", DEFAULT_AGGREGATION)).lower()
    if how not in AGGREGATIONS:
        raise ProfileError(
            f"Unknown registration_aggregation {how!r}; expected one of "
            f"{list(AGGREGATIONS)}"
        )

    settings = _settings(spec)
    measured, failures = measure_edges(edges, **settings)
    metadata["edges_measured"] = len(measured)
    if failures:
        metadata["failed_edges"] = failures
    if not measured:
        return MetricResult(
            metric_id=spec.metric_id, outcome="incomplete", units=units,
            condition=spec.pass_condition, metadata=metadata,
            reason="No edge could be located in all three color records. "
                   + "; ".join(failures[:3]),
        )

    # What the chart offers on each axis, measured or not, so a lone edge is only
    # remarked on where its partner was printed and failed.
    declared: dict[tuple[str, str], int] = {}
    for edge in edges:
        axis = "x" if getattr(edge, "orientation", "") == "vertical" else "y"
        key = (edge.region_id, axis)
        declared[key] = declared.get(key, 0) + 1
    points = field_points(measured, pairing=pairing, declared=declared)
    vectors = field_vectors(points)
    placement = ctx.extras.get("placement")
    per_point, notes = _readings(
        points, component, vectors,
        placement=placement,
        bounds={
            e.edge_id: tuple(e.bounds_px)
            for e in edges if getattr(e, "bounds_px", None)
        },
    )
    if component == "vector" and not vectors:
        # No region gave both axes, so there is no vector to judge. Recorded as the
        # component actually used, not as the one that was asked for.
        component = "per_axis"

    array = np.asarray([p.value for p in per_point], dtype=np.float64)
    index = int(np.argmax(array))
    if how == "worst":
        value = float(array[index])
    elif how == "mean":
        value = float(array.mean())
    else:
        value = float(np.median(array))

    floor = max((p.standard_error_px for p in points), default=0.0)
    metadata.update({
        "pairing": pairing,
        "component": component,
        "registration_aggregation": how,
        "points_reported": len(per_point),
        "points_paired": sum(1 for p in points if p.paired),
        "worst_point": per_point[index].patch_id,
        "min": float(array.min()), "max": float(array.max()),
        "mean": float(array.mean()), "median": float(np.median(array)),
        "spread": float(array.max() - array.min()),
        "measurement_floor_px": floor,
        "linearization": measured[0].linearization,
        "analysis_settings": settings or "defaults",
        "channel_pairs": [pair_name(p) for p in CHANNEL_PAIRS],
        "translation_by_point": {
            f"{p.region_id}:{p.axis}": {k: round(v, 4) for k, v in p.translation.items()}
            for p in points
        },
        "rotation_between_records_deg": rotation_between_records(measured),
    })
    if vectors:
        metadata["displacement_vectors"] = {
            region: {
                "worst_magnitude_px": round(entry["worst_magnitude_px"], 4),
                "pairs": {
                    k: {kk: round(vv, 4) for kk, vv in v.items()}
                    for k, v in entry["pairs"].items()
                },
            }
            for region, entry in vectors.items()
        }

    antisymmetric = {
        pair_name(pair): float(
            np.mean([abs(p.antisymmetric[pair_name(pair)]) for p in points
                     if p.antisymmetric])
        )
        for pair in CHANNEL_PAIRS
        if any(p.antisymmetric for p in points)
    }
    if antisymmetric:
        metadata["antisymmetric_term_px"] = {k: round(v, 4) for k, v in antisymmetric.items()}

    missing = sorted({p.region_id for p in points if p.notes and not p.paired})
    if pairing == "opposite_edges" and missing:
        notes.append(f"Edge not measured on {', '.join(missing)}.")

    passed, threshold = evaluate_condition(value, spec.pass_condition)
    if threshold is not None:
        metadata["threshold"] = threshold

    decomposition = _radial_decomposition(vectors, placement)
    if decomposition:
        metadata["radial_decomposition"] = decomposition
        notes.append(_decomposition_note(decomposition))

    return MetricResult(
        metric_id=spec.metric_id, outcome="evaluated", units=units, value=value,
        passed=passed, condition=spec.pass_condition,
        aggregation=spec.aggregation or how,
        per_patch=per_point, vectors=_vector_field(vectors, placement),
        notes=notes, metadata=metadata,
    )
