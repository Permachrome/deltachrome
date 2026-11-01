"""Per-edge SFR records.

A pass/fail verdict reduces a field of edges to one number, and that number is not
what tells you *why* a system underperforms. The differences between edges are the
diagnosis:

* **Orientation.** A lens with astigmatism focuses sagittal and tangential rays at
  different distances, so edges running one way resolve better than edges running
  the other at the same point in the field. One aggregate number cannot show that;
  a horizontal-versus-vertical split does.
* **Field position.** Softness that grows from center to corner is field curvature
  or a decenterd element. Softness on one side only is tilt -- of the sensor, the
  lens, or the copy stand.
* **Spread at one position.** Four edges of one square should agree closely; when
  they do not, either the measurement or the local focus is unstable.

So every edge is recorded in full, whatever the verdict, including a decimated MTF
curve so the field can be re-plotted from a saved run without the original capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Frequency step, in cycles per pixel, at which the stored curve is sampled. Fine
#: enough to re-plot and to re-read any threshold; small enough that twenty curves
#: do not dominate a run record.
CURVE_STEP = 0.02
CURVE_MAX = 1.0


@dataclass
class EdgeRecord:
    """One edge, measured. Everything needed to diagnose without the capture."""

    edge_id: str
    region: str
    orientation: str
    center_px: list[float]
    center_relative: list[float]
    bounds_px: list[int]
    edge_angle_deg: float
    phase_coverage_px: float
    fit_residual_px: float
    rows_used: int
    linearized: bool
    binning: str
    sampling_efficiency_pct: float | None
    sfr50_pct: float | None
    mtf_at_nyquist: float
    peak_modulation: float
    peak_frequency: float
    overshoot_pct: float
    mtf_curve: dict[str, list[float]] = field(default_factory=dict)
    #: The same edge measured again in each color record, on the frequency grid
    #: of ``mtf_curve`` so the axis is not repeated three times. Diagnostic only:
    #: every verdict is formed on the luminance measurement above, and nothing
    #: here can move one.
    #:
    #: This is how lateral chromatic aberration reads. A lens that focuses red
    #: and blue at different distances from the axis resolves them differently,
    #: and the difference grows towards the corners -- which one aggregate curve
    #: cannot show and three can. It is also the other half of channel
    #: registration: that measures how far apart the records sit, this measures
    #: whether they are equally sharp, and a system can fail either alone.
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def channel_spread_pct(self) -> float | None:
        """Widest SFR50 disagreement between the color records, in points.

        The single number that says whether this position has a chromatic
        problem. Reported per edge rather than pooled, because growing from
        center to corner is what distinguishes lateral chromatic aberration from
        a system that is simply softer in one record everywhere.
        """
        values = [
            c["sfr50_pct"] for c in self.channels.values()
            if c.get("sfr50_pct") is not None
        ]
        return round(max(values) - min(values), 3) if len(values) > 1 else None


def build_edge_record(edge, result, center_relative, channels=None) -> EdgeRecord:
    frequency = np.arange(0.0, CURVE_MAX + CURVE_STEP / 2, CURVE_STEP)
    efficiency = result.sampling_efficiency()
    fifty = result.sfr50()
    return EdgeRecord(
        edge_id=edge.edge_id,
        region=edge.region_id,
        orientation=edge.orientation,
        center_px=[round(float(v), 1) for v in edge.center_px],
        center_relative=[round(float(v), 5) for v in center_relative],
        bounds_px=[int(v) for v in edge.bounds_px],
        edge_angle_deg=round(float(result.edge_angle_deg), 3),
        phase_coverage_px=round(float(result.phase_coverage_px), 2),
        fit_residual_px=float(result.settings.get("edge_fit_residual_px", float("nan"))),
        rows_used=int(result.rows_used),
        linearized=bool(result.settings.get("linearized", False)),
        binning=str(result.settings.get("binning", "distance")),
        sampling_efficiency_pct=None if efficiency is None else round(efficiency * 100, 3),
        sfr50_pct=None if fifty is None else round(fifty * 100, 3),
        mtf_at_nyquist=round(float(result.response_at_nyquist()), 4),
        peak_modulation=round(float(result.peak_modulation()), 4),
        peak_frequency=round(float(result.peak_frequency() or 0.0), 4),
        overshoot_pct=round(float(result.overshoot()), 3),
        mtf_curve={
            "frequency_cyc_px": [round(float(f), 3) for f in frequency],
            "mtf": [round(float(result.mtf_at(f)), 4) for f in frequency],
        },
        channels={
            name: _channel_entry(channel_result, frequency)
            for name, channel_result in (channels or {}).items()
        },
        warnings=list(result.warnings),
    )


def _channel_entry(result, frequency: np.ndarray) -> dict[str, Any]:
    """One color record's curve and the two readings taken off it.

    The frequency axis is deliberately absent: it is the one in ``mtf_curve``,
    and three copies of an identical axis in every edge of every run is storage
    spent on nothing.
    """
    fifty = result.sfr50()
    return {
        "mtf": [round(float(result.mtf_at(f)), 4) for f in frequency],
        "sfr50_pct": None if fifty is None else round(fifty * 100, 3),
        "mtf_at_nyquist": round(float(result.response_at_nyquist()), 4),
    }


def _chromatic_summary(records: list[EdgeRecord]) -> dict[str, Any]:
    """How far apart the color records' sharpness runs, and whether it grows.

    Lateral chromatic aberration is a *radial* effect: it is nil on the axis and
    grows towards the corners, which is what separates it from a system that is
    simply softer in one record everywhere. So the spread is correlated against
    distance from the center of the field rather than only averaged -- an average
    is the one shape of summary that cannot tell those two apart.

    A pointer, not a diagnosis, on the same terms as the orientation difference
    beside it. A record-dependent difference is also what a chart printed with
    chromatic ink edges would produce, and that is a property of the target.
    """
    placed = [
        r for r in records
        if r.channel_spread_pct is not None and len(r.center_relative) == 2
    ]
    if len(placed) < 2:
        return {}

    spreads = np.asarray([r.channel_spread_pct for r in placed], dtype=float)
    radius = np.asarray([
        float(np.hypot(r.center_relative[0] - 0.5, r.center_relative[1] - 0.5))
        for r in placed
    ])
    worst = max(placed, key=lambda r: r.channel_spread_pct or 0.0)

    growth = None
    if radius.std() > 1e-9 and spreads.std() > 1e-9:
        growth = round(float(np.corrcoef(radius, spreads)[0, 1]), 3)

    return {
        "sfr50_channel_spread_pct": {
            "n": int(spreads.size),
            "mean": round(float(spreads.mean()), 3),
            "max": round(float(spreads.max()), 3),
        },
        "worst_edge": worst.edge_id,
        "grows_with_field_radius": growth,
        "note": (
            "Widest disagreement in SFR50 between the color records, per edge. "
            "Correlation with distance from the center of the field is given "
            "because lateral chromatic aberration is radial -- near zero on the "
            "axis, growing to the corners -- while a record that is softer "
            "everywhere by the same amount is a different fault with a different "
            "fix. Treat as a pointer: chromatic ink in the chart's own printed "
            "edges would look the same, and that is the target, not the system."
        ),
    }


def field_summary(records: list[EdgeRecord]) -> dict[str, Any]:
    """Group the edges the ways that make an aberration visible.

    Reported alongside the raw records rather than instead of them: these are
    pointers to look at, not diagnoses. A horizontal/vertical difference is
    *consistent with* astigmatism, and would also be consistent with motion during
    capture or a line-scan artefact.
    """
    if not records:
        return {}

    def stats(subset: list[EdgeRecord], key: str) -> dict[str, float] | None:
        values = [getattr(r, key) for r in subset if getattr(r, key) is not None]
        if not values:
            return None
        array = np.asarray(values, dtype=float)
        return {
            "n": int(array.size),
            "mean": round(float(array.mean()), 3),
            "min": round(float(array.min()), 3),
            "max": round(float(array.max()), 3),
        }

    metrics = ("sfr50_pct", "mtf_at_nyquist", "sampling_efficiency_pct", "peak_modulation")
    summary: dict[str, Any] = {"edges": len(records)}

    by_orientation = {}
    for orientation in ("horizontal", "vertical"):
        subset = [r for r in records if r.orientation == orientation]
        if subset:
            by_orientation[orientation] = {m: stats(subset, m) for m in metrics}
    summary["by_orientation"] = by_orientation

    summary["by_region"] = {
        region: {m: stats([r for r in records if r.region == region], m) for m in metrics}
        for region in sorted({r.region for r in records})
    }

    chromatic = _chromatic_summary(records)
    if chromatic:
        summary["chromatic_difference"] = chromatic

    horizontal = [r.sfr50_pct for r in records if r.orientation == "horizontal" and r.sfr50_pct]
    vertical = [r.sfr50_pct for r in records if r.orientation == "vertical" and r.sfr50_pct]
    if horizontal and vertical:
        difference = float(np.mean(horizontal) - np.mean(vertical))
        spread = float(np.std([r.sfr50_pct for r in records if r.sfr50_pct]))
        summary["orientation_difference"] = {
            "sfr50_horizontal_minus_vertical_pct": round(difference, 3),
            "as_fraction_of_field_spread": round(abs(difference) / spread, 2) if spread else None,
            "note": (
                "A consistent difference between edge orientations at the same field "
                "positions is what astigmatism looks like. It is also what motion "
                "during capture and line-scan artefacts look like, so treat it as a "
                "pointer rather than a diagnosis. Compare it against the spread across "
                "the whole field: a difference smaller than that spread is not "
                "evidence of anything."
            ),
        }
    return summary


@dataclass
class RegistrationRecord:
    """Where each color record put one edge. Kept whatever the profile asks for.

    Same reasoning as the SFR records above: the aggregate cannot show which
    records are displaced, in which direction, or whether the pattern is radial --
    and the raw locations are what makes a saved run re-scorable against a profile
    that judges registration differently.
    """

    edge_id: str
    region: str
    axis: str
    polarity: str
    center_px: list[float]
    linearization: str
    channel_locations_px: dict[str, float]
    channel_angles_deg: dict[str, float]
    standard_error_px: float
    shifts_px: dict[str, float]


def build_registration_records(measurements) -> list[RegistrationRecord]:
    return [
        RegistrationRecord(
            edge_id=m.edge_id,
            region=m.region_id,
            axis=m.axis,
            polarity=m.polarity,
            center_px=[round(float(v), 1) for v in m.center_px],
            linearization=m.linearization,
            channel_locations_px={k: round(v.location_px, 5) for k, v in m.channels.items()},
            channel_angles_deg={k: round(v.angle_deg, 4) for k, v in m.channels.items()},
            standard_error_px=round(m.standard_error_px, 5),
            shifts_px={k: round(v, 5) for k, v in m.shifts.items()},
        )
        for m in measurements
    ]


def registration_summary(measurements) -> dict[str, Any]:
    """The reduced form: displacement per position, and the terms it separates out."""
    from deltachrome.imaging.registration import (
        field_points,
        field_vectors,
        rotation_between_records,
    )

    if not measurements:
        return {}
    points = field_points(measurements)
    vectors = field_vectors(points)
    return {
        "positions": [
            {
                "region": p.region_id,
                "axis": p.axis,
                "edges": p.edge_ids,
                "paired": p.paired,
                "translation_px": {k: round(v, 5) for k, v in p.translation.items()},
                "antisymmetric_px": {k: round(v, 5) for k, v in p.antisymmetric.items()},
                "standard_error_px": round(p.standard_error_px, 5),
            }
            for p in points
        ],
        "displacement_vectors": {
            region: {
                "center_px": [round(v, 1) for v in entry["center_px"]],
                "worst_magnitude_px": round(entry["worst_magnitude_px"], 5),
                "pairs": {
                    k: {kk: round(vv, 5) for kk, vv in v.items()}
                    for k, v in entry["pairs"].items()
                },
            }
            for region, entry in vectors.items()
        },
        "rotation_between_records_deg": {
            k: round(v, 5) for k, v in rotation_between_records(measurements).items()
        },
        "worst_position_px": round(max((p.spread_px for p in points), default=0.0), 5),
    }
