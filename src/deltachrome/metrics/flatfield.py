"""Illumination and white balance across the whole image plane.

Metamorfoze measures these on a **frame-filling white sheet** -- or a frame-filling
UTT -- rather than on a chart's patches: *"Differences in luminance and white
balance must be measured using a frame-filling white sheet with an L* value of
approximately 95."* That is a different capture from every other one here, and
modelling it as a variant of the chart-based uniformity metric would have been
wrong in both directions.

The difference from FADGI's lightness uniformity is not cosmetic:

* FADGI takes the **standard deviation over the mean, as a percentage**, across
  repeated neutrals distributed over the *chart's* area.
* Metamorfoze takes a **Delta L\\*** -- the spread between spots -- across the *whole
  image plane*, with tolerances banded by the size of the original: 3 / 4 / 5 / 6
  for A3 / A2 / A1 / A0.

They share a name, measure different things over different areas, and answer to
different numbers. So this is its own metric, and the chart-based one is untouched.

**"Randomly selected spots" is read as a regular grid.** The guideline's wording is
about where a *person* would look; a grid covers the plane without the sampling
itself becoming a variable between runs, and is reported so the choice is visible.

**Spots at the frame edge are half-spots.** A spot is clipped where it runs off the
image rather than being nudged inward, so the sample still reaches the very corner --
which is where fall-off is worst and the whole reason for measuring. The cost is that
a clipped spot's mean sits inward by about a quarter of its width, so a gradient is
under-reported by that much: on a synthetic ramp of exactly 3.0 dL* the metric
returns 2.94. That is inherent to a spot having extent, not to the clipping; moving
the spot inward to keep it whole would sample further from the corner and report
less. The alternative worth having is a smaller ``spot_size``, at more noise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from deltachrome.imaging.color import delta_e_2000_ab, rgb_to_lab
from deltachrome.metrics.base import (
    MetricContext,
    MetricResult,
    MetricSpec,
    PatchValue,
    evaluate_condition,
    register_metric,
    resolve_condition,
)

#: Sample points across the plane, per axis. Enough to describe a smooth falloff
#: and its corners; few enough that every one can be listed in a report.
DEFAULT_GRID = 7
#: Each sample's size, as a fraction of the shorter side of the placed area. Small
#: enough to be a "spot", large enough to average out sensor noise.
DEFAULT_SPOT = 0.04
#: A flat field at the encoding limit reads as perfectly uniform, which is the most
#: dangerous possible failure for this measurement: it turns an overexposed capture
#: into a perfect score. Flagged above this fraction of any sample's pixels.
CLIPPING_LIMIT = 0.001


def _samples(ctx: MetricContext, spec: MetricSpec) -> tuple[list[dict[str, Any]], list[str]]:
    image = ctx.extras.get("image")
    placement = ctx.extras.get("placement")
    if image is None or placement is None:
        return [], ["No image or placement was supplied to the metric context."]

    grid = int(spec.algorithm.get("grid", DEFAULT_GRID))
    spot = float(spec.algorithm.get("spot_size", DEFAULT_SPOT))
    if grid < 2:
        return [], [f"A grid of {grid} cannot describe a plane."]

    warnings: list[str] = []
    step = np.linspace(0.0, 1.0, grid)
    high = 1.0 - image.quantisation_step / 2.0
    low = image.quantisation_step / 2.0

    out: list[dict[str, Any]] = []
    for row, v in enumerate(step):
        for col, u in enumerate(step):
            corners = np.array([
                [u - spot / 2, v - spot / 2], [u + spot / 2, v - spot / 2],
                [u + spot / 2, v + spot / 2], [u - spot / 2, v + spot / 2],
            ])
            pixels = placement.to_pixel(np.clip(corners, 0.0, 1.0))
            x0, x1 = int(pixels[:, 0].min()), int(pixels[:, 0].max())
            y0, y1 = int(pixels[:, 1].min()), int(pixels[:, 1].max())
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, image.width), min(y1, image.height)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            window = image.window(y0, y1, x0, x1).reshape(-1, image.channels)
            clipped = float(
                np.mean(np.any((window >= high) | (window <= low), axis=-1))
            )
            out.append({
                "id": f"r{row}c{col}",
                "u": round(float(u), 3),
                "v": round(float(v), 3),
                "lab": rgb_to_lab(window.mean(axis=0), image.profile),
                "clipped_fraction": clipped,
                # The requested spot, not the clipped read: a spot at the frame
                # edge is deliberately truncated rather than nudged inward, and
                # outlining the truncated part is what shows that it was.
                "roi_px": [[float(px), float(py)] for px, py in pixels],
            })

    worst_clipping = max((s["clipped_fraction"] for s in out), default=0.0)
    if worst_clipping > CLIPPING_LIMIT:
        clipped = sum(1 for s in out if s["clipped_fraction"] > CLIPPING_LIMIT)
        warnings.append(
            f"{clipped} of {len(out)} spots clipped; uniformity may read better "
            "than it is."
        )
    return out, warnings


#: What one of these metrics' readings is in. See MetricResult.reading_units.
READING_UNITS = "L*"


def _result(
    spec: MetricSpec, value: float | None, units: str, samples: list[dict[str, Any]],
    metadata: dict[str, Any], notes: list[str], reason: str | None = None,
    condition: str | None = None,
) -> MetricResult:
    if value is None:
        return MetricResult(
            metric_id=spec.metric_id, outcome="incomplete", units=units,
            condition=condition, reason=reason, metadata=metadata,
        )
    passed, threshold = evaluate_condition(value, condition) if condition else (None, None)
    if threshold is not None:
        metadata["threshold"] = threshold
    return MetricResult(
        metric_id=spec.metric_id, outcome="evaluated", units=units, value=value,
        # Each reading is one spot's L*; the verdict is a spread across them.
        reading_units=READING_UNITS,
        passed=passed, condition=condition,
        aggregation=spec.aggregation or "max spread across the plane",
        per_patch=[
            PatchValue(
                patch_id=s["id"], role="flat_field",
                value=float(s["lab"][0]),
                measured_lab=tuple(float(v) for v in s["lab"]),
                reference_lab=None,
                flags=["clipped"] if s["clipped_fraction"] > CLIPPING_LIMIT else [],
                position_relative=(float(s["u"]), float(s["v"])),
                roi_px=s.get("roi_px"),
            )
            for s in samples
        ],
        notes=notes, metadata=metadata,
    )


def _qualification(spec: MetricSpec, value: float) -> list[str]:
    """Which bands this reading meets, when no band was declared.

    Deliberately *not* phrased as "qualifies for originals up to A0". The bands do
    not nest: the tolerance **loosens** as originals get larger, because holding a
    bigger plane even is harder. So a reading can meet A0 and fail A3, and saying
    "up to A0" would imply everything smaller came with it.
    """
    banded = spec.pass_condition_by
    if banded is None:
        return []
    cleared = banded.clears(value)
    parameter = banded.parameter.replace("_", " ")
    if not cleared:
        return [
            f"This reading meets none of the bands the profile declares, so there is "
            f"no {parameter} at which it conforms. Declaring one would give a FAIL "
            "rather than change the outcome."
        ]

    names = [b.band for b in cleared]
    missed = [b.band for b in banded.bands if b not in cleared]
    met = (
        f"meets the requirement for {', '.join(names[:-1])} and {names[-1]}"
        if len(names) > 1 else f"meets the requirement for {names[0]}"
    )
    note = (
        f"No {parameter} was declared for this run, so this is measured but not "
        f"judged. On the profile's own bands the reading {met}"
        + (f", but not {' or '.join(missed)}. " if missed else ". ")
        + f"Declare {banded.parameter} to turn it into a verdict."
    )
    out = [note]
    if missed:
        out.append(
            "Those bands do not nest: the tolerance loosens as originals get larger, "
            "because holding a bigger image plane even is harder. So meeting a larger "
            "band is not a weaker result than meeting a smaller one, and it does not "
            "carry the smaller one with it. The reading describes the frame as set up "
            "for this capture -- a system reconfigured for smaller originals should be "
            "re-measured with a sheet filling that frame."
        )
    return out


@register_metric("illumination_non_uniformity")
def illumination_non_uniformity(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    """Spread in L* across the image plane, as a Delta L*."""
    units = spec.units or "dL*"
    samples, warnings = _samples(ctx, spec)
    metadata: dict[str, Any] = {"spots": len(samples), "warnings": warnings}
    if len(samples) < 4:
        return _result(spec, None, units, samples, metadata, [],
                       reason="Too few usable spots to describe the image plane. "
                              + " ".join(warnings))

    condition, band_notes, band_metadata = resolve_condition(spec, ctx)
    metadata.update(band_metadata)
    notes_prefix = list(band_notes)

    lightness = np.array([s["lab"][0] for s in samples])
    brightest = samples[int(np.argmax(lightness))]
    darkest = samples[int(np.argmin(lightness))]
    value = float(lightness.max() - lightness.min())

    metadata.update({
        "grid": int(spec.algorithm.get("grid", DEFAULT_GRID)),
        "brightest": {"at": [brightest["u"], brightest["v"]],
                      "L": round(float(brightest["lab"][0]), 3)},
        "darkest": {"at": [darkest["u"], darkest["v"]],
                    "L": round(float(darkest["lab"][0]), 3)},
        "mean_L": round(float(lightness.mean()), 3),
        "std_L": round(float(lightness.std(ddof=1)), 4),
    })
    notes = [
        f"Spread between the brightest spot (L* {brightest['lab'][0]:.2f} at "
        f"{brightest['u']:.2f},{brightest['v']:.2f}) and the darkest (L* "
        f"{darkest['lab'][0]:.2f} at {darkest['u']:.2f},{darkest['v']:.2f}) over a "
        f"{metadata['grid']}x{metadata['grid']} grid across the image plane.",
        "This is a Delta L* across the whole plane, which is what Metamorfoze asks "
        "for. It is not FADGI's lightness uniformity: that is a standard deviation "
        "over a mean, as a percentage, across a chart's own repeated neutrals. The "
        "two share a name and measure different things over different areas.",
        "Metamorfoze bands the tolerance by the size of the original -- 3 / 4 / 5 / "
        "6 for A3 / A2 / A1 / A0. That is a property of the work the system is being "
        "qualified for rather than of this capture, so it is declared per run "
        "(--original-size) against the table the profile carries.",
    ]
    notes = notes_prefix + notes
    notes.extend(_qualification(spec, value))
    notes.extend(warnings)
    if float(lightness.mean()) < 80.0:
        notes.append(
            f"White sheet reads L* {lightness.mean():.1f}; expected L* 95."
        )
    return _result(spec, value, units, samples, metadata, notes, condition=condition)


@register_metric("white_balance_non_uniformity")
def white_balance_non_uniformity(ctx: MetricContext, spec: MetricSpec) -> MetricResult:
    """Largest white-balance difference between any two spots on the plane.

    A color cast that varies across the frame -- one side warmer than the other --
    is invisible to a white balance measured at the center, and is exactly what
    Metamorfoze means by "a white balance deviation that should not occur anywhere
    in the image plane".
    """
    units = spec.units or "dE2000(a*b*)"
    samples, warnings = _samples(ctx, spec)
    metadata: dict[str, Any] = {"spots": len(samples), "warnings": warnings}
    if len(samples) < 4:
        return _result(spec, None, units, samples, metadata, [],
                       reason="Too few usable spots to describe the image plane. "
                              + " ".join(warnings))

    condition, band_notes, band_metadata = resolve_condition(spec, ctx)
    metadata.update(band_metadata)

    labs = np.array([s["lab"] for s in samples], dtype=np.float64)
    worst, pair = 0.0, (samples[0], samples[0])
    for i in range(len(labs)):
        differences = delta_e_2000_ab(labs, labs[i])
        j = int(np.argmax(differences))
        if float(differences[j]) > worst:
            worst, pair = float(differences[j]), (samples[i], samples[j])

    metadata.update({
        "grid": int(spec.algorithm.get("grid", DEFAULT_GRID)),
        "worst_pair": [
            {"at": [s["u"], s["v"]], "a": round(float(s["lab"][1]), 3),
             "b": round(float(s["lab"][2]), 3)}
            for s in pair
        ],
    })
    notes = [
        f"Largest difference between any two of {len(samples)} spots: "
        f"({pair[0]['u']:.2f},{pair[0]['v']:.2f}) a*{pair[0]['lab'][1]:+.2f} "
        f"b*{pair[0]['lab'][2]:+.2f} against ({pair[1]['u']:.2f},"
        f"{pair[1]['v']:.2f}) a*{pair[1]['lab'][1]:+.2f} b*{pair[1]['lab'][2]:+.2f}.",
        "Measured spot against spot rather than against a reference, because the "
        "fault being looked for is a cast that varies across the frame -- one side "
        "warmer than the other -- which a white balance read at the center cannot "
        "see.",
        "CIEDE2000 with the lightness term removed, which is what Metamorfoze 2.0 "
        "specifies. Not the Euclidean Delta E(a*b*) FADGI asks for: the two differ "
        "by CIEDE2000's chroma and hue weighting.",
    ]
    notes = band_notes + notes
    notes.extend(_qualification(spec, worst))
    notes.extend(warnings)
    return _result(spec, worst, units, samples, metadata, notes, condition=condition)
