"""Locate the twenty printed rules on the ISA Golden Thread, once.

The chart prints a centimetre and an inch rule on all four sides of each of its
five modules. Two were declared by hand; the rest are found here and pasted into
the target definition, because hand-placing eighteen bands against a grid overlay
is slow and wrong in ways that do not announce themselves -- a band a few pixels
off reads the printed digits as ticks and returns a confident wrong pitch.

The search is deliberately dumb: sweep the band across and along, read it, and
keep whatever yields the most divisions at the lowest fit residual. What makes
that safe is the plausibility gate. A centimetre division and a sixteenth of an
inch differ by 59% in pixels, so at a known approximate sampling rate only one
unit can be right, and a band that has locked onto digits or onto a neighbouring
feature lands nowhere near either.

Run from the repository root with a capture and the four fiducial corners:

    python tools/locate_dlt_rulers.py ISA_Device_Level_Target.tif
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from deltachrome.imaging.geometry import place_chart
from deltachrome.imaging.loader import load_image
from deltachrome.imaging.rulers import RulerError, measure_ruler
from deltachrome.targets.registry import load_target

#: Candidate units, as millimetres per division.
UNITS = {"mm": 1.0, "in": 25.4 / 16}
#: The band has to land on a rule, and a rule reads at the capture's own sampling
#: rate. Anything outside this has locked onto something else.
PPI_RANGE = (520.0, 700.0)

SIDES = {
    #        along-axis, which box edge, which way is "out"
    "top":    ("u", "v0", -1),
    "bottom": ("u", "v1", +1),
    "left":   ("v", "u0", -1),
    "right":  ("v", "u1", +1),
}
#: How far out from the module's own box to look for a rule, in box units.
SEARCH_OUT = (0.08, 0.36)
#: Along-rule spans to try, as fractions of the module's side.
ALONG_SPANS = ((0.05, 0.95), (0.10, 0.90), (0.16, 0.84), (0.22, 0.78))
#: How far the located rate may sit from the chart's own, before the band is
#: assumed to have locked onto something that is not the rule. Wider than any
#: distortion worth reporting, far narrower than the gap between the two units.
RATE_TOLERANCE = 0.04


def module_box(geometry: dict) -> tuple[float, float, float, float]:
    (cu, cv), (hu, hv) = geometry["center"], geometry["half_size"]
    return cu - hu, cv - hv, cu + hu, cv + hv


def _rectify(image, placement, bounds, axis, samples=900):
    """A strip of the search zone, resampled so the rule runs down the rows."""
    u0, v0, u1, v1 = bounds
    across = samples
    along = int(samples * 2.2)
    us = np.linspace(u0, u1, across if axis == "v" else along)
    vs = np.linspace(v0, v1, along if axis == "v" else across)
    grid_u, grid_v = np.meshgrid(us, vs)
    points = placement.to_pixel(np.stack([grid_u.ravel(), grid_v.ravel()], axis=-1))
    xs = points[:, 0].reshape(grid_u.shape)
    ys = points[:, 1].reshape(grid_u.shape)
    x0, x1 = int(xs.min()) - 2, int(xs.max()) + 2
    y0, y1 = int(ys.min()) - 2, int(ys.max()) + 2
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return None
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, image.width), min(y1, image.height)
    window = image.window(y0, y1, x0, x1).mean(axis=2)
    sampled = window[
        np.clip(np.round(ys - y0).astype(int), 0, window.shape[0] - 1),
        np.clip(np.round(xs - x0).astype(int), 0, window.shape[1] - 1),
    ]
    # Rows along the rule, columns across it.
    return sampled if axis == "v" else sampled.T


def tick_zone(image, placement, box, side):
    """Where the ticks are, across the rule.

    Three things live in a rule's search zone and they separate cleanly by the
    variation *down* the rule: the baseline is dark and uniform, plain card is
    light and uniform, and only the ticks alternate. So the tick zone is the run
    of across-positions with real variation in it, and the baseline is whatever
    dark uniform column sits beside it.
    """
    bu0, bv0, bu1, bv1 = box
    w, h = bu1 - bu0, bv1 - bv0
    axis, edge, direction = SIDES[side]
    near, far = SEARCH_OUT
    if axis == "u":
        base = bv1 if edge == "v1" else bv0
        a, b = base + direction * near, base + direction * far
        bounds = (bu0 + 0.08 * w, min(a, b), bu0 + 0.92 * w, max(a, b))
    else:
        base = bu1 if edge == "u1" else bu0
        a, b = base + direction * near, base + direction * far
        bounds = (min(a, b), bv0 + 0.08 * h, max(a, b), bv0 + 0.92 * h)

    strip = _rectify(image, placement, bounds, axis)
    if strip is None:
        return None
    variation = strip.std(axis=0)
    if variation.max() <= 0:
        return None
    lit = variation > 0.45 * variation.max()
    # longest contiguous run of modulated columns
    best_run, run_start = None, None
    for i, on in enumerate(np.append(lit, False)):
        if on and run_start is None:
            run_start = i
        elif not on and run_start is not None:
            if best_run is None or i - run_start > best_run[1] - best_run[0]:
                best_run = (run_start, i)
            run_start = None
    if best_run is None or best_run[1] - best_run[0] < 8:
        return None

    lo, hi = best_run
    span = bounds[2] - bounds[0] if axis == "v" else bounds[3] - bounds[1]
    origin = bounds[0] if axis == "v" else bounds[1]
    to_relative = lambda i: origin + span * i / (len(variation) - 1)  # noqa: E731
    return to_relative(lo), to_relative(hi), bounds


#: Where a rule sits relative to its module's own box, in box units, measured from
#: the bands verified by hand on the TL and center modules. The modules are printed
#: alike, so this is a starting point for every one of them; the search below
#: refines it, because print registration and the fiducial fit both move it a little.
TEMPLATE = {
    "top":    (-0.262, -0.228),
    "bottom": (1.228, 1.262),
    "left":   (-0.256, -0.196),
    "right":  (1.196, 1.256),
}


def locate(image, placement, box, side, reference_ppi):
    """Find one rule: start from the template, then refine.

    Scoring is fit residual first and division count second, deliberately. A band
    that has caught a printed digit usually reads *more* ticks, not fewer, and the
    extra one bends the comb -- so preferring the longer read picks the wrong band.
    The rate gate does the rest: at a known sampling rate a centimetre and a
    sixteenth of an inch are 59% apart, so only one unit can fit.
    """
    bu0, bv0, bu1, bv1 = box
    w, h = bu1 - bu0, bv1 - bv0
    axis = SIDES[side][0]
    across_size, along_size = (h, w) if axis == "u" else (w, h)
    t0, t1 = TEMPLATE[side]

    zone = tick_zone(image, placement, box, side)
    starts = [(t0, t1)]
    if zone is not None:
        zone_lo, zone_hi, _ = zone
        base = bv0 if axis == "u" else bu0
        starts.append(((zone_lo - base) / across_size, (zone_hi - base) / across_size))

    best = None
    for s0, s1 in starts:
        for shift in np.arange(-0.05, 0.051, 0.005):
            for grow in (-0.015, -0.008, 0.0, 0.008, 0.02, 0.035):
                c0 = s0 + shift - grow / 2
                c1 = s1 + shift + grow / 2
                if c1 - c0 < 0.012:
                    continue
                for a_lo, a_hi in ALONG_SPANS:
                    if axis == "u":
                        candidate = (bu0 + a_lo * w, bv0 + c0 * h,
                                     bu0 + a_hi * w, bv0 + c1 * h)
                    else:
                        candidate = (bu0 + c0 * w, bv0 + a_lo * h,
                                     bu0 + c1 * w, bv0 + a_hi * h)
                    for unit, division_mm in UNITS.items():
                        try:
                            reading = measure_ruler(
                                image, placement, bounds_relative=candidate,
                                division_mm=division_mm, axis=axis, unit=unit,
                            )
                        except (RulerError, ValueError):
                            continue
                        if abs(reading.ppi - reference_ppi) / reference_ppi > RATE_TOLERANCE:
                            continue
                        score = (-round(reading.residual_px, 2), reading.divisions_used)
                        if best is None or score > best[0]:
                            best = (score, candidate, axis, unit, division_mm, reading)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--corners", type=Path, default=Path("/tmp/dlt_corners.npy"),
                        help="four fiducial corners, TL TR BR BL, as a .npy array")
    parser.add_argument("--target", default="isa_golden_thread_dlt_v1")
    args = parser.parse_args()

    image = load_image(args.image)
    placement = place_chart(np.load(args.corners), (image.width, image.height))
    target = load_target(args.target)
    modules = {r.region_id: r.geometry for r in target.regions if r.kind == "slanted_edge"}

    declared = [r for r in target.regions if r.kind == "scale"]
    rates = []
    for region in declared:
        g = region.geometry
        rates.append(measure_ruler(
            image, placement, bounds_relative=tuple(g["bounds_relative"]),
            division_mm=g["division_mm"], axis=g["axis"], unit=g["unit"],
        ).ppi)
    reference_ppi = float(np.mean(rates))
    print(f"  reference rate from {len(rates)} hand-declared rule(s): "
          f"{reference_ppi:.2f} ppi", file=sys.stderr)

    regions, readings = [], []
    for module_id, geometry in modules.items():
        box = module_box(geometry)
        position = module_id.replace("module_", "")
        for side in SIDES:
            found = locate(image, placement, box, side, reference_ppi)
            if found is None:
                print(f"  {module_id:16}{side:8} NOT FOUND", file=sys.stderr)
                continue
            _, bounds, axis, unit, division_mm, reading = found
            region_id = f"ruler_{position}_{side}"
            print(f"  {region_id:26}{unit:4}{axis} divs={reading.divisions_used:3} "
                  f"ppi={reading.ppi:7.2f} resid={reading.residual_px:.2f}", file=sys.stderr)
            readings.append((region_id, axis, reading.ppi))
            regions.append({
                "region_id": region_id,
                "kind": "scale",
                "geometry": {
                    "unit": unit,
                    "division_mm": round(division_mm, 6),
                    "axis": axis,
                    "bounds_relative": [round(float(b), 5) for b in bounds],
                    "printed_label": "cm." if unit == "mm" else "in.",
                },
                "notes": (
                    f"{'Centimetre' if unit == 'mm' else 'Inch'} rule on the {side} of the "
                    f"{position.upper()} module, running along {axis}. Band located by "
                    "tools/locate_dlt_rulers.py against a real capture."
                ),
            })

    horizontal = [p for _, a, p in readings if a == "u"]
    vertical = [p for _, a, p in readings if a == "v"]
    print(f"\n  {len(regions)}/20 rules located", file=sys.stderr)
    for name, values in (("horizontal", horizontal), ("vertical", vertical)):
        if values:
            print(f"  {name:11} n={len(values):2} mean {np.mean(values):7.2f} ppi  "
                  f"spread {max(values) - min(values):.2f} "
                  f"({(max(values) - min(values)) / np.mean(values):.2%})", file=sys.stderr)
    print(json.dumps(regions, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
