"""Measure the Image Engineering UTT's geometry off a capture, once.

Image Engineering ship a layout file with iQ-Analyzer (``UTT_A3.chart``) and it is
not in the material this project has. So the chart's regions are measured here the
way the Golden Thread's rules were: from a real capture, by features rather than by
eye, and printed as the ``regions`` block to paste into the target definition.

What makes that safe on this chart is that it registers itself. The UTT prints
**thirty-six crosshairs** -- four at the corners of each of the nine modules' panel
pairs -- and they are sharp, isolated and unambiguous. Found and rectified through,
they turn the capture into the target's own coordinate frame, and every other
feature is then read off in that frame rather than in pixels. The lattice is the
check as well as the frame: the four corner crosshairs *define* (0,0) to (1,1), and
the remaining thirty-two then land on a grid regular to 0.0003 of the chart, which
a mis-detection could not do.

Three independent confirmations that the frame is right, none of them circular:

* The four gray ramps come out 1720, 1724, 1718 and 1726 rectified pixels long.
  They are the same printed bar in four orientations, so an aspect error would
  separate the horizontal pair from the vertical pair.
* The color patches come out square -- 121.3 px across against 122.3 down.
* The two printed rules, read as metric and again as imperial, agree on the
  sampling rate to 0.02 ppi, and land on 397.5 / 398.0 against the 397.7 that
  iQ-Analyzer reported for this capture.

Run from the repository root:

    python tools/locate_utt_geometry.py qa62_..._0411.jp2
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import cv2
import numpy as np

from deltachrome.imaging.geometry import place_chart
from deltachrome.imaging.loader import load_image
from deltachrome.imaging.rulers import RulerError, measure_ruler

#: Rectified pixels per unit of relative u. Fine enough that a boundary is located
#: to well under a source pixel, coarse enough to warp the whole frame quickly.
RECTIFIED_SCALE = 4000.0
#: How far outside the fiducial quad the rectified raster reaches. The rules and the
#: chart's own edges live outside it, so it has to.
RECTIFIED_PAD = 0.30

#: A crosshair is a small isolated dark plus on a light panel. The checkerboard's
#: squares are far larger and its grid lines far longer, so size and fill separate
#: them without any template matching.
CROSS_SIZE_PX = (40, 65)
CROSS_SQUARENESS_PX = 10
CROSS_FILL = (0.14, 0.30)
CROSS_DARK_LEVEL = 90

#: Gap, in relative units, above which crosshairs belong to different modules. The
#: widest module spans 0.145; the closest two modules are 0.28 apart.
MODULE_GAP = 0.15

#: The sampling rate a located rule has to land near before it is believed. Wide
#: enough for any distortion worth reporting, far narrower than the 59% gap between
#: a millimetre and a sixteenth of an inch -- which is what stops a band that has
#: drifted onto the printed digits from reporting a confident wrong unit.
PPI_RANGE = (380.0, 415.0)
UNITS = {"mm": 1.0, "in": 25.4 / 16}

#: Each printed rule is declared in three consecutive segments rather than as one
#: band. Two reasons, and the second is the load-bearing one:
#:
#: * The rule is not exactly parallel to the module lattice -- its baseline drifts
#:   about 6 px across the chart -- so one thin band spanning the whole rule falls
#:   off the ticks at one end and loses a third of them.
#: * Curvature *within* a comb's span is invisible: a short comb absorbs bowing into
#:   its pitch. Splitting the rule is what makes scale variation along it readable
#:   at all, and variation across the field is what geometric distortion is.
RULE_SEGMENTS = 3


def find_crosshairs(rgb: np.ndarray) -> np.ndarray:
    """Every printed registration crosshair in the capture, as (x, y) centroids."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < CROSS_DARK_LEVEL).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    out: list[tuple[float, float]] = []
    lo, hi = CROSS_SIZE_PX
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if not (lo <= w <= hi and lo <= h <= hi):
            continue
        if abs(w - h) > CROSS_SQUARENESS_PX:
            continue
        if not (CROSS_FILL[0] <= area / float(w * h) <= CROSS_FILL[1]):
            continue
        # Center of mass of this component alone, so a neighbouring checker square
        # touching the bounding box cannot drag it.
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == i)
        out.append((x + float(xs.mean()), y + float(ys.mean())))
    return _keep_grouped(np.array(sorted(out, key=lambda p: (p[1], p[0])), dtype=np.float64))


def _keep_grouped(points: np.ndarray, minimum: int = 4) -> np.ndarray:
    """Drop marks that stand alone. A crosshair never does.

    Size and fill alone are not enough to say a small dark mark is a crosshair: on
    this capture, where the chart's bottom edge crosses a border stripe, the two
    make a corner that passes both. One stray is enough to matter, because the
    fiducial corners are picked as the outermost marks and an outlier outside the
    chart wins that by definition -- which is how a single fragment 880 px past the
    bottom-left module skewed the whole frame.

    What separates them is that crosshairs are printed at the corners of a panel and
    so arrive in fours; nothing else on the chart does. The linking radius is set
    from the marks' own nearest-neighbour distances rather than in pixels, so it
    follows the capture's sampling rate instead of assuming one: within a panel that
    distance is the panel's side, and the nearest mark in the next module along is
    more than twice as far.
    """
    if len(points) < minimum:
        return points
    separation = np.hypot(
        points[:, 0][:, None] - points[:, 0][None, :],
        points[:, 1][:, None] - points[:, 1][None, :],
    )
    np.fill_diagonal(separation, np.inf)
    radius = 1.6 * float(np.median(separation.min(axis=1)))

    linked = separation <= radius
    group = -np.ones(len(points), dtype=int)
    current = 0
    for start in range(len(points)):
        if group[start] >= 0:
            continue
        stack = [start]
        group[start] = current
        while stack:
            i = stack.pop()
            for j in np.nonzero(linked[i] & (group < 0))[0]:
                group[j] = current
                stack.append(int(j))
        current += 1
    sizes = np.bincount(group)
    return points[sizes[group] >= minimum]


def fiducial_quad(crosses: np.ndarray) -> dict[str, np.ndarray]:
    """The four corner-module crosshairs nearest the chart's own corners."""
    return {
        name: crosses[int(np.argmax(crosses @ np.array([sx, sy], dtype=np.float64)))]
        for name, sx, sy in (("TL", -1, -1), ("TR", 1, -1), ("BL", -1, 1), ("BR", 1, 1))
    }


class Frame:
    """The capture rectified into the fiducial frame, and back again."""

    def __init__(self, rgb: np.ndarray, quad: dict[str, np.ndarray]):
        src = np.array([quad["TL"], quad["TR"], quad["BR"], quad["BL"]], dtype=np.float32)
        width = float(np.linalg.norm(src[1] - src[0]) + np.linalg.norm(src[2] - src[3])) / 2
        height = float(np.linalg.norm(src[3] - src[0]) + np.linalg.norm(src[2] - src[1])) / 2
        self.aspect = width / height
        # Isotropic: v is scaled by the quad's own aspect, so a square printed on the
        # chart comes out square here and a tilt is a tilt rather than a shear.
        self.su = RECTIFIED_SCALE
        self.sv = RECTIFIED_SCALE / self.aspect
        offset = np.array([RECTIFIED_PAD * self.su, RECTIFIED_PAD * self.sv], dtype=np.float32)
        dst = np.array(
            [[0, 0], [self.su, 0], [self.su, self.sv], [0, self.sv]], dtype=np.float32
        ) + offset
        self.matrix = cv2.getPerspectiveTransform(src, dst)
        size = (
            int(self.su * (1 + 2 * RECTIFIED_PAD)),
            int(self.sv * (1 + 2 * RECTIFIED_PAD)),
        )
        self.image = cv2.warpPerspective(rgb, self.matrix, size, flags=cv2.INTER_AREA)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY).astype(np.float64)

    def to_pixel(self, u: float, v: float) -> tuple[float, float]:
        return ((RECTIFIED_PAD + u) * self.su, (RECTIFIED_PAD + v) * self.sv)

    def to_relative(self, x: float, y: float) -> tuple[float, float]:
        return (x / self.su - RECTIFIED_PAD, y / self.sv - RECTIFIED_PAD)

    def relative_of(self, points: np.ndarray) -> np.ndarray:
        """Capture pixel coordinates into the fiducial frame."""
        warped = cv2.perspectiveTransform(
            points.astype(np.float32).reshape(-1, 1, 2), self.matrix
        ).reshape(-1, 2)
        return np.stack(
            [warped[:, 0] / self.su - RECTIFIED_PAD, warped[:, 1] / self.sv - RECTIFIED_PAD],
            axis=1,
        )

    def box(self, u0: float, v0: float, u1: float, v1: float) -> np.ndarray:
        x0, y0 = self.to_pixel(u0, v0)
        x1, y1 = self.to_pixel(u1, v1)
        return self.gray[int(y0):int(y1), int(x0):int(x1)]

    def profile(
        self, u0: float, v0: float, u1: float, v1: float, axis: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean intensity along ``axis``, with each sample's relative coordinate."""
        x0, y0 = self.to_pixel(u0, v0)
        window = self.box(u0, v0, u1, v1)
        values = window.mean(axis=0 if axis == "u" else 1)
        base, scale = (x0, self.su) if axis == "u" else (y0, self.sv)
        coord = (base + np.arange(len(values))) / scale - RECTIFIED_PAD
        return coord, values


def cluster(values: np.ndarray, gap: float) -> list[list[int]]:
    """Split sorted indices wherever the gap between neighbours exceeds ``gap``."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups: list[list[int]] = []
    current = [order[0]]
    for previous, i in zip(order, order[1:]):
        if values[i] - values[previous] > gap:
            groups.append(current)
            current = []
        current.append(i)
    groups.append(current)
    return groups


def modules(frame: Frame, crosses_relative: np.ndarray) -> dict[str, dict[str, Any]]:
    """Each module's crosshair box, and the slanted square inside it."""
    out: dict[str, dict[str, Any]] = {}
    rows = sorted(
        cluster(crosses_relative[:, 1], MODULE_GAP),
        key=lambda r: crosses_relative[r, 1].mean(),
    )
    for row_name, row in zip("TCB", rows):
        columns = sorted(
            cluster(crosses_relative[row, 0], MODULE_GAP),
            key=lambda c: crosses_relative[row][c, 0].mean(),
        )
        for column_name, group in zip("LCR", columns):
            index = [row[j] for j in group]
            us, vs = crosses_relative[index, 0], crosses_relative[index, 1]
            box = (us.min(), vs.min(), us.max(), vs.max())
            out[row_name + column_name] = {
                "crosshairs": len(index),
                "box": [round(float(b), 5) for b in box],
                **slanted_square(frame, box),
            }
    return out


def slanted_square(frame: Frame, box: tuple[float, float, float, float]) -> dict[str, Any]:
    """Locate the tilted dark square inside one module's crosshair box.

    Thresholded between the square and the light panel it sits on, and *not* by a
    percentile over the box: the module also holds a white resolution-wedge panel
    and four black crosshairs, so the box's own midpoint lands nowhere useful.
    """
    x0, y0 = (int(v) for v in frame.to_pixel(box[0], box[1]))
    x1, y1 = (int(v) for v in frame.to_pixel(box[2], box[3]))
    window = frame.gray[y0:y1, x0:x1]
    mask = ((window > 60) & (window < 150)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return {}
    chosen = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours(
        (labels == chosen).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    (cx, cy), (w, h), angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    center = frame.to_relative(x0 + cx, y0 + cy)
    return {
        "center": [round(float(c), 5) for c in center],
        "side_px": round(float((w + h) / 2), 2),
        # minAreaRect reports the angle of one side in (-90, 0]; the printed tilt is
        # the distance from an axis, whichever side got measured.
        "tilt_deg": round(float(angle % 90.0 if angle % 90.0 <= 45 else angle % 90.0 - 90), 3),
    }


def fit_comb(
    coord: np.ndarray, values: np.ndarray, cells: int,
    lo: float, hi: float, pitch_range: tuple[float, float], steps: int = 1200,
) -> tuple[float, float]:
    """Fit an evenly divided bar of ``cells`` cells to an intensity profile.

    Scored on the gradient at the cell boundaries, because the boundary between two
    neighbouring gray steps of nearly equal lightness is invisible -- on the UTT the
    last two steps are L* 5 and L* 3 -- and the even division has to supply the ones
    that cannot be seen from the ones that can.
    """
    gradient = np.abs(np.gradient(values))
    best: tuple[float, float, float] | None = None
    for pitch in np.linspace(pitch_range[0], pitch_range[1], steps):
        for start in np.linspace(lo, hi - cells * pitch, steps // 4):
            index = np.searchsorted(coord, start + np.arange(cells + 1) * pitch)
            index = index[(index > 0) & (index < len(coord))]
            if len(index) < cells:
                continue
            score = float(gradient[index].sum())
            if best is None or score > best[0]:
                best = (score, start, pitch)
    if best is None:
        raise ValueError("no comb fitted")
    return best[1], best[2]


def strong_boundaries(
    frame: Frame, u0: float, v0: float, u1: float, v1: float, axis: str, threshold: float
) -> list[tuple[float, float]]:
    """Where the profile steps hard, with each step's strength."""
    coord, values = frame.profile(u0, v0, u1, v1, axis)
    gradient = np.abs(np.gradient(values))
    out: list[tuple[float, float]] = []
    i = 0
    while i < len(gradient):
        if gradient[i] > threshold:
            j = i
            while j + 1 < len(gradient) and gradient[j + 1] > threshold:
                j += 1
            weight = gradient[i:j + 1]
            out.append(
                (float(np.average(coord[i:j + 1], weights=weight)), float(weight.max()))
            )
            i = j + 1
        else:
            i += 1
    return out


def locate_rule(
    image, placement, axis: str, across: tuple[float, float],
    along: tuple[float, float], unit: str,
) -> tuple[list[float], Any] | None:
    """Sweep a band across the rule and keep the cleanest reading of it."""
    best = None
    for c0 in np.arange(across[0], across[1], 0.0002):
        for c1 in np.arange(c0 + 0.0010, min(c0 + 0.0045, across[1]), 0.0002):
            bounds = (along[0], c0, along[1], c1) if axis == "u" else (c0, along[0], c1, along[1])
            try:
                reading = measure_ruler(
                    image, placement, bounds_relative=bounds,
                    division_mm=UNITS[unit], axis=axis, unit=unit,
                )
            except RulerError:
                continue
            if not (PPI_RANGE[0] <= reading.ppi <= PPI_RANGE[1]):
                continue
            score = (reading.residual_px / reading.pitch_px, -reading.divisions_used)
            if best is None or score < best[0]:
                best = (score, [round(float(b), 5) for b in bounds], reading)
    return (best[1], best[2]) if best else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    args = parser.parse_args()

    image = load_image(args.capture)
    rgb = image.window(0, image.height, 0, image.width)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb

    crosses = find_crosshairs(rgb)
    print(f"  {len(crosses)} crosshair(s) found of the 36 the chart prints", file=sys.stderr)
    quad = fiducial_quad(crosses)
    frame = Frame(rgb, quad)
    print(
        f"  fiducial quad aspect {frame.aspect:.4f}: "
        + ", ".join(f"{k}=({v[0]:.1f},{v[1]:.1f})" for k, v in quad.items()),
        file=sys.stderr,
    )

    relative = frame.relative_of(crosses)
    found = modules(frame, relative)
    print("\n  modules (crosshair box, and the slanted square in it)", file=sys.stderr)
    for name, m in found.items():
        print(
            f"    {name}  n={m['crosshairs']} box={m['box']} "
            f"center={m.get('center')} side={m.get('side_px')}px tilt={m.get('tilt_deg')}deg",
            file=sys.stderr,
        )
    sides = [m["side_px"] for m in found.values() if "side_px" in m]
    tilts = [m["tilt_deg"] for m in found.values() if "tilt_deg" in m]
    print(
        f"    square side: median {np.median(sides):.1f} px "
        f"({np.min(sides):.1f}..{np.max(sides):.1f}); tilt median {np.median(tilts):+.2f} deg",
        file=sys.stderr,
    )

    print("\n  gray ramps (20 steps each)", file=sys.stderr)
    ramps = {
        "top": ("u", (0.22, 0.195, 0.72, 0.245), (0.225, 0.715), (0.0205, 0.0225)),
        "bottom": ("u", (0.22, 0.755, 0.72, 0.810), (0.225, 0.715), (0.0205, 0.0225)),
        "left": ("v", (0.125, 0.15, 0.160, 0.85), (0.155, 0.845), (0.0300, 0.0325)),
        "right": ("v", (0.762, 0.15, 0.800, 0.85), (0.155, 0.845), (0.0300, 0.0325)),
    }
    for name, (axis, window, span, pitch_range) in ramps.items():
        coord, values = frame.profile(*window, axis)
        start, pitch = fit_comb(coord, values, 20, span[0], span[1], pitch_range)
        length = 20 * pitch * (frame.su if axis == "u" else frame.sv)
        print(
            f"    {name:7s} {axis} start={start:.5f} pitch={pitch:.5f} "
            f"end={start + 20 * pitch:.5f}  bar {length:.1f} rectified px",
            file=sys.stderr,
        )

    print("\n  printed rules", file=sys.stderr)
    placement = place_chart(
        [quad["TL"], quad["TR"], quad["BR"], quad["BL"]], (image.width, image.height)
    )
    regions: list[dict[str, Any]] = []
    rates: list[float] = []
    layout = [
        ("top", "u", "mm", (-0.0560, -0.0505), (-0.09, 0.99)),
        ("top", "u", "in", (-0.0500, -0.0455), (-0.09, 0.99)),
        ("left", "v", "mm", (-0.0940, -0.0893), (-0.09, 1.09)),
        ("left", "v", "in", (-0.0874, -0.0840), (-0.09, 1.09)),
    ]
    for side, axis, unit, across, (a0, a1) in layout:
        edges = np.linspace(a0, a1, RULE_SEGMENTS + 1)
        for k in range(RULE_SEGMENTS):
            region_id = f"ruler_{side}{k + 1}_{unit}"
            located = locate_rule(
                image, placement, axis, across,
                (float(edges[k]) + 0.005, float(edges[k + 1]) - 0.005), unit,
            )
            if located is None:
                print(f"    {region_id:20} NOT FOUND", file=sys.stderr)
                continue
            bounds, reading = located
            rates.append(reading.ppi)
            print(
                f"    {region_id:20} divs={reading.divisions_used:3d} "
                f"pitch={reading.pitch_px:7.3f} "
                f"resid={reading.residual_px / reading.pitch_px:.2%} "
                f"-> {reading.ppi:.2f} ppi",
                file=sys.stderr,
            )
            regions.append({
                "region_id": region_id,
                "kind": "scale",
                "geometry": {
                    "unit": unit,
                    "division_mm": UNITS[unit],
                    "axis": axis,
                    "bounds_relative": bounds,
                },
            })
    if rates:
        spread = (max(rates) - min(rates)) / float(np.mean(rates)) * 100.0
        print(
            f"    {len(regions)} rule(s): {min(rates):.2f}..{max(rates):.2f} ppi, "
            f"spread {spread:.3f}%",
            file=sys.stderr,
        )
    print(json.dumps(regions, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
