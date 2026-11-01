"""Measure the ISA FADGI-ISO 19264 chart's geometry off a capture.

The chart in both its versions -- the early one that numbers its patches 1 to 168,
and the commercial release that labels them A-N by 1-16 -- is the same layout, and
this tool measures it. Three things come out, in the order they depend on each
other:

**The fiducial quad.** Five modules print a white "dots per inch (optical)" field.
The four in the corners are the largest, highest-contrast features on the chart and
their centres are what a placement keys on, exactly as on the Golden Thread Device
Level Target. Nothing else here would do: the chart carries three black registration
squares rather than four, and only the commercial version adds crosshairs.

**The patch lattice.** 168 circular patches, and they sit on ONE regular 14 x 21
lattice -- not on sixteen unevenly spaced rows, which is what the printed row
numbers suggest. The five empty rows are where the grey wedge, the printed rules and
the serial box sit. The lattice fit is what establishes that: 168 patches through
eight parameters, residual under 0.1% of the quad on both versions.

**The printed rules.** One horizontal and one vertical, each carrying a millimetre
comb at 1 mm divisions and an inch comb at 1/8 inch. Each is declared in three
segments rather than as one band, for the reason the UTT's are: curvature within one
comb's span is invisible, because a short comb absorbs bowing into its pitch, and
variation along the rule is what geometric distortion is.

Run from the repository root:

    python tools/locate_fadgi19264_geometry.py FADGI_19264.tif
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from locate_patch_lattice import fit_lattice, load_working

#: Rows and columns of the lattice the patches sit on, including the empty rows.
LATTICE_ROWS, LATTICE_COLUMNS = 21, 14

#: Lattice row index for each printed row number 1..16. The gaps are the grey
#: wedge (after printed row 6), the printed rules (after 14) and the serial box
#: (after 15) -- all of which occupy whole row slots of the same lattice.
PRINTED_ROWS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20]

#: Which lattice columns carry a patch, per lattice row. Measured, not assumed:
#: the early version's capture yields exactly this and nothing else.
OCCUPIED = {
    0: range(3, 11), 1: range(3, 11), 2: range(3, 11), 3: range(3, 11), 4: range(3, 11),
    5: range(0, 14),
    8: [*range(0, 6), *range(8, 14)],
    9: [*range(0, 5), *range(9, 14)],
    10: [*range(0, 5), *range(9, 14)],
    11: [*range(0, 5), *range(9, 14)],
    12: range(0, 14), 13: range(0, 14), 14: range(0, 14), 15: range(0, 14),
    17: range(3, 11), 20: range(3, 11),
}

#: Percentile of gray above which a pixel counts as part of a module's white
#: field. High, because the chart's mount is already a light mid-grey and only the
#: dpi fields and the lightest patches are brighter.
MODULE_BRIGHT_PERCENTILE = 95.0

#: How many segments each printed rule is declared in. See the module docstring.
RULE_SEGMENTS = 3
#: Fraction of a rule's length each segment spans. Under a third so the three do
#: not touch, which keeps them three independent readings.
SEGMENT_SPAN = 0.30


def module_fields(rgb: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
    """The five modules' white dpi fields, as x0, y0, x1, y1, keyed by position.

    Found by brightness and then by size: the fields are the only large bright
    rectangles on the chart, and all five are the same printed block, so four come
    out portrait and the centre one comes out as its exact transpose. That
    transposition is the check -- it is not something the detector was told to
    expect, and a false positive would not have it.
    """
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bright = (gray > np.percentile(gray, MODULE_BRIGHT_PERCENTILE)).astype(np.uint8)
    # Close over the printed numerals and wedge lines inside the field, which would
    # otherwise cut it into a dozen pieces.
    span = max(int(min(width, height) * 0.012) | 1, 9)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((span, span), np.uint8))

    _, _, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    found = []
    for x, y, w, h, area in stats[1:]:
        if area < 0.002 * width * height:
            continue
        # Bounded on the box, not only on the area. The chart's mount is bright
        # enough in places to leave one sprawling component whose bounding box is
        # the whole frame; its area is unremarkable and its aspect is fine, and
        # only its extent gives it away.
        if w > 0.4 * width or h > 0.4 * height:
            continue
        if not 0.45 < w / h < 2.2:
            continue
        if area / (w * h) < 0.5:
            continue
        found.append((float(area), float(x), float(y), float(w), float(h)))
    if len(found) < 4:
        raise SystemExit(f"Found {len(found)} module fields, expected at least 4.")

    # The four corner modules are one printed block repeated, so they are the four
    # candidates that agree best on size. Picking the four largest instead lets a
    # merged blob -- the centre module joined to the grey wedge above it, on the
    # commercial capture -- displace a real corner.
    quads = _most_alike(found, 4)
    boxes = [(x, y, x + w, y + h) for _, x, y, w, h in quads]
    mid_x = float(np.mean([(b[0] + b[2]) / 2 for b in boxes]))
    mid_y = float(np.mean([(b[1] + b[3]) / 2 for b in boxes]))

    named: dict[str, tuple[float, float, float, float]] = {}
    for box in boxes:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        named[("T" if cy < mid_y else "B") + ("L" if cx < mid_x else "R")] = box
    missing = {"TL", "TR", "BR", "BL"} - set(named)
    if missing:
        raise SystemExit(f"Corner module fields not resolved: missing {sorted(missing)}.")

    centre = _centre_field(gray, quads[0][3], quads[0][4], named)
    if centre is not None:
        named["CENTER"] = centre
    return named


def _centre_field(
    gray: np.ndarray, width: float, height: float, corners: dict
) -> tuple[float, float, float, float] | None:
    """The centre module's field, which needs a higher threshold than the corners.

    It is the same printed block turned ninety degrees, so what is being looked for
    is known exactly: a component the transpose of a corner field's size. That is
    the acceptance test, and it is a strong one -- an 8% tolerance on both sides at
    once is not something a merge or a stray highlight passes.

    Searched by sweeping the threshold rather than at one setting, because on the
    commercial capture the centre field merges with the wedge above it at the
    percentile the corners are found at, and separates by the 97th. Taking the
    corners' setting and hoping produced a box 647x451 where the answer is 595x404
    -- close enough to look plausible, far enough to put the module's search window
    over the Siemens star, whose radial lines include several at exactly the angle
    a slanted edge is looked for at.
    """
    frame_h, frame_w = gray.shape
    span = max(int(min(frame_w, frame_h) * 0.012) | 1, 9)
    middle = np.mean([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in corners.values()], axis=0)

    best = None
    for percentile in np.arange(MODULE_BRIGHT_PERCENTILE, 99.0, 0.5):
        bright = (gray > np.percentile(gray, percentile)).astype(np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((span, span), np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
        for x, y, w, h, area in stats[1:]:
            if abs(w - height) > 0.08 * height or abs(h - width) > 0.08 * width:
                continue
            if np.hypot(x + w / 2 - middle[0], y + h / 2 - middle[1]) > 0.25 * frame_w:
                continue
            error = abs(w - height) / height + abs(h - width) / width
            if best is None or error < best[0]:
                best = (error, (float(x), float(y), float(x + w), float(y + h)))
    return best[1] if best else None


def _most_alike(found: list[tuple[float, ...]], count: int) -> list[tuple[float, ...]]:
    """The ``count`` candidates whose widths and heights agree most closely."""
    if len(found) <= count:
        return list(found)
    best, best_spread = list(found[:count]), float("inf")
    for start in range(len(found) - count + 1):
        for group in _combinations(found, count, start):
            widths = np.array([g[3] for g in group])
            heights = np.array([g[4] for g in group])
            spread = float(np.ptp(widths) / widths.mean() + np.ptp(heights) / heights.mean())
            if spread < best_spread:
                best, best_spread = list(group), spread
    return best


def _combinations(items, count, start):
    if count == 0:
        yield ()
        return
    for index in range(start, len(items) - count + 1):
        for rest in _combinations(items, count - 1, index + 1):
            yield (items[index], *rest)


def rule_bands(gray, quad, lattice, convert) -> dict[str, tuple[float, float, float, float]]:
    """The four tick combs, as pixel bounds: two on the horizontal rule, two on
    the vertical one.

    A comb is found by how *periodic* it is rather than by how dark: the chart is
    covered in dark things -- patch borders, numerals, the registration dots --
    and none of them repeats at a millimetre. But periodicity alone is not enough
    either, because the module fields' converging-line wedges are periodic too and
    score higher. So the search is restricted first, using the lattice: the
    horizontal rule occupies one empty row slot of the same lattice the patches
    sit on, and the vertical rule sits outside the array entirely.
    """
    height, width = gray.shape
    left, top, right, bottom = quad_bounds(quad)

    # The horizontal rule fills lattice row 16, between printed rows 14 and 15.
    row_centre = lattice.center(16, LATTICE_COLUMNS // 2)[1]
    pitch = abs(lattice.center(1, 0)[1] - lattice.center(0, 0)[1])
    h_lo = max(int(row_centre - 0.75 * pitch), 0)
    h_hi = min(int(row_centre + 0.75 * pitch), height)

    # The vertical rule is right of everything else on the chart: past the right
    # edge of the array, and past the corner modules.
    array_right = max(lattice.center(r, LATTICE_COLUMNS - 1)[0] for r in (0, 20))
    v_lo = min(int(max(array_right, right) + 0.5 * pitch), width - 1)
    v_hi = width

    bands: dict[str, tuple[float, float, float, float]] = {}
    strength = _periodicity_profile(gray, (left, top, right, bottom), "horizontal")
    for name, (lo, hi) in zip(("mm", "in"), _two_combs(strength, h_lo, h_hi)):
        bands[f"h_{name}"] = (float(left), float(lo), float(right), float(hi))

    strength = _periodicity_profile(gray, (v_lo, top, v_hi, bottom), "vertical")
    for name, (lo, hi) in zip(("mm", "in"), _two_combs(strength, v_lo, v_hi)):
        bands[f"v_{name}"] = (float(lo), float(top), float(hi), float(bottom))
    return bands


def quad_bounds(quad: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(quad[:, 0].min()), int(quad[:, 1].min()),
        int(quad[:, 0].max()), int(quad[:, 1].max()),
    )


def _periodicity_profile(
    gray: np.ndarray, inside: tuple[int, int, int, int], axis: str
) -> np.ndarray:
    """How comb-like each line of the image is, along one axis."""
    x0, y0, x1, y1 = inside
    if axis == "horizontal":
        block = gray[:, x0:x1]
        lines = range(gray.shape[0])
    else:
        block = gray[y0:y1, :]
        lines = range(gray.shape[1])

    out = np.zeros(len(lines), dtype=np.float64)
    for index in lines:
        line = block[index] if axis == "horizontal" else block[:, index]
        if line.size < 64:
            continue
        centred = line.astype(np.float64) - line.mean()
        variance = float(np.dot(centred, centred))
        if variance < 1e-6:
            continue
        # Correlation at the lags a fine comb repeats at. Normalized, so a faint
        # comb scores as highly as a heavy one and only regularity counts.
        best = 0.0
        for lag in range(6, 40):
            correlation = float(np.dot(centred[:-lag], centred[lag:])) / variance
            best = max(best, correlation)
        out[index] = best
    return out


def _two_combs(strength: np.ndarray, lo: int, hi: int) -> list[tuple[int, int]]:
    """The two comb bands inside a window, in order along the axis.

    Peaks rather than runs. A run above a threshold merges the two combs of a rule
    whenever the gap between them also scores -- the inch comb's long ticks reach
    into it -- and then reports one band spanning both, whose pitch is neither.
    """
    window = np.zeros_like(strength)
    window[lo:hi] = strength[lo:hi]
    smooth = cv2.GaussianBlur(window.reshape(-1, 1).astype(np.float32), (0, 0), 1.5).ravel()

    bands: list[tuple[int, int]] = []
    for _ in range(2):
        peak = int(np.argmax(smooth))
        if smooth[peak] <= 0:
            break
        half = 0.5 * float(smooth[peak])
        start = peak
        while start > lo and smooth[start - 1] > half:
            start -= 1
        end = peak
        while end < hi - 1 and smooth[end + 1] > half:
            end += 1
        bands.append((start, end + 1))
        # Suppress this comb and its immediate surround before looking again, so
        # the second peak is the other comb rather than the same one's shoulder.
        smooth[max(start - 4, 0) : end + 5] = 0.0
    if len(bands) < 2:
        raise SystemExit(f"Expected two tick combs in {lo}..{hi}, found {len(bands)}.")
    return sorted(bands)


def to_relative(quad: np.ndarray):
    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    matrix, _ = cv2.findHomography(quad.astype(np.float64), unit)

    def convert(points) -> np.ndarray:
        return cv2.perspectiveTransform(
            np.asarray(points, dtype=np.float64).reshape(-1, 1, 2), matrix
        ).reshape(-1, 2)

    return convert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--long-edge", type=int, default=3600)
    args = parser.parse_args()

    rgb, scale = load_working(args.capture, args.long_edge)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    print(f"capture  {args.capture}, working {rgb.shape[1]}x{rgb.shape[0]}, "
          f"{scale:.3f} full px per working px")

    fields = module_fields(rgb)
    for name in ("TL", "TR", "CENTER", "BL", "BR"):
        if name not in fields:
            print(f"  module {name:6s} NOT LOCATED")
            continue
        x0, y0, x1, y1 = fields[name]
        print(f"  module {name:6s} {x1 - x0:6.1f} x {y1 - y0:6.1f} px, "
              f"centre ({(x0 + x1) / 2:8.2f}, {(y0 + y1) / 2:8.2f})")

    quad = np.array([
        [(fields[n][0] + fields[n][2]) / 2, (fields[n][1] + fields[n][3]) / 2]
        for n in ("TL", "TR", "BR", "BL")
    ])
    print("\nfiducial quad (working px), the four corner dpi-field centres:")
    for name, point in zip(("TL", "TR", "BR", "BL"), quad):
        print(f"  {name}  {point[0]:9.2f} {point[1]:9.2f}   full: "
              f"{point[0] * scale:9.2f} {point[1] * scale:9.2f}")

    lattice = fit_lattice(rgb, LATTICE_ROWS, LATTICE_COLUMNS, "circle")
    print(f"\nlattice  {len(lattice.centers)} patches located, residual "
          f"{lattice.residual_px:.2f} px "
          f"({100 * lattice.residual_px / abs(quad[1][0] - quad[0][0]):.3f}% of the quad width)")
    print(f"         patch diameter {lattice.patch_size_px[0]:.1f} x "
          f"{lattice.patch_size_px[1]:.1f} px, spread {100 * lattice.size_spread:.2f}%")

    convert = to_relative(quad)
    corners = convert([
        lattice.center(0, 0),
        lattice.center(0, LATTICE_COLUMNS - 1),
        lattice.center(LATTICE_ROWS - 1, 0),
    ])
    spacing_u = (corners[1][0] - corners[0][0]) / (LATTICE_COLUMNS - 1)
    spacing_v = (corners[2][1] - corners[0][1]) / (LATTICE_ROWS - 1)
    wu, wv = lattice.patch_size_px
    centre = lattice.center(LATTICE_ROWS // 2, LATTICE_COLUMNS // 2)
    span = convert([centre - (wu / 2, wv / 2), centre + (wu / 2, wv / 2)])
    size_u, size_v = float(span[1][0] - span[0][0]), float(span[1][1] - span[0][1])

    grid = {
        "patch_size_relative": {"width": round(size_u, 5), "height": round(size_v, 5)},
        "patch_spacing_relative": {
            "horizontal": round(float(spacing_u), 5),
            "vertical": round(float(spacing_v), 5),
        },
        "grid_origin_relative": [
            round(float(corners[0][0] - size_u / 2), 5),
            round(float(corners[0][1] - size_v / 2), 5),
        ],
    }
    print("\npatch_grid geometry:")
    print(json.dumps(grid, indent=2))

    module_geometry = {}
    for name, box in fields.items():
        rel = convert([(box[0], box[1]), (box[2], box[3])])
        module_geometry[name] = {
            "field_bounds_relative": [round(float(v), 5) for v in (*rel[0], *rel[1])],
        }
    print("\nmodule dpi fields, relative:")
    print(json.dumps(module_geometry, indent=2))

    bands = rule_bands(gray, quad, lattice, convert)
    print("\nrule tick combs, relative bounds (u0, v0, u1, v1):")
    for name, box in sorted(bands.items()):
        rel = convert([(box[0], box[1]), (box[2], box[3])])
        print(f"  {name}: {[round(float(v), 5) for v in (*rel[0], *rel[1])]}")


if __name__ == "__main__":
    main()
