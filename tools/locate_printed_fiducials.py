"""Measure a chart's four placement fiducials off its own capture.

Three of the shipped targets place on features that no existing tool could find, and
each declares a different kind of feature. This finds all three, because they turn
out to be two ideas rather than three:

* ``crosshairs`` -- a small printed **plus** just outside each corner of the patch
  array. The ColorChecker SG and the DT Next Generation Target V2 both use them, and
  the marks could hardly look less alike: the SG's is a hairline cross 94 px across
  with 6 px strokes, the DT's a stubby grey plus 57 px across with 15 px strokes.
  Nothing about fill ratio or size separates either from the printed letters beside
  it -- see below for what does.
* ``frame`` -- the four corners of the thin black rule enclosing the LaserSoft ISO
  12641-2 target, taken as the intersections of the four fitted lines.
* ``strip`` -- the four outer corners of the Golden Thread Object Level Target's
  patch strip, which are then re-derived from the strip's own patch lattice for the
  reason in the third trap below.
* ``discs`` -- the four solid dark circles the Golden Thread Device Level Target
  prints, which sit as a **diamond** about the chart's centre rather than as a
  rectangle at its corners. They are chosen by circularity: a disc fills pi/4 of
  its bounding box and nothing else on that chart does. See the fourth trap.

``frame`` and ``strip`` share a computation. A boundary is scanned for along a few
hundred lines, a straight line is fitted through what was found with the strays
thrown out, and adjacent lines are intersected. What differs is only what counts as
a boundary, and each answer carries the fit's own residual so a bad one is visible
as a number rather than as a picture that looks about right.

Run from the repository root::

    python tools/locate_printed_fiducials.py crosshairs DT-NGT-v2.tif
    python tools/locate_printed_fiducials.py frame LaserSoft_4x5.tif
    python tools/locate_printed_fiducials.py strip ObjectDICE.tif
    python tools/locate_printed_fiducials.py discs ISA_Device_Level_Target.tif

Output is the quad in the order ``place_chart`` wants it -- TL, TR, BR, BL in the
TARGET's own frame -- ready to paste into ``tools/render_target_overlays.py``.

Three traps, one per mode, all of them found the hard way.

**A plus is not a blob.** Choosing crosshairs by size, squareness and fill ratio
picks the printed column letters instead: an ``H`` at 60x70 px with fill 0.51 is a
better match for those three tests than the crosshair is. What actually distinguishes
a plus is its *shape*: the four corners of its bounding box are background and the
four arm midpoints are not. That test is scale-free and passes both charts' marks
unchanged, which is why one function serves a hairline cross and a stubby one. The
one thing it must not do is measure the centre over a box sized from the bounding
box -- on the SG's 6 px stroke that box is mostly background and the real crosshair
is rejected while a letter survives.

**The LaserSoft prints a barcode below its frame.** Scanning up from the card for
the first dark thing finds the bars, not the rule, and the bottom line then fits
through nothing. The rule is 6 px thick and a bar is 300, so a boundary is only
accepted when its dark run is between 2 and 20 px -- which rejects the bars, walks
past them, and finds the rule above.

**A diamond is not a rectangle, and the ordering that suits one breaks the other.**
Every other mode sorts its four points into top row and bottom row and reads off
TL, TR, BR, BL. A diamond has one point at each extreme, so that sort puts two of
them in the same "row" by accident of a few pixels and can return a bow tie -- which
``place_chart`` accepts as a valid projective transform and which makes every ROI
read the frame's average. ``discs`` therefore orders by angle about the centroid,
which is correct for any convex quad including a rectangle, and names the points
TOP, RIGHT, BOTTOM, LEFT because that is what they are.

**The two Object Level units are printed differently, and the strip's quad is 34
times longer than it is tall.** The older unit puts a white keyline around its patch
block and the current one does not, so a boundary scan lands on the keyline's outer
edge on one and on the patch's own edge on the other -- about two pixels. On any
other target that would not be worth a sentence; here the quad's short side is one
patch, and both printed rules' bands are declared as fractions of it and are twelve
pixels tall. So the boundaries only seed, and the quad comes from fitting each
patch's centre against its slot index in both axes, where thirty patches across
3400 px fix the pitch to a tenth of a pixel. Landing the boundary on the patch's own
edge instead was tried first and is worse: taking the last gradient before the patch
interior took the fit's residual from 0.7 px to 17, because inside a patch there is
texture and no edge. Both quads are reported, so the difference between them is a
number.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: ITU-R BT.709 luma. Only ever used to find a printed feature, never to measure one,
#: so the exact weights do not matter -- but they should not be a surprise either.
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

# -- crosshairs ---------------------------------------------------------------

#: Bounds on a crosshair's bounding box, the lower one absolute and the upper one a
#: fraction of the capture's short side. A registration mark is small by definition:
#: anything approaching a twentieth of the frame is a patch or a label.
CROSS_MIN_PX = 8
CROSS_MAX_FRACTION = 0.05
#: How much of the bounding box may be ink. A hairline cross fills 0.13 of its box
#: and a stubby plus 0.40, so this is wide on purpose -- it is a sanity bound, and
#: the shape test below is what actually decides.
CROSS_FILL = (0.05, 0.70)

# -- frame --------------------------------------------------------------------

#: How dark, as a fraction of the card's own level, counts as the printed rule.
#:
#: It was 0.65, which is a hair too tight for one of the four charts that use this
#: mode. Their keylines sit at 0.51 of the card (ISA 4x5), 0.60 (LaserSoft), 0.62
#: (ISA 120) and **0.665** (ISA 35mm) -- so the 35mm's fell on the wrong side by
#: fifteen thousandths, and was found only on the scan lines where noise happened to
#: dip it under. Ninety-four of six hundred and sixty-three on its left side, fitted
#: at -2.1 degrees, and the resulting "quad" was 2091 x 617 px on a chart that is
#: 5336 x 3453: not a near miss but nonsense, which is at least the safe way to fail.
#:
#: Swept from 0.65 to 0.85 the other three charts' quads do not move by a tenth of a
#: pixel -- their keylines are far enough from the bound that it never binds -- while
#: the 35mm fails below 0.70 and is stable from 0.75 up, at a fit residual of 0.38 px,
#: the best of the four. 0.80 is the middle of that flat region.
FRAME_DARK_FRACTION = 0.80
#: Thickest run of dark that can still be a rule. The LaserSoft's is 6 px; its
#: barcode's bars are nearer 300, and this is the whole reason the bottom rule can
#: be found at all.
FRAME_MAX_THICKNESS = 20

# -- strip --------------------------------------------------------------------

#: How far a pixel must sit from the card colour, per channel, to be on the strip.
STRIP_THRESHOLD = 0.06
#: Saturation levels the patch blobs are sought at. The seed is the union across all
#: of them rather than the best single one: a brown patch and a cyan one do not
#: appear at the same threshold, and a seed missing the strip's last patch clips the
#: search window short of the corner it exists to find.
STRIP_SATURATIONS = (0.04, 0.06, 0.08, 0.10, 0.14, 0.20)
#: How many patch pitches the Object Level strip's fiducial quad spans. Thirty
#: patches are printed, in two blocks of fifteen with the module between them, and
#: the gap is four pitches wide -- so the quad is 34 long and one tall.
STRIP_SLOTS = 34


def load_rgb(path: Path) -> np.ndarray:
    """The capture as float RGB in 0..1, by whatever route reads it."""
    if path.suffix.lower() in {".jp2", ".jpf", ".j2k"}:
        sys.path.insert(0, str(ROOT / "src"))
        from deltachrome.imaging.loader import load_image, to_display_records

        image = load_image(path)
        pixels = to_display_records(image.pixels)
        if pixels.dtype.kind in "ui":
            return pixels.astype(np.float32) / float(image.scale)
        return np.asarray(pixels, np.float32)

    import tifffile

    with tifffile.TiffFile(path) as handle:
        array = handle.pages[0].asarray()
    scale = float(np.iinfo(array.dtype).max) if array.dtype.kind in "ui" else 1.0
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    return array[..., :3].astype(np.float32) / scale


def to_luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ LUMA


# -- crosshairs ---------------------------------------------------------------


def _is_cross(mask: np.ndarray) -> bool:
    """Whether a component is a plus rather than a glyph, a disc or a patch.

    Two tests, and between them they are the whole finder. The four corners of the
    bounding box must be clear of ink -- which no letter with a bowl or a stem in a
    corner survives -- and the centre and the four arm midpoints must be solid ink.

    Both probes are **three pixels across regardless of the mark's size**. That is
    the point: a probe scaled to the bounding box is wider than the SG crosshair's
    6 px stroke, so most of it falls on background and the real mark scores worse
    than a letter does.
    """
    height, width = mask.shape
    corner = max(2, min(width, height) // 6)
    if max(mask[:corner, :corner].mean(), mask[:corner, -corner:].mean(),
           mask[-corner:, :corner].mean(), mask[-corner:, -corner:].mean()) > 0.10:
        return False

    def solid(row: int, col: int) -> bool:
        return mask[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2].mean() > 0.99

    mid_y, mid_x = height // 2, width // 2
    arm_y, arm_x = int(0.15 * height), int(0.15 * width)
    return (solid(mid_y, mid_x)
            and solid(arm_y, mid_x) and solid(height - 1 - arm_y, mid_x)
            and solid(mid_y, arm_x) and solid(mid_y, width - 1 - arm_x))


def find_crosshairs(gray: np.ndarray) -> tuple[list[tuple[float, float]], dict]:
    """The four printed pluses, as centroids.

    Swept over threshold and both polarities, keeping the sweep step that yields
    exactly four and whose four marks are most alike in area -- because four marks
    of one printed size are the answer and four unrelated blobs are not, and the
    spread says which happened without anyone looking at the picture.
    """
    height, width = gray.shape
    upper = CROSS_MAX_FRACTION * min(width, height)
    best = None
    for level in np.linspace(0.08, 0.92, 43):
        for polarity in (1, -1):
            mask = ((gray > level) if polarity > 0 else (gray < level)).astype(np.uint8)
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            found = []
            for index in range(1, count):
                x, y, w, h, area = stats[index]
                if not (CROSS_MIN_PX < max(w, h) < upper): continue
                if not (0.75 < w / max(h, 1) < 1.33): continue
                fill = area / float(w * h)
                if not (CROSS_FILL[0] < fill < CROSS_FILL[1]): continue
                if not _is_cross(labels[y:y + h, x:x + w] == index): continue
                found.append((tuple(centroids[index]), w, h, fill))
            if len(found) != 4:
                continue
            areas = np.array([w * h for _, w, h, _ in found], float)
            spread = float(np.ptp(areas) / areas.mean())
            if best is None or spread < best[0]:
                best = (spread, level, polarity, found)

    if best is None:
        raise SystemExit(
            "no four-crosshair solution. Either the marks are outside the capture, "
            "or this chart's are not pluses -- check with --debug."
        )
    spread, level, polarity, found = best
    return ([point for point, _, _, _ in found],
            {"level": level, "polarity": polarity, "area_spread": spread,
             "sizes": [f"{w}x{h} fill {fill:.2f}" for _, w, h, fill in found]})


# -- a boundary, fitted -------------------------------------------------------


def _runs(line: np.ndarray) -> list[tuple[int, int]]:
    where = np.flatnonzero(line)
    if not len(where):
        return []
    breaks = np.flatnonzero(np.diff(where) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(where) - 1]))
    return [(int(where[a]), int(where[b])) for a, b in zip(starts, ends)]


def _fit_line(points: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """A straight line through the found boundary, with the strays thrown out.

    Three tightening passes rather than one: the first is dragged by whatever the
    scan mistook for the boundary, and the residual it reports is only meaningful
    once those are gone.
    """
    keep = points[np.abs(points[:, 1] - np.median(points[:, 1])) < tolerance]
    for _ in range(3):
        fit = np.polyfit(keep[:, 0], keep[:, 1], 1)
        residual = keep[:, 1] - np.polyval(fit, keep[:, 0])
        keep = keep[np.abs(residual) < max(0.8, 2.5 * residual.std())]
    fit = np.polyfit(keep[:, 0], keep[:, 1], 1)
    return fit, keep


def _intersect(horizontal: np.ndarray, vertical: np.ndarray) -> tuple[float, float]:
    """Where a fitted row-line and a fitted column-line meet.

    The two are parameterised differently on purpose -- ``y = a*x + b`` for a nearly
    horizontal rule and ``x = a*y + b`` for a nearly vertical one -- because each is
    then fitted along the axis it actually extends in and neither goes singular.
    """
    x = (vertical[0] * horizontal[1] + vertical[1]) / (1.0 - vertical[0] * horizontal[0])
    return float(x), float(horizontal[0] * x + horizontal[1])


def _scan(measure, window, side: str, step: int) -> np.ndarray:
    """Walk in from one side of a window, collecting where the boundary was found."""
    x0, y0, x1, y1 = window
    vertical = side in {"top", "bottom"}
    across = range(x0, x1, step) if vertical else range(y0, y1, step)
    points = []
    for coordinate in across:
        if vertical:
            profile, base, sign = measure(coordinate, vertical=True), y0, 1
            if side == "bottom":
                profile, base, sign = profile[::-1], y1 - 1, -1
        else:
            profile, base, sign = measure(coordinate, vertical=False), x0, 1
            if side == "right":
                profile, base, sign = profile[::-1], x1 - 1, -1
        found = measure.locate(profile)
        if found is not None:
            points.append((coordinate, base + sign * found))
    return np.array(points, float)


def _corners(fits: dict) -> list[tuple[float, float]]:
    return [_intersect(fits["top"], fits["left"]), _intersect(fits["top"], fits["right"]),
            _intersect(fits["bottom"], fits["right"]), _intersect(fits["bottom"], fits["left"])]


# -- frame --------------------------------------------------------------------


def _card(gray: np.ndarray) -> tuple[float, tuple[int, int, int, int]]:
    """The chart's own body: its modal level, and the box it occupies.

    Found on a decimated copy, because a hundred megapixels of connected components
    to locate something a thousand pixels across is work for nothing.
    """
    small = gray[::8, ::8]
    level = float(np.median(small[(small > 0.30) & (small < 0.80)]))
    mask = ((small > level - 0.12) & (small < level + 0.12)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        raise SystemExit("no card found -- the capture may be cropped to the chart")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) * 8 for v in stats[index][:4])
    return level, (x, y, x + w, y + h)


def find_frame(gray: np.ndarray) -> tuple[list[tuple[float, float]], dict]:
    level, (x0, y0, x1, y1) = _card(gray)
    dark = level * FRAME_DARK_FRACTION
    margin = 40
    window = (x0 + margin, y0 + margin, x1 - margin, y1 - margin)

    class Measure:
        def __call__(self, coordinate, *, vertical):
            return (gray[window[1]:window[3], coordinate] if vertical
                    else gray[coordinate, window[0]:window[2]])

        @staticmethod
        def locate(profile):
            for start, end in _runs(profile < dark):
                if not (2 <= end - start + 1 <= FRAME_MAX_THICKNESS):
                    continue
                lo, hi = max(start - 1, 0), min(end + 2, len(profile))
                weight = np.clip(level - profile[lo:hi], 0.0, None)
                if weight.sum() <= 0:
                    continue
                return float(np.arange(lo, hi) @ weight / weight.sum())
            return None

    return _fit_sides(Measure(), window, gray.shape,
                      extra={"card_level": level, "card_box": (x0, y0, x1, y1)})


# -- strip --------------------------------------------------------------------


def find_strip(rgb: np.ndarray) -> tuple[list[tuple[float, float]], dict]:
    gray_shape = rgb.shape[:2]
    height = gray_shape[0]
    saturation = rgb.max(2) - rgb.min(2)

    blobs = []
    for level in STRIP_SATURATIONS:
        mask = cv2.morphologyEx((saturation > level).astype(np.uint8),
                                cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        blobs += [s for s in stats[1:]
                  # A patch is a solid square. The current unit's yellow border is
                  # saturated too, and is one component whose bounding box spans the
                  # whole card -- it fails on fill, which is the only test it fails.
                  if min(s[2], s[3]) > 0.06 * height
                  and 0.5 < s[2] / max(s[3], 1) < 2.0
                  and s[4] > 0.7 * s[2] * s[3]]
    if len(blobs) < 4:
        raise SystemExit(f"found {len(blobs)} patch blobs, need at least four")

    found = np.array(blobs, float)
    patch = float(np.median(found[:, 3]))
    middle = float(np.median(found[:, 1] + found[:, 3] / 2.0))
    found = found[(np.abs(found[:, 3] - patch) < 0.25 * patch)
                  & (np.abs(found[:, 1] + found[:, 3] / 2.0 - middle) < 0.5 * patch)]
    x0, y0 = int(found[:, 0].min()), int(found[:, 1].min())
    x1, y1 = int((found[:, 0] + found[:, 2]).max()), int((found[:, 1] + found[:, 3]).max())

    band = y1 - y0
    above = rgb[max(y0 - int(0.35 * band), 0):max(y0 - int(0.12 * band), 1), x0:x1]
    card = np.median(above.reshape(-1, 3), axis=0)
    departure = np.abs(rgb - card).max(2)
    pad = max(20, band // 3)
    run = max(6, band // 12)
    window = (max(x0 - pad, 0), max(y0 - pad, 0),
              min(x1 + pad, rgb.shape[1]), min(y1 + pad, rgb.shape[0]))

    class Measure:
        def __call__(self, coordinate, *, vertical):
            return (departure[window[1]:window[3], coordinate] if vertical
                    else departure[coordinate, window[0]:window[2]])

        @staticmethod
        def locate(profile):
            over = (profile > STRIP_THRESHOLD).astype(np.int16)
            if len(over) < run:
                return None
            hit = np.flatnonzero(np.convolve(over, np.ones(run, np.int16), "valid") == run)
            if not len(hit):
                return None
            index = int(hit[0])
            while index > 0 and profile[index - 1] > STRIP_THRESHOLD:
                index -= 1
            lo, hi = max(index - 6, 1), min(index + 7, len(profile) - 1)
            gradient = np.abs(np.diff(profile[lo - 1:hi + 1]))
            if gradient.sum() <= 0:
                return None
            return lo - 1 + float(np.arange(len(gradient)) @ gradient / gradient.sum()) + 0.5

    # The scans run along the strip and no further. Padding the window is what
    # lets the boundary be found at all -- the scan has to start on the card -- but
    # a scan STARTING beyond the strip's end has no strip to find and reports
    # whatever else is out there, which on the older unit is a printed disc.
    edges, report = _fit_sides(Measure(), window, gray_shape,
                               across=(x0 + 5, y0 + 5, x1 - 5, y1 - 5),
                               tolerance=0.25 * band,
                               extra={"patch_height_px": patch,
                                      "card_rgb": np.round(card, 4)})
    corners, lattice = _fit_slots(rgb, saturation, edges)
    report.update(lattice)
    report["boundary_quad"] = edges
    return corners, report


def _fit_slots(rgb, saturation, seed):
    """The quad the strip's own patch lattice implies, which is the one to use.

    Fitting the four boundaries gets the ends of a 3400 px strip to a fraction of a
    pixel and its *height* to about two, because the height is one patch and the
    older unit prints a white keyline around the block that the current one does
    not. On a quad 34 times longer than it is tall that is the wrong error to carry:
    every feature off the patch row -- both printed rules especially -- is declared
    as a fraction of the short side, so the rules' bands are twelve pixels tall and
    two pixels of quad height moves them a sixth of their own width.

    So the boundaries are used only to seed, and the answer comes from fitting each
    patch's centre against its slot index in both axes. Thirty patches across 3400 px
    determine the pitch to a tenth of a pixel; the quad is then 34 pitches long, one
    pitch tall (the patches are square and butt together), and lies on the fitted
    centre line. This is the construction the definition's own geometry was measured
    with, and it recovers its recorded 99.59 px pitch to 0.04 px.

    **Neighbouring patches must not be allowed to merge.** They butt together with
    no gutter, so two adjacent saturated patches threshold into one component whose
    centroid sits on the boundary between them, half a slot from either. Admitting
    those took the fit's residual along the strip from 2.1 px to 14.
    """
    height = rgb.shape[0]
    quad = np.array(seed, float)
    pitch = float(np.linalg.norm(quad[1] - quad[0])) / STRIP_SLOTS

    blobs = []
    for level in STRIP_SATURATIONS:
        mask = cv2.morphologyEx((saturation > level).astype(np.uint8),
                                cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        blobs += [(centroids[i][0], centroids[i][1], stats[i][3])
                  for i in range(1, count)
                  if min(stats[i][2], stats[i][3]) > 0.06 * height
                  and 0.82 < stats[i][2] / max(stats[i][3], 1) < 1.22
                  and stats[i][4] > 0.85 * stats[i][2] * stats[i][3]]
    found = np.array(blobs, float)
    tall = np.median(found[:, 2])
    middle = np.median(found[:, 1])
    found = found[(np.abs(found[:, 2] - tall) < 0.25 * tall)
                  & (np.abs(found[:, 1] - middle) < 0.5 * tall)]

    origin = quad[0][0] + pitch / 2.0
    for _ in range(3):
        slots = np.round((found[:, 0] - origin) / pitch).astype(int)
        inside = (slots >= 0) & (slots < STRIP_SLOTS)
        used = sorted(set(slots[inside].tolist()))
        index = np.array(used, float)
        centre_x = np.array([found[inside][slots[inside] == k, 0].mean() for k in used])
        centre_y = np.array([found[inside][slots[inside] == k, 1].mean() for k in used])
        along = np.polyfit(index, centre_x, 1)
        across = np.polyfit(index, centre_y, 1)
        pitch = float(along[0])
    residual = np.hypot(centre_x - np.polyval(along, index),
                        centre_y - np.polyval(across, index))

    direction = np.array([along[0], across[0]], float)
    direction /= np.linalg.norm(direction)
    half = np.array([-direction[1], direction[0]]) * (pitch / 2.0)
    ends = [np.array([np.polyval(along, t), np.polyval(across, t)])
            for t in (-0.5, STRIP_SLOTS - 0.5)]
    corners = [tuple(ends[0] - half), tuple(ends[1] - half),
               tuple(ends[1] + half), tuple(ends[0] + half)]
    return corners, {"slots_used": len(used), "pitch_px": pitch,
                     "lattice_residual_px": float(residual.std()),
                     "deg": float(np.degrees(np.arctan2(across[0], along[0])))}


def _fit_sides(measure, window, shape, *, across=None, tolerance=None,
               extra=None):
    """The four boundaries of a rectangle, fitted and intersected.

    ``across`` is where the scans run, which is not the same as where they look:
    the strip's left and right boundaries are only 100 px of capture, so the scan
    that finds them runs along the band and no further.
    """
    limits = across or window
    x0, y0, x1, y1 = window
    tolerance = tolerance if tolerance is not None else 0.02 * max(shape)
    fits, report = {}, {}
    for side in ("top", "bottom", "left", "right"):
        vertical = side in {"top", "bottom"}
        span = ((limits[0], limits[2]) if vertical else (limits[1], limits[3]))
        step = max(1, (span[1] - span[0]) // 1200)
        scan_window = ((limits[0], y0, limits[2], y1) if vertical
                       else (x0, limits[1], x1, limits[3]))
        points = _scan(measure, scan_window, side, step)
        if len(points) < 8:
            raise SystemExit(f"the {side} boundary was found on only {len(points)} lines")
        fit, kept = _fit_line(points, tolerance)
        residual = kept[:, 1] - np.polyval(fit, kept[:, 0])
        fits[side] = fit
        report[side] = {"found": len(points), "kept": len(kept),
                        "deg": float(np.degrees(np.arctan(fit[0]))),
                        "rms_px": float(residual.std())}
    return _corners(fits), {"sides": report, **(extra or {})}


# -- output -------------------------------------------------------------------


#: A circle fills pi/4 of its bounding box. Both bounds are wide enough to survive
#: the threshold moving thirty levels and tight enough that nothing else on the
#: Device Level Target -- not a patch, not a letter, not a resolution field -- gets
#: through: measured, the four discs sit at 0.783 to 0.787 across that sweep.
DISC_FILL = math.pi / 4
DISC_FILL_TOLERANCE = 0.06
#: How square a disc's bounding box must be, and how far its area may sit from the
#: others'. A disc is round, so both are tight.
DISC_ASPECT_TOLERANCE = 0.12
DISC_AREA_SPREAD = 0.15


def find_discs(luma: np.ndarray) -> tuple[list[tuple[float, float]], dict]:
    """The four solid dark circles the Device Level Target prints as a diamond.

    Swept over a range of thresholds rather than fixed at one, and the answer taken
    as the mean where the sweep is flat. That is not caution for its own sake: the
    discs are printed on a mid-grey card whose level varies with the illumination,
    so any one threshold is a guess about the capture. Over the swept range the four
    centroids move about a tenth of a pixel while each disc's measured area grows by
    nine percent -- the mark stays put while its apparent size changes, which is
    what says the threshold does not matter and therefore that the marks are real.
    A single threshold could not tell that from having found something else.
    """
    # Thresholded against the CARD's own level, not the frame's. A chart shot on a
    # dark surround puts the frame's darkest levels far below anything printed on
    # it, so a threshold picked from the whole image lands in the surround and finds
    # the chart itself as one enormous "disc".
    card_level, (x0, y0, x1, y1) = _card(luma)
    window = luma[y0:y1, x0:x1]
    report: dict = {"levels": [], "card_level": card_level, "card_box": (x0, y0, x1, y1)}
    blurred = cv2.GaussianBlur(window, (9, 9), 0)
    per_level: dict[float, list[tuple[float, float]]] = {}
    # Swept as a fraction of the card's own level. The useful range was measured
    # rather than guessed: below 0.45 the discs have not fully separated from the
    # card and the centroid wanders a third of a pixel; from 0.45 to 0.80 the four
    # centroids hold to about a tenth while each disc's area grows 9%, which is the
    # signature of a real edge being crossed rather than a threshold finding noise.
    for fraction in np.arange(0.45, 0.80, 0.025):
        level = card_level * fraction
        mask = (blurred < level).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        hits = []
        for i in range(1, count):
            x, y, width, height, area = stats[i]
            if width < 12 or height < 12:
                continue
            if abs(width / height - 1.0) > DISC_ASPECT_TOLERANCE:
                continue
            if abs(area / (width * height) - DISC_FILL) > DISC_FILL_TOLERANCE:
                continue
            hits.append((float(centroids[i][0]) + x0, float(centroids[i][1]) + y0,
                         float(area)))
        # Only a set of four whose areas agree can be one chart's own marks.
        if len(hits) != 4:
            continue
        areas = np.array([h[2] for h in hits])
        if float(np.ptp(areas) / areas.mean()) > DISC_AREA_SPREAD:
            continue
        points = [(h[0], h[1]) for h in hits]
        per_level[float(level)] = _order_by_angle(points)
        report["levels"].append(float(level))

    if not per_level:
        raise SystemExit(
            "No set of four agreeing circles was found. This mode looks for solid "
            "discs filling pi/4 of their bounding box; a chart that prints none, or "
            "a capture whose card is darker than its marks, will find nothing."
        )

    stacked = np.array([per_level[k] for k in sorted(per_level)])
    best = [tuple(map(float, p)) for p in stacked.mean(axis=0)]
    report["stability_px"] = float(np.max(np.linalg.norm(stacked - stacked.mean(axis=0), axis=2)))
    report["level_count"] = len(per_level)
    return best, report


def _order_by_angle(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Round the quad, starting from the topmost point.

    Correct for a diamond and for a rectangle alike, which the row-and-column sort
    below is not: it assumes two points share a top row, and a diamond's do not.
    """
    centre = np.mean(np.array(points, dtype=np.float64), axis=0)
    ordered = sorted(points, key=lambda p: math.atan2(p[1] - centre[1], p[0] - centre[0]))
    start = min(range(4), key=lambda i: ordered[i][1])
    return ordered[start:] + ordered[:start]


def _order(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    by_row = sorted(points, key=lambda p: p[1])
    top = sorted(by_row[:2], key=lambda p: p[0])
    bottom = sorted(by_row[2:], key=lambda p: p[0])
    return [top[0], top[1], bottom[1], bottom[0]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("crosshairs", "frame", "strip", "discs"))
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()

    rgb = load_rgb(args.capture)
    print(f"\n{args.capture.name}: {rgb.shape[1]} x {rgb.shape[0]}")

    if args.mode == "crosshairs":
        points, report = find_crosshairs(to_luma(rgb))
        print(f"  threshold {report['level']:.3f}, "
              f"{'bright' if report['polarity'] > 0 else 'dark'} marks, "
              f"areas agreeing to {report['area_spread'] * 100:.1f}%")
        for size in report["sizes"]:
            print(f"    {size}")
    elif args.mode == "discs":
        points, report = find_discs(to_luma(rgb))
        print(f"  card level {report['card_level']:.3f}, four circles agreeing across "
              f"{report['level_count']} thresholds, centroids stable to "
              f"{report['stability_px']:.2f} px")
    elif args.mode == "frame":
        points, report = find_frame(to_luma(rgb))
        print(f"  card level {report['card_level']:.3f}, box {report['card_box']}")
    else:
        points, report = find_strip(rgb)
        print(f"  patch {report['patch_height_px']:.0f} px tall, "
              f"card {report['card_rgb']}")

    if "pitch_px" in report:
        print(f"    lattice: {report['slots_used']} of {STRIP_SLOTS} slots, pitch "
              f"{report['pitch_px']:.3f} px, {report['deg']:+.3f} deg, residual "
              f"{report['lattice_residual_px']:.2f} px")
        seed = np.array(report["boundary_quad"], float)
        print("    the four boundaries alone would have said: "
              + " ".join(f"{x:.1f},{y:.1f}" for x, y in seed))
    for side, entry in (report.get("sides") or {}).items():
        print(f"    {side:6} {entry['kept']:5} of {entry['found']:5} lines, "
              f"{entry['deg']:+.3f} deg, residual {entry['rms_px']:.3f} px")

    quad = np.array(points if args.mode == "discs" else _order(points))
    names = "TOP, RIGHT, BOTTOM, LEFT" if args.mode == "discs" else "TL, TR, BR, BL"
    print(f"\nFIDUCIAL QUAD ({names}), ready to paste:")
    print("        (" + ", ".join(f"({x:.1f}, {y:.1f})" for x, y in quad) + "),")
    if args.mode == "discs":
        # A diamond has no width and height; its diagonals are what describe it.
        across = float(np.linalg.norm(quad[1] - quad[3]))
        down = float(np.linalg.norm(quad[2] - quad[0]))
        print(f"  diagonals {across:.2f} x {down:.2f} px")
    else:
        width = float(np.mean([np.linalg.norm(quad[1] - quad[0]),
                               np.linalg.norm(quad[2] - quad[3])]))
        height = float(np.mean([np.linalg.norm(quad[3] - quad[0]),
                                np.linalg.norm(quad[2] - quad[1])]))
        print(f"  {width:.2f} x {height:.2f} px, aspect {width / height:.4f}")


if __name__ == "__main__":
    main()
