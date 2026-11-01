"""Measure the Applied Image QA-62's geometry off a capture holding several of them.

Every other tool here measures one chart in one frame. This one measures *four*, and
that is the point rather than a convenience: the QA-62 is a 60 mm-ish card that reads
one field position, so it is shot several to a board, and a capture of four units is
four independent measurements of one printed model under one illumination. Where a
figure agrees across all four to five decimal places -- and every figure below does --
that is what says it describes the model rather than the card that happened to be
measured.

Three things are measured, in this order, because each is found inside the last:

* **The four printed crosshairs**, per card, giving that unit's fiducial quad. Found
  by the same shape test ``locate_printed_fiducials.py crosshairs`` uses -- the
  corners of the bounding box clear of ink, the centre and the arm midpoints solid --
  swept over threshold and kept where exactly four marks of one size survive.
* **The OEC field's outer boundary**, as four fitted lines in the quad's own relative
  frame. This is what fixes the 6 x 6 lattice: the field is exactly six cells across
  and six down with the steps butted, so origin and pitch follow from the extent and
  no interior boundary has to be found. Which matters, because two of them cannot
  be: neighbouring steps of nearly equal density show no boundary at all.
* **The slanted square**, as four sides fitted from perpendicular profiles at their
  50% crossings, then intersected. Reported as centre, side and tilt because that is
  what the definition declares.

Run from the repository root::

    python tools/locate_qa62_geometry.py QA62_2.jp2

**Why the boundary and not the lattice.** The obvious measurement is to find all five
interior boundaries per axis and fit a comb through them. On this chart that is worse:
step 2 against step 3 and step 15 against step 16 differ by about ten code values, so
a gradient search finds three or four of the five and the comb is fitted to whichever
ones happened to have contrast. The field's outer boundary is one long high-contrast
step against the white card on all four sides, it is found to a fifth of a pixel, and
the interior boundaries that *can* be found then agree with the six-cell prediction to
0.13% -- which is the check rather than the measurement.

**The trap this tool exists to catch.** The QA-62's fiducial quad is square and its
cells are square, so nothing geometric can tell one rotation of it from another. Only
the 20-step ring can, and only if the ring's declared order is right. This prints the
ladder as read, in the declared order, so that a definition claiming the wrong
direction shows up here as a sequence that is not monotone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from locate_printed_fiducials import LUMA, _is_cross, load_rgb  # noqa: E402

#: How large a printed mark may be, as a fraction of the card's shorter side. A
#: crosshair here is 108 px on a card 1970 px wide -- 5.5%, which is over the 5% the
#: whole-frame finder allows, because that bound is a fraction of the *frame* and here
#: the card is the frame.
CROSS_MAX_FRACTION = 0.12

#: Where the OEC field's outer boundary is looked for, as a fraction of the fiducial
#: quad. Only has to bracket it: the search walks 14 px either side of these and takes
#: the 50% crossing, so being a percent out costs nothing.
FIELD_GUESS = (0.065, 0.060, 0.945, 0.940)  # x0, y0, x1, y1

#: How far apart the two plateaux either side of a profile must sit before that
#: profile is believed to hold a boundary. The QA-62's lightest step reads 0.90 against
#: a card at 0.93, so this cannot be generous -- and it does not need to be, since
#: nothing else lies along these lines.
STEP_GATE = 0.02

#: Cells across and down the OEC lattice, and how many of them are steps.
GRID = 6
STEPS = 20


def find_cards(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """The bounding boxes of the individual cards, brightest-thing-first.

    Deliberately crude. Every measurement below is made from the crosshairs, so this
    only has to hand each card's neighbourhood to the finder without cutting a mark
    off -- which is why the boxes are grown by a tenth before they are returned.
    """
    small = gray[::8, ::8]
    level = float(np.percentile(small, 99)) * 0.75
    mask = (small > level).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for index in range(1, count):
        x, y, w, h, _area = (int(v) for v in stats[index])
        if min(w, h) < min(small.shape) // 20:
            continue
        pad = max(w, h) // 10
        boxes.append((
            max(0, (x - pad) * 8), max(0, (y - pad) * 8),
            min(gray.shape[1], (x + w + pad) * 8), min(gray.shape[0], (y + h + pad) * 8),
        ))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def find_crosshairs(gray: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """The four printed pluses in one card's neighbourhood, as TL, TR, BR, BL.

    Kept from the threshold whose four marks agree most closely in area, which on this
    chart is decisive: all four printed marks come out within a pixel of 108 x 108 and
    the spread is exactly zero, because nothing else on the card passes the shape test
    at all.
    """
    height, width = gray.shape
    upper = CROSS_MAX_FRACTION * min(width, height)
    best = None
    for level in np.linspace(0.10, 0.90, 41):
        mask = (gray < level).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        found = []
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if not (8 < max(w, h) < upper):
                continue
            if not (0.75 < w / max(h, 1) < 1.33):
                continue
            if not (0.05 < area / float(w * h) < 0.70):
                continue
            if not _is_cross(labels[y:y + h, x:x + w] == index):
                continue
            found.append((tuple(centroids[index]), float(w * h)))
        if len(found) != 4:
            continue
        areas = np.array([area for _, area in found])
        spread = float(np.ptp(areas) / areas.mean())
        if best is None or spread < best[0]:
            best = (spread, level, [point for point, _ in found])
    if best is None:
        return None
    spread, level, points = best
    top, bottom = sorted(sorted(points, key=lambda p: p[1])[:2]), sorted(
        sorted(points, key=lambda p: p[1])[2:]
    )
    quad = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float64)
    return quad, level, spread


class Rectified:
    """One card's pixels addressed in its own fiducial-relative coordinates.

    Nothing is resampled. The homography is carried around and points are mapped
    through it on demand, so every offset below is measured on the capture's own
    pixels and only the *answer* is expressed relative to the quad.
    """

    def __init__(self, gray: np.ndarray, quad: np.ndarray) -> None:
        unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        self.gray = gray
        self.to_relative = cv2.getPerspectiveTransform(quad.astype(np.float32), unit)
        self.to_pixel_matrix = np.linalg.inv(self.to_relative)

    def _through(self, matrix: np.ndarray, points) -> np.ndarray:
        points = np.asarray(points, np.float64).reshape(-1, 2)
        mapped = np.c_[points, np.ones(len(points))] @ matrix.T
        return mapped[:, :2] / mapped[:, 2:3]

    def to_pixel(self, points) -> np.ndarray:
        return self._through(self.to_pixel_matrix, points)

    def to_rel(self, points) -> np.ndarray:
        return self._through(self.to_relative, points)

    def at(self, point: np.ndarray) -> float:
        """Bilinear sample, because a 50% crossing found on integer pixels is not one."""
        x, y = point
        xi, yi = int(x), int(y)
        fx, fy = x - xi, y - yi
        g = self.gray
        return float(
            g[yi, xi] * (1 - fx) * (1 - fy) + g[yi, xi + 1] * fx * (1 - fy)
            + g[yi + 1, xi] * (1 - fx) * fy + g[yi + 1, xi + 1] * fx * fy
        )

    def fit_step(self, start, end, *, samples: int = 200, reach: float = 14.0):
        """Fit the boundary running from ``start`` to ``end``, both relative.

        Walks the declared line, takes a profile perpendicular to it at each station,
        and records where that profile crosses halfway between its own two plateaux.
        A station whose plateaux are too alike is skipped rather than guessed at --
        which is what lets one call fit the QA-62's top boundary, where the contrast
        against the card runs from 36 code values at one end to 3 at the other.

        Returns the fitted line as a point and a direction in *pixels*, how many
        stations contributed, and the fit's rms in pixels.
        """
        a, b = self.to_pixel([start, end])
        along = b - a
        length = float(np.hypot(*along))
        unit = along / length
        normal = np.array([-unit[1], unit[0]])
        offsets = np.linspace(-reach, reach, int(4 * reach) + 1)
        crossings = []
        for t in np.linspace(0.03, 0.97, samples):
            base = a + along * t
            profile = np.array([self.at(base + normal * s) for s in offsets])
            low, high = profile[:8].mean(), profile[-8:].mean()
            if abs(low - high) < STEP_GATE:
                continue
            half = 0.5 * (low + high)
            index = int(np.argmax(np.sign(high - low) * (profile - half) > 0))
            if index == 0:
                continue
            step = (half - profile[index - 1]) / (profile[index] - profile[index - 1])
            between = offsets[index - 1] + step * (offsets[index] - offsets[index - 1])
            crossings.append(base + normal * between)
        if len(crossings) < 20:
            raise RuntimeError(f"only {len(crossings)} usable stations on this boundary")
        points = np.array(crossings)
        centre = points.mean(axis=0)
        _u, singular, right = np.linalg.svd(points - centre)
        return centre, right[0], len(points), float(singular[1] / np.sqrt(len(points)))


def intersect(first, second) -> np.ndarray:
    (p, u), (q, v) = first, second
    t = np.linalg.solve(np.array([u, -v]).T, q - p)
    return p + u * t[0]


def measure_field(view: Rectified) -> tuple[np.ndarray, list[float]]:
    """The OEC field's four outer boundaries, as x0, y0, x1, y1 relative."""
    x0, y0, x1, y1 = FIELD_GUESS
    sides = {
        "top": view.fit_step((x0 + 0.02, y0), (x1 - 0.02, y0)),
        "bottom": view.fit_step((x0 + 0.02, y1), (x1 - 0.02, y1)),
        "left": view.fit_step((x0, y0 + 0.02), (x0, y1 - 0.02)),
        "right": view.fit_step((x1, y0 + 0.02), (x1, y1 - 0.02)),
    }
    edges = {}
    for name, (centre, direction, _n, _rms) in sides.items():
        span = direction * 0.4 * min(view.gray.shape)
        ends = view.to_rel([centre - span, centre + span])
        edges[name] = ends[:, 1].mean() if name in ("top", "bottom") else ends[:, 0].mean()
    return (
        np.array([edges["left"], edges["top"], edges["right"], edges["bottom"]]),
        [sides[name][3] for name in ("top", "bottom", "left", "right")],
    )


def measure_square(view: Rectified) -> tuple[np.ndarray, float, float, list[float]]:
    """The slanted square's corners (relative), its side, its tilt, and four rms."""
    box = view.to_pixel([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
    x0, y0 = box.min(axis=0).astype(int)
    x1, y1 = box.max(axis=0).astype(int)
    window = view.gray[y0:y1, x0:x1]
    low, high = np.percentile(window, [5, 95])
    mask = (window < 0.5 * (low + high)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours(
        (labels == index).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    outline += np.array([x0, y0], np.float32)
    seed = cv2.boxPoints(cv2.minAreaRect(outline)).astype(np.float64)

    fits, rms = [], []
    for i in range(4):
        a, b = view.to_rel([seed[i], seed[(i + 1) % 4]])
        centre, direction, _n, residual = view.fit_step(
            a + (b - a) * 0.15, a + (b - a) * 0.85, samples=120, reach=12.0
        )
        fits.append((centre, direction))
        rms.append(residual)
    corners_px = np.array([intersect(fits[i - 1], fits[i]) for i in range(4)])
    corners = view.to_rel(corners_px)
    sides = [float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)]
    angles = [
        float(np.degrees(np.arctan2(*(corners[(i + 1) % 4] - corners[i])[::-1])) % 90)
        for i in range(4)
    ]
    return corners, float(np.mean(sides)), float(np.mean(angles)), rms


def read_ladder(view: Rectified, field: np.ndarray) -> list[float]:
    """The twenty steps as captured, in the declared clockwise-from-top-left order.

    Read as luma rather than colour-managed lightness, and only to check that the
    declared order is monotone -- no number printed here is a measurement of anything.
    """
    x0, y0, x1, y1 = field
    pitch_x, pitch_y = (x1 - x0) / GRID, (y1 - y0) / GRID
    order = (
        [(0, c) for c in range(GRID)]
        + [(r, GRID - 1) for r in range(1, GRID - 1)]
        + [(GRID - 1, c) for c in range(GRID - 1, -1, -1)]
        + [(r, 0) for r in range(GRID - 2, 0, -1)]
    )
    values = []
    for row, col in order:
        centre = np.array([x0 + (col + 0.5) * pitch_x, y0 + (row + 0.5) * pitch_y])
        half = np.array([pitch_x, pitch_y]) * 0.55 / 2.0
        box = view.to_pixel([centre - half, centre + half])
        (px0, py0), (px1, py1) = box.min(axis=0), box.max(axis=0)
        patch = view.gray[int(py0):int(py1), int(px0):int(px1)]
        values.append(float(patch.mean()) * 255.0)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()

    rgb = load_rgb(args.capture)
    gray = rgb @ LUMA
    print(f"\n{args.capture.name}: {rgb.shape[1]} x {rgb.shape[0]}")

    fields, squares, sides, tilts = [], [], [], []
    for number, (x0, y0, x1, y1) in enumerate(find_cards(gray), start=1):
        found = find_crosshairs(gray[y0:y1, x0:x1])
        if found is None:
            print(f"\n  card {number} at {x0},{y0}: no four crosshairs -- skipped")
            continue
        quad, level, spread = found
        view = Rectified(gray[y0:y1, x0:x1], quad)
        absolute = quad + np.array([x0, y0], float)
        width = float(np.mean([
            np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])
        ]))
        height = float(np.mean([
            np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])
        ]))
        print(f"\n  card {number}: crosshairs at threshold {level:.2f}, "
              f"areas agreeing to {spread * 100:.1f}%")
        print("    quad (TL, TR, BR, BL): "
              + " ".join(f"{x:.1f},{y:.1f}" for x, y in absolute))
        print(f"    {width:.1f} x {height:.1f} px, aspect {width / height:.4f}")

        field, field_rms = measure_field(view)
        fields.append(field)
        print(f"    OEC field  x {field[0]:.5f}..{field[2]:.5f}  "
              f"y {field[1]:.5f}..{field[3]:.5f}   "
              f"boundary rms " + " ".join(f"{r:.2f}" for r in field_rms) + " px")

        corners, side, tilt, square_rms = measure_square(view)
        squares.append(corners.mean(axis=0))
        sides.append(side)
        tilts.append(tilt)
        print(f"    square     centre {corners.mean(axis=0)[0]:.5f},"
              f"{corners.mean(axis=0)[1]:.5f}  side {side:.5f}  tilt {tilt:.3f} deg   "
              f"side rms " + " ".join(f"{r:.2f}" for r in square_rms) + " px")

        ladder = read_ladder(view, field)
        run = "monotone" if all(
            b < a for a, b in zip(ladder, ladder[1:])
        ) else "NOT MONOTONE -- the declared step order is wrong"
        print(f"    ladder     " + " ".join(f"{v:.0f}" for v in ladder))
        print(f"               {run}")

    if not fields:
        raise SystemExit("\nNo card in this capture yielded four crosshairs.")

    field = np.array(fields)
    origin, extent = field[:, :2].mean(axis=0), (field[:, 2:] - field[:, :2]).mean(axis=0)
    centre = np.array(squares)
    print(f"\nACROSS {len(fields)} UNIT(S), ready to paste:")
    print(f'  "grid_origin_relative": [{origin[0]:.5f}, {origin[1]:.5f}],'
          f'   (spread {field[:, :2].std(axis=0)[0]:.5f}, {field[:, :2].std(axis=0)[1]:.5f})')
    print(f'  "patch_size_relative":  {{"width": {extent[0] / GRID:.6f}, '
          f'"height": {extent[1] / GRID:.6f}}}   '
          f'(field {extent[0]:.5f} x {extent[1]:.5f}, aspect {extent[0] / extent[1]:.5f})')
    print(f'  "center":    [{centre[:, 0].mean():.5f}, {centre[:, 1].mean():.5f}]'
          f'   (spread {centre[:, 0].std():.5f}, {centre[:, 1].std():.5f})')
    print(f'  "half_size": [{np.mean(sides) / 2:.5f}, {np.mean(sides) / 2:.5f}]'
          f'   (side {np.mean(sides):.5f}, spread {np.std(sides):.5f})')
    print(f'  "tilt_deg":  {np.mean(tilts):.3f}   (spread {np.std(tilts):.3f})')
    print(f"\n  the ring carries {STEPS} steps of a {GRID} x {GRID} lattice; the "
          f"remaining {GRID * GRID - STEPS} are the square's ground and are not patches")


if __name__ == "__main__":
    main()
