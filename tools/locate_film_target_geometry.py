"""Measure the film targets' geometry off their own captures.

Two chart families, one tool, because both are located the same way: find the
fiducials, rectify the capture to the quad they bound, and read everything else off
the rectified frame in relative coordinates. What differs is only which features are
the fiducials.

``isa`` -- the ISA transmissive line (4x5, 35mm, 120). Ten dark quarter discs in
five apex-to-apex pairs; the four corner pairs' midpoints are the quad. The pair
midpoint is the *symmetry centre* of the two discs, so it is the mean of their
centroids and needs no apex to be identified -- which matters, because identifying
an apex by eye is a coin toss and pairing a real one with a corner across the arc
produces ROIs holding no single edge.

``microform`` -- the 35mm microform resolution chart. Four printed crosses, which
are the only marks on it whose position is two lines crossing rather than an
outline. They come out one inch apart in both axes on both of OpenDICE's specimens.

Both print a block to paste into a target definition. Run from the repository root:

    python tools/locate_film_target_geometry.py isa 120_target.tif
    python tools/locate_film_target_geometry.py microform "Positive 35mm.tif"

**Two things cost time here and are worth knowing before re-measuring anything.**

A **thresholded connected component is not a comb**. The step boundaries inside a
neutral wedge are thin dark lines between patches that are themselves sometimes
darker than the line, so picking troughs by depth finds the darkest *patch* rather
than the boundary. The line filter below subtracts a wide moving average first, so
what is picked is narrowness rather than darkness.

And a **quarter disc's centroid is not its bounding square's centre**. The centroid
of a quarter disc of side s sits 0.4244s from the apex along each axis where the
square's centre sits 0.5s, so a definition written from centroids puts every disc
12% too close to its pair's midpoint. ``minAreaRect`` gives the square directly and
is what this uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

#: Side of the rectified working frame, in pixels, for the fiducial quad itself.
#: Large enough that a relative coordinate is quoted to better than half a pixel of
#: the original capture on everything here.
UNIT = 2400
#: Margin around the unit square, in the same pixels. Both families put features
#: outside their fiducial quad -- every one of the ISA line's rules, and all four of
#: the microform chart's -- so a rectification cropped to the quad would show none of
#: them.
MARGIN = 800


# -- shared ------------------------------------------------------------------


def load_luma(path: Path) -> np.ndarray:
    """The capture as one normalized plane. Structure only; nothing colorimetric."""
    import tifffile

    with tifffile.TiffFile(path) as handle:
        array = handle.pages[0].asarray()
    scale = float(np.iinfo(array.dtype).max) if array.dtype.kind in "ui" else 1.0
    array = array.astype(np.float32) / scale
    if array.ndim == 2:
        return array
    return array[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def rectify(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    matrix = cv2.getPerspectiveTransform(
        quad.astype(np.float32),
        np.array([[0, 0], [UNIT, 0], [UNIT, UNIT], [0, UNIT]], np.float32),
    )
    shift = np.array([[1, 0, MARGIN], [0, 1, MARGIN], [0, 0, 1]], np.float64)
    size = UNIT + 2 * MARGIN
    return cv2.warpPerspective(gray, shift @ matrix, (size, size), flags=cv2.INTER_AREA)


def relative(pixels: float) -> float:
    return (pixels - MARGIN) / UNIT


def line_comb(profile: np.ndarray, offset: float, count: int, separation: int) -> list[float]:
    """The ``count`` strongest *narrow dark lines* in a 1-D profile, sub-pixel.

    Narrowness, not depth: a wide moving average is subtracted first, so a boundary
    line between two patches is found even when one of the patches is darker than the
    line is.
    """
    window = 41
    padded = np.pad(profile, window // 2, mode="edge")
    background = np.convolve(padded, np.ones(window) / window, mode="valid")
    signal = background - profile
    remaining = signal.copy()
    picks: list[float] = []
    for _ in range(count):
        peak = int(np.argmax(remaining))
        lo, hi = max(0, peak - 5), min(len(signal), peak + 6)
        weight = np.clip(signal[lo:hi], 0.0, None)
        if weight.sum() <= 0:
            break
        picks.append(offset + float((np.arange(lo, hi) * weight).sum() / weight.sum()))
        remaining[max(0, peak - separation): peak + separation] = -np.inf
    return sorted(picks)


def fit_comb(lines: list[float]) -> tuple[float, float, float]:
    """Pitch, first line and worst residual, from a least-squares fit by index."""
    values = np.asarray(lines, dtype=np.float64)
    index = np.arange(len(values))
    design = np.vstack([index, np.ones_like(index)]).T
    (pitch, first), *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(pitch), float(first), float(np.abs(values - (index * pitch + first)).max())


# -- the ISA transmissive line ----------------------------------------------


def find_quarter_discs(gray: np.ndarray, stride: int = 8) -> list[dict]:
    """Every quarter disc in the capture, as its bounding square.

    Found coarsely first and refined at full resolution, because a 100-megapixel
    master thresholded whole costs more memory than the file.
    """
    small = gray[::stride, ::stride]
    body = ((small > 0.35) & (small < 0.60)).astype(np.uint8)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(body, 8)
    if count < 2:
        raise SystemExit("no chart body found: the card is not a mid-grey field here")
    card = stats[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    cx, cy, cw, ch = card[:4]

    window = small[cy: cy + ch, cx: cx + cw]
    mask = ((window > 0.14) & (window < 0.30)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    seeds = [stats[i] for i in range(1, count) if stats[i][4] > 5000]

    found: list[dict] = []
    for seed in seeds:
        x, y, w, h = seed[:4]
        x0, y0 = (cx + x - 3) * stride, (cy + y - 3) * stride
        x1, y1 = (cx + x + w + 3) * stride, (cy + y + h + 3) * stride
        patch = gray[y0:y1, x0:x1]
        level = (float(patch.min()) + float(np.median(patch[patch > 0.35]))) / 2.0
        blob = cv2.morphologyEx(
            (patch < level).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
        )
        n, labels, blob_stats, _ = cv2.connectedComponentsWithStats(blob, 8)
        if n < 2:
            continue
        chosen = 1 + int(np.argmax(blob_stats[1:, cv2.CC_STAT_AREA]))
        points = np.column_stack(np.nonzero(labels == chosen))[:, ::-1].astype(np.float32)
        rect = cv2.minAreaRect(points)
        found.append(
            {
                "center_px": (x0 + rect[0][0], y0 + rect[0][1]),
                "side_px": float(np.mean(rect[1])),
                # minAreaRect reports the angle of one side; a quarter disc's square
                # is symmetric under 90 degrees, so fold it into 0..90 and take the
                # slant, which is the quantity the definition states.
                "tilt_deg": float(rect[2]) % 90.0,
            }
        )
    return found


def pair_discs(discs: list[dict]) -> dict[str, tuple[dict, dict]]:
    """Group ten discs into five apex-to-apex pairs, keyed by field position."""
    if len(discs) != 10:
        raise SystemExit(f"expected ten quarter discs, found {len(discs)}")
    centers = np.array([d["center_px"] for d in discs])
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i in range(len(discs)):
        if i in used:
            continue
        distances = np.linalg.norm(centers - centers[i], axis=1)
        distances[list(used) + [i]] = np.inf
        j = int(np.argmin(distances))
        used |= {i, j}
        pairs.append((i, j))

    midpoints = np.array([centers[[i, j]].mean(axis=0) for i, j in pairs])
    order = np.argsort(midpoints[:, 0] + midpoints[:, 1])
    center_index = int(np.argmin(np.linalg.norm(midpoints - midpoints.mean(axis=0), axis=1)))
    corners = [k for k in order if k != center_index]
    # Clockwise from top-left, which is the order the four placed points are matched
    # against. Listing them across the chart instead maps it onto a bow tie.
    labels = ["TL", "TR", "BL", "BR"]
    by_row = sorted(corners, key=lambda k: midpoints[k][1])
    top = sorted(by_row[:2], key=lambda k: midpoints[k][0])
    bottom = sorted(by_row[2:], key=lambda k: midpoints[k][0])
    ordered = dict(zip(labels, top + bottom))
    ordered["CENTER"] = center_index
    return {name: (discs[pairs[k][0]], discs[pairs[k][1]]) for name, k in ordered.items()}


def report_isa(path: Path) -> None:
    gray = load_luma(path)
    discs = find_quarter_discs(gray)
    pairs = pair_discs(discs)

    def midpoint(name: str) -> np.ndarray:
        a, b = pairs[name]
        return (np.asarray(a["center_px"]) + np.asarray(b["center_px"])) / 2.0

    quad = np.array([midpoint(n) for n in ("TL", "TR", "BR", "BL")])
    print(f"\n{path.name}: {gray.shape[1]} x {gray.shape[0]}")
    print("\nFIDUCIAL QUAD (paste as --corners):")
    print("  " + " ".join(f"{x:.1f},{y:.1f}" for x, y in quad))
    width = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    height = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    print(f"  {width:.1f} x {height:.1f} px, aspect {width / height:.4f}")

    # The check that the quad is right: the centre pair is the intersection of the
    # diagonals, and it was measured independently of them.
    measured = midpoint("CENTER")
    expected = quad.mean(axis=0)
    print(f"  centre pair measured at {measured[0]:.1f},{measured[1]:.1f}; "
          f"the quad's own centre is {expected[0]:.1f},{expected[1]:.1f} "
          f"({np.linalg.norm(measured - expected):.2f} px apart)")

    matrix = cv2.getPerspectiveTransform(
        quad.astype(np.float32), np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    ).astype(np.float64)
    sides = [d["side_px"] for d in discs]
    tilts = [d["tilt_deg"] for d in discs]
    half = float(np.mean(sides)) / 2.0 / float(width)
    print(f"\nQUARTER DISCS: bounding square {np.mean(sides):.1f} px "
          f"(spread {np.ptp(sides):.1f}), tilt {np.mean(tilts):.2f} deg "
          f"(spread {np.ptp(tilts):.2f})")
    print(f"  half_size relative: [{half:.5f}, {half:.5f}]")
    for name in ("TL", "TR", "CENTER", "BL", "BR"):
        centers = np.array([[d["center_px"] for d in pairs[name]]], dtype=np.float64)
        (a, b) = cv2.perspectiveTransform(centers, matrix)[0]
        print(f"  {name:<7} discs at ({a[0]:+.4f}, {a[1]:+.4f}) and ({b[0]:+.4f}, {b[1]:+.4f})")

    print("\nRECTIFIED FRAME written beside the capture; measure wedges and rule "
          "bands off it, in the same relative coordinates.")
    _write_rectified(path, gray, quad)


# -- the microform chart -----------------------------------------------------


def find_crosses(gray: np.ndarray) -> np.ndarray:
    """The four corner crosses, whichever polarity the chart is printed in.

    A cross is found by its fill ratio rather than its size: it is the only mark
    here that occupies roughly an eighth of its own bounding box. Both polarities are
    tried and the one yielding exactly four is taken, because this chart ships as a
    positive and as a negative and nothing in the file says which.
    """
    ground = float(np.median(gray))
    for sign, name in ((-1.0, "dark on light"), (+1.0, "light on dark")):
        for level in (0.25, 0.35, 0.45, 0.55):
            mask = (sign * (gray - ground) > level * abs(sign)).astype(np.uint8)
            count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            hits = [
                centroids[i]
                for i in range(1, count)
                if 0.02 < stats[i][2] / gray.shape[1] < 0.06
                and 0.02 < stats[i][3] / gray.shape[0] < 0.06
                and 0.08 < stats[i][4] / (stats[i][2] * stats[i][3]) < 0.30
            ]
            if len(hits) == 4:
                print(f"  crosses found {name} at threshold {level}")
                points = np.array(hits)
                by_row = points[np.argsort(points[:, 1])]
                top = by_row[:2][np.argsort(by_row[:2, 0])]
                bottom = by_row[2:][np.argsort(by_row[2:, 0])]
                return np.array([top[0], top[1], bottom[1], bottom[0]])
    raise SystemExit("could not find four corner crosses")


def report_microform(path: Path) -> None:
    gray = load_luma(path)
    print(f"\n{path.name}: {gray.shape[1]} x {gray.shape[0]}")
    quad = find_crosses(gray)
    print("\nFIDUCIAL QUAD (paste as --corners):")
    print("  " + " ".join(f"{x:.1f},{y:.1f}" for x, y in quad))
    width = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    height = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    print(f"  {width:.1f} x {height:.1f} px, aspect {width / height:.4f}")
    print(f"  at one inch across the crosses that is {width:.0f} x {height:.0f} ppi -- "
          "compare the file's own header before believing either")
    _write_rectified(path, gray, quad)


# -- output ------------------------------------------------------------------


def _write_rectified(path: Path, gray: np.ndarray, quad: np.ndarray) -> None:
    out = path.with_name(path.stem + "_rectified.png")
    rect = rectify(gray, quad)
    cv2.imwrite(str(out), np.clip(rect * 255.0, 0, 255).astype(np.uint8))
    print(f"  wrote {out}")
    print(f"  the fiducial quad is the square from ({MARGIN}, {MARGIN}) to "
          f"({MARGIN + UNIT}, {MARGIN + UNIT}); relative = (pixel - {MARGIN}) / {UNIT}")


def measure_lattice(
    rectified: np.ndarray, box: tuple[int, int, int, int], columns: int, rows: int
) -> dict:
    """Fit a patch lattice inside a region of the rectified frame.

    ``box`` is (x0, y0, x1, y1) in rectified pixels, comfortably around the block.
    Returns the origin and spacing a definition wants, plus the residuals -- which
    are the reason to look at the numbers rather than paste them: a fit that locked
    onto a neighbouring feature is a large residual and an entirely plausible pitch.
    """
    x0, y0, x1, y1 = box
    block = rectified[y0:y1, x0:x1]
    if min(block.shape) < 40:
        raise SystemExit(
            f"the box {box} is {block.shape[1]}x{block.shape[0]} in the rectified "
            f"frame, too small to fit a lattice in. Remember the frame is "
            f"{UNIT + 2 * MARGIN} px square with the fiducial quad at "
            f"({MARGIN}, {MARGIN})-({MARGIN + UNIT}, {MARGIN + UNIT}), and that the "
            "box is X0 Y0 X1 Y1."
        )
    inset = max(4, min(block.shape) // 10)
    verticals = line_comb(block[inset:-inset, :].mean(axis=0), x0, columns + 1, 100)
    horizontals = line_comb(block[:, inset:-inset].mean(axis=1), y0, rows + 1, 100)
    pitch_u, first_u, worst_u = fit_comb(verticals)
    pitch_v, first_v, worst_v = fit_comb(horizontals)
    return {
        "grid_origin_relative": [round(relative(first_u), 5), round(relative(first_v), 5)],
        "patch_spacing_relative": {
            "horizontal": round(pitch_u / UNIT, 5),
            "vertical": round(pitch_v / UNIT, 5),
        },
        "patch_size_relative": {
            "width": round(pitch_u / UNIT, 5),
            "height": round(pitch_v / UNIT, 5),
        },
        "_residual_px": {"u": round(worst_u, 2), "v": round(worst_v, 2)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("isa", "microform"))
    parser.add_argument("capture", type=Path)
    parser.add_argument(
        "--lattice", nargs=6, metavar=("X0", "Y0", "X1", "Y1", "COLS", "ROWS"),
        action="append", default=[],
        help="fit a patch lattice inside a rectified-pixel box; repeatable",
    )
    args = parser.parse_args()

    if args.family == "isa":
        report_isa(args.capture)
    else:
        report_microform(args.capture)

    if args.lattice:
        gray = load_luma(args.capture)
        quad = (
            np.array([
                (np.asarray(a["center_px"]) + np.asarray(b["center_px"])) / 2.0
                for a, b in (
                    pair_discs(find_quarter_discs(gray))[n]
                    for n in ("TL", "TR", "BR", "BL")
                )
            ])
            if args.family == "isa"
            else find_crosses(gray)
        )
        rect = rectify(gray, quad)
        print("\nLATTICES:")
        for box in args.lattice:
            x0, y0, x1, y1, columns, rows = (int(v) for v in box)
            print(json.dumps(
                measure_lattice(rect, (x0, y0, x1, y1), columns, rows), indent=2
            ))


if __name__ == "__main__":
    main()
