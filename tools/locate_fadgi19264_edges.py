"""Locate the ISA FADGI-ISO 19264 modules' slanted edges, once per version.

Each of the chart's five resolution modules stacks two separate printed figures: a
band whose lower boundary slants a few degrees, and a block beside it whose side
boundary slants by the same amount. They share no centre, no size and no tilt, so
they are two edges rather than the sides of one shape, and the definition declares
them one at a time.

**The selection rule is the slant itself.** The module also contains the white
"dots per inch" field, a rectangle whose borders are the highest-contrast
boundaries anywhere in it -- and they are axis-aligned, so they repeat one
sub-pixel phase on every row and cannot be oversampled. Picking edges by contrast
would choose exactly the wrong two. Picking them by being a few degrees off the
axis chooses the right two and can hardly choose anything else, because nothing
else in the module is printed at an angle.

Run from the repository root:

    python tools/locate_fadgi19264_edges.py FADGI_19264.tif
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from locate_fadgi19264_geometry import module_fields, to_relative
from locate_patch_lattice import load_working

#: A printed slanted edge is a few degrees off the axis. The window excludes the
#: resolution field's own borders, which are axis-aligned, and excludes anything
#: turned far enough to be a wedge line or a numeral stroke.
SLANT_RANGE_DEG = (2.5, 9.0)
#: Shortest run that counts as a fragment of one of the module's edges, as a
#: fraction of the module field's short side. Deliberately well below the whole
#: edge: Hough returns one printed boundary as a scatter of collinear fragments,
#: and asking for the whole thing in one run yields a single fragment -- two
#: endpoints, too few to fit a line through. Fragments are merged below.
MIN_LENGTH_FRACTION = 0.15
#: How far outside the module field to search. The band sits above the field on a
#: corner module, so the window has to reach past it.
SEARCH_MARGIN = 0.45


def candidate_lines(gray: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
    """Long straight runs in a window, as (start, end, angle-off-axis, orientation)."""
    scaled = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(scaled, (0, 0), 1.2), 40, 120)
    span = min(gray.shape)
    found = cv2.HoughLinesP(
        edges, 1, np.pi / 1800, threshold=40,
        minLineLength=int(span * MIN_LENGTH_FRACTION), maxLineGap=10,
    )
    if found is None or len(found) == 0:
        return []
    # OpenCV has returned this as (N, 1, 4) and as (N, 4) across versions.
    segments = found.reshape(-1, 4)
    out = []
    for x1, y1, x2, y2 in segments:
        start = np.array([float(x1), float(y1)])
        end = np.array([float(x2), float(y2)])
        direction = end - start
        if np.linalg.norm(direction) < 1:
            continue
        vertical = abs(direction[1]) >= abs(direction[0])
        # Angle away from the nearest axis, which is what "slanted" means here.
        off = abs(np.degrees(np.arctan2(
            abs(direction[0]) if vertical else abs(direction[1]),
            abs(direction[1]) if vertical else abs(direction[0]),
        )))
        if not SLANT_RANGE_DEG[0] <= off <= SLANT_RANGE_DEG[1]:
            continue
        out.append((start, end, off, "vertical" if vertical else "horizontal"))
    return out


def merge(lines, orientation: str) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Reduce the runs Hough found along one edge to a single fitted segment.

    Hough returns a scatter of short collinear fragments for one printed boundary.
    Fitting one line through all of their endpoints at once, rather than taking the
    longest fragment, uses the whole edge and is what makes the reported angle
    stable to a hundredth of a degree between modules.
    """
    chosen = [line for line in lines if line[3] == orientation]
    if not chosen:
        return None
    points = np.vstack([np.vstack([s, e]) for s, e, _, _ in chosen])

    # Cluster each fragment by **its own line extrapolated to a common abscissa**,
    # not by its raw perpendicular coordinate and not by a shared slope.
    #
    # The raw coordinate does not work because a slanted edge's own two ends differ
    # in it by as much as two separate boundaries do -- every edge on this chart
    # slants 25 px over its length, which is the separation being tested for, so one
    # edge splits in half and half of it is then discarded.
    #
    # A single shared slope does not work either, and fails worse. The Siemens star
    # beside the centre module radiates fragments at every angle, and their median
    # slope comes out at -3.3 degrees -- the wrong sign for the real edge, whose two
    # endpoints are then pushed 59 px apart in the key and split again.
    #
    # Each fragment's own slope has neither problem: two fragments of one printed
    # boundary agree however that boundary is turned, and fragments of different
    # features do not.
    centre = float(np.mean(points[:, 1] if orientation == "vertical" else points[:, 0]))
    key = np.empty(len(points))
    for index, (start, end, _, _) in enumerate(chosen):
        direction = end - start
        run = direction[1] if orientation == "vertical" else direction[0]
        rise = direction[0] if orientation == "vertical" else direction[1]
        slope = rise / run if abs(run) > 1 else 0.0
        for half, point in enumerate((start, end)):
            along = point[1] if orientation == "vertical" else point[0]
            perpendicular = point[0] if orientation == "vertical" else point[1]
            key[index * 2 + half] = perpendicular + slope * (centre - along)

    order = np.argsort(key)
    groups: list[list[int]] = [[int(order[0])]]
    for index in order[1:]:
        if key[index] - key[groups[-1][-1]] > 20:
            groups.append([])
        groups[-1].append(int(index))

    # Every cluster is fitted and judged, and the longest one that is still slanted
    # after fitting wins. Taking the most populous cluster instead is what put an
    # ROI on the Siemens star: the star radiates lines at every angle, so a handful
    # land inside the slant window individually, outvote the real edge on fragment
    # count, and fit to a line that is not slanted at all. The fitted angle is the
    # check the individual fragments cannot provide.
    best = None
    for members in groups:
        group = points[members]
        if len(group) < 2:
            continue
        if orientation == "vertical":
            slope, intercept = np.polyfit(group[:, 1], group[:, 0], 1)
            lo, hi = group[:, 1].min(), group[:, 1].max()
            a = np.array([slope * lo + intercept, lo])
            b = np.array([hi * 0 + slope * hi + intercept, hi])
        else:
            slope, intercept = np.polyfit(group[:, 0], group[:, 1], 1)
            lo, hi = group[:, 0].min(), group[:, 0].max()
            a = np.array([lo, slope * lo + intercept])
            b = np.array([hi, slope * hi + intercept])
        angle = float(abs(np.degrees(np.arctan(slope))))
        if not SLANT_RANGE_DEG[0] <= angle <= SLANT_RANGE_DEG[1]:
            continue
        extent = float(hi - lo)
        if best is None or extent > best[0]:
            best = (extent, a, b, angle)
    if best is None:
        return None
    return best[1], best[2], best[3]


def module_edges(gray: np.ndarray, box: tuple[float, float, float, float]):
    """The two slanted edges of one module, in pixels."""
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    wx, wy = width * SEARCH_MARGIN, height * SEARCH_MARGIN
    sx0 = int(max(x0 - wx, 0)); sx1 = int(min(x1 + wx, gray.shape[1]))
    sy0 = int(max(y0 - wy, 0)); sy1 = int(min(y1 + wy, gray.shape[0]))
    window = gray[sy0:sy1, sx0:sx1]
    lines = candidate_lines(window)
    offset = np.array([sx0, sy0], dtype=np.float64)

    out = {}
    for orientation in ("horizontal", "vertical"):
        merged = merge(lines, orientation)
        if merged is None:
            continue
        a, b, angle = merged
        out[orientation] = (a + offset, b + offset, angle)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--long-edge", type=int, default=6000)
    args = parser.parse_args()

    rgb, scale = load_working(args.capture, args.long_edge)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fields = module_fields(rgb)
    quad = np.array([
        [(fields[n][0] + fields[n][2]) / 2, (fields[n][1] + fields[n][3]) / 2]
        for n in ("TL", "TR", "BR", "BL")
    ])
    convert = to_relative(quad)

    print(f"capture {args.capture}, working {rgb.shape[1]}x{rgb.shape[0]}\n")
    regions = {}
    for name in ("TL", "TR", "CENTER", "BL", "BR"):
        if name not in fields:
            print(f"  module {name}: field not located"); continue
        edges = module_edges(gray, fields[name])
        declared = []
        for orientation, (a, b, angle) in sorted(edges.items()):
            rel = convert([a, b])
            length = float(np.linalg.norm(b - a))
            print(f"  module {name:6s} {orientation:10s} {angle:5.2f} deg off axis, "
                  f"{length:5.0f} px, ({a[0]:7.1f},{a[1]:7.1f}) -> ({b[0]:7.1f},{b[1]:7.1f})")
            declared.append({
                "id": orientation[0],
                "orientation": orientation,
                "start": [round(float(rel[0][0]), 5), round(float(rel[0][1]), 5)],
                "end": [round(float(rel[1][0]), 5), round(float(rel[1][1]), 5)],
            })
        if len(declared) != 2:
            print(f"     WARNING: {len(declared)} slanted edge(s) found, expected 2")
        regions[f"module_{name.lower()}"] = declared

    print("\nstraight_edges blocks, relative to the four corner dpi-field centres:")
    print(json.dumps(regions, indent=2))


if __name__ == "__main__":
    main()
