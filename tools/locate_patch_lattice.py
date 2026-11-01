"""Measure a regular patch lattice off a capture, and print it as a target region.

Four of the targets OpenDICE ships samples for are a single regular array of
patches and nothing else -- no slanted edges, no printed rules. For those, the
whole geometry problem is: where does the lattice sit, how big is a patch, and
what does that look like in the frame the operator's four clicks define.

The method is the one the ColorChecker SG's geometry was measured with, and the
reason it is trustworthy is that the lattice over-determines itself. A 13 x 10
array has 130 patches and a projective fit from index space to pixel space has
eight parameters, so 122 of the patches are a check rather than an input. The
residual is printed, and a residual much above a tenth of a patch means the fit
found something other than the array.

Detection is deliberately blind to color. Patches are found as locally distinct
regions rather than as "things brighter than the surround", because on every one
of these charts some patches are darker than their own mount -- an array found by
thresholding is an array missing its blacks, and the blacks are where a lattice
fit is most constrained.

Run from the repository root, e.g.:

    python tools/locate_patch_lattice.py NGT.tif \
        --rows 10 --columns 13
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

#: Long edge of the working raster used for detection. Full resolution buys
#: nothing here -- a patch is hundreds of pixels across and its boundary is
#: located from a gradient ridge, not from a single pixel -- and it costs a
#: minute per capture.
WORKING_LONG_EDGE = 2400

#: A detected blob is kept if its area sits within this factor of the median
#: blob area. Patch arrays are uniform by construction, so anything well off the
#: median is lettering, a mount screw or a merged pair.
AREA_TOLERANCE = (0.55, 1.8)

#: How square a blob must be. Circular patches are square in bounding box too,
#: so this separates patches from printed labels and rules without caring which
#: shape the chart uses.
ASPECT_TOLERANCE = 0.22

#: How far a stacked cell reaches from its patch center, in pitches. Just over
#: half, so the cell holds the whole patch and the gap around it and stops in the
#: middle of its neighbours -- any further and a neighbour's boundary lands in
#: the composite too.
CELL_REACH = 0.62
#: Resolution of a stacked cell. Finer than the patch pitch on any of these
#: captures, so the boundary is not quantized by the stacking.
CELL_SAMPLES = 257

#: How much of its own bounding box a patch fills, by shape. A square fills all of
#: it and a circle pi/4 = 0.785, so this separates the two -- which matters on a
#: chart carrying both. The ISA FADGI-ISO 19264 commercial version prints a
#: fifteen-step grey wedge whose cells are rectangles sitting at the patch pitch:
#: without a shape to exclude them they read as a sixteenth row of patches and
#: shift every row index below them.
SHAPE_FILL = {
    "any": (0.60, 1.05),
    "circle": (0.65, 0.88),
    "square": (0.85, 1.05),
}


@dataclass(frozen=True)
class Lattice:
    """A fitted index-space-to-pixel map, with the evidence it was fitted from."""

    homography: np.ndarray
    rows: int
    columns: int
    #: (col, row) index assigned to each accepted blob, and where it was found.
    indices: np.ndarray
    centers: np.ndarray
    residual_px: float
    patch_size_px: tuple[float, float]
    detection_percentile: float = 0.0
    #: Coefficient of variation of the per-patch size measurements. The check on
    #: the size, not a property of the chart: a regular array's patches agree.
    size_spread: float = float("nan")

    def with_size(self, width: float, height: float, spread: float) -> "Lattice":
        return Lattice(
            homography=self.homography,
            rows=self.rows,
            columns=self.columns,
            indices=self.indices,
            centers=self.centers,
            residual_px=self.residual_px,
            patch_size_px=(width, height),
            detection_percentile=self.detection_percentile,
            size_spread=spread,
        )

    def center(self, row: int, col: int) -> np.ndarray:
        point = np.array([[[float(col), float(row)]]], dtype=np.float64)
        return cv2.perspectiveTransform(point, self.homography)[0, 0]

    def corner_box(self) -> np.ndarray:
        """Outer corners of the four extreme patches, TL, TR, BR, BL."""
        wu, wv = self.patch_size_px
        half = np.array([wu / 2.0, wv / 2.0])
        tl = self.center(0, 0) + half * (-1, -1)
        tr = self.center(0, self.columns - 1) + half * (1, -1)
        br = self.center(self.rows - 1, self.columns - 1) + half * (1, 1)
        bl = self.center(self.rows - 1, 0) + half * (-1, 1)
        return np.array([tl, tr, br, bl])


def load_working(path: str, long_edge: int = WORKING_LONG_EDGE) -> tuple[np.ndarray, float]:
    """The capture as an RGB working raster, with the scale back to full size."""
    image = Image.open(path)
    full_w, full_h = image.size
    shrink = max(full_w, full_h) / long_edge
    if shrink > 1.0:
        # draft() only helps JPEG and JPEG 2000; reduce() covers everything else
        # and is an exact box filter, which is what averaging a patch wants.
        image = image.reduce(max(int(shrink), 1))
    rgb = np.asarray(image.convert("RGB"))
    if rgb.dtype != np.uint8:
        rgb = (rgb.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    return rgb, full_w / rgb.shape[1]


def find_patch_blobs(
    rgb: np.ndarray, expected: int, shape: str = "any"
) -> tuple[np.ndarray, float]:
    """Centroids and bounding boxes of everything shaped like a patch.

    Patches are found as the regions *between* edges rather than as regions of
    some lightness, because on every one of these charts a patch is darker than
    its own mount somewhere in the array. What counts as an edge is not a fixed
    number either: it depends on how much of the frame the chart fills and on how
    much surface texture the print has. So the threshold is swept, and the
    setting that harvests the most patch-shaped regions without exceeding the
    array's own size wins. That is a self-tuning step with a stated criterion,
    not a constant tuned on one capture and left to rot.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    magnitude = np.hypot(
        cv2.Scharr(blur, cv2.CV_32F, 1, 0), cv2.Scharr(blur, cv2.CV_32F, 0, 1)
    )

    best: np.ndarray = np.empty((0, 5))
    best_percentile = 0.0
    for percentile in np.arange(78.0, 96.0, 1.0):
        edge = (magnitude > np.percentile(magnitude, percentile)).astype(np.uint8)
        edge = cv2.dilate(edge, np.ones((3, 3), np.uint8))
        found = _regions_between(1 - edge, shape)
        if len(found) > len(best) and len(found) <= expected:
            best, best_percentile = found, float(percentile)
    return best, best_percentile


def _regions_between(mask: np.ndarray, shape: str = "any") -> np.ndarray:
    """Patch-shaped connected components of ``mask``, as (x, y, w, h, area)."""
    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 4)
    boxes, centers = stats[1:], centroids[1:]
    if len(boxes) == 0:
        return np.empty((0, 5))

    areas = boxes[:, cv2.CC_STAT_AREA].astype(float)
    widths = boxes[:, cv2.CC_STAT_WIDTH].astype(float)
    heights = boxes[:, cv2.CC_STAT_HEIGHT].astype(float)

    seed = _modal_area(areas)
    lo, hi = AREA_TOLERANCE
    keep = (areas > seed * lo) & (areas < seed * hi)
    keep &= np.abs(widths - heights) / np.maximum(widths, heights) < ASPECT_TOLERANCE
    # How much of its own bounding box the region fills. Bounded on both sides,
    # not just below: the upper bound is what tells a circular patch from a
    # rectangle sitting at the same pitch.
    fill_low, fill_high = SHAPE_FILL[shape]
    fill = areas / np.maximum(widths * heights, 1.0)
    keep &= (fill > fill_low) & (fill < fill_high)
    return np.column_stack([centers[keep], widths[keep], heights[keep], areas[keep]])


def _modal_area(areas: np.ndarray) -> float:
    """The area a patch has, taken as the commonest area in the frame.

    The array's defining property is that it repeats: a hundred and thirty
    components share one area to within the print tolerance, and nothing else in
    a capture does. So the mode is the patch, and it survives a frame component
    larger than every patch put together and a scattering of dust smaller than
    any of them -- neither of which a median or a rank could be told apart from
    the array itself.
    """
    large = areas[areas >= 64.0]
    if len(large) == 0:
        return float(np.median(areas))
    log_area = np.log10(large)
    bins = np.arange(log_area.min(), log_area.max() + 0.08, 0.08)
    counts, _ = np.histogram(log_area, bins=bins)
    if len(counts) == 0:
        return float(np.median(large))
    # Widen by one bin either side: a patch's area varies a little with its own
    # boundary contrast, and a peak that straddles a bin edge would read half.
    padded = np.convolve(counts, [1, 1, 1], mode="same")
    peak = int(np.argmax(padded))
    lo, hi = 10 ** bins[max(peak - 1, 0)], 10 ** bins[min(peak + 2, len(bins) - 1)]
    within = large[(large >= lo) & (large <= hi)]
    return float(np.median(within)) if len(within) else float(np.median(large))


def assign_indices(centers: np.ndarray, rows: int, columns: int) -> np.ndarray:
    """Give each detected center its (col, row) place in the lattice.

    The array is close enough to axis-aligned on a flatbed or copy-stand capture
    that ranking by coordinate is enough to index it, and the projective fit that
    follows is what absorbs the residual rotation and keystone. Rank rather than
    absolute position, so an array that does not start at the frame edge still
    indexes correctly.
    """
    xs, ys = centers[:, 0], centers[:, 1]
    col_edges = _cluster_1d(xs, columns)
    row_edges = _cluster_1d(ys, rows)
    if col_edges is None or row_edges is None:
        raise SystemExit(
            f"Could not separate {columns} columns and {rows} rows from "
            f"{len(centers)} detected patches. Check --rows/--columns."
        )
    col_index = np.argmin(np.abs(xs[:, None] - col_edges[None, :]), axis=1)
    row_index = np.argmin(np.abs(ys[:, None] - row_edges[None, :]), axis=1)
    return np.column_stack([col_index, row_index])


def _cluster_1d(values: np.ndarray, groups: int) -> np.ndarray | None:
    """Where each of ``groups`` lines of the lattice sits along one axis.

    Not k-means. A lattice is regular, and that is worth more than a general
    clustering: the values are split at their own large gaps, the pitch is read
    from the run of resulting group centers, and the full set of line positions
    is then *predicted* from the pitch. A column with no detected patch in it --
    an all-dark column, or one a mount screw crosses -- therefore still gets its
    place, where a clustering forced to return k groups would have split a
    neighbour in two to make up the number.
    """
    if len(values) < 3:
        return None
    ordered = np.sort(values)
    span = float(ordered[-1] - ordered[0])
    if span <= 0:
        return None
    gaps = np.diff(ordered)
    # A gap belongs *between* lines if it is a large fraction of the smallest
    # plausible pitch. With `groups` lines across `span`, the pitch is at most
    # span/(groups-1); half of that separates lines from within-line scatter.
    threshold = 0.5 * span / max(groups - 1, 1)
    boundaries = np.where(gaps > threshold)[0]
    starts = np.concatenate([[0], boundaries + 1])
    ends = np.concatenate([boundaries + 1, [len(ordered)]])
    seen = np.array([float(np.mean(ordered[s:e])) for s, e in zip(starts, ends)])
    if len(seen) < 2:
        return None

    # The pitch is the spacing between adjacent detected lines, taken as the
    # median of each gap divided by however many pitches it spans.
    steps = np.diff(seen)
    unit = float(np.median(steps))
    counts = np.maximum(np.round(steps / unit), 1)
    pitch = float(np.median(steps / counts))
    origin = float(seen[0] - round((seen[0] - ordered[0]) / pitch) * pitch)
    return origin + pitch * np.arange(groups, dtype=float)


def fit_lattice(rgb: np.ndarray, rows: int, columns: int, shape: str = "any") -> Lattice:
    blobs, percentile = find_patch_blobs(rgb, rows * columns, shape)
    if len(blobs) < 8:
        raise SystemExit(f"Only {len(blobs)} patch-like regions found; cannot fit a lattice.")
    centers = blobs[:, :2]
    indices = assign_indices(centers, rows, columns)

    # Duplicates mean two blobs landed in one cell -- keep the one nearest the
    # cell's own median, since the other is a highlight or a split patch.
    keep = _drop_duplicate_cells(indices, centers)
    indices, centers = indices[keep], centers[keep]

    homography, _ = cv2.findHomography(indices.astype(np.float64), centers, cv2.RANSAC, 3.0)
    if homography is None:
        raise SystemExit("Homography fit failed; the detected patches are not a lattice.")
    predicted = cv2.perspectiveTransform(
        indices.astype(np.float64).reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    residual = float(np.sqrt(np.mean(np.sum((predicted - centers) ** 2, axis=1))))

    rough = Lattice(
        homography=homography,
        rows=rows,
        columns=columns,
        indices=indices,
        centers=centers,
        residual_px=residual,
        patch_size_px=(float(np.median(blobs[:, 2])), float(np.median(blobs[:, 3]))),
        detection_percentile=percentile,
    )
    return refine(rgb, rough)


def refine(rgb: np.ndarray, rough: Lattice) -> Lattice:
    """Measure the patch size by stacking every patch on top of every other.

    The regions the rough lattice came from were carved out of a *dilated* edge
    mask, so each is a few pixels smaller than the patch that produced it, and by
    a different amount each time -- a black patch on a black mount has a fainter
    boundary than a white one and loses more of it. That is fine for finding the
    array and useless as a measurement, and the size is what decides how far a
    sampling ROI reaches.

    Measuring each patch separately does not work either, and it is worth saying
    why rather than leaving the next reader to rediscover it: a single profile
    across one patch crosses that patch's own dust, texture and bevel, and on a
    chart whose patches sit in recessed wells the well's shoulder is a stronger
    ridge than the patch boundary about half the time. Per-patch, that lost two
    thirds of the array and left a 10% spread.

    So the patches are stacked instead. Each is resampled into a common cell,
    scaled to its own contrast so a pale patch counts as much as a saturated one,
    and the gradient of the average is taken. The boundary is the one feature all
    hundred and thirty share; everything else averages away. What comes back is
    the median patch outline, which is exactly what a definition wants.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    pitch_u = float(np.hypot(*(rough.center(0, 1) - rough.center(0, 0))))
    pitch_v = float(np.hypot(*(rough.center(1, 0) - rough.center(0, 0))))

    stack = _stack_cells(gray, rough, pitch_u, pitch_v)
    if stack is None:
        return rough.with_size(*rough.patch_size_px, float("nan"))

    scale_u = (pitch_u * CELL_REACH * 2) / CELL_SAMPLES
    scale_v = (pitch_v * CELL_REACH * 2) / CELL_SAMPLES
    size = _size_of(stack, scale_u, scale_v)
    if size is None:
        return rough.with_size(*rough.patch_size_px, float("nan"))

    # Split the array into quarters and measure each on its own. Four
    # independent readings of the same printed patch have to agree; if they do
    # not, the composite is averaging over something that varies across the
    # chart and a single median is the wrong summary of it.
    quarters = [
        _size_of(stack[part], scale_u, scale_v)
        for part in np.array_split(np.arange(len(stack)), 4)
    ]
    measured = [q for q in quarters if q is not None]
    if len(measured) < 2:
        return rough.with_size(*size, float("nan"))
    spread = max(
        float(np.ptp([m[axis] for m in measured]) / size[axis]) for axis in (0, 1)
    )
    return rough.with_size(*size, spread)


def _size_of(
    stack: np.ndarray, scale_u: float, scale_v: float
) -> tuple[float, float] | None:
    composite = stack.mean(axis=0)
    span_u = _boundary_from_composite(composite.mean(axis=0))
    span_v = _boundary_from_composite(composite.mean(axis=1))
    if span_u is None or span_v is None:
        return None
    return span_u * scale_u, span_v * scale_v


def _stack_cells(
    gray: np.ndarray, lattice: Lattice, pitch_u: float, pitch_v: float
) -> np.ndarray | None:
    """Every patch resampled into one common cell, contrast-normalized."""
    reach_u, reach_v = pitch_u * CELL_REACH, pitch_v * CELL_REACH
    axis = np.linspace(-1.0, 1.0, CELL_SAMPLES, dtype=np.float32)
    offset_u, offset_v = np.meshgrid(axis * reach_u, axis * reach_v)

    cells: list[np.ndarray] = []
    for (col, row) in lattice.indices:
        cx, cy = lattice.center(int(row), int(col))
        map_x = (offset_u + cx).astype(np.float32)
        map_y = (offset_v + cy).astype(np.float32)
        if (
            map_x.min() < 0 or map_y.min() < 0
            or map_x.max() >= gray.shape[1] or map_y.max() >= gray.shape[0]
        ):
            continue
        cell = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR)
        # Scale each cell to its own contrast, and flip so the patch is always
        # the bright side. Without the flip a dark patch on a light mount and a
        # light patch on a dark mount cancel at the boundary instead of adding.
        middle = cell[CELL_SAMPLES // 3 : 2 * CELL_SAMPLES // 3, CELL_SAMPLES // 3 : 2 * CELL_SAMPLES // 3]
        surround = np.concatenate([cell[0], cell[-1], cell[:, 0], cell[:, -1]])
        contrast = float(middle.mean() - surround.mean())
        if abs(contrast) < 1.0:
            continue
        span = float(cell.max() - cell.min())
        if span <= 0:
            continue
        cells.append(np.sign(contrast) * (cell - cell.mean()) / span)

    return np.stack(cells) if len(cells) >= 8 else None


def _boundary_from_composite(profile: np.ndarray) -> float | None:
    """Width of the patch in a composite profile, in cell samples.

    The composite runs from the middle of one gap, across the patch, to the
    middle of the next, so its gradient has exactly two ridges and they sit
    either side of the center. Nothing has to be rejected: any feature that was
    not on every patch is no longer in the picture.
    """
    smooth = cv2.GaussianBlur(profile.reshape(-1, 1).astype(np.float32), (0, 0), 1.0).ravel()
    slope = np.abs(np.gradient(smooth))
    middle = len(slope) // 2
    margin = 2
    left, right = slope[margin:middle], slope[middle:-margin]
    if len(left) == 0 or len(right) == 0:
        return None
    near = margin + _sub_sample_peak(left)
    far = middle + _sub_sample_peak(right)
    span = far - near
    return span if span > CELL_SAMPLES * 0.15 else None


def _sub_sample_peak(slope: np.ndarray) -> float:
    """Where a gradient ridge peaks, to better than a sample, by parabolic fit."""
    peak = int(np.argmax(slope))
    if peak == 0 or peak == len(slope) - 1:
        return float(peak)
    before, here, after = slope[peak - 1], slope[peak], slope[peak + 1]
    denominator = before - 2.0 * here + after
    if denominator == 0:
        return float(peak)
    return float(peak + 0.5 * (before - after) / denominator)


def _drop_duplicate_cells(indices: np.ndarray, centers: np.ndarray) -> np.ndarray:
    seen: dict[tuple[int, int], list[int]] = {}
    for i, (c, r) in enumerate(indices):
        seen.setdefault((int(c), int(r)), []).append(i)
    keep = np.zeros(len(indices), dtype=bool)
    for members in seen.values():
        if len(members) == 1:
            keep[members[0]] = True
            continue
        pick = min(members, key=lambda i: float(np.sum(centers[i] ** 2)))
        keep[pick] = True
    return keep


def relative_geometry(lattice: Lattice, quad: np.ndarray) -> dict:
    """Express the lattice in the 0..1 frame the four fiducial points define."""
    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    to_relative, _ = cv2.findHomography(quad.astype(np.float64), unit)

    def rel(points: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(
            np.asarray(points, dtype=np.float64).reshape(-1, 1, 2), to_relative
        ).reshape(-1, 2)

    corners = np.array(
        [
            lattice.center(0, 0),
            lattice.center(0, lattice.columns - 1),
            lattice.center(lattice.rows - 1, lattice.columns - 1),
            lattice.center(lattice.rows - 1, 0),
        ]
    )
    r = rel(corners)
    spacing_u = (r[1][0] - r[0][0]) / (lattice.columns - 1)
    spacing_v = (r[3][1] - r[0][1]) / (lattice.rows - 1)

    wu, wv = lattice.patch_size_px
    centre = lattice.center(lattice.rows // 2, lattice.columns // 2)
    span = rel(np.array([centre - (wu / 2, wv / 2), centre + (wu / 2, wv / 2)]))
    size_u = float(span[1][0] - span[0][0])
    size_v = float(span[1][1] - span[0][1])

    return {
        "patch_size_relative": {"width": round(size_u, 5), "height": round(size_v, 5)},
        "patch_spacing_relative": {
            "horizontal": round(float(spacing_u), 5),
            "vertical": round(float(spacing_v), 5),
        },
        "grid_origin_relative": [
            round(float(r[0][0] - size_u / 2), 5),
            round(float(r[0][1] - size_v / 2), 5),
        ],
        "provisional": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument(
        "--long-edge",
        type=int,
        default=WORKING_LONG_EDGE,
        help=(
            "long edge of the working raster. Raise it when the patches come "
            "out too small to separate; the default trades detail for speed."
        ),
    )
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--columns", type=int, required=True)
    parser.add_argument(
        "--shape",
        choices=sorted(SHAPE_FILL),
        default="any",
        help=(
            "what shape the patches are. Bounds how much of its bounding box a "
            "region must fill, which is what separates circular patches from a "
            "rectangle printed at the same pitch."
        ),
    )
    parser.add_argument(
        "--quad",
        help=(
            "Fiducial quad in FULL-RESOLUTION pixels as x,y;x,y;x,y;x,y (TL,TR,BR,BL). "
            "Omit to use the patch array's own bounding box, which is what an "
            "operator places on a chart carrying no registration marks."
        ),
    )
    args = parser.parse_args()

    rgb, scale = load_working(args.capture, args.long_edge)
    lattice = fit_lattice(rgb, args.rows, args.columns, args.shape)
    print(f"capture      {args.capture}")
    print(f"working      {rgb.shape[1]}x{rgb.shape[0]} px, {scale:.4f} full px per working px")
    print(f"patches used {len(lattice.centers)} of {args.rows * args.columns}")
    print(f"edge cutoff  {lattice.detection_percentile:.0f}th percentile of gradient")
    print(
        f"patch size   {lattice.patch_size_px[0]:.1f} x {lattice.patch_size_px[1]:.1f} "
        f"working px, spread {100 * lattice.size_spread:.2f}%"
    )
    print(
        f"lattice fit  residual {lattice.residual_px:.2f} working px "
        f"({100 * lattice.residual_px / lattice.patch_size_px[0]:.2f}% of a patch)"
    )

    if args.quad:
        quad = np.array(
            [[float(v) for v in point.split(",")] for point in args.quad.split(";")]
        ) / scale
        print("quad         supplied, in full-resolution pixels")
    else:
        quad = lattice.corner_box()
        print("quad         patch-array bounding box (no registration marks used)")
    print("quad (full px)")
    for name, point in zip(("TL", "TR", "BR", "BL"), quad * scale):
        print(f"  {name}  {point[0]:9.2f} {point[1]:9.2f}")

    print("\npatch_grid geometry:")
    print(json.dumps(relative_geometry(lattice, quad), indent=2))


if __name__ == "__main__":
    main()
