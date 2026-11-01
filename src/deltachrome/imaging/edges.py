"""Locating slanted edges on a target and cutting ROIs from them.

A slanted-edge region declares roughly where its edge feature sits; the exact edge
is found in the capture. Declaring exact positions instead would bake one
specimen's print registration into a definition meant to describe a model, and
would break the moment a chart was photographed at a slightly different scale.

For a target like the ISA Golden Thread the feature is a dark square rotated a few
degrees, giving four usable edges per module. Its deliberate tilt is what makes the
measurement possible at all: an edge aligned to the pixel grid repeats the same
sub-pixel phase on every row and cannot be oversampled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from deltachrome.imaging.geometry import ChartPlacement
from deltachrome.imaging.loader import LoadedImage, to_display_records, to_luma
from deltachrome.targets.schema import TargetDefinition

#: Upper bound on how much of an edge is used, centerd on its midpoint. Corners are
#: where two edges interact and the profile stops being a single step, so the ends
#: are never taken however long the ROI needs to be.
DEFAULT_EDGE_LENGTH_FRACTION = 0.55
#: Perpendicular half-width, in pixels either side of the edge. It has to hold the
#: whole transition plus flat plateau on both sides -- the plateaux set the levels
#: the response is normalized against. Wider than that only invites a neighboring
#: feature or an illumination gradient into the ROI. Published practice sits in this
#: range: Burns reports 60x40 and 75-pixel-wide ROIs, and FADGI's own profile
#: suggests 100 px.
DEFAULT_ROI_HALF_WIDTH = 50
#: Pixels of sub-pixel phase the ROI should cover. One is the floor below which the
#: profile has gaps; asking for several averages over more of the pixel and is
#: cheap when the edge is long enough.
DEFAULT_TARGET_PHASE_COVERAGE = 4.0
#: No ROI shorter than this, whatever the geometry suggests.
MIN_ROI_ROWS = 24
#: A quarter disc fills pi/4 = 0.785 of its bounding square. The band allows for
#: threshold and antialiasing slack while still excluding a full square (1.0) or a
#: half (0.5), either of which would mean the wrong feature was found.
QUARTER_DISC_FILL = (0.68, 0.90)

#: How far either side of a declared straight edge to look for the real one, in
#: pixels. It bounds the placement error the refinement can absorb, and it is also
#: the guard against snapping onto a *different* edge: on the ISA FADGI-ISO 19264
#: module the nearest other boundary is about 60 px away, so a search wider than
#: this would let one edge capture its neighbour.
STRAIGHT_EDGE_SEARCH_PX = 22
#: Perpendicular profiles taken along a declared edge when refining it. Enough to
#: fit a line through robustly; the ends are skipped because a corner is where two
#: features meet and the profile stops being a single step.
STRAIGHT_EDGE_SAMPLES = 48
#: How far the refined line may turn from the declared one before the refinement is
#: assumed to have found something else and is discarded.
STRAIGHT_EDGE_MAX_ROTATION_DEG = 3.0
#: Scatter of the located points about the fitted line, above which the same.
STRAIGHT_EDGE_MAX_RESIDUAL_PX = 1.5


#: The direction a slanted edge measures the response *in*, given the way the
#: edge itself runs. A near-vertical edge is crossed left to right, so it reports
#: the horizontal response; a near-horizontal edge reports the vertical one.
#:
#: ISO 12233:2017 names the direction rather than the edge. Its definitions read
#: horizontal resolution as the value "measured in the longer image dimension ...
#: typically using a vertical or near vertical oriented test-edge", and vertical
#: resolution as the converse. So :attr:`EdgeROI.orientation` -- which is the
#: edge's own geometry, and has to be, since it decides which way the ROI is cut
#: -- is not the word to show a reader in this trade, who reads "vertical"
#: as vertical resolution.
#:
#: This lives in the engine rather than in the frontend because it is a fact
#: about the standard, and because the console report and the desktop report have
#: to agree: one application using one word for opposite things is worse than
#: either convention on its own.
SFR_DIRECTIONS = {"vertical": "horizontal", "horizontal": "vertical"}


def sfr_direction(orientation: str | None) -> str:
    """Which resolution an edge of this geometry measures.

    Presentation only. Nothing in a record changes: ``orientation`` stays the
    edge's geometry, because a year of saved runs has to keep meaning what it
    meant, and because eighteen target definitions declare it that way.
    """
    text = str(orientation or "").lower()
    return SFR_DIRECTIONS.get(text, str(orientation or ""))


@dataclass
class EdgeROI:
    """One extracted slanted edge, ready for SFR analysis."""

    edge_id: str
    region_id: str
    pixels: np.ndarray
    orientation: str          # "vertical" or "horizontal" in image terms
    center_px: tuple[float, float]
    bounds_px: tuple[int, int, int, int]
    #: Maps this ROI's encoded RGB to linear luminance, through the capture's own
    #: ICC transform. The edge analysis needs a profile linear in luminance.
    linearize: Callable[[np.ndarray], np.ndarray] | None = None
    #: Maps encoded RGB to a linear signal **per record**, without mixing the
    #: channels. Registration is measured on the device's own color records, and
    #: the luminance transform above would blend them together.
    linearize_channels: Callable[[np.ndarray], np.ndarray] | None = None
    linearization: str = "none"
    notes: list[str] = field(default_factory=list)


def _square_corners(center: np.ndarray, half: np.ndarray, tilt_deg: float) -> np.ndarray:
    """The four corners of a rotated square in target-relative coordinates."""
    t = np.radians(tilt_deg)
    rotation = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    unit = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    # Rotate first, then scale into the (anisotropic) relative frame. Scaling before
    # rotating shears the square, because the fiducial quad is not square: on the ISA
    # target its aspect is about 1.75, which turns a 4.8 degree tilt into edges tens
    # of degrees off and destroys the measurement.
    return center + (unit @ rotation.T) * half


def extract_edges(
    image: LoadedImage,
    placement: ChartPlacement,
    target: TargetDefinition,
    *,
    roi_half_width: int = DEFAULT_ROI_HALF_WIDTH,
    length_fraction: float = DEFAULT_EDGE_LENGTH_FRACTION,
    target_phase_coverage: float = DEFAULT_TARGET_PHASE_COVERAGE,
    refine: bool = True,
) -> list[EdgeROI]:
    """Cut an ROI from every edge of every slanted-edge region on the target.

    ROI length is derived, not fixed. The method needs each row to catch the edge at
    a different sub-pixel phase, and the edge drifts by ``rows * tan(angle)`` pixels
    down the ROI -- so the rows required follow from the edge's own tilt, and a
    single "use N rows" constant would be too short on a shallow edge and wastefully
    long on a steep one. The requested coverage is capped by how much of the edge is
    usable, since the corners have to stay out.
    """
    profile = image.profile

    def linearize(rgb: np.ndarray) -> np.ndarray:
        """Encoded RGB to relative luminance, via the profile's own transform.

        Y from the PCS XYZ, so it works for matrix/TRC and LUT profiles alike rather
        than assuming a single gamma exponent.
        """
        return profile.to_xyz(rgb)[..., 1]

    linearize_channels, linearization = profile.channel_linearizer()

    rois: list[EdgeROI] = []
    for region in target.regions:
        if region.kind != "slanted_edge":
            continue
        for edge_id, region_id, a, b, narrow in _edge_segments(
            image, placement, region, refine
        ):
            roi = _cut_edge(
                image, a, b, edge_id, region_id,
                narrow or roi_half_width, length_fraction, target_phase_coverage,
            )
            if roi is not None:
                roi.linearize = linearize
                roi.linearize_channels = linearize_channels
                roi.linearization = linearization
                rois.append(roi)
    return rois


def _edge_segments(image, placement, region, refine: bool):
    """The edges a region offers, as (edge_id, region_id, start, end, half_width).

    ``half_width`` is ``None`` unless the definition declares one, in which case it
    overrides the caller's default for that edge alone.

    Three feature shapes, because the targets modelled here use three and the
    differences are not cosmetic. A **rotated square** gives four edges, all of them
    real. A **quarter disc** gives only two: the arc is not a straight edge, and an
    ROI cut across it holds no single transition at all -- fit residuals of 27 px
    against 0.06 for a real edge. So the apex has to be identified and only the two
    sides meeting there used.

    **Straight edges** are declared one at a time, by their endpoints, and exist
    because some charts do not build their edges out of a closed shape at all. The
    ISA FADGI-ISO 19264 module stacks two separate figures -- a band whose lower
    boundary slants, and a block beside it whose left boundary slants -- so its
    horizontal and vertical edges belong to different objects and share no centre,
    no size and no tilt. Describing that as a rotated square would be describing a
    figure the chart does not print, and would emit two further "edges" that are
    really the borders of the white resolution field: axis-aligned, and so with no
    sub-pixel phase to oversample.
    """
    geometry = region.geometry or {}
    discs = geometry.get("discs")
    straight = geometry.get("straight_edges")

    if straight:
        for index, edge in enumerate(straight):
            corners = placement.to_pixel(
                np.array([edge["start"], edge["end"]], dtype=np.float64)
            )
            a, b = corners[0], corners[1]
            if refine:
                a, b = _refine_straight_edge(image, a, b)
            # A feature can be too tight for the default half-width, and only the
            # definition knows that. The Golden Thread Object Level module's block
            # is 60 px deep with the chart's printed value table 25 px below its
            # boundary, so a 50 px ROI takes in both and the edge-line fit scatters
            # 17 px instead of 0.06. Declaring the constraint beside the edge is
            # better than lowering it for every target in the profile.
            yield (
                f"{region.region_id}_{edge.get('id', index)}", region.region_id, a, b,
                edge.get("roi_half_width"),
            )
        return

    if discs:
        for index, disc in enumerate(discs):
            corners_px = placement.to_pixel(
                _square_corners(
                    np.asarray(disc["center"], dtype=np.float64),
                    np.asarray(disc["half_size"], dtype=np.float64),
                    float(disc.get("tilt_deg", 0.0)),
                )
            )
            found = _refine_quarter_disc(image, corners_px) if refine else None
            if found is None:
                # Without the apex located there is no way to tell a straight side
                # from the arc, and guessing would put an ROI across the curve.
                continue
            box, apex = found
            for j, other in enumerate(((apex + 1) % 4, (apex - 1) % 4)):
                yield (f"{region.region_id}_d{index}_e{j}", region.region_id,
                       box[apex], box[other], geometry.get("roi_half_width"))
        return

    center, half = geometry.get("center"), geometry.get("half_size")
    if center is None or half is None:
        return
    corners_px = placement.to_pixel(
        _square_corners(
            np.asarray(center, dtype=np.float64),
            np.asarray(half, dtype=np.float64),
            float(geometry.get("tilt_deg", 0.0)),
        )
    )
    if refine:
        corners_px = _refine_square(image, corners_px)
    for i in range(4):
        yield (f"{region.region_id}_e{i}", region.region_id,
               corners_px[i], corners_px[(i + 1) % 4], geometry.get("roi_half_width"))


def _refine_quarter_disc(
    image: LoadedImage, corners_px: np.ndarray
) -> tuple[np.ndarray, int] | None:
    """Snap a declared square onto the quarter disc in the capture, and find its apex.

    Returns the four corners of the disc's bounding square and the index of the
    apex -- the corner where its two straight edges meet.

    **The apex is the corner nearest the mask's centroid.** A quarter disc's
    centroid sits about 0.42r from the apex along each axis, so the apex is 0.60r
    away while its two neighbors are 0.72r and the opposite corner 0.82r: a clean
    separation. Deciding instead by sampling the mask at each corner looks more
    direct and is a coin toss, because the apex is slightly rounded and the corners
    are antialiased -- and getting it wrong pairs a real apex with a corner across
    the arc, producing an ROI that holds no single edge while looking entirely
    plausible.
    """
    margin = int(0.3 * float(np.linalg.norm(corners_px[1] - corners_px[0])))
    x0 = int(max(corners_px[:, 0].min() - margin, 0))
    x1 = int(min(corners_px[:, 0].max() + margin, image.width))
    y0 = int(max(corners_px[:, 1].min() - margin, 0))
    y1 = int(min(corners_px[:, 1].max() + margin, image.height))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None

    window = image.window(y0, y1, x0, x1)
    gray = to_luma(window).astype(np.float32)
    level = np.uint8(np.clip(gray * 255.0, 0, 255))
    _, mask = cv2.threshold(level, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    declared = corners_px - np.array([x0, y0], dtype=np.float64)
    center = declared.mean(axis=0)
    at_center = labels[
        int(np.clip(center[1], 0, gray.shape[0] - 1)),
        int(np.clip(center[0], 0, gray.shape[1] - 1)),
    ]
    chosen = int(at_center) if at_center != 0 else 1 + int(
        np.argmax(stats[1:, cv2.CC_STAT_AREA])
    )
    points = np.column_stack(np.nonzero(labels == chosen))[:, ::-1].astype(np.float32)
    if len(points) < 100:
        return None

    rect = cv2.minAreaRect(points)
    width, height = rect[1]
    declared_side = float(np.linalg.norm(corners_px[1] - corners_px[0]))
    if not 0.6 < float(np.mean([width, height])) / max(declared_side, 1e-6) < 1.6:
        return None
    if not 0.8 < width / max(height, 1e-6) < 1.25:
        return None
    fill = len(points) / max(width * height, 1e-6)
    if not QUARTER_DISC_FILL[0] <= fill <= QUARTER_DISC_FILL[1]:
        # A quarter disc fills pi/4 of its bounding square. Anything else in the
        # window -- a solid square, a ruler, a text block -- is not this feature.
        return None

    box = cv2.boxPoints(rect).astype(np.float64)
    distances = np.linalg.norm(box - points.mean(axis=0), axis=1)
    apex = int(np.argmin(distances))
    return box + np.array([x0, y0], dtype=np.float64), apex


def _refine_straight_edge(
    image: LoadedImage, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Snap a declared straight edge onto the transition actually in the capture.

    A closed feature can be refined by finding the shape and taking its corners.
    A lone straight edge has no shape to find, so it is refined the way the edge
    itself is measured: perpendicular profiles are taken along the declared line,
    the strongest gradient in each locates the boundary to sub-pixel, and a line is
    fitted through those points.

    The fit is of the **offset from the declared line** against position along it,
    which is what lets one step absorb both a placement that is bodily out and one
    that is slightly turned. Both endpoints then move by their own fitted offset,
    so the segment keeps its declared length and its edge id stays stable.

    Refinement is discarded rather than trusted when it disagrees with the
    declaration too much -- more than a few degrees of rotation, or points that do
    not lie on a line. On a lone edge there is no shape to confirm the find, so the
    declaration is the only check available, and a refinement that has wandered onto
    a neighbouring boundary is worse than none.
    """
    direction = b - a
    length = float(np.linalg.norm(direction))
    if length < 40:
        return a, b
    unit = direction / length
    normal = np.array([-unit[1], unit[0]])

    reach = STRAIGHT_EDGE_SEARCH_PX
    x0 = int(max(min(a[0], b[0]) - reach - 4, 0))
    x1 = int(min(max(a[0], b[0]) + reach + 4, image.width))
    y0 = int(max(min(a[1], b[1]) - reach - 4, 0))
    y1 = int(min(max(a[1], b[1]) + reach + 4, image.height))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return a, b
    gray = image.luma_window(y0, y1, x0, x1).astype(np.float32)

    # Skip the ends: a corner is where two features meet, and the profile there is
    # not a single step.
    along = np.linspace(0.12, 0.88, STRAIGHT_EDGE_SAMPLES)
    across = np.arange(-reach, reach + 1, dtype=np.float64)
    origin = np.array([x0, y0], dtype=np.float64)

    points = a[None, :] + direction[None, :] * along[:, None]      # (S, 2)
    samples = points[:, None, :] + normal[None, None, :] * across[None, :, None]
    grid = (samples - origin).astype(np.float32)
    profiles = cv2.remap(
        gray, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    profiles = cv2.GaussianBlur(profiles, (3, 1), 0)

    slope = np.abs(np.gradient(profiles, axis=1))
    floor = float(np.median(slope))
    kept_t: list[float] = []
    kept_offset: list[float] = []
    for index, row in enumerate(slope):
        peak = int(np.argmax(row))
        if peak < 1 or peak > len(row) - 2 or row[peak] < floor * 3.0:
            continue
        before, here, after = row[peak - 1], row[peak], row[peak + 1]
        curve = before - 2.0 * here + after
        shift = 0.5 * (before - after) / curve if curve else 0.0
        kept_t.append(float(along[index]))
        kept_offset.append(float(across[peak] + shift))

    if len(kept_t) < STRAIGHT_EDGE_SAMPLES // 2:
        return a, b

    t = np.array(kept_t)
    offset = np.array(kept_offset)
    for _ in range(2):
        gradient, intercept = np.polyfit(t, offset, 1)
        residual = offset - (gradient * t + intercept)
        keep = np.abs(residual) < 2.5 * max(residual.std(), 0.3)
        if keep.all():
            break
        t, offset = t[keep], offset[keep]
        if len(t) < 8:
            return a, b

    if float(residual.std()) > STRAIGHT_EDGE_MAX_RESIDUAL_PX:
        return a, b
    rotation = abs(math.degrees(math.atan(gradient / length)))
    if rotation > STRAIGHT_EDGE_MAX_ROTATION_DEG:
        return a, b

    return a + normal * intercept, b + normal * (gradient + intercept)


#: How far past the declared square to look, as a fraction of its own side, and the
#: pixel ceiling on that. The margin has to hold enough of the surround to measure a
#: level against, and no more: whatever else is printed nearby comes in with it, and
#: the surround's median is what sets the threshold.
#:
#: It was a flat 60 px, which is a fair margin on a large feature and most of a small
#: one. Every feature that existed when this was written is 238 px a side or more, so
#: 60 was 25% or less and the fraction never binds; the ISA FADGI-ISO 19264 charts'
#: corner squares are 70 px, where a flat 60 reaches 86% of the feature's own size
#: past it and takes in the chart's dark border. That dragged the surround's median
#: down toward the square's own level, and the top-left mark -- the one with border
#: on two sides rather than one -- declined on BOTH charts while the other two
#: snapped to 0.3 px. Proportional and capped, every pre-existing feature keeps the
#: 60 px it had and only the small ones change.
SQUARE_MARGIN_FRACTION = 0.4
SQUARE_MARGIN_MAX_PX = 60.0
SQUARE_MARGIN_MIN_PX = 20.0

#: The level-sampling annulus, in the normalised units `t` below is built in --
#: 1.0 sits at the declared box's own corner, ``1/sqrt(2)`` at its edge midpoint.
#: The inside band is what a seeded placement error erodes first, well
#: before the connected-component step that uses the threshold these two levels
#: produce -- moving it closer to the centre buys back seed-error tolerance on a
#: small feature without moving it at all on a large one, because the failure is
#: the band's *outer* edge crossing the true feature's own boundary once the box
#: has drifted, and a band that starts closer in has further to drift before that
#: happens. Measured on the early ISA FADGI-ISO 19264's eight registration dots
#: (42 px across) against the real capture, eight directions per seed magnitude:
#: the old (0.55, 0.80) declined outright by 20 px of seed error, after already
#: losing two thirds of its eight directions at 15; (0.20, 0.45) still refines all
#: eight at 15 px and three quarters of them at 20, and never once lands more
#: than 1.5 px from where the un-seeded refinement puts it. Costs nothing at zero
#: seed error: an unseeded dot still refines to 0.10-0.57 px on both charts,
#: matching what it already did.
#:
#: The **outside** band is left alone on purpose. Widening it the same way
#: looked like the obvious symmetric fix and broke a real feature outright: the
#: commercial chart's top-left corner square has chart border on two sides
#: rather than one, and pushing the outside ring out to catch it merges its
#: level with the square's own and the feature declines where it used to snap
#: to 0.34 px. The inside band alone is where the seed-error benefit is, and
#: moving only it is what keeps that regression from happening.
SQUARE_INSIDE_T = (0.20, 0.45)
SQUARE_OUTSIDE_T = 1.15


def _refine_square(image: LoadedImage, corners_px: np.ndarray) -> np.ndarray:
    """Snap the declared square onto the dark square actually in the capture."""
    side = float(np.linalg.norm(corners_px[1] - corners_px[0]))
    margin = min(SQUARE_MARGIN_MAX_PX, max(SQUARE_MARGIN_MIN_PX, SQUARE_MARGIN_FRACTION * side))
    x0 = int(max(corners_px[:, 0].min() - margin, 0))
    x1 = int(min(corners_px[:, 0].max() + margin, image.width))
    y0 = int(max(corners_px[:, 1].min() - margin, 0))
    y1 = int(min(corners_px[:, 1].max() + margin, image.height))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return corners_px

    window = image.window(y0, y1, x0, x1)
    gray = to_luma(window).astype(np.float32)

    # Threshold between the feature and its surround, sampled using the declared
    # position. A percentile split over the whole window does not work here: the
    # module also contains a bright resolution pattern and light registration
    # patches, so the midpoint of the window's range lands above the card gray and
    # the "dark square" component swallows the card with it.
    #
    # **Anisotropic, not circular**. A circular annulus at a fraction of
    # the corner-to-center distance is the right shape for a square feature, where
    # it reduces to the formula below with half_w == half_h. It is the wrong shape
    # for a rectangle: the ISA FADGI-ISO 19264 charts' resolution-field panels are
    # 3:2, and on the short axis a circular annulus scaled to the *long* axis's
    # corner distance sits mostly outside the panel, pulling its own surround's
    # level into what this takes to be "inside". `t` below is the same construction
    # generalised to each declared axis independently, so it draws exactly the same
    # annulus as before on a square and a correctly anisotropic one on a rectangle.
    local = corners_px - np.array([x0, y0], dtype=np.float64)
    center = local.mean(axis=0)
    u_axis = local[1] - local[0]
    v_axis = local[3] - local[0]
    half_w = float(np.linalg.norm(u_axis)) / 2.0
    half_h = float(np.linalg.norm(v_axis)) / 2.0
    if half_w < 1e-6 or half_h < 1e-6:
        return corners_px
    u_hat = u_axis / (2.0 * half_w)
    v_hat = v_axis / (2.0 * half_h)
    yy, xx = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]]
    du = (xx - center[0]) * u_hat[0] + (yy - center[1]) * u_hat[1]
    dv = (xx - center[0]) * v_hat[0] + (yy - center[1]) * v_hat[1]
    t = np.hypot(du / half_w, dv / half_h) / math.sqrt(2.0)
    inside = (t > SQUARE_INSIDE_T[0]) & (t < SQUARE_INSIDE_T[1])
    outside = t > SQUARE_OUTSIDE_T
    if inside.sum() < 50 or outside.sum() < 50:
        return corners_px
    level_in = float(np.median(gray[inside]))
    level_out = float(np.median(gray[outside]))
    if abs(level_in - level_out) < 0.02:
        return corners_px

    threshold = (level_in + level_out) / 2.0
    mask = ((gray < threshold) if level_in < level_out else (gray > threshold))
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return corners_px
    # The component containing the declared center, not merely the largest: the
    # largest can be the card itself if the threshold drifts.
    at_center = labels[int(round(center[1])), int(round(center[0]))]
    chosen = int(at_center) if at_center != 0 else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    points = np.column_stack(np.nonzero(labels == chosen))[:, ::-1].astype(np.float32)
    if len(points) < 100:
        return corners_px
    rect = cv2.minAreaRect(points)
    declared_side = float(np.linalg.norm(corners_px[1] - corners_px[0]))
    found_side = float(np.mean(rect[1]))
    if not 0.6 < found_side / max(declared_side, 1e-6) < 1.6:
        # The detected shape is not the feature the definition describes.
        return corners_px
    box = cv2.boxPoints(rect) + np.array([x0, y0], dtype=np.float32)

    # Keep the declared corner order so edge ids stay stable between captures.
    ordered = np.zeros_like(corners_px)
    used: set[int] = set()
    for i, corner in enumerate(corners_px):
        distances = np.linalg.norm(box - corner, axis=1)
        for j in np.argsort(distances):
            if int(j) not in used:
                used.add(int(j))
                ordered[i] = box[j]
                break
    return ordered


def _cut_edge(
    image: LoadedImage,
    a: np.ndarray,
    b: np.ndarray,
    edge_id: str,
    region_id: str,
    half_width: int,
    length_fraction: float,
    target_phase_coverage: float,
) -> EdgeROI | None:
    direction = b - a
    length = float(np.linalg.norm(direction))
    if length < 40:
        return None
    midpoint = (a + b) / 2.0
    unit = direction / length

    # Use as much of the edge as the corners allow. Every extra row is another
    # independent sample of the same transition, so length buys noise reduction; the
    # limit is that a long edge can bow under lens distortion, which is what the
    # quadratic edge_fit is for. Phase coverage is then a *check* on the result, not
    # the thing that sets the length -- sizing to the bare coverage minimum would
    # throw away most of the edge and leave the estimate noisy.
    vertical = abs(unit[1]) >= abs(unit[0])
    tilt = abs(unit[0] / unit[1]) if vertical else abs(unit[1] / unit[0])
    rows = max(length * length_fraction, MIN_ROI_ROWS)
    half_length = rows / 2.0
    coverage = rows * tilt

    notes: list[str] = []
    if coverage < target_phase_coverage:
        notes.append(
            "Slanted edge is too shallow for reliable analysis."
        )

    # An edge running mostly top-to-bottom in the image is "vertical"; the ROI is
    # taken axis-aligned either way and compute_sfr handles the residual tilt.
    if vertical:
        y0 = int(round(midpoint[1] - half_length))
        y1 = int(round(midpoint[1] + half_length))
        x0 = int(round(midpoint[0] - half_width))
        x1 = int(round(midpoint[0] + half_width))
    else:
        x0 = int(round(midpoint[0] - half_length))
        x1 = int(round(midpoint[0] + half_length))
        y0 = int(round(midpoint[1] - half_width))
        y1 = int(round(midpoint[1] + half_width))

    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x1, image.width), min(y1, image.height)
    if cx1 - cx0 < 16 or cy1 - cy0 < 16:
        return None
    if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
        notes.append("ROI clipped at image border.")

    return EdgeROI(
        edge_id=edge_id,
        region_id=region_id,
        pixels=image.window(cy0, cy1, cx0, cx1),
        orientation="vertical" if vertical else "horizontal",
        center_px=(float(midpoint[0]), float(midpoint[1])),
        bounds_px=(cx0, cy0, cx1, cy1),
        notes=notes,
    )


def render_edge_overlay(
    image: LoadedImage,
    placement: ChartPlacement,
    target: TargetDefinition,
    rois: list[EdgeROI],
    *,
    max_width: int = 1600,
) -> np.ndarray:
    """Draw the located edge features and their ROIs over the capture.

    The same purpose the patch overlay serves: with geometry that is declared
    approximately and refined per capture, this is the only way to confirm the ROIs
    actually straddle the edges before believing an MTF.
    """
    stride = max(1, int(np.ceil(image.width / max_width)))
    preview = to_display_records(image.pixels[::stride, ::stride, :])
    if preview.dtype.kind == "f":
        bgr = (np.clip(preview[..., ::-1], 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        bgr = (preview[..., ::-1].astype(np.float32) * (255.0 / image.scale)).astype(np.uint8)
    bgr = np.ascontiguousarray(bgr)
    thickness = max(1, bgr.shape[1] // 500)

    for region in target.regions:
        if region.kind != "slanted_edge" or not region.geometry.get("center"):
            continue
        corners = placement.to_pixel(
            _square_corners(
                np.asarray(region.geometry["center"], dtype=np.float64),
                np.asarray(region.geometry["half_size"], dtype=np.float64),
                float(region.geometry.get("tilt_deg", 0.0)),
            )
        )
        cv2.polylines(
            bgr, [np.round(corners / stride).astype(np.int32)], True,
            (0, 200, 255), max(1, thickness // 2),
        )

    for roi in rois:
        x0, y0, x1, y1 = (v / stride for v in roi.bounds_px)
        color = (0, 0, 255) if roi.notes else (0, 255, 0)
        cv2.rectangle(
            bgr, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness
        )
    return bgr
