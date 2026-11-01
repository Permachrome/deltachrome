"""Draw every shipped target as a flat illustration, for the documentation.

The documentation needs a picture of each chart saying where its placement marks
are, and the six drawn by hand set the style: a flat
card with a heavy black outline, the patches as butted flat squares, the slanted-edge
figures as dark shapes, and the fiducials -- crosshairs, keylines, discs -- drawn
where the chart prints them. Rules, text and resolution wedges are left out: the
picture is about where to click, and they are not where anybody clicks.

**What is drawn comes from the target definition, and what colour it is comes from
the capture.** Every patch, every slanted-edge feature and every fiducial is at the
position the definition declares, so the picture cannot disagree with what the
engine samples -- a hand-drawn chart could, and nothing would notice. The card's
outline, and every fill, is read off the committed capture through the committed
placement in ``render_target_overlays.PLACEMENTS``, converted through the capture's
own ICC profile to sRGB. Only features the definition does not describe -- the
mount a module sits on, the figure a pair of straight edges belongs to -- are traced
from the pixels, and those are listed per target below rather than guessed at.

Run from the repository root:

    python tools/render_target_illustrations.py [--target ID]

Writes two SVGs per target into ``docs/site/images/targets`` -- the chart, and the
chart with its placement points ringed -- and a PNG of each. The SVG is the
deliverable -- it opens as editable vector artwork in Illustrator, and places
into Photoshop as a smart object. It carries no pixel of any capture, only flat
colours sampled from one, so unlike an overlay it is not the capture redistributed.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

# -- the style, measured off the hand-drawn PSDs ---------------------------------

#: The hand-drawn illustrations sit on a 1630 x 2038 portrait canvas, the chart
#: centred and nearly filling its width. Strips too long for that get a canvas of
#: their own height, as ISAOLT.psd does.
CANVAS_WIDTH = 1630
CANVAS_HEIGHT = 2038
SIDE_MARGIN = 45
#: Black, 9.57 px, on the card and on a printed keyline alike.
STROKE = 9.57
CARD_RADIUS = 17.0
MODULE_STROKE = 5.0
MODULE_RADIUS = 12.0
INK = "#000000"
#: Below this C*ab a sampled colour is drawn as the neutral of the same L*.
NEUTRAL_CHROMA = 6.0


# -- colour ------------------------------------------------------------------------

#: CIE XYZ (D50, the ICC PCS) to linear sRGB, Bradford-adapted to D65.
_XYZ_D50_TO_SRGB = np.array([
    [3.1338561, -1.6168667, -0.4906146],
    [-0.9787684, 1.9161415, 0.0334540],
    [0.0719453, -0.2289914, 1.4052427],
])


def srgb_hex(device: np.ndarray, profile) -> str:
    """One device colour, through the capture's own profile, as an sRGB hex string.

    Display only: this is the colour a patch is painted in a picture, and no number
    anywhere is derived from it.
    """
    from deltachrome.imaging.color import lab_to_xyz, xyz_to_lab

    xyz = profile.to_xyz(np.asarray(device, dtype=np.float64).reshape(1, -1))[0]
    lab = xyz_to_lab(xyz)
    if math.hypot(lab[1], lab[2]) < NEUTRAL_CHROMA:
        # A white card photographed under a slightly cool light is a white card: in
        # a flat illustration the cast reads as a colour the chart does not have.
        xyz = lab_to_xyz(np.array([lab[0], 0.0, 0.0]))
    linear = np.clip(_XYZ_D50_TO_SRGB @ xyz, 0.0, 1.0)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear,
                       1.055 * np.power(linear, 1 / 2.4) - 0.055)
    r, g, b = (int(round(float(c) * 255)) for c in encoded)
    return f"#{r:02x}{g:02x}{b:02x}"


def luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# -- the chart, rectified ------------------------------------------------------------

@dataclass
class Rectified:
    """The capture resampled onto the target's own relative frame, upright.

    ``ax`` and ``ay`` are pixels per relative unit, chosen from the placement so
    that a square on the chart is square here -- the relative frame is anisotropic
    whenever the fiducial quad is not square, which is nearly always.
    """

    pixels: np.ndarray        # (H, W, C) float in [0, 1], NaN outside the capture
    origin: tuple[float, float]
    ax: float
    ay: float
    profile: object

    def to_px(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        return np.stack([(p[..., 0] - self.origin[0]) * self.ax,
                         (p[..., 1] - self.origin[1]) * self.ay], axis=-1)

    def to_rel(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        return np.stack([p[..., 0] / self.ax + self.origin[0],
                         p[..., 1] / self.ay + self.origin[1]], axis=-1)

    def median_in(self, polygon_rel, shrink: float = 0.6) -> np.ndarray | None:
        """Median device colour inside a polygon, shrunk about its centroid."""
        poly = self.to_px(polygon_rel)
        centre = poly.mean(axis=0)
        poly = centre + (poly - centre) * shrink
        mask = np.zeros(self.pixels.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
        values = self.pixels[mask.astype(bool)]
        values = values[np.all(np.isfinite(values), axis=-1)]
        return None if len(values) == 0 else np.median(values, axis=0)

    def median_ring(self, polygon_rel, outer: float = 0.92, inner: float = 0.75) -> np.ndarray | None:
        """Median colour of a band just inside a polygon's outline.

        A module's slanted mount holds a white field of its own, so a median over
        its middle reads the field rather than the mount.
        """
        poly = self.to_px(polygon_rel)
        centre = poly.mean(axis=0)
        mask = np.zeros(self.pixels.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(centre + (poly - centre) * outer).astype(np.int32)], 1)
        cv2.fillPoly(mask, [np.round(centre + (poly - centre) * inner).astype(np.int32)], 0)
        return self.median_mask(mask)

    def flood(self, seed_rel, tolerance: float = 0.05, blur: float = 2.0) -> tuple[np.ndarray, np.ndarray] | None:
        """The region of the seed's own colour connected to it: (mask, colour).

        Used for what a definition does not describe -- the light mount a module
        sits on, the ground a slanted square is printed on.
        """
        grey = np.nan_to_num(self.pixels.mean(axis=-1), nan=-1.0).astype(np.float32)
        smooth = cv2.GaussianBlur(grey, (0, 0), blur) if blur else grey
        x, y = np.round(self.to_px(seed_rel)).astype(int)
        mask = np.zeros((grey.shape[0] + 2, grey.shape[1] + 2), np.uint8)
        cv2.floodFill(smooth, mask, (int(x), int(y)), 0, tolerance, tolerance,
                      4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (1 << 8))
        region = mask[1:-1, 1:-1]
        if blur:
            region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return region, self.median_mask(region)

    def box_of(self, mask: np.ndarray) -> tuple[float, float, float, float]:
        ys, xs = np.nonzero(mask)
        (u0, v0), (u1, v1) = self.to_rel(np.array([[xs.min(), ys.min()], [xs.max() + 1, ys.max() + 1]], float))
        return (float(u0), float(v0), float(u1), float(v1))

    def median_mask(self, mask: np.ndarray) -> np.ndarray | None:
        values = self.pixels[mask.astype(bool)]
        values = values[np.all(np.isfinite(values), axis=-1)]
        return None if len(values) == 0 else np.median(values, axis=0)


def _scale(placement, at) -> tuple[float, float]:
    """Capture pixels per relative unit along each axis, at one point."""
    c, d = np.asarray(at, float), 0.01
    sx = np.linalg.norm(np.diff(placement.to_pixel(np.array([c - [d, 0], c + [d, 0]])), axis=0)) / (2 * d)
    sy = np.linalg.norm(np.diff(placement.to_pixel(np.array([c - [0, d], c + [0, d]])), axis=0)) / (2 * d)
    return float(sx), float(sy)


def rectify(image, placement, window, width_px: int = 2400) -> Rectified:
    u0, v0, u1, v1 = window
    sx, sy = _scale(placement, ((u0 + u1) / 2, (v0 + v1) / 2))
    k = width_px / max((u1 - u0) * sx, (v1 - v0) * sy)
    ax, ay = sx * k, sy * k
    out_w, out_h = int(round((u1 - u0) * ax)), int(round((v1 - v0) * ay))

    # Decimate first by a whole factor with area averaging: a warp samples, and a
    # 100 Mpx master sampled at 2400 px across aliases every fine feature it holds.
    factor = max(1, int(1 / k))
    src = image.pixels
    if src.ndim == 2:
        src = src[..., None]
    if factor > 1:
        src = cv2.resize(src, (src.shape[1] // factor, src.shape[0] // factor),
                         interpolation=cv2.INTER_AREA)
        if src.ndim == 2:
            src = src[..., None]
    src = src.astype(np.float32) / float(image.scale)

    to_rel = np.array([[1 / ax, 0, u0], [0, 1 / ay, v0], [0, 0, 1]])
    shrink = np.diag([1 / factor, 1 / factor, 1.0])
    matrix = shrink @ placement.matrix @ to_rel
    channels = [cv2.warpPerspective(np.ascontiguousarray(src[..., i]), matrix, (out_w, out_h),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                for i in range(src.shape[2])]
    return Rectified(np.stack(channels, axis=-1), (u0, v0), ax, ay, image.profile)


def find_card(rect: Rectified, inside_rel) -> tuple[float, float, float, float] | None:
    """The card's bounding box in relative units, or None if it cannot be separated.

    The background is what the window's own border holds; the card is the connected
    region unlike it that contains the chart's centre. Where the capture is cropped
    tight to the card there is no background to find, and the caller says where the
    card is instead.
    """
    px = rect.pixels
    valid = np.all(np.isfinite(px), axis=-1)
    grey = np.nan_to_num(px.mean(axis=-1), nan=0.0)
    # The background is sampled along the edge of what the capture covers, which
    # is the window's edge only when the capture reaches that far.
    b = max(4, int(0.02 * min(valid.shape)))
    padded = np.pad(valid, 1, constant_values=False).astype(np.uint8)
    border = valid & ~cv2.erode(padded, np.ones((2 * b + 1, 2 * b + 1), np.uint8))[1:-1, 1:-1].astype(bool)
    if border.sum() < 100:
        return None
    background = float(np.median(grey[border]))
    spread = float(np.median(np.abs(grey[border] - background))) + 0.01
    differs = (np.abs(grey - background) > max(0.06, 6 * spread)) & valid
    k = max(3, int(0.01 * min(valid.shape)))
    closed = cv2.morphologyEx(differs.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((k, k), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    cx, cy = np.round(rect.to_px(inside_rel)).astype(int)
    label = labels[min(max(cy, 0), labels.shape[0] - 1), min(max(cx, 0), labels.shape[1] - 1)]
    if label == 0:
        return None
    x, y, w, h = stats[label, :4]
    component = labels == label
    if (component & border).sum() > 0.25 * border.sum():
        return None
    (u0, v0), (u1, v1) = rect.to_rel(np.array([[x, y], [x + w, y + h]], dtype=np.float64))
    return (float(u0), float(v0), float(u1), float(v1))


# -- the scene -----------------------------------------------------------------------

@dataclass
class Scene:
    """Shapes in relative coordinates, drawn in order, then fitted to a canvas."""

    rect: Rectified
    items: list[tuple[str, dict]] = field(default_factory=list)

    def add(self, kind: str, **attrs) -> None:
        self.items.append((kind, attrs))


def square_corners(geometry) -> np.ndarray:
    from deltachrome.imaging.edges import _square_corners

    return _square_corners(np.asarray(geometry["center"], float),
                           np.asarray(geometry["half_size"], float),
                           float(geometry.get("tilt_deg", 0.0)))


def quarter_disc(rect: Rectified, disc, towards) -> np.ndarray:
    """A quarter disc, as a polygon in relative units.

    The definition declares each disc by its bounding square; the apex is the
    square's corner nearest ``towards`` (the pair's common centre), and the arc is
    centred on it with the square's side as its radius.
    """
    corners_px = rect.to_px(square_corners(disc))
    apex_index = int(np.argmin(np.linalg.norm(corners_px - rect.to_px(towards), axis=1)))
    apex = corners_px[apex_index]
    a = corners_px[(apex_index + 1) % 4]
    b = corners_px[(apex_index - 1) % 4]
    radius = (np.linalg.norm(a - apex) + np.linalg.norm(b - apex)) / 2
    ta = math.atan2(*(a - apex)[::-1])
    tb = math.atan2(*(b - apex)[::-1])
    sweep = (tb - ta + math.pi) % (2 * math.pi) - math.pi
    arc = [apex + radius * np.array([math.cos(ta + sweep * s), math.sin(ta + sweep * s)])
           for s in np.linspace(0, 1, 48)]
    return rect.to_rel(np.array([apex, *arc]))


# -- building a target ---------------------------------------------------------------

#: Features a definition declares but that the hand-drawn style leaves out, because
#: nobody places a chart on them.
UNDRAWN_MARKERS = {
    "resolution_field", "bar_pattern_array", "converging_line_wedge", "siemens_star",
}


def build(entry, overrides: dict) -> Scene:
    from deltachrome.imaging.geometry import place_chart
    from deltachrome.imaging.loader import load_image
    from deltachrome.targets.registry import load_target

    image = load_image(entry.path)
    target = load_target(overrides.get("definition", entry.target_id))
    placement = place_chart(
        list(entry.corners), image_size=(image.width, image.height),
        corner_positions_relative=target.fiducials.corner_positions or None,
    )

    fid = np.array(target.fiducials.corner_positions or [[0, 0], [1, 0], [1, 1], [0, 1]], float)
    # Everything the definition declares, so that a patch strip outside the
    # fiducial quad -- an IT8's grey wedge, the Object Level module -- is inside.
    declared = [fid, [[0.0, 0.0], [1.0, 1.0]]]
    declared += [target.patch_corners_relative(p) for p in target.patches]
    for region in target.regions:
        g = region.geometry or {}
        for e in g.get("straight_edges") or []:
            declared.append([e["start"], e["end"]])
    declared = np.vstack([np.asarray(d, float) for d in declared])
    lo, hi = declared.min(axis=0), declared.max(axis=0)
    # Padded by the same number of PIXELS on both axes. The relative frame is
    # anisotropic, and on the Object Level strip -- 34 pitches long, one patch
    # tall -- a pad proportional to each axis leaves the card outside the window.
    sx, sy = _scale(placement, (lo + hi) / 2)
    pad_px = overrides.get("window_pad", 0.3) * max((hi[0] - lo[0]) * sx, (hi[1] - lo[1]) * sy)
    window = (lo[0] - pad_px / sx, lo[1] - pad_px / sy, hi[0] + pad_px / sx, hi[1] + pad_px / sy)
    span = hi - lo
    rect = rectify(image, placement, window)
    scene = Scene(rect)
    profile = image.profile

    def colour(device, fallback="#808080") -> str:
        return fallback if device is None else srgb_hex(device, profile)

    # The card.
    centre = (fid.min(axis=0) + fid.max(axis=0)) / 2
    card = overrides.get("card")
    if card is None and overrides.get("card_px") is not None:
        # Where nothing in the window separates the card, it is given as a box in
        # the capture's own pixels -- or as the frame itself, for a capture
        # cropped inside the card.
        x0, y0, x1, y1 = ((0, 0, image.width, image.height) if overrides["card_px"] == "frame"
                          else overrides["card_px"])
        rel = placement.to_relative(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))
        (u0, v0), (u1, v1) = rel.min(axis=0), rel.max(axis=0)
        card = (float(u0), float(v0), float(u1), float(v1))
    if card is None and overrides.get("card_seed") is not None:
        card = rect.box_of(rect.flood(overrides["card_seed"], overrides.get("card_tolerance", 0.05))[0])
    if card is None:
        card = find_card(rect, centre)
    if card is not None and overrides.get("card_symmetric"):
        # The capture crops the card; its layout is symmetric about the chart's
        # centre, so the side that is present says where the missing one is.
        c = (fid.min(axis=0) + fid.max(axis=0)) / 2
        du = max(c[0] - card[0], card[2] - c[0])
        dv = max(c[1] - card[1], card[3] - c[1])
        card = (c[0] - du, c[1] - dv, c[0] + du, c[1] + dv)
    if card is None:
        # Nothing separates the card from its surround: the declared features' own
        # extent, padded a little, is the least wrong outline available.
        card = (lo[0] - 0.05 * span[0], lo[1] - 0.05 * span[1],
                hi[0] + 0.05 * span[0], hi[1] + 0.05 * span[1])
        print(f"  card not found in the capture; using the fiducial extent padded")
    scene.card = card
    scene.fiducials = fid
    scene.labels = list(target.fiducials.corner_ids or ["TL", "TR", "BR", "BL"])
    card_fill = overrides.get("card_fill") or colour(_card_colour(rect, card, target))
    scene.add("card", box=card, fill=card_fill)

    # A printed keyline is the placement itself on these charts.
    if target.fiducials.method == "printed_frame_corners" and not overrides.get("no_keyline"):
        scene.add("keyline", points=fid)

    for extra in overrides.get("under", ()):
        extra(scene, rect, target, colour)

    # Slanted-edge figures and the markers worth drawing.
    for region in target.regions:
        g = region.geometry or {}
        if region.kind == "slanted_edge":
            if g.get("discs"):
                discs = g["discs"]
                towards = np.mean([d["center"] for d in discs], axis=0)
                for disc in discs:
                    poly = quarter_disc(rect, disc, towards)
                    scene.add("polygon", points=poly,
                              fill=colour(rect.median_in(poly, 0.5)))
            elif g.get("center") and g.get("half_size"):
                poly = square_corners(g)
                scene.add("polygon", points=poly, fill=colour(rect.median_ring(poly)))
            elif g.get("straight_edges") and region.region_id not in overrides.get("skip_regions", ()):
                module = overrides.get("module_box")
                if module is not None:
                    # A module is a small card of its own: boxed from what the
                    # definition declares, then flattened and traced inside.
                    box = module(g)
                    c = scene.card  # a module is printed on the card, never past it
                    box = (max(box[0], c[0]), max(box[1], c[1]), min(box[2], c[2]), min(box[3], c[3]))
                    lines = [(e["start"], e["end"]) for e in g["straight_edges"]]
                    back, shapes = posterise(rect, box, overrides.get("module_tones", 3), lines=lines,
                                             corners=overrides.get("module_corners", 12),
                                             close=overrides.get("module_close", True))
                    scene.add("module", box=box, fill=colour(back))
                    for poly, dev in shapes:
                        scene.add("polygon", points=poly, fill=colour(dev))
                    continue
                for poly, dev in trace_figure(rect, g, overrides.get("light_figures", False)):
                    scene.add("polygon", points=poly, fill=colour(dev))
        elif region.kind == "marker" and g.get("feature") not in UNDRAWN_MARKERS:
            feature = g.get("feature")
            if feature == "registration_dot":
                r = g["printed_diameter_relative"] / 2
                scene.add("ellipse", center=g["center"],
                          radii=(r, r * rect.ax / rect.ay),  # round on the chart
                          fill=colour(rect.median_in(square_corners(g), 0.5)))
            elif feature == "corner_square":
                poly = square_corners(g)
                scene.add("polygon", points=poly, fill=colour(rect.median_in(poly)))
            elif feature == "stepped_neutral_ladder":
                # Not a patch grid -- no metric reads it -- but it is the largest
                # thing on the chart, and a picture without it is not this chart.
                u0, v0, u1, v1 = g["bounds_relative"]
                n = int(g["steps"])
                for i in range(n):
                    a, b = u0 + (u1 - u0) * i / n, u0 + (u1 - u0) * (i + 1) / n
                    step = [[a, v0], [b, v0], [b, v1], [a, v1]]
                    scene.add("polygon", points=step, fill=colour(rect.median_in(step, 0.5)))
            elif feature == "continuous_neutral_wedge":
                u0, v0, u1, v1 = g["bounds_relative"]
                stops = []
                for t in np.linspace(0.02, 0.98, 9):
                    u = u0 + (u1 - u0) * t
                    d = 0.01 * (u1 - u0)
                    probe = [[u - d, v0], [u + d, v0], [u + d, v1], [u - d, v1]]
                    stops.append((float(t), colour(rect.median_in(probe, 0.5))))
                scene.add("gradient", box=(u0, v0, u1, v1), stops=stops)
            elif feature == "concentric_rings":
                for position in g["positions"]:
                    scene.add("rings", center=position)

    # Patches, at their full printed size rather than the inset that is sampled.
    # Some charts print round patches and sample a square inside each; their
    # diameter is not declared, so it is measured -- the median across the
    # patches of the region of each one's own colour.
    shape = overrides.get("patch_shape", "square")
    diameter = _patch_diameter(rect, target) if shape == "circle" else None
    for patch in target.patches:
        corners = np.array(target.patch_corners_relative(patch), float)
        fill = colour(rect.median_in(corners, 0.5))
        if diameter is None:
            scene.add("polygon", points=corners, fill=fill)
        else:
            scene.add("ellipse", center=corners.mean(axis=0),
                      radii=(diameter / 2 / rect.ax, diameter / 2 / rect.ay), fill=fill)

    for extra in overrides.get("over", ()):
        extra(scene, rect, target, colour)

    # The fiducials a person places on, last so nothing covers them.
    method = target.fiducials.method
    if method in {"printed_crosshairs", "module_corner_crosshairs", "corner_crosses"}:
        ink = "#ffffff" if luminance(card_fill) < 0.4 else INK
        for point in fid:
            scene.add("cross", center=point, ink=overrides.get("cross_ink", ink),
                      size=overrides.get("cross_size", 0.045))
    elif method == "printed_discs":
        # Each disc is measured where it is: the dark region connected to the point
        # the placement was made on.
        for point in fid:
            mask, device = rect.flood(point, 0.08)
            u0, v0, u1, v1 = rect.box_of(mask)
            scene.add("ellipse", center=((u0 + u1) / 2, (v0 + v1) / 2),
                      radii=((u1 - u0) / 2, (v1 - v0) / 2), fill=colour(device))
    return scene


def posterise(rect: Rectified, box, tones: int = 3, min_fraction: float = 0.02, lines=(),
               corners: int = 12, close: bool = True):
    """A small printed figure, flattened to a few tones and traced.

    For what a definition declares only as edges -- a module's slanted band and
    block. Inside the module's own box the figure is the only large structure;
    printed text and scales are small, and are dropped by area.

    Two things keep the result looking printed rather than traced. The tones are
    **nested**: each one is drawn as everything at least that far from the
    background, darker over lighter, so neighbouring shapes overlap instead of
    meeting at two separately traced outlines with a sliver of background between.
    And every outline is **regularised** (:func:`regularise`) onto the edges the
    definition declares, ``lines`` -- pairs of relative-coordinate endpoints -- so
    a slanted edge is drawn exactly where the engine measures it.

    Returns (background colour, [(polygon, colour), ...]) in drawing order.
    """
    (x0, y0), (x1, y1) = np.round(rect.to_px([box[:2], box[2:]])).astype(int)
    crop = rect.pixels[y0:y1, x0:x1]
    grey = np.nan_to_num(crop.mean(axis=-1), nan=0.5).astype(np.float32)
    grey = cv2.medianBlur((grey * 255).astype(np.uint8), 5).astype(np.float32) / 255
    _, labels, centres = cv2.kmeans(grey.reshape(-1, 1), tones, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3),
                                    4, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(grey.shape)
    level = centres[:, 0]
    counts = np.bincount(labels.ravel(), minlength=tones)
    background = int(np.argmax(counts))
    k = max(3, int(0.02 * min(grey.shape)))
    area = grey.size * min_fraction
    h, w = rect.pixels.shape[:2]
    guides = [rect.to_px(np.asarray(line, float)) - [x0, y0] for line in lines]

    lighter = sorted((t for t in range(tones) if level[t] > level[background]), key=lambda t: level[t])
    darker = sorted((t for t in range(tones) if level[t] < level[background]), key=lambda t: -level[t])
    shapes = []
    # Lighter tones first, then darker over them, each side working outward from
    # the background: a dark figure printed on a white field sits on top of it.
    for side, ordered in ((1, lighter), (-1, darker)):
        for tone in ordered:
            nested = (level[labels] >= level[tone]) if side > 0 else (level[labels] <= level[tone])
            # Closed as well as opened: printed text touching a field's edge bites
            # notches into its outline, which a close fills and a real corner,
            # being larger than the kernel, survives.
            kernel = np.ones((k, k), np.uint8)
            mask = nested.astype(np.uint8)
            if close:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            exact = (labels == tone).astype(np.uint8)
            for contour in contours:
                if cv2.contourArea(contour) < area:
                    continue
                poly = merge_short_sides(regularise(simplify(contour, corners), guides,
                                                    tolerance=0.03 * min(grey.shape)),
                                         guides, shortest=0.04 * min(grey.shape))
                one = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(one, [contour], -1, 1, -1)
                full = np.zeros((h, w), np.uint8)
                full[y0:y1, x0:x1] = cv2.erode(one & exact, np.ones((3, 3), np.uint8))
                shapes.append((rect.to_rel(poly + [x0, y0]), rect.median_mask(full)))
    back = np.zeros((h, w), np.uint8)
    back[y0:y1, x0:x1] = (labels == background)
    return rect.median_mask(back), shapes


def simplify(contour, most: int = 12) -> np.ndarray:
    """A traced outline reduced to the few corners a printed shape has."""
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.008 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    while len(approx) > most and epsilon < 0.05 * perimeter:
        epsilon *= 1.4
        approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx[:, 0, :].astype(float)


def merge_short_sides(poly: np.ndarray, guides, shortest: float) -> np.ndarray:
    """Drop the stubs a trace leaves at a corner -- a chamfer where a printed shape
    has a square corner.

    A stub is removed by extending the two sides either side of it until they meet,
    never by deleting a vertex and joining its neighbours: that cuts a diagonal
    across a real corner, which is what it did to an L-shaped module until this was
    written down. A stub between two parallel sides is a genuine step, and stays.
    """
    tolerance = 0.75 * shortest
    while len(poly) > 3:
        n = len(poly)
        lengths = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
        merged = False
        for i in np.argsort(lengths):
            if lengths[i] >= shortest:
                break
            a, b = poly[(i - 1) % n], poly[i]
            c, d = poly[(i + 1) % n], poly[(i + 2) % n]
            d1, d2 = b - a, d - c
            denom = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(denom) < 0.3 * np.linalg.norm(d1) * np.linalg.norm(d2):
                continue
            t = ((c[0] - a[0]) * d2[1] - (c[1] - a[1]) * d2[0]) / denom
            meet = a + t * d1
            if np.linalg.norm(meet - b) > 3 * shortest or np.linalg.norm(meet - c) > 3 * shortest:
                continue
            keep = [j for j in range(n) if j not in (i, (i + 1) % n)]
            insert_at = keep.index((i - 1) % n) + 1
            new = [poly[j] for j in keep]
            new.insert(insert_at, meet)
            poly = regularise(np.array(new), guides, tolerance)
            merged = True
            break
        if not merged:
            break
    return poly


def regularise(poly: np.ndarray, guides, tolerance: float, axis_deg: float = 4.0) -> np.ndarray:
    """Put a traced outline's sides where a printed shape's sides are.

    Each side is given a line to lie on: a declared edge if it runs along one
    (within ``axis_deg`` and ``tolerance`` pixels), else the horizontal or vertical
    if it is within ``axis_deg`` of one -- printed figures are drawn square to the
    chart except where they are deliberately slanted -- else its own line. Each
    corner is then where its two sides' lines cross. A traced edge wanders by a
    pixel or two along its length; this is what makes it read as ruled.
    """
    n = len(poly)
    if n < 3:
        return poly
    limit = math.radians(axis_deg)

    def angle_between(d1, d2) -> float:
        a = abs(math.atan2(d1[1], d1[0]) - math.atan2(d2[1], d2[0])) % math.pi
        return min(a, math.pi - a)

    sides = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        d = b - a
        mid = (a + b) / 2
        line = None
        for p, q in guides:
            e = q - p
            if np.linalg.norm(e) == 0 or angle_between(d, e) > limit:
                continue
            u = e / np.linalg.norm(e)
            offset = abs(u[0] * (mid - p)[1] - u[1] * (mid - p)[0])
            if offset <= tolerance:
                line = (p, e)
                break
        if line is None:
            if angle_between(d, (1.0, 0.0)) <= limit:
                line = (np.array([0.0, (a[1] + b[1]) / 2]), np.array([1.0, 0.0]))
            elif angle_between(d, (0.0, 1.0)) <= limit:
                line = (np.array([(a[0] + b[0]) / 2, 0.0]), np.array([0.0, 1.0]))
            else:
                line = (a, d)
        sides.append(line)

    out = []
    for i in range(n):
        (p1, d1), (p2, d2) = sides[i - 1], sides[i]
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-6 * np.linalg.norm(d1) * np.linalg.norm(d2):
            out.append(poly[i])     # two sides on one line: the corner is not one
            continue
        t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / denom
        out.append(p1 + t * d1)
    return np.array(out)


def _patch_diameter(rect: Rectified, target) -> float:
    """Median printed diameter of round patches, in rectified pixels."""
    sizes = []
    for patch in target.patches:
        centre = np.array(target.patch_corners_relative(patch), float).mean(axis=0)
        mask, _ = rect.flood(centre, 0.04, blur=1.0)
        ys, xs = np.nonzero(mask)
        if len(xs):
            sizes.append(((xs.max() - xs.min()) + (ys.max() - ys.min())) / 2 + 1)
    pitch = min(np.diff(rect.to_px([[0, 0], list(target.grid_regions[0].grid.patch_spacing_relative)]), axis=0)[0])
    sizes = [v for v in sizes if 0.3 * pitch < v < 1.05 * pitch]
    return float(np.median(sizes))


def _card_colour(rect: Rectified, card, target) -> np.ndarray | None:
    """The card's own colour: its body with every declared feature masked out."""
    px = rect.pixels
    mask = np.zeros(px.shape[:2], np.uint8)
    x0, y0 = rect.to_px([card[0], card[1]])
    x1, y1 = rect.to_px([card[2], card[3]])
    inset = 0.03 * min(x1 - x0, y1 - y0)
    mask[int(y0 + inset):int(y1 - inset), int(x0 + inset):int(x1 - inset)] = 1
    for patch in target.patches:
        poly = rect.to_px(np.array(target.patch_corners_relative(patch), float))
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 0)
    for region in target.regions:
        if region.kind == "patch_grid" and region.grid is not None:
            g = region.grid
            ox, oy = g.grid_origin_relative
            ex, ey = g.extent_relative
            poly = rect.to_px([[ox, oy], [ox + ex, oy], [ox + ex, oy + ey], [ox, oy + ey]])
            cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 0)
    return rect.median_mask(mask)


def trace_figure(rect: Rectified, geometry, invert: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """The dark printed shape a set of straight edges belongs to, traced.

    A definition declares straight edges by their endpoints and says nothing about
    the figure they bound, so the figure is found: the darker side of an Otsu split
    of the neighbourhood, the component touching the middle of each declared edge.
    """
    ends = np.array([[e["start"], e["end"]] for e in geometry["straight_edges"]], float)
    pts = rect.to_px(ends.reshape(-1, 2))
    if geometry.get("field_bounds_relative"):
        f = geometry["field_bounds_relative"]
        pts = np.vstack([pts, rect.to_px([[f[0], f[1]], [f[2], f[3]]])])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    margin = 0.35 * max(x1 - x0, y1 - y0)
    h, w = rect.pixels.shape[:2]
    X0, Y0 = int(max(0, x0 - margin)), int(max(0, y0 - margin))
    X1, Y1 = int(min(w, x1 + margin)), int(min(h, y1 + margin))
    crop = rect.pixels[Y0:Y1, X0:X1]
    grey = np.nan_to_num(crop.mean(axis=-1), nan=1.0)
    grey8 = np.clip(grey * 255, 0, 255).astype(np.uint8)
    # Not Otsu: a module holds three tones -- a white field, a grey mount and the
    # dark figure -- and a two-way split puts the mount in with the figure. The
    # figure is what sits well away from the neighbourhood's median.
    finite = grey[np.isfinite(crop).all(axis=-1)]
    median = float(np.median(finite))
    if invert:
        dark = (grey > (np.percentile(finite, 97) + median) / 2).astype(np.uint8)
    else:
        dark = (grey < (np.percentile(finite, 3) + median) / 2).astype(np.uint8)
    k = max(3, int(0.004 * max(crop.shape[:2])))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    count, labels = cv2.connectedComponents(dark, 8)

    wanted = set()
    for (s, e) in ends:
        mid = rect.to_px((np.asarray(s) + np.asarray(e)) / 2) - [X0, Y0]
        seg = rect.to_px(np.asarray(e)) - rect.to_px(np.asarray(s))
        normal = np.array([-seg[1], seg[0]]) / (np.linalg.norm(seg) + 1e-9)
        for side in (-1, 1):
            for step in (4, 8, 12):
                x, y = np.round(mid + side * step * normal).astype(int)
                if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] and labels[y, x]:
                    wanted.add(int(labels[y, x]))
    shapes = []
    for label in sorted(wanted):
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, 0.004 * cv2.arcLength(contour, True), True)[:, 0, :]
        device = rect.median_mask(np.pad(cv2.erode(mask, np.ones((k, k), np.uint8)),
                                         ((Y0, h - Y1), (X0, w - X1))))
        shapes.append((rect.to_rel(approx.astype(float) + [X0, Y0]), device))
    return shapes


# -- output --------------------------------------------------------------------------

#: The placement canvas's handle colour, and its halo. A callout in the
#: documentation is a picture of the handle the operator will drag there, so it
#: wears the handle's own yellow and the handle's own label.
CALLOUT = "#ffd600"
CALLOUT_HALO = "#000000"
CALLOUT_RADIUS = 42.0
CALLOUT_STROKE = 6.0
CALLOUT_FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"


def to_svg(scene: Scene, title: str, *, callouts: bool = False) -> str:
    """The scene on its canvas, upright as the chart is printed.

    Always upright, never turned to fit the page: the placement canvas names its
    handles by the chart's own corners, so a chart drawn a quarter turn round puts
    the ring marked TL at the picture's top right -- and a reader who drags TL to
    the top left of what they see places the chart mirrored, which measures wrong
    without looking wrong. ``callouts`` rings each point the chart is placed on.
    """
    rect = scene.rect
    base = rect.to_px

    u0, v0, u1, v1 = scene.card
    corners = base([[u0, v0], [u1, v1]])
    lo_px, hi_px = corners.min(axis=0), corners.max(axis=0)
    cw, ch = hi_px - lo_px
    scale = (CANVAS_WIDTH - 2 * SIDE_MARGIN - STROKE) / cw
    if ch * scale > CANVAS_HEIGHT - 2 * 237:
        scale = (CANVAS_HEIGHT - 2 * 237) / ch
    width = CANVAS_WIDTH
    # A strip gets a canvas of its own height; with callouts, room for the tags.
    margin = 160 if callouts else 90
    height = CANVAS_HEIGHT if cw / ch < 2.4 else int(round(ch * scale + 2 * margin))
    offset = np.array([width / 2, height / 2]) - (lo_px + hi_px) / 2 * scale

    def P(points) -> np.ndarray:
        return base(points) * scale + offset

    def path(points) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in P(points))

    def box(b) -> tuple[float, float, float, float]:
        pts = P([[b[0], b[1]], [b[2], b[3]]])
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        return x0, y0, x1 - x0, y1 - y0

    short = min(cw, ch) * scale
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}">',
           f"<title>{escape(title)}</title>",
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    for kind, a in scene.items:
        if kind == "card":
            x, y, w, h = box(a["box"])
            out.append(f'<rect id="card" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                       f'height="{h:.1f}" rx="{CARD_RADIUS}" fill="{a["fill"]}" '
                       f'stroke="{INK}" stroke-width="{STROKE}"/>')
        elif kind == "keyline":
            out.append(f'<polygon id="keyline" points="{path(a["points"])}" fill="none" '
                       f'stroke="{INK}" stroke-width="{STROKE}"/>')
        elif kind == "module":
            x, y, w, h = box(a["box"])
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                       f'rx="{MODULE_RADIUS}" fill="{a["fill"]}" stroke="{INK}" '
                       f'stroke-width="{MODULE_STROKE}"/>')
        elif kind == "box":
            x, y, w, h = box(a["box"])
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                       f'height="{h:.1f}" fill="{a["fill"]}"/>')
        elif kind == "gradient":
            gid = f"g{len(out)}"
            x, y, w, h = box(a["box"])
            direction = 'x1="0" y1="0" x2="1" y2="0"'
            stops = "".join(f'<stop offset="{t:.3f}" stop-color="{c}"/>' for t, c in a["stops"])
            out.append(f'<defs><linearGradient id="{gid}" {direction}>{stops}'
                       f'</linearGradient></defs><rect x="{x:.1f}" y="{y:.1f}" '
                       f'width="{w:.1f}" height="{h:.1f}" fill="url(#{gid})"/>')
        elif kind == "qr":
            # A label, not a QR: a light square with the three finder marks a
            # reader recognises a QR code by. Drawing a real code would invite
            # somebody to scan a picture of a chart.
            x, y, w, h = box(a["box"])
            side = min(w, h)
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                       f'rx="3" fill="{a["fill"]}"/>')
            m = 0.12 * side
            f = 0.26 * side
            for fx, fy in ((x + m, y + m), (x + w - m - f, y + m), (x + m, y + m + f * 1.9)):
                out.append(f'<rect x="{fx:.1f}" y="{fy:.1f}" width="{f:.1f}" height="{f:.1f}" '
                           f'fill="none" stroke="{INK}" stroke-width="{0.06 * side:.1f}"/>')
                out.append(f'<rect x="{fx + 0.3 * f:.1f}" y="{fy + 0.3 * f:.1f}" '
                           f'width="{0.4 * f:.1f}" height="{0.4 * f:.1f}" fill="{INK}"/>')
        elif kind == "polygon":
            out.append(f'<polygon points="{path(a["points"])}" fill="{a["fill"]}"/>')
        elif kind == "ellipse":
            (cx, cy), = P([a["center"]])
            rx = a["radii"][0] * rect.ax * scale  # each radius in its own axis's units
            ry = a["radii"][1] * rect.ay * scale
            out.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
                       f'fill="{a["fill"]}"/>')
        elif kind == "rings":
            (cx, cy), = P([a["center"]])
            r = 0.025 * short
            for f in (1.0, 0.6):
                out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r * f:.1f}" fill="none" '
                           f'stroke="{INK}" stroke-width="{STROKE * 0.6:.2f}"/>')
        elif kind == "cross":
            (cx, cy), = P([a["center"]])
            arm = a["size"] * short / 2
            t = max(4.0, arm / 3.5)
            out.append(f'<g class="fiducial" fill="{a["ink"]}">'
                       f'<rect x="{cx - arm:.1f}" y="{cy - t / 2:.1f}" width="{2 * arm:.1f}" height="{t:.1f}"/>'
                       f'<rect x="{cx - t / 2:.1f}" y="{cy - arm:.1f}" width="{t:.1f}" height="{2 * arm:.1f}"/>'
                       f"</g>")
    if callouts:
        out.append(_callouts(P(scene.fiducials), scene.labels))
    out.append("</svg>")
    return "\n".join(out)


def _callouts(points, labels) -> str:
    """A ring round each placement point, and the handle's label outside it.

    The label goes outward along whichever axis the point is further out on,
    measured against the spread of the four points themselves: a corner is
    labelled above or below, so a wide chart's labels stay on the page, and the
    Device Level Target's side discs are labelled beside them. The ring shrinks
    where points crowd -- the Object Level strip's are one patch apart.
    """
    points = np.asarray(points, float)
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    half = np.maximum((points.max(axis=0) - points.min(axis=0)) / 2, 1.0)
    gaps = [np.linalg.norm(p - q) for i, p in enumerate(points) for q in points[i + 1:]]
    radius = min(CALLOUT_RADIUS, 0.42 * min(gaps))
    parts = ['<g id="placement-callouts">']
    for (x, y), label in zip(points, labels):
        nx, ny = (x - centre[0]) / half[0], (y - centre[1]) / half[1]
        if abs(ny) >= 0.8 * abs(nx):
            dx, dy = 0.0, (1.0 if ny >= 0 else -1.0)
        else:
            dx, dy = (1.0 if nx >= 0 else -1.0), 0.0
        tag_w = 26 + 19 * len(label)
        tag_h = 44
        reach = radius + 12 + (tag_w / 2 if dx else tag_h / 2)
        tx, ty = x + dx * reach, y + dy * reach
        for colour, width in ((CALLOUT_HALO, CALLOUT_STROKE + 6), (CALLOUT, CALLOUT_STROKE)):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="none" '
                         f'stroke="{colour}" stroke-width="{width}"/>')
        parts.append(f'<rect x="{tx - tag_w / 2:.1f}" y="{ty - tag_h / 2:.1f}" width="{tag_w}" '
                     f'height="{tag_h}" rx="6" fill="{CALLOUT_HALO}"/>')
        parts.append(f'<text x="{tx:.1f}" y="{ty + 11:.1f}" text-anchor="middle" '
                     f'font-family="{CALLOUT_FONT}" font-weight="bold" font-size="31" '
                     f'fill="{CALLOUT}">{escape(label)}</text>')
    parts.append("</g>")
    return "\n".join(parts)


def rasterise(svg_path: Path, png_path: Path) -> None:
    """A PNG beside the SVG, for pages that want a bitmap.

    Qt's renderer rather than ImageMagick's: the latter drops ``stroke-width`` on
    a rounded rectangle, which is the one stroke this style is built on.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    global _APP
    _APP = QGuiApplication.instance() or QGuiApplication([])
    renderer = QSvgRenderer(str(svg_path))
    image = QImage(renderer.defaultSize(), QImage.Format.Format_ARGB32)
    image.fill(0xFFFFFFFF)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    image.save(str(png_path))


_APP = None


# -- per target ----------------------------------------------------------------------

def isa_19264_module(g):
    """The ISA FADGI-ISO 19264 module: its declared field, plus the dark band
    printed beyond it.

    The band lies past whichever declared edge falls outside the field -- the top
    on the corner modules, the left on the centre one, which is printed turned --
    and reaches about 1.75 times that edge's distance from the field.
    """
    u0, v0, u1, v1 = g["field_bounds_relative"]
    for edge in g["straight_edges"]:
        mu, mv = (np.asarray(edge["start"]) + np.asarray(edge["end"])) / 2
        if mv < v0:
            v0 -= 1.75 * (v0 - mv)
        elif mv > v1:
            v1 += 1.75 * (mv - v1)
        elif mu < u0:
            u0 -= 1.75 * (u0 - mu)
        elif mu > u1:
            u1 += 1.75 * (mu - u1)
    return (u0, v0, u1, v1)


def edges_box(pad_u: float, pad_v: float):
    """A module boxed round its declared edges, padded in relative units."""
    def box(g):
        ends = np.array([[e["start"], e["end"]] for e in g["straight_edges"]], float).reshape(-1, 2)
        (u0, v0), (u1, v1) = ends.min(axis=0), ends.max(axis=0)
        return (u0 - pad_u, v0 - pad_v, u1 + pad_u, v1 + pad_v)
    return box


def mounts(tolerance: float = 0.05, stroke: bool = True):
    """Each square slanted-edge feature's light mount, found by flooding outward.

    A module on the Golden Thread and ISA charts is a light square card with the
    dark slanted square printed on it; the definition declares the slanted square
    and not the card, which is exactly the part a picture needs to show.
    """
    def hook(scene, rect, target, colour):
        card_area = np.prod(np.diff(rect.to_px([scene.card[:2], scene.card[2:]]), axis=0))
        for region in target.regions:
            g = region.geometry or {}
            if region.kind != "slanted_edge" or not g.get("half_size") or g.get("discs"):
                continue
            corners = rect.to_px(square_corners(g))
            centre = corners.mean(axis=0)
            top = (corners[0] + corners[1]) / 2
            seed = centre + (top - centre) * 1.08
            found = rect.flood(rect.to_rel(seed), tolerance)
            if found is None:
                continue
            mask, device = found
            if mask.sum() > 0.2 * card_area or mask.sum() == 0:
                print(f"  {region.region_id}: mount not separable, not drawn")
                continue
            box = rect.box_of(mask)
            # The flood stops at the uniformity squares printed in the mount's
            # corners, so its box falls short of the mount by about their width
            # and they were drawn hanging off its edge. The mount is grown to hold
            # its own squares -- the grid named for this module's position, which
            # is how the definition says which squares are whose. Distance cannot
            # say it: a colour strip sits within one patch of two of the mounts.
            position = str(g.get("position", "")).lower()
            own = [np.array(target.patch_corners_relative(p), float)
                   for p in target.patches if p.region == f"uniformity_{position}"]
            if own:
                lo = np.min([q.min(axis=0) for q in own], axis=0)
                hi = np.max([q.max(axis=0) for q in own], axis=0)
                box = (min(box[0], lo[0]), min(box[1], lo[1]), max(box[2], hi[0]), max(box[3], hi[1]))
            scene.add("module" if stroke else "box", box=box, fill=colour(device))
    return hook


def printed_dots():
    """Every round black dot the chart prints, not only the ones it declares.

    The ISA FADGI-ISO 19264 charts print 75 dots on a regular grid between their
    patches and declare eight of them as markers. A picture drawing only the eight
    makes each look unmistakable, and on the early chart one of them is a placement
    point: on the card it is the last of a column of ten, and the picture has to
    show that or it teaches a placement the chart does not support. Found in the
    capture as dark components the size and roundness of a declared dot.
    """
    def hook(scene, rect, target, colour):
        declared = [r.geometry for r in target.regions
                    if (r.geometry or {}).get("feature") == "registration_dot"]
        if not declared:
            return
        radius = declared[0]["printed_diameter_relative"] / 2
        across = 2 * radius * rect.ax
        valid = np.isfinite(rect.pixels).all(axis=-1)
        grey = np.nan_to_num(rect.pixels.mean(axis=-1), nan=1.0)
        dark = (grey < 0.5 * np.median(grey[valid])).astype(np.uint8)
        count, _, stats, centres = cv2.connectedComponentsWithStats(dark, 8)
        u0, v0, u1, v1 = scene.card
        fill = colour(rect.median_in(square_corners(declared[0]), 0.5))
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            round_enough = area > 0.6 * math.pi * (w / 2) * (h / 2)
            if 0.6 * across < w < 1.5 * across and 0.6 * across < h < 1.5 * across and round_enough:
                u, v = rect.to_rel(centres[i])
                if u0 < u < u1 and v0 < v < v1:
                    scene.add("ellipse", center=(u, v),
                              radii=(radius, radius * rect.ax / rect.ay), fill=fill)
    return hook


def qr_label(row: int, col: int):
    """A QR label in a cell of a patch lattice, drawn where this unit carries it.

    The DT Nexus charts hold an OpenQualia QR in their top-right cell. It is a
    label rather than part of the printed lattice, and it sits off the cell's
    centre by a different amount on each unit -- measured at 0.02 to 0.10 of a
    pitch on three -- so it is found in the capture, not placed from the
    definition. It is drawn because it is the one thing on the chart that says
    which corner is which; it is not a placement point, for the same reason.
    """
    def hook(scene, rect, target, colour):
        grid = target.grid_regions[0].grid
        mask, device = rect.flood(grid.center_relative(row, col), 0.25, blur=3)
        u0, v0, u1, v1 = rect.box_of(mask)
        scene.add("qr", box=(u0, v0, u1, v1), fill=colour(device))
    return hook


def keyline(seed_rel, tolerance: float = 0.04):
    """A printed keyline the placement is not made on, found by flooding its inside.

    Blur is off: the line is a pixel or two wide at this scale, and smoothing it
    into its surround opens the gap the flood would leak through.
    """
    def hook(scene, rect, target, colour):
        mask, _ = rect.flood(seed_rel, tolerance, blur=0.7)
        u0, v0, u1, v1 = rect.box_of(mask)
        scene.add("keyline", points=np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]))
    return hook


def fill_region(seed_rel, tolerance: float = 0.05):
    """The region of one colour around a seed, traced and drawn flat."""
    def hook(scene, rect, target, colour):
        mask, device = rect.flood(seed_rel, tolerance)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, 0.004 * cv2.arcLength(contour, True), True)[:, 0, :]
        scene.add("polygon", points=rect.to_rel(approx.astype(float)), fill=colour(device))
    return hook


#: What each target needs beyond its definition. Keyed by the output name.
OVERRIDES: dict[str, dict] = {
    "isa_golden_thread_dlt_v1": {"under": [keyline((0.3, 0.15)), mounts()]},
    # The card is white and lies on a dark board beside three more; flooding the
    # white is what separates it.
    "applied_image_qa62_sfr_v1": {"card_seed": (0.5, -0.03),
                                  "under": [fill_region((0.22, 0.5))]},
    # Round patches, a square sampled inside each.
    "dt_nexus_transmissive_ntpt_rev_a_v1": {"patch_shape": "circle", "over": [qr_label(0, 12)]},
    "dt_nexus_transmissive_ntvt_rev_a_v1": {"patch_shape": "circle", "over": [qr_label(0, 12)]},
    "dt_nexus_transmissive_ntpt_rev_b_v1": {"patch_shape": "circle", "over": [qr_label(0, 12)]},
    "dt_nexus_transmissive_ntvt_rev_b_v1": {"patch_shape": "circle", "over": [qr_label(0, 12)]},
    # The strip lies between a white specimen sheet and black velvet; measured on
    # the capture rather than found.
    # The 0.5X's patches stand taller, so the same reach in patch heights would
    # take the module's box up into the card's own top edge.
    "isa_object_level_05x_v1": {"module_box": edges_box(0.03, 0.9),
                                "module_corners": 8, "module_close": False,
                                "card_px": (1021, 7170, 10650, 8491)},
    # Cropped inside its own card: the grey ramp runs out past the colour block
    # towards both edges of the frame, and a card found any tighter cuts it off.
    "it8_7_1_transmissive_v1": {"card_px": "frame"},
    # Cropped inside its own card, and printed with round patches.
    "avian_dt_ngt_v2": {"patch_shape": "circle", "card_px": "frame"},
    "isa_fadgi_19264_commercial_v1": {"patch_shape": "circle", "module_box": isa_19264_module,
                                      "under": [printed_dots()],
                                      "card_seed": (0.5, 0.03)},
    "isa_fadgi_19264_v1": {"patch_shape": "circle", "module_box": isa_19264_module,
                           "under": [printed_dots()],
                           "card_seed": (0.5, 0.2), "card_symmetric": True},
    # The strip's module is three simple shapes with printed numbers crowding their
    # edges: fewer corners and no closing keep the numbers from being traced in.
    "isa_object_level_1x_v1": {"module_box": edges_box(0.03, 1.2),
                               "module_corners": 8, "module_close": False},
    # A positive: the slanted figure is clear film on a dark ground.
    "microform_35mm_positive_v1": {"light_figures": True},
}


def extra_placements():
    """The two targets with no placement in ``render_target_overlays``.

    Kept here rather than added there, because that list is where a definition's
    geometry is checked against the capture it was measured from, and neither of
    these is that.
    """
    from render_target_overlays import PLACEMENTS, Placement

    rev_a = next(p for p in PLACEMENTS if p.target_id == "dt_nexus_transmissive_ntvt_rev_a_v1")
    return (
        # The 0.5X strip, placed by identify_target inside a region drawn round
        # its card -- what Search a Region does. Whole-frame identification does
        # not find it on this frame.
        Placement("isa_object_level_05x_v1", "NMNH-USNMENT00317979.tif",
                  ((1411.1, 7465.3), (10265.1, 7492.1), (10263.9, 7885.1), (1409.9, 7858.3)),
                  "identify_target, extent (900, 7050, 10800, 8600)"),
        # No capture of a Revision B Validation target exists. Its lattice is the
        # Revision A one to 0.23%, so the Revision A Validation capture is drawn
        # through the Revision B definition: the placement is right, and the patch
        # colours are the Revision A unit's -- a different population, which a
        # picture of where to click does not depend on.
        Placement("dt_nexus_transmissive_ntvt_rev_b_v1", rev_a.capture, rev_a.corners,
                  "the Revision A Validation placement"),
    )


def entries():
    from render_target_overlays import PLACEMENTS

    seen = set()
    for entry in PLACEMENTS + extra_placements():
        # One picture per model: the QA-62 board is four cards of one chart, and
        # the Object Level Target has two specimens.
        if entry.target_id in seen:
            continue
        seen.add(entry.target_id)
        yield entry.target_id, entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/site/images/targets")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from deltachrome.targets.registry import load_target

    for name, entry in entries():
        if args.target and name != args.target:
            continue
        if not (entry.path).exists():
            print(f"{name}: capture not present, skipped")
            continue
        print(name)
        scene = build(entry, OVERRIDES.get(name, {}))
        title = load_target(entry.target_id).display_name
        # Two pictures per chart: the chart, and the chart with the points it is
        # placed on ringed and named as the placement canvas names its handles.
        for suffix, callouts in (("", False), ("__placement", True)):
            svg = args.out / f"{name}{suffix}.svg"
            svg.write_text(to_svg(scene, title, callouts=callouts))
            rasterise(svg, svg.with_suffix(".png"))


if __name__ == "__main__":
    main()
