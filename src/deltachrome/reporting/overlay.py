"""Everything a run measured, drawn on the capture it measured it from.

Every target here has its geometry **measured off one capture** and written into a
definition as relative coordinates. That is the only practical way to describe a
chart nobody publishes a layout file for -- and it is also a class of error that
does not announce itself. A patch grid shifted by a third of a patch still samples
140 regions and still returns 140 plausible Delta E values; a rule band a few pixels
off the ticks reads the printed digits and reports a confident wrong sampling rate;
an edge ROI straddling the wrong boundary returns an MTF curve.

So this draws all of it at once, in one picture, over the pixels it came from:

* the **placement quad**, so a misordered corner shows as a bow tie;
* every **patch ROI**, at the inset actually sampled, coloured by whether the sample
  was flagged;
* every **declared slanted-edge feature** and the **ROI cut from each edge**, which
  are different things -- the declaration is approximate on purpose and the ROI is
  where the capture's own boundary was found, so the gap between them is the
  refinement working;
* every **rule band**, whether or not its ticks could be read;
* every **marker**, which is a feature the chart carries that no metric reads.

Two decisions worth stating. The image is decimated **before** any float conversion,
so an overlay of a hundred-megapixel master costs a preview's worth of memory rather
than a multiple of the file. And nothing here is drawn from the *definition* alone
where a measurement exists: the patch quadrilaterals come from the sampler and the
edge rectangles from the edge extractor, so what is shown is what was measured and
not what was intended.
"""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from deltachrome.imaging.geometry import ChartPlacement
from deltachrome.imaging.loader import LoadedImage, to_display_records
from deltachrome.targets.schema import TargetDefinition

#: BGR, because OpenCV. Chosen to stay apart on a grey chart and in a screenshot:
#: the quad and the declarations are warm, what was actually measured is cool, and
#: anything flagged is red.
COLORS = {
    "quad": (0, 220, 255),        # amber -- the four placed corners
    "patch": (120, 220, 120),     # green -- a clean sample
    "flagged": (60, 60, 255),     # red -- clipped, contaminated, or off the image
    "declared": (0, 170, 255),    # orange -- the feature as the definition states it
    "edge": (255, 200, 0),        # cyan -- the ROI an edge was actually measured in
    "rule": (255, 120, 0),        # blue -- a band a printed scale was read from
    "rule_failed": (60, 60, 255),
    "marker": (200, 130, 200),    # mauve -- carried, but read by no metric
}


def render_geometry(
    image: LoadedImage,
    placement: ChartPlacement,
    target: TargetDefinition,
    *,
    sampling: Any = None,
    edges: Sequence[Any] = (),
    rulers: Sequence[Any] = (),
    max_width: int = 2000,
    label_patches: bool = False,
) -> np.ndarray:
    """One BGR image showing every region a run touched."""
    stride = max(1, int(np.ceil(image.width / max_width)))
    preview = to_display_records(image.pixels[::stride, ::stride, :])
    if preview.dtype.kind == "f":
        bgr = (np.clip(preview[..., ::-1], 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        bgr = (preview[..., ::-1].astype(np.float32) * (255.0 / image.scale)).astype(np.uint8)
    bgr = np.ascontiguousarray(bgr)
    thick = max(1, bgr.shape[1] // 700)

    def scaled(points) -> np.ndarray:
        return np.round(np.asarray(points, dtype=np.float64) / stride).astype(np.int32)

    def band(bounds, color, width=None):
        u0, v0, u1, v1 = (float(v) for v in bounds)
        quad = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=np.float64)
        cv2.polylines(bgr, [scaled(placement.to_pixel(quad))], True, color,
                      width or thick)

    # -- rule bands and markers first, so patches and edges draw over them -----
    read = {getattr(r, "region_id", "") for r in rulers}
    for region in target.regions:
        geometry = region.geometry or {}
        if region.kind == "scale" and geometry.get("bounds_relative"):
            band(geometry["bounds_relative"],
                 COLORS["rule"] if region.region_id in read else COLORS["rule_failed"])
        elif region.kind == "marker":
            if geometry.get("bounds_relative"):
                band(geometry["bounds_relative"], COLORS["marker"])
            elif geometry.get("center") and geometry.get("half_size"):
                _square(bgr, placement, geometry, COLORS["marker"], thick, scaled)
            for position in geometry.get("positions") or []:
                centre = scaled(placement.to_pixel(np.array([position], dtype=np.float64)))[0]
                cv2.circle(bgr, tuple(centre), max(4, thick * 4), COLORS["marker"], thick)

    # -- the declared slanted-edge features -----------------------------------
    for region in target.regions:
        if region.kind != "slanted_edge":
            continue
        geometry = region.geometry or {}
        for disc in geometry.get("discs") or []:
            _square(bgr, placement, disc, COLORS["declared"], thick, scaled,
                    tilt_default=geometry.get("tilt_deg", 0.0))
        for edge in geometry.get("straight_edges") or []:
            ends = placement.to_pixel(
                np.array([edge["start"], edge["end"]], dtype=np.float64)
            )
            cv2.line(bgr, tuple(scaled(ends)[0]), tuple(scaled(ends)[1]),
                     COLORS["declared"], thick)
        if geometry.get("center") and geometry.get("half_size"):
            _square(bgr, placement, geometry, COLORS["declared"], thick, scaled)

    # -- what was actually measured -------------------------------------------
    for roi in edges:
        x0, y0, x1, y1 = (v / stride for v in roi.bounds_px)
        cv2.rectangle(bgr, (int(x0), int(y0)), (int(x1), int(y1)),
                      COLORS["flagged"] if roi.notes else COLORS["edge"], thick)

    if sampling is not None:
        for sample in sampling.samples.values():
            if not sample.usable:
                continue
            color = COLORS["flagged"] if sample.flags else COLORS["patch"]
            cv2.polylines(bgr, [scaled(sample.polygon_px)], True, color,
                          max(1, thick // 2))
            if label_patches:
                at = scaled(sample.polygon_px)[0]
                cv2.putText(bgr, sample.patch_id, (at[0], at[1] - 2),
                            cv2.FONT_HERSHEY_PLAIN, 0.6, color, 1, cv2.LINE_AA)

    cv2.polylines(bgr, [scaled(placement.corners_px)], True, COLORS["quad"], thick + 1)
    for index, point in enumerate(scaled(placement.corners_px)):
        cv2.circle(bgr, tuple(point), max(5, thick * 5), COLORS["quad"], -1)
        cv2.putText(bgr, "ABCD"[index % 4], (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_PLAIN, 1.4, COLORS["quad"], max(1, thick), cv2.LINE_AA)
    return bgr


def _square(bgr, placement, geometry, color, thick, scaled, tilt_default=0.0) -> None:
    from deltachrome.imaging.edges import _square_corners

    corners = placement.to_pixel(
        _square_corners(
            np.asarray(geometry["center"], dtype=np.float64),
            np.asarray(geometry["half_size"], dtype=np.float64),
            float(geometry.get("tilt_deg", tilt_default)),
        )
    )
    cv2.polylines(bgr, [scaled(corners)], True, color, thick)


def legend() -> list[tuple[str, tuple[int, int, int]]]:
    """What each colour means, for a caller writing a caption."""
    return [
        ("placed corners", COLORS["quad"]),
        ("patch ROI", COLORS["patch"]),
        ("declared feature", COLORS["declared"]),
        ("edge ROI measured", COLORS["edge"]),
        ("rule band read", COLORS["rule"]),
        ("marker (no metric reads it)", COLORS["marker"]),
        ("flagged or unread", COLORS["flagged"]),
    ]
