"""Mapping between target-relative space and image pixel space.

Everything geometric in Deltachrome is expressed in target-relative coordinates
(0..1 across the fiducial quadrilateral, origin top-left). A single homography
converts those to pixels. Manual corner placement today and automatic fiducial
detection later differ only in how the four corner points are obtained -- all the
downstream maths is shared, which is the whole point of not storing pixel
coordinates in target definitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from deltachrome.errors import GeometryError


@dataclass
class ChartPlacement:
    """A resolved target-relative -> pixel mapping, plus sanity metrics."""

    matrix: np.ndarray  # 3x3 homography, relative -> pixel
    corners_px: np.ndarray  # (4, 2) in TL TR BR BL order
    image_size: tuple[int, int]  # (width, height)

    def to_pixel(self, points: np.ndarray) -> np.ndarray:
        """Map relative coordinates (..., 2) to pixel coordinates (..., 2)."""
        return self._project(points, self.matrix)

    def to_relative(self, points: np.ndarray) -> np.ndarray:
        """Map pixel coordinates (..., 2) back to target-relative coordinates.

        The inverse of :meth:`to_pixel`. Needed wherever a measurement was made
        at a pixel location and has to be reported at a place on the chart --
        which is every measurement whose *position* is part of the diagnosis, and
        so every measurement that can be drawn as a field. Provided here rather
        than inverted by hand at each call site so that one definition of the
        mapping serves both directions.
        """
        return self._project(points, np.linalg.inv(self.matrix))

    @staticmethod
    def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        shape = pts.shape
        flat = pts.reshape(-1, 2)
        homogeneous = np.column_stack([flat, np.ones(len(flat))])
        projected = homogeneous @ matrix.T
        projected = projected[:, :2] / projected[:, 2:3]
        return projected.reshape(shape)

    # -- sanity metrics -----------------------------------------------------

    @property
    def edge_lengths_px(self) -> tuple[float, float, float, float]:
        c = self.corners_px
        return tuple(  # type: ignore[return-value]
            float(np.linalg.norm(c[(i + 1) % 4] - c[i])) for i in range(4)
        )

    @property
    def area_px(self) -> float:
        """Shoelace area of the placed quadrilateral."""
        c = self.corners_px
        x, y = c[:, 0], c[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @property
    def frame_edge_lengths_px(self) -> tuple[float, float, float, float]:
        """Edge lengths of the chart's own frame -- the relative unit square -- as placed.

        Not of the four points the placement was made on, and the difference is the
        point. Physical size and proportion are statements about the frame, and on
        most targets the placement points are its corners, so the two coincide. They
        need not: the DT Nexus charts place on four circles one in from the corners,
        because a corner cell is opaque on every unit, and measuring those four
        points against the frame's millimetres would misreport the sampling rate
        with nothing to say it had.
        """
        c = self.to_pixel(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
        return tuple(  # type: ignore[return-value]
            float(np.linalg.norm(c[(i + 1) % 4] - c[i])) for i in range(4)
        )

    @property
    def aspect_ratio(self) -> float:
        """Width over height of the chart's frame as placed."""
        top, right, bottom, left = self.frame_edge_lengths_px
        height = (right + left) / 2.0
        return ((top + bottom) / 2.0) / height if height else float("nan")

    def effective_ppi(self, physical_size_mm: tuple[float, float] | None) -> float | None:
        """Sampling rate implied by the placement, in pixels per inch.

        ``physical_size_mm`` is the size of the chart's frame (see
        :attr:`frame_edge_lengths_px`). Reported rather than enforced: FADGI star levels carry resolution
        requirements that depend on material type, which is a profile concern.
        """
        if not physical_size_mm:
            return None
        top, right, bottom, left = self.frame_edge_lengths_px
        width_px = (top + bottom) / 2.0
        height_px = (right + left) / 2.0
        width_in = physical_size_mm[0] / 25.4
        height_in = physical_size_mm[1] / 25.4
        if width_in <= 0 or height_in <= 0:
            return None
        return float((width_px / width_in + height_px / height_in) / 2.0)

    def aspect_deviation(self, physical_size_mm: tuple[float, float] | None) -> float | None:
        """Fractional disagreement between placed and physical aspect ratio.

        A large value usually means the corners were clicked in the wrong order or
        on the wrong features -- worth catching before 140 patches are sampled from
        the wrong places.
        """
        if not physical_size_mm or physical_size_mm[1] <= 0:
            return None
        expected = physical_size_mm[0] / physical_size_mm[1]
        if not np.isfinite(self.aspect_ratio) or expected <= 0:
            return None
        return float(abs(self.aspect_ratio - expected) / expected)


#: How far past the frame's edge a corner may sit, as a fraction of its longer
#: side, before it is outside the image rather than on its border.
OUTSIDE_MARGIN = 0.02


def corners_outside(corners, image_size: tuple[int, int]) -> list[tuple[float, float]]:
    """The corners lying outside an image of ``image_size`` (width, height)."""
    width, height = image_size
    margin = OUTSIDE_MARGIN * max(width, height)
    return [
        (float(x), float(y))
        for x, y in np.asarray(corners, float).reshape(-1, 2)
        if x < -margin or y < -margin or x > width + margin or y > height + margin
    ]


def place_chart(
    corners_px: np.ndarray | list[tuple[float, float]],
    image_size: tuple[int, int],
    *,
    corner_positions_relative: np.ndarray | list[tuple[float, float]] | None = None,
) -> ChartPlacement:
    """Build a homography from four placed corners.

    ``corners_px`` are image pixel coordinates in the same order as
    ``corner_positions_relative`` (default TL, TR, BR, BL).
    """
    corners = np.asarray(corners_px, dtype=np.float64)
    if corners.shape != (4, 2):
        raise GeometryError(
            f"Expected four (x, y) corner points, got array of shape {corners.shape}."
        )
    if not np.all(np.isfinite(corners)):
        raise GeometryError("Corner coordinates contain NaN or infinity.")

    relative = (
        np.asarray(corner_positions_relative, dtype=np.float64)
        if corner_positions_relative is not None
        else np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    )
    if relative.shape != (4, 2):
        raise GeometryError(f"Expected four relative corner positions, got {relative.shape}.")

    _check_quadrilateral(corners)

    width, height = image_size
    outside = corners_outside(corners, image_size)
    if outside:
        raise GeometryError(
            f"Corner(s) {outside} fall outside the {width}x{height} image. "
            "Check that the coordinates are in pixels, x,y, and in TL TR BR BL order."
        )

    matrix = cv2.getPerspectiveTransform(
        relative.astype(np.float32), corners.astype(np.float32)
    )
    return ChartPlacement(
        matrix=np.asarray(matrix, dtype=np.float64),
        corners_px=corners,
        image_size=(int(width), int(height)),
    )


def _check_quadrilateral(corners: np.ndarray) -> None:
    """Reject degenerate, self-intersecting, or mis-ordered corner sets."""
    x, y = corners[:, 0], corners[:, 1]
    signed_area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if abs(signed_area) < 1.0:
        raise GeometryError(
            "The four corners are collinear or coincident (near-zero area); "
            "no homography exists."
        )

    # Convexity: every cross product of consecutive edges must share a sign. A sign
    # flip means the points are in a bowtie order, which yields a homography that
    # silently mirrors half the chart.
    crosses = []
    for i in range(4):
        a = corners[(i + 1) % 4] - corners[i]
        b = corners[(i + 2) % 4] - corners[(i + 1) % 4]
        crosses.append(float(a[0] * b[1] - a[1] * b[0]))
    if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
        raise GeometryError(
            "The four corners do not form a convex quadrilateral in the order given. "
            "Corners must be listed in sequence around the chart (TL, TR, BR, BL), "
            "not as opposite pairs."
        )
