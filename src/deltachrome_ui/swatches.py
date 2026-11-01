"""The chart, drawn from what was measured.

A table of Delta E values answers "how far off" and never answers "in which
direction" -- and direction is what tells a warm cast from a green one, or one
contaminated patch from a chart photographed under the wrong light. Two hundred
numbers cannot be read at all; two hundred patches can be looked at.

So each patch is drawn where it physically sits, split: what the camera recorded
on one side, what the chart is supposed to be on the other. A seam that
disappears is a patch that matched. The eye finds the ones that did not without
reading a single figure.

**Nothing here is colored by pass or fail**, which is the same refusal the field
maps make and for the same reason. These thresholds are conditions on an
aggregate -- a mean, a 90th percentile -- and a per-patch verdict is one the
profile never formed. What *is* marked is a patch carrying a sampling flag,
because clipping and contamination are facts about the sample rather than
judgements of it.

The conversion to screen color is **display only**. It is not a measurement, it
is not color-managed to the viewer's monitor, and no number anywhere in the
report comes from it.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from deltachrome_ui.charts import ChartSelection
from deltachrome_ui.theme import BG, BORDER, TEXT_DIM

#: sRGB primaries from D50-adapted XYZ (Bradford), which is the white point every
#: Lab value in a run is expressed against.
_XYZ_D50_TO_SRGB = np.array([
    [3.1338561, -1.6168667, -0.4906146],
    [-0.9787684, 1.9161415, 0.0334540],
    [0.0719453, -0.2289914, 1.4052427],
])


def lab_to_display_rgb(lab) -> QColor:
    """A Lab triple as a screen color. Presentation, never a measurement.

    Out-of-gamut values are clipped rather than gamut-mapped: this is a swatch to
    look at, and a patch too saturated for sRGB should read as saturated rather
    than be quietly desaturated into something the chart does not look like.
    """
    from deltachrome.imaging.color import lab_to_xyz

    xyz = np.asarray(lab_to_xyz(np.asarray([lab], dtype=float)), dtype=float)[0]
    linear = _XYZ_D50_TO_SRGB @ xyz
    linear = np.clip(linear, 0.0, 1.0)
    encoded = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1 / 2.4) - 0.055,
    )
    r, g, b = (int(round(float(v) * 255)) for v in np.clip(encoded, 0.0, 1.0))
    return QColor(r, g, b)


def chart_view(run, selection=None) -> QWidget:
    """The chart, with a line saying how much of it was measured."""
    from PySide6.QtWidgets import QVBoxLayout

    from deltachrome_ui.widgets import caveat

    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 4, 0, 0)
    layout.setSpacing(6)
    grid = PatchGrid.for_run(run, selection=selection)
    layout.addWidget(caveat(
        f"{len(grid.patches)} patches, measured against reference"
    ))
    layout.addWidget(grid, 1)
    return holder


class PatchGrid(QWidget):
    """Every patch, where it sits, measured against its reference.

    Takes the run's own patch records rather than a metric's per-patch list: the
    quadrilateral each patch was sampled from is stored once on the record, and
    the point here is the chart, not one metric's reading of it.
    """

    #: Below this the split is not legible and the swatch is drawn whole.
    MIN_SPLIT_PX = 14

    def __init__(
        self,
        patches: list,
        selection: ChartSelection | None = None,
        inset: float | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.selection = selection
        # Each stored quadrilateral is the *sampled* region, which is an inset
        # fraction of the patch -- 0.55 by default, so drawing it directly gives
        # a field of small dots rather than a chart. Undoing the inset recovers
        # the patch's own outline exactly, because the inset is defined as that
        # fraction of it. The drawing is of the chart; the measurement is still
        # of the inset area, which is what the readout says.
        self._expand = 1.0 / inset if inset and 0 < inset <= 1 else 1.0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(280)
        self.setMouseTracking(True)
        self._hover: str | None = None

        self.patches = [
            p for p in patches
            if getattr(p, "measured_lab", None) and getattr(p, "roi_px", None)
        ]
        self._bounds = self._extent()

    @classmethod
    def for_run(cls, run, selection: ChartSelection | None = None) -> "PatchGrid":
        """Built from a run, which is where the inset it has to undo is recorded."""
        return cls(
            run.patches or [], selection=selection,
            inset=(run.sampling or {}).get("roi_inset"),
        )

    def _extent(self) -> tuple[float, float, float, float] | None:
        points = [pt for p in self.patches for pt in self._corners(p)]
        if not points:
            return None
        xs = [float(pt[0]) for pt in points]
        ys = [float(pt[1]) for pt in points]
        return min(xs), min(ys), max(xs), max(ys)

    def _transform(self) -> tuple[float, float, float] | None:
        """Scale and offset taking image pixels to widget pixels, aspect kept."""
        if not self._bounds:
            return None
        x0, y0, x1, y1 = self._bounds
        width, height = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        margin = 10
        scale = min(
            (self.width() - 2 * margin) / width,
            (self.height() - 2 * margin) / height,
        )
        dx = margin + ((self.width() - 2 * margin) - width * scale) / 2 - x0 * scale
        dy = margin + ((self.height() - 2 * margin) - height * scale) / 2 - y0 * scale
        return scale, dx, dy

    def _corners(self, patch) -> list[tuple[float, float]]:
        points = [(float(x), float(y)) for x, y in patch.roi_px]
        if self._expand == 1.0:
            return points
        cx = sum(x for x, _ in points) / len(points)
        cy = sum(y for _, y in points) / len(points)
        return [
            (cx + (x - cx) * self._expand, cy + (y - cy) * self._expand)
            for x, y in points
        ]

    def _polygon(self, patch, transform) -> QPolygonF:
        scale, dx, dy = transform
        return QPolygonF([
            QPointF(x * scale + dx, y * scale + dy)
            for x, y in self._corners(patch)
        ])

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name
        transform = self._transform()
        if transform is None:
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(BG))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont(painter.font())
        font.setPointSizeF(8.0)
        painter.setFont(font)

        for patch in self.patches:
            polygon = self._polygon(patch, transform)
            box = polygon.boundingRect()
            measured = lab_to_display_rgb(patch.measured_lab)
            reference = (
                lab_to_display_rgb(patch.reference_lab)
                if patch.reference_lab else None
            )

            painter.setPen(Qt.PenStyle.NoPen)
            if reference is not None and box.width() >= self.MIN_SPLIT_PX:
                # Measured on the left, reference on the right, sharing a seam.
                # A seam that vanishes is a patch that matched -- which is the
                # whole reason this is a split and not two grids side by side.
                painter.setClipRect(QRectF(
                    box.left(), box.top(), box.width() / 2, box.height()
                ))
                painter.setBrush(measured)
                painter.drawPolygon(polygon)
                painter.setClipRect(QRectF(
                    box.center().x(), box.top(), box.width() / 2 + 1, box.height()
                ))
                painter.setBrush(reference)
                painter.drawPolygon(polygon)
                painter.setClipping(False)
            else:
                painter.setBrush(measured)
                painter.drawPolygon(polygon)

            flagged = bool(getattr(patch, "flags", None))
            focused = self._hover == patch.patch_id
            if focused:
                painter.setPen(QPen(QColor("#ffffff"), 2.0))
            elif flagged:
                # A fact about the sample, not a verdict on it: this patch was
                # clipped or contaminated, so its value means less than the
                # others', whatever the metric made of it.
                pen = QPen(QColor("#ffc44d"), 1.6)
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
            else:
                painter.setPen(QPen(QColor(BORDER), 0.6))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(polygon)

        if self._hover:
            self._draw_readout(painter)
        painter.end()

    def _draw_readout(self, painter: QPainter) -> None:
        patch = next(
            (p for p in self.patches if p.patch_id == self._hover), None
        )
        if patch is None:
            return
        lines = [f"{patch.patch_id}   {patch.role.replace('_', ' ')}"]
        lines.append(
            "measured  L* {:.1f}  a* {:.1f}  b* {:.1f}".format(*patch.measured_lab)
        )
        if patch.reference_lab:
            lines.append(
                "reference L* {:.1f}  a* {:.1f}  b* {:.1f}".format(*patch.reference_lab)
            )
        if patch.flags:
            lines.append(", ".join(patch.flags))

        metrics = painter.fontMetrics()
        width = max(metrics.horizontalAdvance(line) for line in lines) + 16
        height = len(lines) * (metrics.height() + 1) + 12
        box = QRectF(8, self.height() - height - 8, width, height)
        painter.setPen(QPen(QColor(BORDER), 1.0))
        painter.setBrush(QColor(26, 26, 29, 235))
        painter.drawRoundedRect(box, 5, 5)
        painter.setPen(QColor("#e6e6ea"))
        y = box.top() + 8 + metrics.ascent()
        for index, line in enumerate(lines):
            painter.setPen(QColor("#e6e6ea" if index == 0 else TEXT_DIM))
            painter.drawText(QPointF(box.left() + 8, y), line)
            y += metrics.height() + 1

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt's name
        transform = self._transform()
        if transform is None:
            return
        point = event.position()
        found = None
        for patch in self.patches:
            if self._polygon(patch, transform).containsPoint(
                point, Qt.FillRule.OddEvenFill
            ):
                found = patch.patch_id
                break
        if found != self._hover:
            self._hover = found
            if self.selection is not None and found:
                self.selection.set_hovered(found)
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt's name
        if self._hover is not None:
            self._hover = None
            self.update()
