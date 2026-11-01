"""The capture, kept beside the numbers -- and now the only picture of where.

There used to be two drawings of the target on screen at once: this panel,
showing the frame with every sampled region on it, and a "Field" view drawing
the same positions as abstract marks on an empty rectangle. Both were correct
and they competed to be believed, which is worse than either alone.

So this panel absorbed the field map. It carries the two things a marginal
reading raises, and now the third:

* the **whole frame**, with every region this metric sampled drawn where it
  physically sits -- which is whether the ones misbehaving are clustered or
  scattered;
* the **size of each mark following its reading**, spanning the middle 90% of
  the distribution, which is what makes a soft corner visible before anyone
  reads a number;
* the **focused region at full resolution**, padded so its border is visible --
  which is whether that reading is a soft system or a speck of dust.

It follows the same :class:`~deltachrome_ui.charts.ChartSelection` as the plots
and the readings tables, so pointing at a row brings up its pixels with no
separate gesture, and clicking a mark here focuses it everywhere else. That is
the argument for a permanent panel rather than a tab: the pixels are the thing
you want *while* reading a number, not instead of it.

**It opens collapsed, and opening it loads the capture.** At three hundred and
eighty pixels it is a third of a laptop screen, and until a capture is loaded
what it holds is one button and a sentence explaining that the button has not
been pressed -- the widest thing on the report saying the least. So it starts as
a strip, and the strip carries the words **Load Capture** turned on its side,
because a chevron announces that something is folded away and nothing whatever
about what.

Pressing it is the request. It was two presses -- open the panel, then press the
button inside it -- and the second was a formality: nobody opens a panel labelled
*Load Capture* in order to look at the offer. A capture arriving leaves the panel
open with the strip reading *Capture*, since it should not go on offering
something already done.

**And the wait is on screen, because it was not and that cost real time.**
Decoding this repository's ColorChecker SG master takes 24.6 seconds. A button
that stayed enabled and said the same thing throughout invited a second press,
and a second press started a second decode of the same file. The panel now shows
the same indeterminate bar the loading screen uses -- one
:class:`~deltachrome_ui.widgets.BusyStrip`, shared -- and refuses a second
request while the first is running.

**The capture is loaded on request, not on open.** Opening a saved run costs no
decode -- the whole report renders from the record -- and a JP2 preservation
master takes twenty seconds. Making every reader pay that to populate a panel
they may not look at would undo the thing that makes the library worth opening.
A run *analyzed* in this session is the other case and is not the same one: its
master is already decoded, the report is handed it, and the panel takes it
rather than offering to read the same file again.

**Color-managed, on the same terms as the placement canvas**. Both
panels share `ImageProxy`, so a linear-light master no longer looks dark on
either; what is still true on both is that sRGB is assumed as the viewer's own
display space rather than measured, and no number in a report comes from what
is drawn here.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from deltachrome_ui.charts import ChartSelection
from deltachrome_ui.labels import flag_labels
from deltachrome_ui.theme import ACCENT, BG, BG_RAISED, BORDER, TEXT, TEXT_DIM
from deltachrome_ui.widgets import BusyStrip, figures_font

#: Width of the panel when open. Wider than it was: the frame overview is the
#: report's only picture of where a reading sits now, and at three hundred
#: pixels a landscape chart was a hundred and sixty pixels tall.
DOCK_WIDTH = 384

#: What the panel offers when it holds no capture, said the same way by the
#: button inside it and by the strip that opens it -- because they are one
#: offer reached in two clicks, and two wordings would read as two things.
LOAD_LABEL = "Load Capture"

#: Width when collapsed -- enough for the tab that opens it again, and for its
#: own label turned on its side. Wide enough to read, narrow enough that nobody
#: mistakes it for a panel.
COLLAPSED_WIDTH = 34


class DockTab(QPushButton):
    """The collapsed panel, saying what opening it leads to.

    It was a bare chevron. A chevron says that something is folded away and
    nothing whatever about what -- which is the wrong thing to be silent about
    here, because on most reports this panel is folded away and empty, and the
    one thing behind it is the control that decodes the capture.

    So the strip carries its own label, turned on its side because a strip wide
    enough for horizontal text is not a strip. The chevron stays, upright, at the
    top: rotated with the text it would point somewhere the panel does not open.
    """

    #: Room reserved for the chevron above the label.
    CHEVRON_HEIGHT = 22

    def __init__(self, label: str = "") -> None:
        super().__init__()
        self.setObjectName("DockTab")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._label = label

    def set_label(self, label: str) -> None:
        self._label = label
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name
        # The base class first, so the style sheet's background, border and hover
        # colour are all still the ones every other DockTab uses.
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QColor(TEXT if self.underMouse() else TEXT_DIM))

        painter.drawText(
            QRectF(0, 4, self.width(), self.CHEVRON_HEIGHT),
            Qt.AlignmentFlag.AlignCenter, "‹",
        )
        if not self._label:
            return
        font = QFont(painter.font())
        font.setPointSizeF(max(9.0, font.pointSizeF() - 1.0))
        painter.setFont(font)
        # Bottom to top, so the label reads upward along the right-hand edge of
        # the window rather than upside down from the reader's point of view.
        top = self.CHEVRON_HEIGHT + 6
        height = max(0.0, self.height() - top - 6)
        painter.translate(self.width() / 2, top + height / 2)
        painter.rotate(-90)
        painter.drawText(
            QRectF(-height / 2, -self.width() / 2, height, self.width()),
            Qt.AlignmentFlag.AlignCenter, self._label,
        )


class FrameOverview(QWidget):
    """The whole capture, with this metric's sampled regions on it."""

    #: Tall enough that a chart occupying part of a frame is still big enough to
    #: locate a region on. A minimum rather than a fixed height now: the panel
    #: is the report's only picture of where a reading sits, and it was sitting
    #: at a fixed 300 with a third of the panel empty underneath it. The whole
    #: frame is shown rather than the sampled extent because how much of the
    #: frame the chart fills is itself a finding -- it is the sampling rate the
    #: capture achieves at the target plane.
    HEIGHT = 240

    #: How far a wheel notch moves the zoom, and how far it may go.
    ZOOM_STEP = 1.25
    ZOOM_RANGE = (1.0, 12.0)

    def __init__(self, proxy, selection: ChartSelection, regions: dict) -> None:
        super().__init__()
        self.proxy = proxy
        self.selection = selection
        self.regions = dict(regions)
        self.readings: dict = {}
        self._zoom = 1.0
        #: Where the zoom is centred, in fractions of the image.
        self._center = [0.5, 0.5]
        self._drag: QPointF | None = None
        self.setMinimumHeight(self.HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setMouseTracking(True)
        self._pixmap = QPixmap.fromImage(proxy.overview)
        selection.changed.connect(self.update)

    def _extents(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Widget and image extent per axis, in the order the centre is stored."""
        return (
            (self.width(), self.proxy.width),
            (self.height(), self.proxy.height),
        )

    def _fit(self) -> float:
        """Scale at which the whole frame just fits. Zoom 1 is this."""
        return min(
            self.width() / max(1, self.proxy.width),
            self.height() / max(1, self.proxy.height),
        )

    def _placement(self) -> tuple[float, float, float]:
        """Scale and offset taking image pixels to widget pixels.

        Zoom multiplies the fit scale and the offset is derived from where the
        view is centred, so panning and hit-testing cannot disagree about where
        a region is: both ask this.

        There is no special case for zoom 1 any more. There used to be one
        centring the image explicitly, which is what :meth:`_clamp` now
        guarantees on its own -- an axis with the whole image visible has its
        centre pinned to 0.5, and the two expressions are then identical.
        """
        scale = self._fit() * self._zoom
        dx = self.width() / 2 - self._center[0] * self.proxy.width * scale
        dy = self.height() / 2 - self._center[1] * self.proxy.height * scale
        return scale, dx, dy

    def _clamp(self) -> None:
        """Hold the zoom in range and the view over the image, **per axis**.

        The fraction of the image visible along an axis is the widget's extent
        over the image's, and that is *not* ``1 / zoom``: the fit scale is set by
        whichever axis is tighter, so the other one is letterboxed and shows more
        of itself at every zoom. Treating both as ``1 / zoom`` -- which is what
        this did -- clamped the loose axis to a window narrower than what was on
        screen, so panning it fought the pointer and a zoom near the top or
        bottom edge was pulled back toward the middle.
        """
        self._zoom = max(self.ZOOM_RANGE[0], min(self.ZOOM_RANGE[1], self._zoom))
        scale = self._fit() * self._zoom
        for axis, (widget_extent, image_extent) in enumerate(self._extents()):
            visible = widget_extent / max(1e-6, image_extent * scale)
            if visible >= 1.0:
                # The whole of this axis is on screen; there is nowhere to pan to.
                self._center[axis] = 0.5
                continue
            half = visible / 2.0
            self._center[axis] = max(half, min(1.0 - half, self._center[axis]))

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt's name
        """Zoom about the pointer, and mean it.

        The panel is the report's only picture of where a reading sits, and on a
        hundred-and-forty-patch chart at three hundred and eighty pixels wide the
        patches are three pixels across. Zooming is what makes the marks
        something a person can actually aim at, and it is only worth having if
        the thing under the pointer stays under the pointer.

        **It did not.** The new centre was solved against the *widget's* width and
        height where the invariant is in image pixels -- ``image extent x scale``
        -- and those agree only on the axis the fit scale was set by. On this
        panel that is the wide axis of a landscape capture, so horizontal zoom
        was right and vertical zoom crept, by the ratio between the widget's
        height and the letterboxed image's. Zooming in then walked the view off
        the point of interest, worse the further in it already was.
        """
        notches = event.angleDelta().y() / 120.0
        if not notches:
            return
        scale, dx, dy = self._placement()
        # The point of the image under the pointer, in fractions of the image.
        # This is the quantity the whole gesture holds still.
        anchor = (
            (event.position().x() - dx) / max(1e-6, self.proxy.width * scale),
            (event.position().y() - dy) / max(1e-6, self.proxy.height * scale),
        )
        self._zoom = max(self.ZOOM_RANGE[0], min(
            self.ZOOM_RANGE[1], self._zoom * self.ZOOM_STEP ** notches
        ))
        # Clamped before it is used, so the scale solved against is the scale
        # that will be drawn with.
        after = self._fit() * self._zoom
        self._center = [
            anchor[0]
            - (event.position().x() - self.width() / 2)
            / max(1e-6, self.proxy.width * after),
            anchor[1]
            - (event.position().y() - self.height() / 2)
            / max(1e-6, self.proxy.height * after),
        ]
        self._clamp()
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt's name  # noqa: ARG002
        self._zoom, self._center = 1.0, [0.5, 0.5]
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(BG))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        scale, dx, dy = self._placement()
        target = QRectF(
            dx, dy, self.proxy.width * scale, self.proxy.height * scale
        )
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        focus = self.selection.focus()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # **One faint ink for every region, and colour only on the one in
        # focus.** Regions were each outlined in their group's hue with a dot in
        # the middle, and at this size the dots were larger than the squares and
        # the edge hues the loudest thing in the frame -- a picture of the
        # markup rather than of the capture. What a reader needs from the frame
        # is where the regions sit and which one they are asking about, and
        # neither needs a hue. Edges are dashed and patches solid, so the two
        # kinds still read apart. Never coloured by pass or fail either: these
        # thresholds are conditions on the aggregate, and a per-reading verdict
        # is one the profile never formed.
        for focused_pass in (False, True):
            for name, region in self.regions.items():
                if (name in focus) != focused_pass:
                    continue
                if focused_pass:
                    pen = QPen(QColor(ACCENT), 2.0)
                else:
                    ink = QColor(255, 255, 255)
                    ink.setAlphaF(0.45)
                    pen = QPen(ink, 1.0)
                    if region.dashed:
                        pen.setDashPattern([3.0, 2.0])
                painter.setPen(pen)
                # Every outline of the reading -- two for a pair of patches.
                for shape in region.shapes:
                    painter.drawPolygon(QPolygonF([
                        QPointF(float(x) * scale + dx, float(y) * scale + dy)
                        for x, y in shape
                    ]))

        # A ring around whatever is focused. At this scale one outlined patch
        # among a hundred and forty is not findable by eye, and the point of
        # showing the frame is to say *where*.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for name in focus:
            region = self.regions.get(name)
            if region is None:
                continue
            for shape in region.shapes:
                xs = [float(x) * scale + dx for x, _ in shape]
                ys = [float(y) * scale + dy for _, y in shape]
                center = QPointF((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
                radius = max(9.0, max(max(xs) - min(xs), max(ys) - min(ys)))
                painter.setPen(QPen(QColor(ACCENT), 1.4))
                painter.drawEllipse(center, radius, radius)
        painter.end()

    def _at(self, point) -> str | None:
        scale, dx, dy = self._placement()
        best: tuple[float, str] | None = None
        for name, region in self.regions.items():
            for shape in region.shapes:
                polygon = QPolygonF([
                    QPointF(float(x) * scale + dx, float(y) * scale + dy)
                    for x, y in shape
                ])
                if polygon.containsPoint(point, Qt.FillRule.OddEvenFill):
                    return name
                # A patch on a 140-patch chart can be three pixels across here,
                # so falling back to the nearest centre within a small radius is
                # the difference between the marks being clickable and being
                # decorative.
                center = polygon.boundingRect().center()
                distance = (center - point).manhattanLength()
                if distance < 12 and (best is None or distance < best[0]):
                    best = (distance, name)
        return best[1] if best else None

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt's name
        if self._drag is not None and self._zoom > 1.0:
            scale, _, _ = self._placement()
            delta = event.position() - self._drag
            self._drag = event.position()
            self._center[0] -= delta.x() / max(1.0, self.proxy.width * scale)
            self._center[1] -= delta.y() / max(1.0, self.proxy.height * scale)
            self._clamp()
            self.update()
            return
        self.selection.set_hovered(self._at(event.position()))

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt's name
        name = self._at(event.position())
        if name:
            self.selection.toggle_selected(name)
        elif self._zoom > 1.0:
            self._drag = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt's name  # noqa: ARG002
        self._drag = None
        self.unsetCursor()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt's name
        self._drag = None
        self.selection.set_hovered(None)


class RegionLoupe(QWidget):
    """The focused region at the resolution this width can show."""

    HEIGHT = 200
    #: How much context to show around the region, as a fraction of its size.
    #: The question is usually about the *border* -- a speck on the edge of a
    #: patch, a rule sitting off its printed scale -- so showing exactly the
    #: sampled area answers "is the middle clean", which is rarely what was in
    #: doubt. The outline is drawn on top, so what was measured stays plain.
    PAD = 0.35

    def __init__(self, proxy, selection: ChartSelection, regions: dict) -> None:
        super().__init__()
        self.proxy = proxy
        self.selection = selection
        self.regions = dict(regions)
        self.setMinimumHeight(self.HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._pixmap: QPixmap | None = None
        self._outline: list[list[tuple[float, float]]] = []
        self._tile = None
        self._shown: str | None = None
        selection.changed.connect(self._sync)

    def _sync(self) -> None:
        name = self.selection.primary
        if name == self._shown:
            return
        self._shown = name
        region = self.regions.get(name) if name else None
        self._pixmap = None
        self._outline = []
        self._tile = None
        if region is not None:
            bounds = region.bounds(self.PAD)
            stride = self.proxy.stride_for(bounds, self.width() or DOCK_WIDTH)
            tile = self.proxy.tile(bounds, stride=stride)
            self._tile = tile
            self._pixmap = QPixmap.fromImage(tile.image)
            self._outline = [
                [(float(x), float(y)) for x, y in shape] for shape in region.shapes
            ]
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(BG))
        if self._pixmap is None or self._tile is None:
            # Only where there is something to point at. A metric with no
            # regions -- one the target cannot carry, the file's bit depth --
            # offered an instruction nothing on the screen could follow.
            if self.regions:
                painter.setPen(QColor(TEXT_DIM))
                painter.drawText(
                    self.rect().adjusted(14, 0, -14, 0),
                    int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
                    "Hover over a region, chart or value to isolate it.",
                )
            painter.end()
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        pw, ph = self._pixmap.width(), self._pixmap.height()
        scale = min(self.width() / max(1, pw), self.height() / max(1, ph))
        dx = (self.width() - pw * scale) / 2
        dy = (self.height() - ph * scale) / 2
        painter.drawPixmap(
            QRectF(dx, dy, pw * scale, ph * scale),
            self._pixmap, QRectF(self._pixmap.rect()),
        )

        x0, y0, _, _ = self._tile.region
        stride = max(1, self._tile.stride)
        painter.setPen(QPen(QColor(ACCENT), 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for shape in self._outline:
            painter.drawPolygon(QPolygonF([
                QPointF((x - x0) / stride * scale + dx, (y - y0) / stride * scale + dy)
                for x, y in shape
            ]))
        painter.end()


class Readout(QWidget):
    """What the focused region is, at a fixed height.

    Painted rather than laid out, and that is the whole point of the class. As
    labels in a layout it grew a line when a reading arrived and another when
    that reading carried a flag, so pointing at one patch and then at its
    neighbour moved everything under it -- including the thing being pointed
    at. A fixed box cannot shift.

    The two swatches are the aim and the measurement, side by side. For a
    colour patch that comparison *is* the metric, and until now the report
    showed a number for it and never the two colours -- nor, until later, the
    two L*a*b* behind them, which sit at the right as a small table. The
    flags have the bottom line to themselves, the full width, so a patch's
    Lab does not elide what is wrong with it.
    """

    HEIGHT = 70
    SWATCH = 26
    #: The Lab table's columns: a row label, then L*, a* and b*.
    LAB_COLUMNS = (34, 46, 46, 46)

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(self.HEIGHT)
        self.name = ""
        self.reading = ""
        self.flags = ""
        self.aim: QColor | None = None
        self.measured: QColor | None = None
        self.aim_lab: tuple | None = None
        self.measured_lab: tuple | None = None

    def show_region(self, name, reading, flags, aim, measured,
                    aim_lab=None, measured_lab=None) -> None:
        self.name = name or ""
        self.reading = reading or ""
        self.flags = flags or ""
        self.aim, self.measured = aim, measured
        self.aim_lab, self.measured_lab = aim_lab, measured_lab
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        left = 0.0
        if self.aim is not None or self.measured is not None:
            size = self.SWATCH
            for index, color in enumerate((self.aim, self.measured)):
                box = QRectF(index * (size + 3), 2, size, size)
                painter.setPen(QPen(QColor(BORDER), 1))
                painter.setBrush(color or QColor(BG))
                painter.drawRect(box)
            painter.setPen(QPen(QColor(TEXT_DIM)))
            font = painter.font()
            font.setPointSizeF(8.0)
            painter.setFont(font)
            painter.drawText(
                QRectF(0, size + 3, size * 2 + 3, 12),
                int(Qt.AlignmentFlag.AlignLeft), "aim / read",
            )
            left = size * 2 + 14

        right = float(self.width())
        if self.aim_lab is not None or self.measured_lab is not None:
            right -= sum(self.LAB_COLUMNS) + 8
            self._paint_lab(painter, right + 8)

        painter.setPen(QPen(QColor(TEXT)))
        figures = figures_font(10.5, painter.font())
        painter.setFont(figures)
        metrics = painter.fontMetrics()
        for row, text in enumerate((self.name, self.reading)):
            painter.drawText(
                QRectF(left, row * 17, right - left, 17),
                int(Qt.AlignmentFlag.AlignVCenter),
                metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(right - left)),
            )
        if self.flags:
            painter.setPen(QPen(QColor("#ffc44d")))
        figures.setPointSizeF(9.5)
        painter.setFont(figures)
        painter.drawText(
            QRectF(left, 52, self.width() - left, 16),
            int(Qt.AlignmentFlag.AlignVCenter),
            painter.fontMetrics().elidedText(
                self.flags, Qt.TextElideMode.ElideRight, int(self.width() - left)),
        )
        painter.end()

    def _paint_lab(self, painter: QPainter, x: float) -> None:
        """Aim and read L*a*b*, a row each, under a dim heading."""
        figures = figures_font(9.5, painter.font())
        painter.setFont(figures)
        rows = (("", ("L*", "a*", "b*")),
                ("Aim", self.aim_lab), ("Read", self.measured_lab))
        for row, (label, values) in enumerate(rows):
            y = row * 17
            cells = [label] + (
                list(values) if row == 0
                else [f"{v:.2f}" for v in values] if values is not None
                else ["—"] * 3
            )
            left = x
            for column, (width, text) in enumerate(zip(self.LAB_COLUMNS, cells)):
                painter.setPen(QPen(QColor(TEXT if row and column else TEXT_DIM)))
                align = (Qt.AlignmentFlag.AlignLeft if column == 0
                         else Qt.AlignmentFlag.AlignRight)
                painter.drawText(QRectF(left, y, width, 17),
                                 int(align | Qt.AlignmentFlag.AlignVCenter), text)
                left += width


def _panel_heading(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setStyleSheet(
        f"color: {TEXT_DIM}; font-size: 10px; font-weight: 700;"
        "letter-spacing: 0.8px;"
    )
    return label


class CaptureDock(QWidget):
    """Where a reading sits, and what it is made of.

    Lives for the whole report rather than for one view, and is re-pointed at
    whichever metric is selected. The frame does not change; which regions are
    drawn on it does -- and only ever the ones belonging to the metric on
    screen, so a mark on the capture is always something the current page can
    say a number about.
    """

    #: The operator asked for the capture to be decoded.
    loadRequested = Signal()

    def __init__(self, selection: ChartSelection, filename: str = "") -> None:
        super().__init__()
        self.setObjectName("CaptureDock")
        self.setStyleSheet(
            f"QWidget#CaptureDock {{ background: {BG_RAISED}; "
            f"border-left: 1px solid {BORDER}; }}"
        )
        self.selection = selection
        self.regions: dict = {}
        self.readings: dict = {}
        self.swatches: dict = {}
        self.metric_name = ""
        self.proxy = None
        self.overview: FrameOverview | None = None
        self.loupe: RegionLoupe | None = None
        self.collapsed = False
        #: Whether a decode is running. Held here rather than inferred from what
        #: is on screen, because it is what stops a second request -- and a
        #: second request is a second decode of the same three hundred megabytes.
        self.loading = False

        shell = QHBoxLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        # The collapsed state is its own strip rather than the panel resized to
        # nothing: a zero-width panel has no handle to bring it back, and a
        # reader who has just hidden it is exactly the reader who will want it
        # again in a moment.
        self.tab = DockTab(LOAD_LABEL)
        self.tab.setFixedWidth(COLLAPSED_WIDTH)
        # Full height. A button takes its natural height, which put a 30 px chip
        # halfway down the right-hand edge -- a thing to hunt for rather than an
        # edge of the window, and with nowhere to write what it opens.
        self.tab.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )
        self.tab.clicked.connect(self.open_panel)
        self.tab.hide()
        shell.addWidget(self.tab)

        self.body = QWidget()
        self.body.setFixedWidth(DOCK_WIDTH)
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        self._layout = layout
        shell.addWidget(self.body)

        title = QHBoxLayout()
        title.setSpacing(6)
        label = QLabel("Capture")
        label.setStyleSheet("font-size: 13px; font-weight: 600;")
        title.addWidget(label, 0)
        title.addStretch(1)
        hide = QPushButton("›")
        hide.setObjectName("DockTab")
        hide.setFixedSize(22, 22)
        hide.setCursor(Qt.CursorShape.PointingHandCursor)
        hide.setToolTip("Hide capture")
        hide.clicked.connect(lambda: self.set_collapsed(True))
        title.addWidget(hide, 0)
        layout.addLayout(title)

        # What the marks belong to. Without it the panel is a picture of a chart
        # with dots on it, and which dots depends on a selection made elsewhere.
        # One line, never two. Wrapping to a second line moved every pixel view
        # beneath it as the metric changed.
        self.scope = QLabel("")
        self.scope.setWordWrap(False)
        self.scope.setFixedHeight(16)
        self.scope.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        layout.addWidget(self.scope)

        self.frame_heading = _panel_heading("Whole Frame")
        self.frame_heading.hide()
        layout.addWidget(self.frame_heading)

        self.slot = QWidget()
        self.slot_layout = QVBoxLayout(self.slot)
        self.slot_layout.setContentsMargins(0, 0, 0, 0)
        self.slot_layout.setSpacing(8)
        layout.addWidget(self.slot, 1)

        self.prompt = QPushButton(LOAD_LABEL)
        self.prompt.clicked.connect(self.request_load)
        self.slot_layout.addWidget(self.prompt)

        # The same bar the loading screen uses. Narrower, because this is a panel
        # rather than a window, and hidden until there is something to say.
        self.busy = BusyStrip("Loading File\u2026", width=DOCK_WIDTH - 80)
        self.busy.hide()
        # Where the button was, rather than centred in a panel's worth of empty
        # space: the wait replaces the offer, in the place the offer was made.
        self.slot_layout.addWidget(self.busy, 0, Qt.AlignmentFlag.AlignTop)

        self.explain = QLabel(
            f"{filename or 'The capture'} is not open. The report is complete "
            "without it — only the pixels behind a reading are missing."
        )
        self.explain.setWordWrap(True)
        self.explain.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        self.slot_layout.addWidget(self.explain)

        self.region_heading = _panel_heading("Focused Region")
        self.region_heading.hide()
        layout.addWidget(self.region_heading)

        self.readout = Readout()
        layout.addWidget(self.readout, 0)

        selection.changed.connect(self._sync)
        self._sync()
        # Folded away until there is a capture in it. Most reports are opened
        # from the library and never decode one, and a third of a laptop screen
        # spent on a button and a sentence explaining that the button has not
        # been pressed is the widest thing on the screen saying the least.
        self.set_collapsed(True)

    # -- collapsing --------------------------------------------------------

    def set_collapsed(self, collapsed: bool) -> None:
        self.collapsed = bool(collapsed)
        self.body.setVisible(not self.collapsed)
        self.tab.setVisible(self.collapsed)
        self.setFixedWidth(COLLAPSED_WIDTH if self.collapsed else DOCK_WIDTH)

    def open_panel(self) -> None:
        """Unfold, and ask for the capture if there is not one.

        Opening a strip labelled *Load Capture* is the request. Making it two
        presses treated the strip as a disclosure triangle over a button, when
        what it actually is is the button -- turned on its side because a panel
        this wide has no business being open with nothing in it.

        Separate from :meth:`set_collapsed` on purpose: that is also called when
        a capture arrives and when a run is cleared, and neither of those is
        somebody asking for anything.
        """
        self.set_collapsed(False)
        self.request_load()

    def request_load(self) -> None:
        """Ask the shell to decode the capture, at most once at a time.

        The guard is not defensive tidiness. Each request is a full decode --
        24.6 s for this repository's JPEG 2000 sample -- and before the bar
        existed a second press during the first decode was the ordinary thing to
        do, which bought two of them.
        """
        if self.proxy is not None or self.loading:
            return
        self.loadRequested.emit()

    def set_loading(self, loading: bool, message: str = "Loading File\u2026") -> None:
        """Show the wait, or take it away.

        Taking it away without a capture is a *failed* load, which is the one
        state where the button inside the panel is the point: it is the retry,
        and the reason the panel does not fold itself back up on a failure is
        that folding it would hide the only way to try again.
        """
        self.loading = bool(loading)
        if loading:
            self.busy.set_message(message)
        self.busy.setVisible(self.loading)
        showing_offer = self.proxy is None and not self.loading
        self.prompt.setVisible(showing_offer)
        self.explain.setVisible(showing_offer)

    def _label_the_tab(self) -> None:
        """Say what is behind the strip, which is not the same thing twice.

        With no capture open the panel holds one control and the label is that
        control. With one open it holds the frame and the loupe, and offering to
        load a capture that is already loaded would be the strip lying about the
        only thing it says.
        """
        loaded = self.proxy is not None
        self.tab.set_label("Capture" if loaded else LOAD_LABEL)
        self.tab.setToolTip(
            "Show capture" if loaded else ""
        )

    # -- what is on screen -------------------------------------------------

    def reset(self) -> None:
        """Forget the previous run entirely.

        The panel outlives a result -- it is built once and re-pointed -- so
        without this a new run inherited the last one's decoded frame and drew
        one capture's pixels beside another capture's numbers, with nothing on
        screen saying so.
        """
        self.set_proxy(None)
        self.set_regions({}, {}, "")

    def set_proxy(self, proxy) -> None:
        """The capture arrived, or went away."""
        self.proxy = proxy
        for widget in (self.overview, self.loupe):
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.overview = None
        self.loupe = None

        self._label_the_tab()
        if proxy is None:
            self.set_loading(False)
            self.frame_heading.hide()
            self.region_heading.hide()
            # Back to a strip. A panel that stayed open holding nothing but the
            # offer to load something is the state this fold exists to avoid, and
            # it is reached whenever a second run opens over the first.
            self.set_collapsed(True)
            self._sync()
            return

        # Note what this does *not* do: open the panel. A capture arriving is
        # not by itself a request to look at it -- a run analyzed in this session
        # is handed its master automatically, and having the panel unfold for it
        # meant the next report opened wide because the previous one had been
        # looked at. Opening is a gesture, and the strip is where it is made.
        self.set_loading(False)
        self.overview = FrameOverview(proxy, self.selection, self.regions)
        self.overview.readings = self.readings
        self.slot_layout.addWidget(self.overview)
        self.loupe = RegionLoupe(proxy, self.selection, self.regions)
        # Under its own heading, and sharing the panel's spare height with the
        # frame: the two pixel views are the panel, and everything else on it is
        # a caption. Each is labelled, because "the whole capture" and "one
        # region of it at 1:1" are not obviously different pictures when the
        # region happens to be a flat grey patch.
        self._layout.insertWidget(
            self._layout.indexOf(self.region_heading) + 1, self.loupe, 1
        )
        self.frame_heading.show()
        self.region_heading.show()
        self._sync()

    def set_regions(
        self, regions: dict, readings: dict | None = None, metric: str = "",
        swatches: dict | None = None,
    ) -> None:
        """Point the panel at another metric's readings.

        ``swatches`` is ``{name: (aim, measured)}`` where the reading is a
        colour patch and empty otherwise -- a flat-field spot and a slanted edge
        have no colour they were aiming at.
        """
        self.regions = dict(regions)
        self.readings = dict(readings or {})
        self.swatches = dict(swatches or {})
        self.metric_name = metric
        if self.overview is not None:
            self.overview.regions = self.regions
            self.overview.readings = self.readings
            self.overview.update()
        if self.loupe is not None:
            self.loupe.regions = self.regions
            self.loupe._shown = None
            self.loupe._sync()
        self._sync()

    def _sync(self) -> None:
        count = len(self.regions)
        if not count:
            self.scope.setText(
                "No placed readings for this metric."
                if self.metric_name else "Nothing here is placed on the chart."
            )
        else:
            self.scope.setText(
                f"{count} region{'s' if count != 1 else ''}  ·  "
                f"{self.metric_name or 'this metric'}"
            )

        name = self.selection.primary
        region = self.regions.get(name) if name else None
        if region is None:
            self.readout.show_region("", "", "", None, None)
            return
        # Identity, reading, flags -- and the two colours where there are two.
        # Centre, size in pixels and the decimation stride used to be here and
        # answered questions nobody had: where is answered by the frame above,
        # and the stride is a property of this panel rather than of the run.
        reading, flags = "", ""
        found = self.readings.get(name)
        if found:
            value, units, raised = found
            reading = f"{value:.3f} {units}".rstrip()
            flags = ", ".join(flag_labels(raised))
        aim, measured, aim_lab, measured_lab = (
            tuple(self.swatches.get(name, ())) + (None,) * 4)[:4]
        self.readout.show_region(name, reading, flags, aim, measured,
                                 aim_lab, measured_lab)
