"""Where each chart of a capture physically sits, and how it came out.

A board of four QA-62 cards is four runs, and the report's stepper names them
``Target 1 of 4`` through ``Target 4 of 4``. That ordinal is the order the
extent search returned the regions in -- it is not a position, not a reading
order, and not anything an operator can predict -- so on four cards of one model
it identifies nothing. The combined verdict is exactly where that bites, because
it is the screen whose job is comparing them.

So the screen opens on the capture with every chart outlined where it was
measured. The argument is the one that took the Field view off the report:
**where a reading sits is answered on the capture, not on a diagram.** This is
that decision one level up, about charts rather than about patches.

**The outline carries the run's own status tone**, through
:mod:`deltachrome_ui.status` like everything else that states a verdict -- which
means all five tones and not a red/green pair. Every ISA target and every QA-62
runs on nominal values, so an advisory pass is the ordinary result here, and a
green outline on one would be the frontend telling a lie the CLI never tells.

**The quadrilaterals come from the record and the pixels do not.** Every chart's
``geometry.corners_px`` was written when the run was made, so a capture reopened
from the library can draw its whole frame -- outlines, ordinals, verdicts, the
frame's own dimensions and aspect -- without decoding anything. Decoding a
preservation master is five to sixteen seconds, and a screen that waited for it
before saying where Target 3 was would be charging that for an answer it already
had. The pixels make the charts recognisable; they do not make them locatable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from deltachrome_ui.status import Status
from deltachrome_ui.theme import BG, BG_INPUT, BORDER, TEXT, TEXT_DIM, tone_colors
from deltachrome_ui.widgets import heading


@dataclass
class ChartInFrame:
    """One chart of a capture: where it was measured, and how it came out."""

    ordinal: int
    corners: list[tuple[float, float]]
    status: Status
    target_name: str
    run_id: str = ""
    #: What qualifies the reading, said in a few words. Never the verdict again.
    note: str = ""


@dataclass
class FrameInfo:
    """One capture and every chart read from it.

    Assembled by the shell rather than by the engine. A
    :class:`~deltachrome.session.CaptureOutcome` deliberately carries no
    placement -- combining readings needs none -- and widening it so that one
    screen could draw a picture would put a presentation concern in the record.
    Everything here is already held by whoever opens the screen: a fresh board
    has its :class:`~deltachrome.pipeline.AnalysisResult` objects, and a saved
    capture has its charts' own ``geometry.corners_px``.
    """

    width: int
    height: int
    #: What the capture is called, for the screens that name it. The session's
    #: own capture labels are *labels* -- "chart 1 (Applied Image QA-62)" -- and
    #: every other screen in this application puts the image's filename in that
    #: slot, so this is where it comes from.
    filename: str = ""
    charts: list[ChartInFrame] = field(default_factory=list)
    #: The decoded frame, when there is one. Absent is the ordinary case for a
    #: capture reopened from the library, and the map works without it.
    image: QImage | None = None

    @property
    def usable(self) -> bool:
        """Whether this says enough to draw.

        Two charts is the floor, not one: on a single-chart capture the quad is
        very nearly the whole frame and the outline is a border round a picture,
        which answers a question nobody asked.
        """
        return (self.width > 0 and self.height > 0
                and len(self.charts) > 1
                and all(len(chart.corners) >= 3 for chart in self.charts))


class FrameMap(QWidget):
    """The capture, with every chart outlined where it was measured."""

    #: A chart was clicked. Carries its ``run_id``, which is what opens a report.
    chartActivated = Signal(str)
    #: The pointer moved onto a chart, or off every chart (``-1``).
    chartHovered = Signal(int)

    #: Screen-space radius of the ordinal badge.
    BADGE = 13

    def __init__(self, frame: FrameInfo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame = frame
        self._hover = -1
        self.setMouseTracking(True)
        self.setMinimumHeight(360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # -- state -------------------------------------------------------------

    def set_image(self, image: QImage | None) -> None:
        """Hand the map its pixels, if and when they arrive."""
        self._frame.image = image
        self.update()

    def set_hover(self, index: int) -> None:
        """Light one chart from outside -- from the list under the frame."""
        if index != self._hover:
            self._hover = index
            self.update()

    @property
    def canvas_width(self) -> int:
        """How wide the frame actually drew, for anything aligning to it."""
        return int(self._canvas().width())

    # -- geometry ----------------------------------------------------------

    def _canvas(self) -> QRectF:
        scale = min(self.width() / max(1, self._frame.width),
                    self.height() / max(1, self._frame.height))
        width = self._frame.width * scale
        height = self._frame.height * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2,
                      width, height)

    def _polygon(self, corners, box: QRectF) -> QPolygonF:
        return QPolygonF([
            QPointF(box.x() + x / self._frame.width * box.width(),
                    box.y() + y / self._frame.height * box.height())
            for x, y in corners
        ])

    # -- interaction -------------------------------------------------------

    def _outlines(self) -> list[QPolygonF]:
        """Where each chart's outline sits right now, at the current size.

        Computed on demand rather than cached by ``paintEvent``. Hit-testing a
        cache that only painting fills means a press lands correctly only after
        a repaint: a click before the first paint finds nothing, and a click
        after a resize is tested against the previous size's outlines, which on
        a frame of four cards can answer with the wrong one.
        """
        box = self._canvas()
        return [self._polygon(c.corners, box) for c in self._frame.charts]

    def _at(self, point: QPointF) -> int:
        for index, polygon in enumerate(self._outlines()):
            if polygon.containsPoint(point, Qt.FillRule.OddEvenFill):
                return index
        return -1

    def mouseMoveEvent(self, event) -> None:
        found = self._at(QPointF(event.position()))
        if found != self._hover:
            self._hover = found
            self.setCursor(Qt.CursorShape.PointingHandCursor if found >= 0
                           else Qt.CursorShape.ArrowCursor)
            self.chartHovered.emit(found)
            self.update()

    def leaveEvent(self, event) -> None:
        if self._hover != -1:
            self._hover = -1
            self.chartHovered.emit(-1)
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        index = self._at(QPointF(event.position()))
        if index >= 0:
            run_id = self._frame.charts[index].run_id
            if run_id:
                self.chartActivated.emit(run_id)

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        box = self._canvas()

        if self._frame.image is not None:
            painter.drawImage(box, self._frame.image)
        else:
            self._paint_empty_frame(painter, box)

        for index, (chart, polygon) in enumerate(
                zip(self._frame.charts, self._outlines())):
            self._paint_chart(painter, chart, polygon, lit=index == self._hover)
        painter.end()

    def _paint_empty_frame(self, painter: QPainter, box: QRectF) -> None:
        """The frame's own shape, while its pixels are elsewhere.

        Drawn at the capture's real aspect and labelled with its real size, so
        what is on screen is a picture of this capture rather than a placeholder
        that happens to be rectangular.
        """
        painter.fillRect(box, QColor(BG_INPUT))
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawRect(box)
        painter.setPen(QColor(TEXT_DIM))
        painter.setFont(QFont(self.font().family(), 11))
        painter.drawText(
            box, Qt.AlignmentFlag.AlignCenter,
            f"{self._frame.width} × {self._frame.height}")

    def _paint_chart(self, painter: QPainter, chart: ChartInFrame,
                     polygon: QPolygonF, *, lit: bool) -> None:
        colour = QColor(tone_colors(chart.status.tone)[0])
        if lit:
            fill = QColor(colour)
            fill.setAlpha(46)
            painter.setBrush(fill)
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(colour, 3 if lit else 2))
        painter.drawPolygon(polygon)

        # The ordinal, always drawn: it is what matches this outline to
        # ``Target n of N`` in the report, and a label that appears on hover
        # cannot be the thing a reader looks something up by. Outside the quad,
        # because on several targets the top-left corner is the printed fiducial
        # the placement was made on.
        corner = polygon.at(0)
        centre = QPointF(corner.x() - self.BADGE + 2, corner.y() - self.BADGE + 2)
        badge = QRectF(centre.x() - self.BADGE, centre.y() - self.BADGE,
                       self.BADGE * 2, self.BADGE * 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        painter.drawEllipse(badge)
        painter.setPen(QColor(BG))
        font = QFont(self.font().family(), 11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, str(chart.ordinal))


class ChartRow(QWidget):
    """One chart under the frame: its ordinal, what it is, how it came out.

    Full width rather than in a column beside the frame, which was measured and
    rejected: a 330 px margin elides every name to ``Applied Image Q…``, and
    telling charts apart is the one thing this list exists for.
    """

    clicked = Signal(int)
    hovered = Signal(int)

    HEIGHT = 38
    BADGE = 22

    def __init__(self, index: int, chart: ChartInFrame,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self._chart = chart
        self._lit = False
        self.setFixedHeight(self.HEIGHT)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(
            f"{chart.status.label} — {chart.target_name}\n\n{chart.status.hint}")

    def set_lit(self, lit: bool) -> None:
        if lit != self._lit:
            self._lit = lit
            self.update()

    def enterEvent(self, event) -> None:
        self.hovered.emit(self._index)

    def leaveEvent(self, event) -> None:
        self.hovered.emit(-1)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._index)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colour = QColor(tone_colors(self._chart.status.tone)[0])
        if self._lit:
            tint = QColor(colour)
            tint.setAlpha(28)
            painter.fillRect(self.rect(), tint)
        painter.setPen(QColor(BORDER))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)

        middle = self.height() / 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        painter.drawEllipse(QRectF(10, middle - self.BADGE / 2,
                                   self.BADGE, self.BADGE))
        bold = QFont(self.font().family(), 11)
        bold.setBold(True)
        painter.setFont(bold)
        painter.setPen(QColor(BG))
        painter.drawText(QRectF(10, middle - self.BADGE / 2, self.BADGE, self.BADGE),
                         Qt.AlignmentFlag.AlignCenter, str(self._chart.ordinal))

        # The verdict is placed first and the description elided into what is
        # left, so a long note cannot run under the answer.
        label = self._chart.status.label
        painter.setPen(colour)
        verdict_width = painter.fontMetrics().horizontalAdvance(label) + 20
        painter.drawText(
            QRectF(self.width() - verdict_width, 0, verdict_width - 10, self.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, label)

        left = 10 + self.BADGE + 12
        room = self.width() - verdict_width - left
        text = self._chart.target_name
        if self._chart.note:
            text += f"   ·   {self._chart.note}"
        painter.setFont(QFont(self.font().family(), 11))
        painter.setPen(QColor(TEXT))
        painter.drawText(
            QRectF(left, 0, room, self.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            painter.fontMetrics().elidedText(
                text, Qt.TextElideMode.ElideRight, int(room)))


class FrameBlock(QWidget):
    """The frame and its chart list, lighting each other.

    One selection shared between a picture and a list is the same arrangement
    the report already uses for its plots, its readings tables and the capture
    panel -- pointing at a row lights its mark and the other way round.
    """

    chartActivated = Signal(str)

    def __init__(self, frame: FrameInfo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame = frame

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)

        self.map = FrameMap(frame)
        self.map.chartActivated.connect(self.chartActivated.emit)
        self.map.chartHovered.connect(self._light)
        column.addWidget(self.map, 1)

        # *Target* is this application's word for the card in front of the
        # camera; *chart* is the plot of numbers on a report. The code says
        # "chart" because the engine does -- `capture_record(charts)` -- and
        # the screen says what a person calls the thing.
        column.addWidget(heading("Targets in This Capture"))
        self._rows: list[ChartRow] = []
        for index, chart in enumerate(frame.charts):
            row = ChartRow(index, chart)
            row.hovered.connect(self._light)
            row.clicked.connect(self._activate)
            column.addWidget(row, 0)
            self._rows.append(row)

    def set_image(self, image: QImage | None) -> None:
        self.map.set_image(image)

    def _light(self, index: int) -> None:
        self.map.set_hover(index)
        for position, row in enumerate(self._rows):
            row.set_lit(position == index)

    def _activate(self, index: int) -> None:
        run_id = self._frame.charts[index].run_id
        if run_id:
            self.chartActivated.emit(run_id)


def capture_key(fingerprint) -> str | None:
    """What says two records describe the same capture.

    ``file_fingerprint`` writes a mapping -- the algorithm, the size and the
    digest -- and the **digest** is the part that identifies the file.
    Comparing the mappings whole would work today and break the day the
    algorithm string changes, which is precisely when a library holds records
    written by two versions of this application.

    ``None`` when there is nothing to compare, and ``None`` never matches:
    records written before schema 7 carry no fingerprint at all, and two of
    those are not thereby one capture.
    """
    if isinstance(fingerprint, dict):
        digest = fingerprint.get("digest")
        return str(digest) if digest else None
    return str(fingerprint) if fingerprint else None


def frame_from_records(charts: Sequence[dict], statuses: Sequence[Status],
                       notes: Sequence[str] = ()) -> FrameInfo | None:
    """A frame map from saved run records, which carry everything but pixels.

    Returns ``None`` when the records do not describe one frame -- charts read
    from different captures, or a chart whose placement was not recorded. The
    caller then shows the screen it always showed.
    """
    if not charts:
        return None
    keys = {capture_key((record.get("image") or {}).get("fingerprint"))
            for record in charts}
    # One capture, or nothing to draw. A missing fingerprint is not a match:
    # older records carry none, and two of those are not thereby one frame.
    if len(keys) != 1 or not next(iter(keys)):
        return None
    image = charts[0].get("image") or {}
    width, height = int(image.get("width") or 0), int(image.get("height") or 0)
    placed: list[ChartInFrame] = []
    for index, record in enumerate(charts):
        corners = (record.get("geometry") or {}).get("corners_px") or []
        target = record.get("target") or {}
        placed.append(ChartInFrame(
            ordinal=index + 1,
            corners=[(float(x), float(y)) for x, y in corners],
            status=statuses[index],
            target_name=str(target.get("display_name")
                            or target.get("target_id") or "Unnamed target"),
            run_id=str(record.get("run_id") or ""),
            note=notes[index] if index < len(notes) else "",
        ))
    frame = FrameInfo(
        width=width, height=height,
        filename=str(image.get("filename") or ""), charts=placed,
    )
    return frame if frame.usable else None
