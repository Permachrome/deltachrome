"""What a run contains, as a flat list.

This was a tree: family nodes that expanded to metrics, metrics that expanded to
the representations each one had. Every level was real -- the grouping is read
from each metric's declared capability, the view leaves came from the same
function that built the pages -- and it was still the wrong control.

The reason is worth stating, because "it was accurate" was the argument for it.
A tree asks the reader to *operate* it before it will answer them: two
disclosure triangles between arriving and seeing the fifteen things a run
measured. And the level it disclosed last -- Chart, Field, Readings -- was not
navigation at all. Those are representations of one metric, they appear as tabs
the moment the metric is on screen, and putting them in the sidebar meant the
sidebar's deepest level duplicated a control already visible beside it.

So: one flat list. Overview first, then every metric the run judged, then the
pages that belong to the run rather than to a metric. The families survive as
**headings** -- they still say what is being measured and they still come from
the capability rather than from a list kept here -- but a heading is a label and
not a control. Nothing to expand, nothing that can be collapsed into hiding the
thing you came for, and every metric one click from every other.

The reading each metric got stays on its row. That is what makes the list a
summary as well as a route: fifteen rows, fifteen verdicts, no scrolling and no
opening.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from deltachrome_ui.labels import metric_title, unit_label
from deltachrome_ui.status import metric_status
from deltachrome_ui.theme import BORDER, TEXT_DIM, tone_colors

#: What a row is. The tuple in each item's data is ``(kind, *identifiers)``.
OVERVIEW = "overview"
METRIC = "metric"
PAGE = "page"
#: Kept so an older caller naming a family or a view does not fail silently --
#: both now resolve to the metric or page that replaced them.
FAMILY = "family"
VIEW = "view"

#: Width of the list. Narrower than the tree it replaces: nothing is indented
#: any more, so the width only has to hold a metric name and its verdict.
NAV_WIDTH = 250

#: Marks the rows that are labels rather than destinations.
HEADING_ROLE = Qt.ItemDataRole.UserRole + 8


class ReportNav(QListWidget):
    """The run's metrics, and where in them the reader is."""

    #: ``(kind, *identifiers)`` -- a row was chosen.
    navigate = Signal(tuple)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Nav")
        self.setFixedWidth(NAV_WIDTH)
        self.setFrameShape(QListWidget.Shape.NoFrame)
        self.setUniformItemSizes(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.currentItemChanged.connect(self._on_current)
        #: Set while the list is being driven from outside, so following the
        #: page does not immediately drive the page back.
        self._echo = False

    # -- construction ------------------------------------------------------

    def build(self, run, families, extras: list[tuple[str, str, tuple]]) -> None:
        """``extras`` are the pages that belong to no metric, in order."""
        self.clear()
        self._row("Overview", "", (PAGE, OVERVIEW))
        for family in families:
            self._heading(family.name)
            for metric in family.metrics:
                metric_id = metric.get("metric_id", "")
                status = metric_status(metric.get("status", ""))
                value = metric.get("value")
                reading = (
                    "—" if value is None
                    else f"{value:.3f} {unit_label(metric.get('units'))}".strip()
                )
                name = metric_title(metric_id)
                item = self._row(
                    name, status.label,
                    (METRIC, family.key, metric_id), tone=status.tone,
                )
                item.setData(Qt.ItemDataRole.UserRole + 3, name)
        if extras:
            self._heading("This Run")
            for name, note, target in extras:
                self._row(name, note, target)

    def _heading(self, text: str) -> None:
        """A label, not a control.

        Selectable headings were the whole problem: a reader clicks the thing
        that names what they want, and a heading that answers is one more place
        to be, showing a summary of the rows already visible underneath it.
        """
        item = QListWidgetItem(text.upper())
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        item.setForeground(Qt.GlobalColor.gray)
        font = item.font()
        font.setPointSizeF(max(8.5, font.pointSizeF() - 2.0))
        font.setBold(True)
        item.setFont(font)
        item.setData(HEADING_ROLE, True)
        # Taller than a row, and paired with the rule drawn under it: a heading
        # set only in small caps sat at the same weight as the metric names
        # beneath it and the eye read the list as one undifferentiated column.
        item.setSizeHint(QSize(0, 34))
        self.addItem(item)

    #: Room kept at the right of a row for the verdict painted there. The name
    #: is elided to fit it: a long metric name was running underneath its own
    #: status and the two overprinted.
    #: How much smaller the verdict is set than the name beside it.
    NOTE_SHRINK = 1.5

    def _row(
        self, name: str, note: str, target: tuple, tone: str | None = None
    ) -> QListWidgetItem:
        from PySide6.QtGui import QFontMetrics

        budget = NAV_WIDTH - 30 - (self._note_width(note) if note else 10)
        shown = QFontMetrics(self.font()).elidedText(
            name, Qt.TextElideMode.ElideRight, budget
        )
        item = QListWidgetItem("   " + shown)
        if shown != name:
            # A long metric name against a long verdict elides, and where this
            # application elides it puts the whole of it on hover.
            item.setToolTip(name)
        item.setData(Qt.ItemDataRole.UserRole, tuple(target))
        item.setData(Qt.ItemDataRole.UserRole + 1, note)
        item.setData(Qt.ItemDataRole.UserRole + 2, tone or "")
        item.setSizeHint(QSize(0, 26))
        self.addItem(item)
        return item

    def _note_font(self):
        """The face the verdict is painted in. One definition, two callers.

        The elision budget and the painter used to work from different numbers --
        a fixed 62 px against the note's real advance at a smaller size -- so any
        verdict wider than the constant was drawn straight over the name it
        belonged to. ``INCOMPLETE`` is 70 px, and read as
        ``Gain Modulation HighlightINCOMPLETE``.
        """
        from PySide6.QtGui import QFont

        font = QFont(self.font())
        font.setPointSizeF(max(8.5, self.font().pointSizeF() - self.NOTE_SHRINK))
        return font

    def _note_width(self, note: str) -> int:
        from PySide6.QtGui import QFontMetrics

        return QFontMetrics(self._note_font()).horizontalAdvance(note) + 24

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:
        """The verdict is painted at the right edge of its own row.

        A second column would need a tree or a table; the note is one short word
        and the row already knows where its right edge is.
        """
        super().paintEvent(event)
        from PySide6.QtGui import QColor, QPainter, QPen

        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # A rule under each heading. Shading the metric rows instead was tried
        # and read as a table with no columns; a hairline says "these belong to
        # that" with nothing else on the screen having to change.
        for index in range(self.count()):
            item = self.item(index)
            if item.data(HEADING_ROLE):
                rect = self.visualItemRect(item)
                painter.setPen(QPen(QColor(BORDER), 1))
                painter.drawLine(
                    rect.left() + 12, rect.bottom() - 2,
                    rect.right() - 12, rect.bottom() - 2,
                )
        for index in range(self.count()):
            item = self.item(index)
            note = item.data(Qt.ItemDataRole.UserRole + 1)
            if not note:
                continue
            rect = self.visualItemRect(item)
            if not rect.intersects(self.viewport().rect()):
                continue
            tone = item.data(Qt.ItemDataRole.UserRole + 2)
            painter.setPen(QColor(tone_colors(tone)[0] if tone else TEXT_DIM))
            painter.setFont(self._note_font())
            width = self._note_width(note)
            painter.drawText(
                rect.adjusted(rect.width() - width, 0, -10, 0),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                note,
            )
        painter.end()

    # -- selection ---------------------------------------------------------

    def _on_current(self, item: QListWidgetItem | None, _previous) -> None:
        if item is None or self._echo:
            return
        target = item.data(Qt.ItemDataRole.UserRole)
        if target:
            self.navigate.emit(tuple(target))

    def show_target(self, target: tuple) -> None:
        """Move the cursor to a row without emitting -- the page moved first."""
        item = self._find(target)
        if item is None:
            return
        self._echo = True
        try:
            self.setCurrentItem(item)
            self.scrollToItem(item)
        finally:
            self._echo = False

    def _find(self, target: tuple) -> QListWidgetItem | None:
        target = tuple(target)
        for index in range(self.count()):
            item = self.item(index)
            data = item.data(Qt.ItemDataRole.UserRole)
            if data and tuple(data) == target:
                return item
        return None

    def targets(self) -> list[tuple[Any, ...]]:
        """Every row's target, in order. For tests and for keyboard walking."""
        found: list[tuple[Any, ...]] = []
        for index in range(self.count()):
            data = self.item(index).data(Qt.ItemDataRole.UserRole)
            if data:
                found.append(tuple(data))
        return found
