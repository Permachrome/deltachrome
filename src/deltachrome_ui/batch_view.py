"""A folder of captures, running and then run.

**One screen in two states rather than two screens**, because the thing a person
wants to look at is the same in both: the list of captures and what became of each.
While the run is going that list fills a row at a time; when it finishes it gains
the one column that could not be filled until every capture was in. Splitting them
would throw away the list at the moment it became complete and redraw it, and would
put a transition between the operator watching a frame fail and reading about it.

**Everything this screen says, it says in the table.** It used to say a good deal
around it -- a headline count, the last filename, a conformance caveat, and a block
of findings beside the list -- and each was either a restatement of a column or a
sentence the table could carry. What is left above the table is the two things it
cannot say: how far through the run is, and how a hundred rows are coming out so
far. Everything per-capture is a cell.

**The table does not move.** It used to be sized to its own contents after every
row, so a longer filename re-measured it and the centred layout re-centred under
the pointer. Its width is the body's width now and nothing in a row can change it:
the columns that vary are given the slack, and the table only ever gains rows.

**The count is the progress.** A determinate bar over a folder of a hundred JPEG
2000 masters is a lie told sixteen seconds at a time -- the masters are not the same
size, the charts are not equally easy to place, and the honest answer to "is this
working" is which capture it is on and how many are left. So the bar is
indeterminate and the number beside it is the reading.

**Stopping ends the run after the capture in flight**, never inside one. That keeps
every record that exists complete, which matters more here than stopping promptly:
a batch's whole output is a folder of records, and a truncated one would be a file
that looks like a result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtGui import QColor

from deltachrome_ui.labels import metric_title
from deltachrome_ui.run_table import wrapped_tooltip
from deltachrome_ui.status import Status, overall_status
from deltachrome_ui.theme import BAR_HEIGHT, TEXT_DIM, tone_colors
from deltachrome_ui.widgets import (
    BusyStrip, Callout, IconButton, elided_subtitle,
)

#: How many warnings one cell's tooltip prints before it stops and gives a count.
#: The same cap the library's own warnings column uses, for the same reason: a run
#: can raise one per patch and a tooltip is not a report.
MOST_WARNINGS = 8

#: Widest the body of this screen is allowed to get. It is the table's width too,
#: and the table's width is *not* its contents: measured across the twenty Odonata
#: captures, every row carries the same five warnings and the same empty notes, so
#: a width that followed the contents followed a filename -- which is the one thing
#: in a row that changes, and which re-centred the whole screen once per capture.
TABLE_WIDTH = 1100

#: Columns of the capture list. This is the run's own account of itself, and the
#: place to ask anything more of a capture is its record, which the library opens.
#:
#: **A finding is three columns rather than a sentence.** It used to read
#: ``Noise 4.8σ above``, and the σ was the wrong number to print: it is
#: ``|reading − batch median| / (MAD × 1.4826)``, so on twenty frames of one chart
#: that agree closely the ratio is enormous for a small difference -- 27.6 on a
#: gain modulation reading 1.143 against 0.786. Read as a normal distribution that
#: is a probability of one in 10^168, which is not what it means, so the figure
#: misleads the reader most likely to think they understand it. What a technician
#: can act on is the reading and what the rest of the batch read, in the unit they
#: already read on the report, and the direction falls out of the pair. The σ is
#: still computed, still ranks the findings, and is still in the record and the
#: console report.
COLUMNS = ("", "Capture", "Verdict", "Metric", "Value", "Median", "Warnings")
DOT, CAPTURE, VERDICT, METRIC, VALUE, MEDIAN, WARNINGS = range(len(COLUMNS))

#: What each column gets, in pixels; ``None`` is the one given the slack. Declared
#: rather than measured from the contents, which is the whole point: a column sized
#: to what is in it moves every time a longer filename arrives, and a boundary that
#: steps sideways under the pointer is the table moving whatever its outer width
#: does. 330 px holds the longest filename in the Odonata batch -- a longer one
#: elides, with the whole path on the pointer -- and 130 holds the longest verdict
#: this screen can print, which is ``not measured``.
WIDTHS = {
    DOT: 26, CAPTURE: 330, VERDICT: 130, METRIC: None,
    VALUE: 100, MEDIAN: 100, WARNINGS: 90,
}

#: How each column's cells and its heading are set. They take the same value, as
#: they do in the library's table: a right-aligned count under a centred heading
#: reads as two columns.
_RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
ALIGN = {
    DOT: Qt.AlignmentFlag.AlignCenter,
    VALUE: _RIGHT, MEDIAN: _RIGHT, WARNINGS: _RIGHT,
}
DEFAULT_ALIGN = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

#: Decimals a value and a median are printed to. Three, which is what the report's
#: own Values tables print, because a person reads one after the other and a figure
#: that changes precision between them reads as a different quantity.
PLACES = 3

#: What the three finding columns are, which never changes and is therefore on the
#: column headings rather than on the screen. The findings themselves change and
#: are in the rows.
METRIC_HINT = (
    "The reading that stands out most from the rest of the batch, even if it "
    "passed."
)


class ElideMiddle(QStyledItemDelegate):
    """A filename that will not fit elides in the **middle**.

    Qt's elide mode is a property of the whole view, and the right answer is not
    the same for every column here: a refusal's reason reads from the left and
    should be cut at the end, while a capture is named
    ``NMNH-USNMENT01845607.tif`` beside nineteen others identical for their
    first eighteen characters. Cut at the end, the column whose job is telling
    them apart prints the same text twenty times.
    """

    def initStyleOption(self, option, index) -> None:  # noqa: N802 - Qt's name
        super().initStyleOption(option, index)
        option.textElideMode = Qt.TextElideMode.ElideMiddle


class Tally(QWidget):
    """What the batch has produced so far, one entry per verdict.

    **It is the headline.** The screen used to open with a sentence -- "20 of 20
    captures measured, 6 failed" -- above a tally saying the same counts in more
    detail, and above a table stating each of them a row at a time. The tally is
    the one of the three that says something the table cannot: how a hundred rows
    have come out, which is what an operator watching a long batch is watching
    for, and which read off the list means counting dots.

    **Grouped by tone and label rather than by verdict**, through the same
    :func:`_status_for` the rows are drawn with, so the demotion rule reaches it:
    a batch of advisory passes reads ``PASS (ADVISORY)`` in the qualified tone
    and never puts a hundred captures behind a green number. It is a tally and
    not a verdict -- a batch states conformance per image, so there is nothing
    here to pool.

    **Laid out across rather than down**, because it shares a line with the
    progress count now instead of standing in a column beside the table. Three
    words and a number are a poor use of a column, and the column they were in
    was the awkward right-hand half of the screen.
    """

    def __init__(self) -> None:
        super().__init__()
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(18)
        self._cells: list[QWidget] = []
        self._rows: list[tuple[QLabel, QLabel, QLabel]] = []
        self.hide()

    def _entry(self, index: int) -> tuple[QLabel, QLabel, QLabel]:
        """The widgets for one verdict, made on first use and then reused.

        Rebuilding the whole strip per capture would be a few thousand labels
        over a folder of a thousand; there are never more than a handful of
        distinct verdicts, so the entries outlive the counts in them.
        """
        while len(self._rows) <= index:
            cell = QWidget()
            inner = QHBoxLayout(cell)
            inner.setContentsMargins(0, 0, 0, 0)
            inner.setSpacing(6)
            dot = QLabel("●")
            name = QLabel("")
            name.setStyleSheet("font-size: 12px;")
            count = QLabel("")
            count.setStyleSheet("font-size: 12px; font-weight: 600;")
            for widget in (dot, name, count):
                inner.addWidget(widget)
            self._row.addWidget(cell)
            self._cells.append(cell)
            self._rows.append((dot, name, count))
        return self._rows[index]

    def set_captures(self, captures: list[Any], *, running: bool) -> None:
        counted: dict[str, tuple[Status, int]] = {}
        for capture in captures:
            status = _status_for(capture)
            _, so_far = counted.get(status.label, (status, 0))
            counted[status.label] = (status, so_far + 1)
        # Worst first, which is the order the library's own status column sorts
        # in and for the same reason: what needs attention gathers at the front.
        ordered = sorted(counted.values(), key=lambda item: (item[0].rank, item[0].label))
        for index, (status, number) in enumerate(ordered):
            dot, name, count = self._entry(index)
            dot.setStyleSheet(f"color: {tone_colors(status.tone)[0]};")
            name.setText(status.label)
            name.setToolTip(status.hint)
            count.setText(str(number))
            self._cells[index].show()
        for index in range(len(ordered), len(self._rows)):
            # Blanked as well as hidden. A stale label left in a cell nobody can
            # see is a count that stopped being true, and the only thing standing
            # between it and the screen is one call to ``show``.
            for widget in self._rows[index]:
                widget.setText("")
            self._cells[index].hide()
        self.setVisible(bool(ordered))


class BatchView(QWidget):
    """Every capture in one folder, as it runs and once it has."""

    backRequested = Signal()
    stopRequested = Signal()
    #: The batch is finished and the operator wants to see its records.
    libraryRequested = Signal(object)  # BatchResult

    def __init__(self) -> None:
        super().__init__()
        self._result: Any = None
        self._total = 0
        #: Every capture the run has finished, for the tally. Statuses only --
        #: a :class:`~deltachrome.batch.BatchCapture` holds no run and no pixels,
        #: which is why a folder of a thousand can be kept here at all.
        self._captures: list[Any] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())

        # CENTRED, not left-packed. Nothing on this screen wants a
        # two-thousand-pixel measure: the capture list is a filename, a verdict
        # and a short note. Packed to the left it left two thirds of a wide
        # window empty on one side, which reads as a panel that failed to draw
        # rather than as a margin, so the space is given to both sides instead.
        #
        # The body takes as much of it as ``TABLE_WIDTH`` allows and no more, and
        # a high stretch is what makes that happen: a widget given no stretch
        # sits at its size hint, which here follows the contents -- the very
        # thing that used to move the screen under the pointer.
        body = QWidget()
        around = QHBoxLayout(body)
        around.setContentsMargins(20, 16, 20, 16)
        around.addStretch(1)
        column = QWidget()
        column.setMaximumWidth(TABLE_WIDTH)
        around.addWidget(column, 100)
        around.addStretch(1)
        inner = QVBoxLayout(column)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(12)

        # ONE ROW ABOVE THE TABLE, holding the three things a list of captures
        # cannot say about itself: that the run is still going, how far through
        # it is, and how the rows are coming out. The bar sits *in* that row
        # rather than centred above it, which is what keeps the table still --
        # the bar is the one thing on this screen that disappears, and a band of
        # its own would be either a jump when it went or a void once it had.
        # Its height is the row's, so the row is the same height throughout.
        status = QWidget()
        across = QHBoxLayout(status)
        across.setContentsMargins(0, 0, 0, 0)
        across.setSpacing(14)
        self.strip = BusyStrip(width=160)
        across.addWidget(self.strip)
        self.count = QLabel("")
        self.count.setStyleSheet(f"font-size: 12px; color: {TEXT_DIM};")
        across.addWidget(self.count)
        across.addStretch(1)
        self.tally = Tally()
        across.addWidget(self.tally)
        status.setFixedHeight(max(24, self.strip.sizeHint().height()))
        inner.addWidget(status)

        #: The one thing above the table that the table cannot say. Everything
        #: else on this screen was taken off for restating a column -- but this is
        #: a fact about what was in the *frames*, not about any capture's verdict,
        #: and no row carries it: a batch measures one chart per capture, so a
        #: board's other three cards appear nowhere at all. A rule and text in the
        #: qualified tone, like every other statement of fact here, rather than a
        #: filled box, which is the shape of a button.
        self.notice = Callout()
        self.notice.hide()
        inner.addWidget(self.notice)

        self.table = QTableWidget(0, len(COLUMNS))
        # THE APPLICATION'S OWN TABLE STYLE, which this one had never worn: the
        # style sheet names ``QTableView#DataTable`` and nothing here set the
        # name, so the cells had no padding and sat against the frame, the rows
        # did not band, and a selected row came up in Qt's own bright blue
        # rather than in the theme's. Reported as the warnings column touching
        # the edge, which was the visible corner of it.
        self.table.setObjectName("DataTable")
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setShowGrid(False)
        self._elide_middle = ElideMiddle(self.table)
        self.table.setItemDelegateForColumn(CAPTURE, self._elide_middle)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        for column, width in WIDTHS.items():
            item = self.table.horizontalHeaderItem(column)
            if item is not None:
                item.setTextAlignment(ALIGN.get(column, DEFAULT_ALIGN))
            if width is None:
                # The one column given the slack, and the one whose contents are
                # worth the room: a refusal's reason, and -- once the batch is
                # complete enough to compare its captures against each other --
                # what makes one unlike the rest.
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
                continue
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(column, width)
        for column in (METRIC, VALUE, MEDIAN):
            self._set_header_hint(column, METRIC_HINT)
        self._set_header_hint(
            WARNINGS,
            "Hover over a cell to read warnings.",
        )
        inner.addWidget(self.table, 1)

        layout.addWidget(body, 1)

    def _set_header_hint(self, column: int, text: str) -> None:
        item = self.table.horizontalHeaderItem(column)
        if item is not None:
            item.setToolTip(wrapped_tooltip([text]))

    # -- the bar ---------------------------------------------------------------

    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("HeaderBar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setFixedHeight(BAR_HEIGHT)
        header = QHBoxLayout(bar)
        header.setContentsMargins(20, 0, 0, 0)
        header.setSpacing(12)

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(0)
        self.title = QLabel("Batch Analysis")
        self.title.setStyleSheet("font-size: 14px; font-weight: 600;")
        text.addWidget(self.title)
        self.folder = elided_subtitle("")
        text.addWidget(self.folder)
        header.addLayout(text, 1)
        header.addSpacing(12)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(0)
        self.back_button = IconButton("back", "Back to the library", size=BAR_HEIGHT)
        self.back_button.clicked.connect(self.backRequested.emit)
        actions.addWidget(self.back_button)

        # One button in two roles, because they are the same place in the bar and
        # the same question -- what do you want to do about this run. Stopping it
        # and leaving it are never both available.
        self.primary = QPushButton("Stop")
        self.primary.setObjectName("PrimaryBar")
        self.primary.setFixedHeight(BAR_HEIGHT)
        self.primary.setCursor(Qt.CursorShape.PointingHandCursor)
        self.primary.clicked.connect(self._on_primary)
        actions.addWidget(self.primary)
        header.addLayout(actions)
        return bar

    def _on_primary(self) -> None:
        if self._result is None:
            self.primary.setEnabled(False)
            self.primary.setText("Stopping…")
            self.stopRequested.emit()
        else:
            self.libraryRequested.emit(self._result)

    # -- running ---------------------------------------------------------------

    def start(self, folder: Path, total: int, target_name: str, profile_name: str,
              *, local_criteria: bool = False) -> None:
        """Reset for a run about to begin.

        ``local_criteria`` is named **in the header**, beside the guideline it is
        true of, rather than as a sentence over the table. It is a fact about the
        guideline and not about the batch -- and it is the one qualification that
        nothing else on this screen would carry, because a local guideline
        deliberately does not demote a verdict's tone: every row and the tally
        would read a plain ``PASS`` and no reader would learn what it passed.
        """
        self._result = None
        self._total = total
        self._captures = []
        self.tally.set_captures([], running=True)
        self.folder.setText(str(folder))
        guideline = profile_name + (" (local guideline)" if local_criteria else "")
        self.title.setText(f"{target_name}   ·   {guideline}")
        self.title.setToolTip(
            "Judged against a local guideline, not a published standard."
            if local_criteria else ""
        )
        self.notice.hide()
        self.table.clearSpans()
        self.table.setRowCount(0)
        self.strip.show()
        self.back_button.setEnabled(False)
        self.primary.setEnabled(True)
        self.primary.setText("Stop")
        self.primary.setToolTip("")
        self._say(f"0 of {total} captures")

    def advance(self, done: int, total: int, name: str, capture: Any) -> None:
        """One capture finished. The count is the progress; see the module docstring."""
        self._say(f"{done} of {total} captures")
        if capture is not None:
            self._captures.append(capture)
            self.tally.set_captures(self._captures, running=True)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._fill_row(row, capture, None)
            self.table.scrollToBottom()

    def _say(self, text: str, tone: str | None = None) -> None:
        """The left of the status line: progress, or the one thing that outlives it."""
        color = tone_colors(tone)[0] if tone else TEXT_DIM
        self.count.setStyleSheet(f"font-size: 12px; color: {color};")
        self.count.setText(text)

    def _fill_row(self, row: int, capture: Any, unusual: list[Any] | None) -> None:
        """One capture's row, built the same way whenever it is built.

        ``unusual`` is what the batch found out about this capture by comparing it
        against the others, which cannot be known until the others exist -- so a
        row is written once as it arrives and once more when the run ends. The
        row's *position* never changes, which is the property that matters.
        """
        status = _status_for(capture)
        self._cell(row, DOT, "●", wrapped_tooltip([status.label, "", status.hint])
                   ).setForeground(QColor(tone_colors(status.tone)[0]))
        self._cell(row, CAPTURE, capture.name, str(capture.path))
        self._cell(
            row, VERDICT,
            capture.overall or ("not measured" if not capture.measured else ""),
        )
        self._finding(row, capture, unusual)
        # Blank rather than zero. A ``0`` in every clean row is a column of noise
        # the eye has to read past to find the rows that have something in them.
        self._cell(
            row, WARNINGS,
            str(len(capture.warnings)) if capture.warnings else "",
            _warning_tooltip(capture),
        )

    def _finding(self, row: int, capture: Any, unusual: list[Any] | None) -> None:
        """The three finding columns, or the one thing that is not a finding.

        **The worst finding only**, where a capture is unusual on more than one
        metric. Three columns hold one metric, one value and one median, and the
        grouping argument says this is the right thing to lose: one frame soft on
        two sharpness metrics is one soft frame corroborated twice rather than two
        problems, and the row is a pointer to a frame rather than an inventory of
        it. Every finding is in the record.

        **A refusal is not a measurement**, so it has no metric, no value and no
        median -- it takes the room they would have had, by spanning them. That is
        why there is no separate notes column: a column that is empty on every
        measured row and holds a sentence on the one that failed is a column sized
        for its rarest case, and it starved the three that carry the findings.
        """
        self.table.setSpan(row, METRIC, 1, 1)
        if unusual:
            worst = unusual[0]
            self._cell(row, METRIC, metric_title(worst.metric_id))
            self._cell(row, VALUE, f"{worst.value:.{PLACES}f}")
            self._cell(row, MEDIAN, f"{worst.median:.{PLACES}f}")
            return
        for column in (METRIC, VALUE, MEDIAN):
            self._cell(row, column, "")
        if capture.reason:
            self._cell(row, METRIC, capture.reason)
            self.table.setSpan(row, METRIC, 1, 3)

    def _cell(self, row: int, column: int, text: str,
              tooltip: str = "") -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setToolTip(tooltip)
        item.setTextAlignment(ALIGN.get(column, DEFAULT_ALIGN))
        self.table.setItem(row, column, item)
        return item

    # -- finished --------------------------------------------------------------

    def show_result(self, result: Any) -> None:
        """The run is over. The list stays where it is and gains its last column.

        Nothing is said above the table that the table now says. The count goes
        quiet -- the tally beside it holds every number the old summary line
        carried, and holds them per verdict -- except where the run was stopped,
        which is the one fact no row and no tally can state.
        """
        self._result = result
        notes = list(getattr(result, "notes", ()) or ())
        self.notice.set_state("\n\n".join(notes) if notes else "")
        self.strip.hide()
        self.back_button.setEnabled(True)
        self.primary.setEnabled(True)
        self.primary.setText("Open in Library")
        self.primary.setToolTip(
            f"Saved to:\n{result.directory}"
        )
        # From the result rather than from what was watched, so a batch opened
        # having been resumed tallies every capture and not only this session's.
        self._captures = list(result.captures)
        self.tally.set_captures(self._captures, running=False)
        self._say(
            f"Stopped after {len(self._captures)} of {self._total} captures."
            if getattr(result, "stopped_early", False) else "",
            tone="qualified" if getattr(result, "stopped_early", False) else None,
        )

        # The one thing a batch says that its captures do not, and said BY
        # CAPTURE, in the capture's own row: the question this answers is which
        # frames to go and look at, and one frame soft on two sharpness metrics
        # is one soft frame corroborated twice rather than two problems.
        unusual = dict(result.unusual_captures())
        if self.table.rowCount() != len(self._captures):
            self.table.clearSpans()
            self.table.setRowCount(0)
            for capture in self._captures:
                self.table.insertRow(self.table.rowCount())
        for row, capture in enumerate(self._captures):
            self._fill_row(row, capture, unusual.get(capture.name))


def _warning_tooltip(capture: Any) -> str:
    """The warnings themselves, which is the point of keeping the column.

    A count on a screen that cannot show what was counted is a dead end. Capped
    because a run can raise one per patch and a tooltip is not a report; the
    record holds all of them and the library opens it.
    """
    if not capture.warnings:
        return ""
    shown = list(capture.warnings[:MOST_WARNINGS])
    if len(capture.warnings) > MOST_WARNINGS:
        shown.append(
            f"…and {len(capture.warnings) - MOST_WARNINGS} more."
        )
    return wrapped_tooltip(shown, bullet=True)


def _status_for(capture: Any) -> Status:
    """The tone a capture's row is drawn in.

    A capture that could not be measured is *unknown* rather than failed: nothing
    judged it, and drawing it in the failure tone would say the system missed a
    requirement when what happened is that a frame could not be read.
    """
    if not capture.measured:
        return Status(
            label="not measured",
            tone="unknown",
            hint=capture.reason or "This capture could not be measured.",
        )
    return overall_status(
        capture.overall or "", conformance_claim=capture.conformance_claim,
        evidence_sound=capture.reference_basis == "per_unit_measurement",
    )
