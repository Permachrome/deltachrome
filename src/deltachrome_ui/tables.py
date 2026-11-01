"""One table, used everywhere something is a list.

The run library got a table because a year of records is not read, it is
queried. The same is true one level down and for the same reason: a hundred and
forty patch readings, twenty edges, fifteen metrics. Those were a strip of tiles,
a grid of boxes and a hand-laid monospace block respectively -- three ways of
showing a list, none of which could be sorted, none of which lined up with the
others, and all of which had to be re-read as a new layout each time.

So the parts of :mod:`deltachrome_ui.run_table` that were not about runs live
here: a column declared once as what it shows, what it sorts on and how it reads,
a model that asks the column, and a view that looks like the library's.

The library's table keeps its own module because its columns carry things no
other table has -- painted row actions, a trash gesture, a search across every
column at once. What it shares is this -- including, now, the sorting.

**These tables sort.** They did not, and the argument for that was their length:
a profile's fifteen metrics and one chart's readings, in an order the profile or
the chart chose. It does not survive the readings. A hundred and forty patches
is a list nobody reads either, and the questions asked of it -- which patch is
worst, are the flagged ones the failing ones, is one column of the chart adrift
-- are each one click on a heading. So the header is a control again, and this
time it does what it draws: a proxy that orders on :attr:`Col.sort`, which every
column already declared and nothing used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QRect,
    QRectF,
    QSortFilterProxyModel,
    Qt,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QStyledItemDelegate,
    QTableView,
)

from deltachrome_ui.theme import ACCENT, BG_RAISED, TEXT_DIM, tone_colors

#: The row object behind a cell, so a delegate does not reconstruct one.
ROW_ROLE = Qt.ItemDataRole.UserRole + 1
#: What a column sorts on, which is rarely what it displays.
SORT_ROLE = Qt.ItemDataRole.UserRole + 2
#: A semantic tone name, for the column that paints a status dot.
TONE_ROLE = Qt.ItemDataRole.UserRole + 3

#: Everything numeric is set in the mono face. A column of proportional digits
#: does not line up at the decimal point, and comparing two readings down a
#: column is the entire reason they are in one.
MONO = "Menlo, Monaco, Consolas, monospace"

LEFT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
CENTER = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter


@dataclass(frozen=True)
class Col:
    """One column: what it shows, what it sorts on, how it reads."""

    title: str
    text: Callable[[Any], str]
    #: Defaults to the display text, which is right for names and wrong for
    #: everything numeric -- ``10`` sorts before ``9`` as a string.
    sort: Callable[[Any], Any] | None = None
    tooltip: Callable[[Any], str] | None = None
    #: A semantic tone name. When given, the column paints a dot instead of text.
    tone: Callable[[Any], str] | None = None
    width: int = 120
    #: ``fixed`` for painted columns, ``stretch`` for the one that absorbs the
    #: window, ``interactive`` for anything a person may want to widen.
    resize: str = "interactive"
    #: Never narrower than its own widest cell. :attr:`width` is then a floor
    #: rather than the answer, which is what a column of names needs: the
    #: longest metric name in a FADGI profile is ``Color Accuracy 90th
    #: Percentile``, and a width guessed in pixels against a font the machine
    #: chooses is a guess that elides exactly the row a reader was looking for.
    fit: bool = False
    align: Qt.AlignmentFlag = LEFT
    mono: bool = False
    #: Secondary text, for a column that is present but not the point.
    dim: bool = False

    def sort_key(self, row: Any) -> Any:
        return self.sort(row) if self.sort else self.text(row)


class DataModel(QAbstractTableModel):
    """Rows of anything, read through :class:`Col`."""

    def __init__(self, columns: Sequence[Col], rows: Sequence[Any] = ()) -> None:
        super().__init__()
        self.columns = list(columns)
        self._rows = list(rows)

    def set_rows(self, rows: Sequence[Any]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.columns)

    def row_at(self, index: QModelIndex) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        return self._rows[index.row()]

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        row = self.row_at(index)
        if row is None:
            return None
        column = self.columns[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return "" if column.tone else column.text(row)
        if role == Qt.ItemDataRole.ToolTipRole:
            if column.tooltip:
                return column.tooltip(row) or None
            return None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(column.align)
        if role == Qt.ItemDataRole.FontRole and column.mono:
            font = QFont(MONO)
            font.setPointSizeF(11.5)
            return font
        if role == Qt.ItemDataRole.ForegroundRole and column.dim:
            return QColor(TEXT_DIM)
        if role == TONE_ROLE and column.tone:
            return column.tone(row)
        if role == SORT_ROLE:
            return column.sort_key(row)
        if role == ROW_ROLE:
            return row
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.columns[section].title
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(self.columns[section].align)
        return None

    def flags(self, index: QModelIndex):
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable


class DotDelegate(QStyledItemDelegate):
    """A status, as a filled circle in its tone.

    Five tones and not two, for the reason :mod:`deltachrome_ui.status` states:
    a run or a metric that measured something real without earning a pass has to
    have somewhere to go that is not green.
    """

    RADIUS = 4.5

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        super().paint(painter, option, index)
        tone = index.data(TONE_ROLE)
        if not tone:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(tone_colors(str(tone))[0]))
        center = option.rect.center()
        painter.drawEllipse(QRectF(
            center.x() - self.RADIUS + 0.5, center.y() - self.RADIUS + 0.5,
            self.RADIUS * 2, self.RADIUS * 2,
        ))
        painter.restore()


class SortProxy(QSortFilterProxyModel):
    """Ordering, on the key the column declared rather than on its text.

    The distinction is the whole point of :attr:`Col.sort`: a reading shown as
    ``10.500 ΔE*2000`` sorts before ``9.100`` as a string, and a table that
    sorts by accident is one that will silently stop.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSortRole(SORT_ROLE)
        # Rows carrying no key at all -- a patch with no role, an edge with no
        # warnings -- must not jump about when two of them compare equal.
        self.setDynamicSortFilter(False)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        """Compare keys of mixed types without raising.

        A column can legitimately return ``None`` for a row that has nothing to
        sort on -- an edge with no channel data, a patch the metric skipped --
        and ``None < 3.5`` is a TypeError that takes the window with it.

        Missing sorts **last whichever way the header points**, which needs the
        order asked for: "no reading" is not a small reading, and a readings
        table opens worst-first, so treating missing as smallest would put the
        rows that measured nothing at the top of the one view that exists to
        show the worst.
        """
        a = left.data(SORT_ROLE)
        b = right.data(SORT_ROLE)
        if a is None or b is None:
            if a is None and b is None:
                return False
            descending = self.sortOrder() == Qt.SortOrder.DescendingOrder
            return (a is None) if descending else (b is None)
        try:
            return bool(a < b)
        except TypeError:
            return str(a) < str(b)


class SortHeader(QHeaderView):
    """A header that paints its own sort indicator.

    The same trap the combo box chevron fell into: once a style sheet claims
    ``QHeaderView::section``, Qt paints a sort arrow only if it was handed one as
    an image file, so the styled header silently loses the one mark that says
    which column the table is ordered by. Painting it keeps the application a
    single installable package with no asset directory beside it.
    """

    #: How much of a section the indicator takes, at whichever end it is drawn.
    INDICATOR = 19

    def __init__(self) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.setSectionsClickable(True)

    def _indicator_rect(self, rect: QRect, index: int) -> QRect:
        """Which end of the section the chevron goes at.

        The end the heading is *not* against. A right-aligned column -- every
        column of numbers is one -- puts its title hard against the right edge,
        which is where the chevron was drawn: sorting on ``SFR50`` cut it to
        ``SFR5`` and on ``L*`` to ``L``, so the one heading a reader is looking
        at is the one that becomes unreadable. Widening the column cannot fix
        that, because the collision is with the text's own alignment.
        """
        model = self.model()
        alignment = 0
        if model is not None:
            value = model.headerData(
                index, Qt.Orientation.Horizontal,
                Qt.ItemDataRole.TextAlignmentRole,
            )
            alignment = int(value) if value is not None else 0
        right_aligned = bool(alignment & int(Qt.AlignmentFlag.AlignRight))
        left = rect.left() + 1 if right_aligned else rect.right() - self.INDICATOR
        return QRect(left, rect.top() + 1, self.INDICATOR, rect.height() - 2)

    def paintSection(self, painter: QPainter, rect: QRect, index: int) -> None:
        super().paintSection(painter, rect, index)
        if not self.isSortIndicatorShown() or self.sortIndicatorSection() != index:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Clear the corner first. The label is drawn by the base call across the
        # whole section, so on a narrow column the chevron would otherwise land
        # on top of its own header's letters.
        box = self._indicator_rect(rect, index)
        painter.fillRect(box, QColor(BG_RAISED))
        pen = QPen(QColor(ACCENT))
        pen.setWidthF(1.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        down = self.sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
        size, right = 7.0, box.center().x() + 3.5
        mid = rect.center().y() + (1 if down else -1)
        path = QPainterPath()
        path.moveTo(right - size, mid - (size / 2.4 if down else -size / 2.4))
        path.lineTo(right - size / 2, mid + (size / 2.4 if down else -size / 2.4))
        path.lineTo(right, mid - (size / 2.4 if down else -size / 2.4))
        painter.drawPath(path)
        painter.restore()

class DataTable(QTableView):
    """The library's table, for anything that is a list.

    **The header sorts**, on :attr:`Col.sort` through :class:`SortProxy`, and
    paints its own indicator through :class:`SortHeader` for the reason that
    class gives. Every column declared a sort key already; what was missing was
    a proxy to ask.

    :meth:`order_by` is still the table saying what order its contents belong in
    -- worst reading first -- and it is now the table's *initial* order rather
    than its only one.
    """

    #: How wide a fitted column is allowed to get before it elides instead.
    FIT_MAXIMUM = 340

    #: Room a heading needs beyond its own letters: the style sheet's 8 px of
    #: padding either side, the 1 px separator, and the 19 px the sort chevron
    #: is painted into. Without it a narrow column's title is cut off by its own
    #: indicator the moment it is sorted -- ``SFR50`` became ``SFR5``, ``L*``
    #: became ``L`` -- which is the one column whose name a reader is looking
    #: at when it happens.
    HEADING_MARGIN = 17 + 1 + 19

    _MODES = {
        "fixed": QHeaderView.ResizeMode.Fixed,
        "stretch": QHeaderView.ResizeMode.Stretch,
        "interactive": QHeaderView.ResizeMode.Interactive,
    }

    def __init__(self, columns: Sequence[Col], rows: Sequence[Any] = ()) -> None:
        super().__init__()
        self.setObjectName("DataTable")
        self.setShowGrid(True)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        # Horizontally too, which was the one that showed. Per *item* means a
        # wheel tick jumps a whole column, and the widest table here -- twenty
        # slanted edges in three colour records -- has columns wide enough that
        # one tick moves the whole view, so there is no position between two
        # columns to stop at.
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(28)

        header = SortHeader()
        header.setSectionsMovable(True)
        header.setHighlightSections(False)
        header.setDefaultAlignment(LEFT)
        header.setStretchLastSection(False)
        self.setHorizontalHeader(header)

        self.set_columns(columns, rows)
        # Enabled *after* the model, and with no column claimed. Qt sorts on the
        # current indicator the moment sorting is turned on, and the default
        # indicator is section 0 -- which on the overview is the verdict dot, so
        # every report would have opened with its metrics reordered by status
        # rather than in the order the guidelines declare them.
        self.setSortingEnabled(True)
        header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)

    # -- the model, behind a proxy that sorts ------------------------------

    def set_columns(self, columns: Sequence[Col], rows: Sequence[Any] = ()) -> None:
        """Attach a fresh model, wrapped in the sorting proxy."""
        source = DataModel(columns, rows)
        proxy = SortProxy(self)
        proxy.setSourceModel(source)
        self.setModel(proxy)

    @property
    def source(self) -> DataModel:
        """The model holding the rows, whichever side of the proxy is asked."""
        model = self.model()
        return model.sourceModel() if isinstance(model, SortProxy) else model

    def setModel(self, model) -> None:
        """Attach, then lay out.

        A header has no sections until it has a model, so widths applied in a
        constructor land on columns that do not exist yet and are discarded --
        every column then arrives at the same default width.
        """
        super().setModel(model)
        columns = getattr(model, "columns", None)
        if columns is None:
            source = getattr(model, "sourceModel", lambda: None)()
            columns = getattr(source, "columns", None)
        if not columns:
            return
        header = self.horizontalHeader()
        # A *last* column asking to stretch means "take up the slack", not "never
        # let this table scroll" -- and Qt's Stretch mode means the second. It
        # redistributes the viewport among the stretch sections whatever the
        # other columns need, so once the fixed columns are wider than the pane
        # the stretch column is squeezed towards its minimum and no horizontal
        # scrollbar ever appears. That is how the edge table's Warnings column
        # became a few unreadable pixels the moment a column was added to its
        # left. ``stretchLastSection`` does what was meant: it fills leftover
        # space when there is any and leaves the section alone when there is
        # not, so the table scrolls. A stretch column that is *not* last is a
        # different request and keeps Stretch mode.
        last = len(columns) - 1
        fills = bool(columns) and columns[last].resize == "stretch"
        header.setStretchLastSection(fills)
        for i, column in enumerate(columns):
            stretching = column.resize == "stretch" and not (fills and i == last)
            header.setSectionResizeMode(
                i, self._MODES["interactive" if fills and i == last
                               else column.resize]
            )
            if not stretching:
                header.resizeSection(i, self._width(column, i))
            if column.tone:
                self.setItemDelegateForColumn(i, DotDelegate(self))

    def _width(self, column: Col, index: int) -> int:
        """The declared width, widened to whatever must fit inside it.

        Two things must: the column's own heading, always, because a title
        clipped by the chevron beside it is unreadable exactly when the reader
        is looking at it; and the cells, where the column asks with
        :attr:`Col.fit`. Measured through the view, so it accounts for the font
        the platform actually chose and for the padding the style sheet sets.

        Capped, because one long ICC profile description should not push the
        verdict off the right-hand edge -- past the cap the cell elides and its
        tooltip carries the rest, which is what a tooltip is for. A painted
        column has no heading of its own and keeps its declared width.
        """
        if column.tone:
            return column.width
        wanted = column.width
        heading = self.fontMetrics().horizontalAdvance(
            column.title.upper()
        ) + self.HEADING_MARGIN
        wanted = max(wanted, heading)
        if column.fit:
            wanted = max(wanted, self.sizeHintForColumn(index))
        return min(self.FIT_MAXIMUM, wanted)

    def set_rows(self, rows: Sequence[Any]) -> None:
        self.source.set_rows(rows)

    def order_by(self, column: int, *, descending: bool = False) -> "DataTable":
        """The order the table opens in, which a reader may then change.

        For a list of readings that is worst first: a failing aggregate is a
        question about the tail. The header's indicator turns with it, so the
        order on screen is always one the table has said out loud.
        """
        self.sortByColumn(
            column,
            Qt.SortOrder.DescendingOrder if descending
            else Qt.SortOrder.AscendingOrder,
        )
        return self

    def sized_to_contents(self, *, maximum: int = 420) -> "DataTable":
        """Height enough for the rows, up to a cap.

        A table of four metrics inside a scrolling page should be four rows tall
        and not a pane with an acre of empty grid under it. Past the cap it
        scrolls on its own, which is the right behaviour for a hundred and forty
        patches.
        """
        rows = self.model().rowCount()
        height = self.horizontalHeader().height() + 2
        height += rows * self.verticalHeader().defaultSectionSize()
        self.setMinimumHeight(min(maximum, max(90, height + 2)))
        self.setMaximumHeight(min(maximum, max(90, height + 2)))
        return self
