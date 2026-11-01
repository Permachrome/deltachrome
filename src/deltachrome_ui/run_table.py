"""The run library as a table.

The library was a list of cards, one composed widget per run, and that shape
carried an assumption worth naming: that a person reads their runs. They do for
the first fortnight. A department shooting a few targets a day accumulates on the
order of a thousand records a year, and at that size nobody reads the library --
they *query* it. "Every Golden Thread run this month." "Everything that failed."
"Everything judged on nominal values." A card cannot be queried; a column can.

So every visible quantity is declared once, in :data:`COLUMNS`, as the four
things a column has to be able to do: show itself, sort itself, be searched, and
explain itself. Sorting and filtering then fall out of the declaration rather
than being written per column, which is what stops the date column from being
sorted on the string a human reads -- ``2026-08-22 14:32`` happens to sort
correctly and ``22 Aug 2026`` does not, and a table that sorts by accident is one
that will silently stop.

Two things are painted rather than lettered, and both for the same reason: at a
thousand rows the eye scans and does not read.

**The status dot** takes its color from :func:`deltachrome_ui.theme.tone_colors`,
which means it has five states and not two. A red/green dot would put an
advisory pass in the pass color, and refusing to do that is the one rule
:mod:`deltachrome_ui.status` exists to state. Amber is not decoration here: on
the ISA targets an advisory pass is the ordinary result.

**The row actions** are painted by a delegate on the hovered or selected row
rather than built as buttons. Four ``QPushButton`` per row is four thousand
widgets for a year of work, constructed afresh on every refresh, and it was
already the reason the card list slowed down. A delegate paints the row under
the mouse and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PySide6.QtCore import (
    QAbstractItemModel,
    QEvent,
    QModelIndex,
    QObject,
    QPoint,
    QRect,
    QRectF,
    QSortFilterProxyModel,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QStyle,
    QStyledItemDelegate,
    QToolTip,
    QTreeView,
)

from deltachrome_ui.library import BatchGroup, RunSummary
from deltachrome_ui.tables import SortHeader
from deltachrome_ui.status import TONE_FAIL
from deltachrome_ui.theme import TEXT, TEXT_DIM, tone_colors

#: The whole :class:`~deltachrome_ui.library.RunSummary` behind a row, so an
#: action does not have to reconstruct one from the cells it can see.
RUN_ROLE = Qt.ItemDataRole.UserRole + 1
#: What a column sorts on, which is rarely what it displays.
SORT_ROLE = Qt.ItemDataRole.UserRole + 2
#: What a search matches against. Wider than the display text where the display
#: text is a painted mark: a status dot has no letters, but "fail" must find it.
SEARCH_ROLE = Qt.ItemDataRole.UserRole + 3


@dataclass(frozen=True)
class Column:
    """One column, declared once and used by every part of the table."""

    key: str
    title: str
    #: What the cell reads. Empty for a column that is painted.
    text: Callable[[RunSummary], str]
    #: What it sorts on. Ascending must be the order a person wants when they
    #: first click the header, which for a verdict means worst at the top.
    sort: Callable[[RunSummary], Any]
    #: Hover text. Where a cell is elided or painted, this is the only way the
    #: full value is readable at all, so it is never omitted for those.
    tooltip: Callable[[RunSummary], str] = lambda run: ""
    #: What a search query matches. Defaults to the display text.
    search: Callable[[RunSummary], str] | None = None
    width: int = 130
    #: ``fixed`` for the two painted columns, whose width is a property of what
    #: they draw; ``stretch`` for the one that absorbs the window's spare width;
    #: ``interactive`` for everything a person may want to widen.
    resize: str = "interactive"
    #: Which end of an over-long value is dropped. Genuinely per column: a
    #: profile name is distinguished by its front ("FADGI 4-Star — Prints and
    #: Photographs") and a target name by its end ("…Golden Thread Device Level
    #: Target" against "…Transmissive Film Target, 4x5"), so one setting for the
    #: whole table has to be wrong for one of them.
    elide: Qt.TextElideMode = Qt.TextElideMode.ElideRight
    #: Hover text on the header. The only thing that can name a column whose
    #: title is deliberately blank.
    header_tooltip: str = ""
    align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

    def search_text(self, run: RunSummary) -> str:
        return (self.search or self.text)(run)


#: How wide a tooltip in this table is allowed to get, in characters. Qt wraps
#: a plain-text tooltip only when it decides the line is too long, and it decides
#: per line -- so a run whose warnings happen to be short produces a narrow
#: popup and one long warning produces a popup most of the window wide. Wrapping
#: them here is what makes two tooltips on the same column the same shape.
TOOLTIP_WIDTH = 62


def wrapped_tooltip(lines: list[str], *, bullet: bool = False) -> str:
    """Lines wrapped to a consistent width, bullets hanging under themselves.

    Public because the batch screen's own table carries the same warnings
    column and has to wrap them the same way: two tables in one application
    whose tooltips were different shapes would read as two applications.
    """
    import textwrap

    out: list[str] = []
    for line in lines:
        if not line:
            out.append("")
            continue
        out.extend(textwrap.wrap(
            line, width=TOOLTIP_WIDTH,
            initial_indent="•  " if bullet else "",
            subsequent_indent="    " if bullet else "",
        ) or [line])
    return "\n".join(out)


def _warning_text(run: RunSummary) -> str:
    """Blank rather than zero.

    A ``0`` in every clean row is a column of noise that the eye has to read
    past to find the rows that have something in them, which is the opposite of
    what the column is for.
    """
    return str(run.warning_count) if run.warning_count else ""


def _warning_tooltip(run: RunSummary) -> str:
    """The warnings themselves, which is the point of keeping the column.

    A count on a screen that cannot show what was counted is a dead end. Capped
    because a run can raise one per patch and a tooltip is not a report.
    """
    if not run.warnings:
        return "No warnings."
    shown = list(run.warnings[:8])
    if len(run.warnings) > 8:
        shown.append(f"…and {len(run.warnings) - 8} more.")
    return wrapped_tooltip(shown, bullet=True)


def _basis_tooltip(run: RunSummary) -> str:
    """Why *this* run cannot be quoted, in its own words.

    Run-specific qualification rather than standing explanation, so it belongs
    on the cell -- what ``Nominal`` and ``Derived`` mean in general is the
    header's job, and the general half is documentation now.
    """
    if run.caveats:
        return wrapped_tooltip(
            ["Not a conformance result:", ""] + list(run.caveats), bullet=False
        )
    if run.basis == "Measured":
        return wrapped_tooltip(["Measured reference values for this chart."])
    return run.basis


def _status_tooltip(run: RunSummary) -> str:
    return wrapped_tooltip([run.status.label, "", run.status.hint])


def _file_tooltip(run: RunSummary) -> str:
    if isinstance(run, BatchGroup):
        return wrapped_tooltip([
            run.summary or f"{len(run.runs)} records.", "",
            f"Folder: {run.path.name}",
        ])
    if not run.is_readable:
        return wrapped_tooltip([run.path.name, "", run.error or ""])
    return wrapped_tooltip([
        run.image_path or run.image_name, "", f"Record: {run.path.name}",
    ])


def _date_sort(run: RunSummary) -> float:
    stamp = run.created_at
    # A record whose timestamp cannot be parsed sorts to the bottom of a
    # newest-first list rather than to an arbitrary place among real dates.
    return stamp.timestamp() if stamp is not None else float("-inf")


#: The library's columns, in their default order.
def _chart_count(run) -> int:
    """How many targets this record holds. A batch row has no such number."""
    return int(getattr(run, "chart_count", 1) or 1)


def _target_text(run) -> str:
    """The target, and how many of it the frame held.

    On the row rather than in a column of its own: a capture of four QA-62 cards
    is four *targets*, so the count belongs beside the name of the thing being
    counted. It is also the fact a reader needs before clicking, because such a
    row opens on **All Targets** rather than on a report.
    """
    count = _chart_count(run)
    return f"{run.target_name}   \u00d7{count}" if count > 1 else run.target_name


def _target_tooltip(run) -> str:
    count = _chart_count(run)
    if count <= 1:
        return run.target_name
    return f"{run.target_name}\n\n{count} targets, saved as one record."


def _target_search(run) -> str:
    """So a year of runs can be filtered down to the boards.

    ``targets`` is the word somebody would type, and it is not in the cell --
    the cell shows a multiplication sign and a number.
    """
    count = _chart_count(run)
    if count <= 1:
        return run.target_name
    return f"{run.target_name} {count} targets multitarget"


COLUMNS: list[Column] = [
    Column(
        key="status", title="", text=lambda r: "",
        sort=lambda r: (r.status.rank, r.status.label),
        tooltip=_status_tooltip, search=lambda r: r.status.label,
        width=32, resize="fixed",
        header_tooltip="Verdict.",
    ),
    Column(
        key="file", title="Capture", text=lambda r: r.image_name,
        sort=lambda r: r.image_name.lower(), tooltip=_file_tooltip,
        width=260, resize="stretch", elide=Qt.TextElideMode.ElideMiddle,
    ),
    Column(
        key="target", title="Target", text=_target_text,
        sort=lambda r: r.target_name.lower(), tooltip=_target_tooltip,
        search=_target_search,
        width=210, elide=Qt.TextElideMode.ElideMiddle,
        header_tooltip="Target name and how many appear in the capture.",
    ),
    Column(
        key="profile", title="Guidelines", text=lambda r: r.profile_name,
        sort=lambda r: r.profile_name.lower(), tooltip=lambda r: r.profile_name,
        width=230,
        header_tooltip="What the run was judged against.",
    ),
    Column(
        key="basis", title="Basis", text=lambda r: r.basis,
        sort=lambda r: r.basis, tooltip=_basis_tooltip, width=168,
        header_tooltip="Type of reference used and if the guideline is local.",
    ),
    Column(
        key="warnings", title="Warnings", text=_warning_text,
        sort=lambda r: r.warning_count, tooltip=_warning_tooltip,
        search=lambda r: " ".join(r.warnings), width=96,
        header_tooltip="Hover over a cell to read warnings.",
        align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
    ),
    Column(
        key="date", title="Run", text=lambda r: r.created_display,
        sort=_date_sort, tooltip=lambda r: r.created_utc, width=156,
        header_tooltip="Local time.",
    ),
    Column(
        key="actions", title="Actions", text=lambda r: "", sort=lambda r: 0,
        width=150, resize="fixed",
    ),
]

COLUMN_INDEX = {column.key: i for i, column in enumerate(COLUMNS)}
STATUS_COLUMN = COLUMN_INDEX["status"]
ACTION_COLUMN = COLUMN_INDEX["actions"]


class RunTableModel(QAbstractItemModel):
    """The library's rows, read-only, two levels deep.

    **It is a tree because a batch is a folder.** A hundred captures run against
    one guideline write a hundred records at one instant, and in a table sorted
    by date that is a hundred rows between a person and everything else they
    have. They already sit in a folder of their own on disk; this is the same
    fact in the table.

    Two levels and no more, which is the shape the disk has: nothing here writes
    a folder inside a batch's folder.

    **A tree rather than sections in a flat list**, because the library's whole
    argument is that it is queried rather than read -- and a flat list with
    headers in it cannot be sorted without scattering every header away from its
    own rows. ``QSortFilterProxyModel`` sorts each parent's children among
    themselves, so a batch sorts among the loose runs by its own date and its
    records sort inside it.

    The parent of a row is carried in the index's internal id: ``0`` for a row of
    the library itself, and ``parent row + 1`` for a record inside a batch. An id
    rather than the object, because ``createIndex`` holds no reference and a
    model reset would leave indices pointing at rows that had gone.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        #: :class:`~deltachrome_ui.library.RunSummary` or
        #: :class:`~deltachrome_ui.library.BatchGroup`.
        self._runs: list[Any] = []

    def set_runs(self, runs: list[Any]) -> None:
        self.beginResetModel()
        self._runs = list(runs)
        self.endResetModel()

    # -- the tree ----------------------------------------------------------

    def index(self, row: int, column: int, parent=QModelIndex()) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        return self.createIndex(
            row, column, 0 if not parent.isValid() else parent.row() + 1
        )

    def parent(self, index=None):  # noqa: A003 - Qt's name
        """The row above this one, or -- with no argument -- the owning QObject.

        **Both, and that is not tidiness.** ``QAbstractItemModel.parent(index)``
        and ``QObject.parent()`` are the same name in Python, so declaring the
        model's one hides the object's: ``model.parent()`` then raises a
        ``TypeError`` from inside whatever Qt or application code happened to
        ask, which is a crash in a call nobody wrote.
        """
        if index is None:
            return QObject.parent(self)
        if not index.isValid():
            return QModelIndex()
        owner = int(index.internalId())
        if owner == 0:
            return QModelIndex()
        return self.createIndex(owner - 1, 0, 0)

    def rowCount(self, parent=QModelIndex()) -> int:
        if not parent.isValid():
            return len(self._runs)
        if parent.column() != 0:
            return 0
        group = self.run_at(parent)
        return len(group.runs) if isinstance(group, BatchGroup) else 0

    def columnCount(self, parent=QModelIndex()) -> int:
        # Every level has the same columns, which is the point: a record inside a
        # batch is an ordinary record and reads across the table like one.
        return len(COLUMNS)

    def run_at(self, index: QModelIndex) -> Any:
        """The row behind an index, at either level."""
        if not index.isValid():
            return None
        owner = int(index.internalId())
        if owner == 0:
            return (self._runs[index.row()]
                    if 0 <= index.row() < len(self._runs) else None)
        group = (self._runs[owner - 1]
                 if 0 <= owner - 1 < len(self._runs) else None)
        if not isinstance(group, BatchGroup):
            return None
        return (group.runs[index.row()]
                if 0 <= index.row() < len(group.runs) else None)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        run = self.run_at(index)
        if run is None:
            return None
        column = COLUMNS[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return column.text(run)
        if role == Qt.ItemDataRole.ToolTipRole:
            return column.tooltip(run) or None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(column.align)
        if role == Qt.ItemDataRole.ForegroundRole:
            # An unreadable record is still listed -- deliberately, since a
            # record that vanished because its JSON was truncated is worse than
            # a visible bad row -- but it must not read as ordinary data.
            if not run.is_readable:
                return QColor(TEXT_DIM)
            if column.key in ("target", "profile", "basis", "date"):
                return QColor(TEXT_DIM)
            return None
        if role == SORT_ROLE:
            return column.sort(run)
        if role == SEARCH_ROLE:
            return column.search_text(run)
        if role == RUN_ROLE:
            return run
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section].title
        if role == Qt.ItemDataRole.ToolTipRole:
            tip = COLUMNS[section].header_tooltip
            return wrapped_tooltip([tip]) if tip else None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(COLUMNS[section].align)
        return None

    def flags(self, index: QModelIndex):
        return (
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        )


class RunFilterProxy(QSortFilterProxyModel):
    """Search across every column at once.

    Words rather than a substring, and every word has to appear somewhere in the
    row. ``sg fadgi`` is a question a person actually has -- "the SG runs judged
    against FADGI" -- and a plain substring match cannot answer it because those
    two words are never adjacent in any one field.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSortRole(SORT_ROLE)
        # A batch is kept whenever one of its records matches, or the whole of a
        # search's answer could sit inside a folder the search had hidden.
        self.setRecursiveFilteringEnabled(True)
        self._terms: list[str] = []

    def set_query(self, text: str) -> None:
        self._terms = text.lower().split()
        # ``invalidate`` rather than ``invalidateFilter``: PySide6 marks the
        # narrower calls deprecated, and re-sorting a few thousand already-keyed
        # rows costs nothing worth keeping a warning for.
        self.invalidate()

    def _haystack(self, row: int, parent: QModelIndex) -> str:
        model = self.sourceModel()
        return " ".join(
            str(model.data(model.index(row, column, parent), SEARCH_ROLE) or "")
            for column in range(model.columnCount(parent))
        ).lower()

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        """Compare in Python, on what the column declared.

        **The verdict column had never sorted at all.** Its key is a tuple --
        ``(rank, label)``, so that a verdict this version does not recognise
        still lands somewhere deliberate -- and Qt's own comparison cannot order
        a Python tuple carried through a QVariant: every comparison came back
        false and the rows stayed in the order they were read. Clicking the
        header drew the chevron and did nothing, which is the shape of failure
        this table is most able to hide, because a list that did not move looks
        like a list that was already sorted.

        Nothing else was affected -- every other column's key is a string or a
        number -- but comparing them all here is what makes that fact stop
        mattering: the declaration in :data:`COLUMNS` is the whole definition of
        the order, and a column may return whatever kind of key says what it
        means.
        """
        a, b = left.data(SORT_ROLE), right.data(SORT_ROLE)
        try:
            return bool(a < b)
        except TypeError:
            # Keys of two shapes in one column, which is a column with a bug in
            # it. Ordering them by their text is arbitrary and stable, which is
            # better than the exception Qt would swallow.
            return str(a) < str(b)

    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:
        if not self._terms:
            return True
        haystack = self._haystack(row, parent)
        if parent.isValid():
            # A record inside a batch is searched by its batch's NAME as well as
            # by its own words. The batch is what a person calls it -- "Odonata"
            # -- and that name appears nowhere on the records inside, so without
            # this, searching for it returned the folder with nothing in it.
            #
            # The name and nothing else. The batch's row also carries the *union*
            # of every warning its records raised and a tally of their verdicts,
            # and inheriting those made every record in a batch match any warning
            # one of them had: searching "clipped" over a folder of a hundred
            # returned the hundred.
            group = self.sourceModel().data(
                self.sourceModel().index(parent.row(), 0, parent.parent()), RUN_ROLE
            )
            if isinstance(group, BatchGroup):
                haystack += " " + group.image_name.lower()
        return all(term in haystack for term in self._terms)


# -- painted cells -----------------------------------------------------------


#: How far a record inside a batch is set in from the left of the capture
#: column. The tree's own indentation is off -- see :class:`RunTableView` -- so
#: this is the one thing that says a row belongs to the folder above it, and it
#: is on the one column whose content is read from the left.
CHILD_INSET = 20


class ElideDelegate(QStyledItemDelegate):
    """Text, elided at the end the column declares, and inset where it is owned.

    ``QAbstractItemView.setTextElideMode`` is one setting for the whole table,
    and this table has columns that need opposite answers -- so the view-wide
    mode is the default and this overrides it where a column says otherwise.

    The inset is the other half. Qt's own indentation moves *column zero* of a
    child row, which here is the painted status column: the dots would come out
    ragged down a list where their whole job is to be scanned in one line. So
    the tree is flat and the capture column says the nesting instead, which is
    also the column a reader is looking at when they ask what a row is.
    """

    def __init__(self, mode: Qt.TextElideMode, parent=None,
                 *, inset_children: bool = False) -> None:
        super().__init__(parent)
        self._mode = mode
        self._inset = inset_children

    def initStyleOption(self, option, index: QModelIndex) -> None:
        super().initStyleOption(option, index)
        option.textElideMode = self._mode
        if self._inset and index.parent().isValid():
            option.rect = option.rect.adjusted(CHILD_INSET, 0, 0, 0)


class StatusDotDelegate(QStyledItemDelegate):
    """The verdict, as a filled circle in its tone -- and never for a batch.

    Five colors, not two. The demotion rule in :mod:`deltachrome_ui.status` says
    a run without a conformance claim never renders in the pass tone, and a
    two-color dot is precisely the shortcut that would break it -- an advisory
    pass would have nowhere to go but green.

    **A batch gets a disclosure chevron here instead, and that is a rule rather
    than a layout convenience.** A batch asks whether *each* of a hundred
    captures passed; pooling them would let one bad frame fail the lot and lose
    which one, and conformance in this field is stated per image. One dot cannot
    say that, so there is no dot. What the batch's own row carries in place of a
    verdict is a tally, on the pointer and in what the column searches.
    """

    RADIUS = 4.5

    def __init__(self, view: QTreeView) -> None:
        super().__init__(view)
        self._view = view

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        super().paint(painter, option, index)
        run = index.data(RUN_ROLE)
        if run is None:
            return
        if isinstance(run, BatchGroup):
            self._chevron(painter, option.rect, self._view.isExpanded(index))
            return
        color = QColor(tone_colors(run.status.tone)[0])
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        center = option.rect.center()
        painter.drawEllipse(
            QRectF(
                center.x() - self.RADIUS + 0.5, center.y() - self.RADIUS + 0.5,
                self.RADIUS * 2, self.RADIUS * 2,
            )
        )
        painter.restore()

    def _chevron(self, painter: QPainter, cell: QRect, open_: bool) -> None:
        """Pointing right when the folder is shut and down when it is open."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(TEXT_DIM))
        pen.setWidthF(1.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        cx, cy, arm = cell.center().x() + 0.5, cell.center().y() + 0.5, 3.6
        path = QPainterPath()
        if open_:
            path.moveTo(cx - arm, cy - arm / 1.6)
            path.lineTo(cx, cy + arm / 1.6)
            path.lineTo(cx + arm, cy - arm / 1.6)
        else:
            path.moveTo(cx - arm / 1.6, cy - arm)
            path.lineTo(cx + arm / 1.6, cy)
            path.lineTo(cx - arm / 1.6, cy + arm)
        painter.drawPath(path)
        painter.restore()


#: The row actions, in the order they are painted: ``(key, menu label, hover
#: text)``. The hover text is the only thing that ever names an icon, so it says
#: what will happen rather than what the picture is meant to be.
ACTIONS: list[tuple[str, str, str]] = [
    ("open", "Open", "Open"),
    ("rerun", "Re-run", "Re-run"),
    ("copy", "Copy Command", "Copy command"),
    ("export", "Export Values…", "Export values as TXT, CSV or XML"),
    ("delete", "Move to Trash", "Move to trash"),
]

_ICON_SIZE = 17
_ICON_GAP = 9

#: The icons at rest, on every row that is neither hovered nor selected. They
#: were painted only on the active row, which was clean and read as an empty
#: column: nothing said the controls were there until the pointer happened to
#: cross them.
RESTING_ICON = "#43434d"
#: An action this row cannot take. Dimmer than resting, so a disabled icon on
#: the active row never looks brighter than a live one on a row at rest.
DISABLED_ICON = "#3a3a42"


def _icon_rects(cell: QRect) -> list[QRect]:
    """Where each icon sits, right-aligned in the cell."""
    total = len(ACTIONS) * _ICON_SIZE + (len(ACTIONS) - 1) * _ICON_GAP
    left = cell.right() - total - 10
    top = cell.center().y() - _ICON_SIZE // 2
    return [
        QRect(left + i * (_ICON_SIZE + _ICON_GAP), top, _ICON_SIZE, _ICON_SIZE)
        for i in range(len(ACTIONS))
    ]


def _paint_icon(painter: QPainter, rect: QRect, key: str) -> None:
    """Draw one action glyph.

    Painted rather than shipped as image files, following the same reasoning as
    the combo box chevron: the application stays a single installable package
    with no asset directory beside it. Every glyph is a stroked path in a unit
    box, so they share a weight and stay legible at seventeen pixels.
    """
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

    def px(u: float, v: float) -> tuple[float, float]:
        return x + u * w, y + v * h

    path = QPainterPath()
    if key == "open":
        # A pane with an arrow leaving it: the conventional "open this".
        path.moveTo(*px(0.55, 0.16))
        for point in ((0.86, 0.16), (0.86, 0.47)):
            path.lineTo(*px(*point))
        path.moveTo(*px(0.86, 0.16))
        path.lineTo(*px(0.47, 0.55))
        path.moveTo(*px(0.62, 0.60))
        for point in ((0.62, 0.88), (0.12, 0.88), (0.12, 0.34), (0.40, 0.34)):
            path.lineTo(*px(*point))
    elif key == "rerun":
        # A chart: nine patches. The same picture the report's header uses for
        # the same destination -- re-running a capture opens the screen where the
        # chart is placed -- and it has to change here for a second reason. This
        # was an arc with a head on it, and the library's own header now carries
        # a refresh arrow that means "re-read the folder". Two circular arrows on
        # one screen meaning two different things is worse than either icon.
        #
        # Filled, unlike its neighbours, because at seventeen pixels a nine-cell
        # outline closes up into a smudge. The weight is close enough: the cells
        # are small and the ink adds up to about what a stroked glyph lays down.
        gap = w * 0.09
        cell = (w - gap * 2) / 3
        for row in range(3):
            for col in range(3):
                path.addRect(
                    x + col * (cell + gap), y + row * (cell + gap), cell, cell
                )
        painter.fillPath(path, painter.pen().color())
        path = QPainterPath()
    elif key == "copy":
        path.addRoundedRect(
            QRectF(x + w * 0.10, y + h * 0.10, w * 0.52, h * 0.52), 2.0, 2.0
        )
        path.addRoundedRect(
            QRectF(x + w * 0.38, y + h * 0.38, w * 0.52, h * 0.52), 2.0, 2.0
        )
    elif key == "export":
        # The report bar's export glyph, in this unit box: an arrow out of a
        # tray, so the one action has one picture wherever it is offered.
        path.moveTo(*px(0.50, 0.60))
        path.lineTo(*px(0.50, 0.16))
        path.moveTo(*px(0.32, 0.36))
        for point in ((0.50, 0.16), (0.68, 0.36)):
            path.lineTo(*px(*point))
        path.moveTo(*px(0.18, 0.66))
        for point in ((0.18, 0.84), (0.82, 0.84), (0.82, 0.66)):
            path.lineTo(*px(*point))
    elif key == "delete":
        path.moveTo(*px(0.10, 0.26))
        path.lineTo(*px(0.90, 0.26))
        path.moveTo(*px(0.36, 0.26))
        for point in ((0.36, 0.13), (0.64, 0.13), (0.64, 0.26)):
            path.lineTo(*px(*point))
        path.moveTo(*px(0.20, 0.26))
        for point in ((0.26, 0.90), (0.74, 0.90), (0.80, 0.26)):
            path.lineTo(*px(*point))
        path.moveTo(*px(0.42, 0.42))
        path.lineTo(*px(0.44, 0.76))
        path.moveTo(*px(0.58, 0.42))
        path.lineTo(*px(0.56, 0.76))
    painter.drawPath(path)


class ActionDelegate(QStyledItemDelegate):
    """The action icons, painted on the row under the mouse and on the selected row.

    Not four buttons per row. A thousand records is four thousand widgets built
    on every refresh of the library, which is what the card list was already
    doing and is the reason the table exists.

    This class only paints. Hit-testing lives in :class:`RunTableView`, because
    ``editorEvent`` is not delivered for mouse *movement* -- the view routes
    press, release and double-click to a delegate and nothing else -- so hover
    highlighting would be the one thing that could not be done here. Splitting
    the two would put the geometry in one place and the response to it in
    another; both are in the view, and the delegate reads what it is told.
    """

    def __init__(self, view: QTreeView) -> None:
        super().__init__(view)
        self._view = view

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        super().paint(painter, option, index)
        run = index.data(RUN_ROLE)
        if run is None:
            return
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        on_this_row = self._view.is_hovered(index)
        active = on_this_row or selected
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        batch = isinstance(run, BatchGroup)
        for i, rect in enumerate(_icon_rects(option.rect)):
            key = ACTIONS[i][0]
            # An unreadable record can be deleted and nothing else: there is no
            # report to open and no command to reproduce. A batch is the same
            # shape of answer for a different reason -- there is no one report
            # of a hundred captures and no one command that made them, and
            # clearing a batch out a confirmation at a time is not clearing it.
            enabled = key == "delete" or (run.is_readable and not batch)
            if batch and not enabled:
                # Left out rather than drawn dim. A broken record is a row that
                # went wrong and its greyed icons say which parts of it are
                # missing; a batch is a different *kind* of row, and three
                # permanently dead controls on every one of them is three
                # things to learn to ignore.
                continue
            hovered = enabled and on_this_row and self._view.hovered_icon == i
            if not enabled and active:
                color = QColor(DISABLED_ICON)
            elif not active:
                # Resting. Faint enough not to be four more things to read down
                # a thousand rows, present enough that the column is visibly a
                # column of controls rather than the empty space it read as.
                color = QColor(RESTING_ICON)
            elif hovered:
                color = QColor(tone_colors(TONE_FAIL)[0] if key == "delete" else TEXT)
            else:
                color = QColor(TEXT_DIM)
            pen = QPen(color)
            pen.setWidthF(1.35)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            _paint_icon(painter, rect, key)
        painter.restore()


class RunTableView(QTreeView):
    """The table itself, and the mouse.

    A tree that does not look like one. Batches are the only rows with children
    and they are drawn as a chevron in the status column, so **the indentation
    is off and the branch decoration with it**: Qt indents column zero, which
    here is the painted verdict dot, and a column of dots that steps sideways is
    one nobody can scan. A record inside a batch is set in on the capture column
    instead, by :class:`ElideDelegate`.

    The hovered row is tracked here rather than read from ``State_MouseOver``
    because that flag is set for the cell under the pointer, not for the row:
    hover a filename and the actions cell four columns away is not "under the
    mouse", so the icons would only ever appear once the pointer was already on
    top of them -- which is no use for finding out that they exist. It is kept
    as an *index* and not a row number, because in a tree a row number means
    nothing without its parent: row 2 of a batch and row 2 of the library are
    both "2".
    """

    #: ``(action key, run)`` -- an icon was clicked, or a menu item chosen.
    actionTriggered = Signal(str, object)

    def __init__(self) -> None:
        super().__init__()
        self.hovered_row = QModelIndex()
        self.hovered_icon = -1
        #: Batch folders the reader has opened, by path, so that re-reading the
        #: library does not shut them. A set rather than the indices: a reload
        #: resets the model and every index with it.
        self.expanded_batches: set = set()
        self.setObjectName("RunTable")
        self.setMouseTracking(True)
        self.setIndentation(0)
        self.setRootIsDecorated(False)
        self.setUniformRowHeights(True)
        self.setExpandsOnDoubleClick(False)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        # Elide the tail, not the middle. Every long value in this table is
        # distinguished by its front: "FADGI 4-Star — Prints and Photographs
        # (2023 3rd Ed.)" elided in the middle keeps the edition and drops the
        # material category, which is the only part that differs between rows.
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        # Horizontally too, which was the one that showed. Per *item* means a
        # wheel tick jumps a whole column, and the widest table here -- twenty
        # slanted edges in three colour records -- has columns wide enough that
        # one tick moves the whole view, so there is no position between two
        # columns to stop at.
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.setHeader(SortHeader())
        header = self.header()
        header.setSectionsMovable(True)
        header.setSortIndicatorShown(True)
        header.setHighlightSections(False)
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        header.setStretchLastSection(False)

        self.setItemDelegateForColumn(STATUS_COLUMN, StatusDotDelegate(self))
        self.setItemDelegateForColumn(ACTION_COLUMN, ActionDelegate(self))
        for i, column in enumerate(COLUMNS):
            if column.elide != Qt.TextElideMode.ElideRight or column.key == "file":
                self.setItemDelegateForColumn(i, ElideDelegate(
                    column.elide, self, inset_children=column.key == "file",
                ))
        self.setSortingEnabled(True)
        self.expanded.connect(self._remember_expansion)
        self.collapsed.connect(self._remember_expansion)

    #: Qt's own names for what a :class:`Column` declares.
    _RESIZE_MODES = {
        "fixed": QHeaderView.ResizeMode.Fixed,
        "stretch": QHeaderView.ResizeMode.Stretch,
        "interactive": QHeaderView.ResizeMode.Interactive,
    }

    def setModel(self, model) -> None:
        """Attach the model, then lay the columns out.

        In that order, and not the other way round: a header has no sections
        until it has a model, so widths set in the constructor are applied to
        columns that do not exist yet and are silently discarded. Every column
        then comes up at the default section width, which is how a table ends up
        with the capture name and the warning count given the same hundred
        pixels.
        """
        super().setModel(model)
        header = self.header()
        for i, column in enumerate(COLUMNS):
            header.setSectionResizeMode(i, self._RESIZE_MODES[column.resize])
            if column.resize != "stretch":
                header.resizeSection(i, column.width)
        if model is not None:
            model.modelReset.connect(self.restore_expansion)

    # -- folders -----------------------------------------------------------

    def _remember_expansion(self, index: QModelIndex) -> None:
        row = index.data(RUN_ROLE)
        if not isinstance(row, BatchGroup):
            return
        if self.isExpanded(index):
            self.expanded_batches.add(row.path)
        else:
            self.expanded_batches.discard(row.path)

    def restore_expansion(self) -> None:
        """Re-open the folders the reader had open, after the model was rebuilt.

        The library re-reads itself on every visit and after every delete, and a
        reader who opened a batch to look through it has not asked for it to be
        shut again each time.
        """
        model = self.model()
        if model is None:
            return
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            value = index.data(RUN_ROLE)
            if isinstance(value, BatchGroup) and value.path in self.expanded_batches:
                self.setExpanded(index, True)

    def is_hovered(self, index: QModelIndex) -> bool:
        """Whether this index is on the row the pointer is over."""
        hovered = self.hovered_row
        return (
            hovered.isValid() and index.isValid()
            and hovered.row() == index.row()
            and hovered.parent() == index.parent()
        )

    # -- the mouse ---------------------------------------------------------

    def run_at(self, index: QModelIndex) -> Any:
        """Whatever is behind an index: a record, or a batch's own row."""
        return index.data(RUN_ROLE) if index.isValid() else None

    def selected_rows(self) -> list[Any]:
        """Everything selected, in the order the table shows it.

        Walked rather than read off ``selectedIndexes``, which in a tree gives
        row numbers that are only meaningful beside their parent -- row 2 of a
        batch and row 2 of the library are both "2", and sorting them together
        interleaves two different lists.
        """
        model = self.model()
        selection = self.selectionModel()
        if model is None or selection is None:
            return []
        out: list[Any] = []

        def walk(parent: QModelIndex) -> None:
            for row in range(model.rowCount(parent)):
                index = model.index(row, 0, parent)
                if selection.isSelected(index):
                    value = index.data(RUN_ROLE)
                    if value is not None:
                        out.append(value)
                walk(index)

        walk(QModelIndex())
        return out

    def selected_runs(self) -> list[RunSummary]:
        """Every selected *record*, in the order the table shows them.

        Deliberately not "the current row": the reason a table replaced the card
        list is that a year of records has to be handled in quantity, and
        clearing out forty superseded runs one confirmation at a time is not
        handling it.

        A batch's own row is not a record and is left out. That is what keeps
        "Combine into a Session" honest -- a batch is a hundred independent
        verdicts and pooling them is the one thing it must not become -- and it
        means selecting a folder and its records does not count them twice.
        """
        return [row for row in self.selected_rows()
                if not isinstance(row, BatchGroup)]

    def _icon_at(self, index: QModelIndex, pos: QPoint) -> int:
        if not index.isValid() or index.column() != ACTION_COLUMN:
            return -1
        run = self.run_at(index)
        for i, rect in enumerate(_icon_rects(self.visualRect(index))):
            # An icon a batch's row does not paint is not there to be aimed at
            # either, or the pointer would turn into a hand over empty space.
            if isinstance(run, BatchGroup) and not self._can(run, ACTIONS[i][0]):
                continue
            # Grown a little: a seventeen-pixel glyph is a smaller target than
            # anyone aims at, and the icons are far enough apart to afford it.
            if rect.adjusted(-4, -4, 4, 4).contains(pos):
                return i
        return -1

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        index = self.indexAt(pos)
        icon = self._icon_at(index, pos)
        if not self.is_hovered(index) or icon != self.hovered_icon:
            self.hovered_row, self.hovered_icon = index, icon
            self.viewport().update()
        over_chevron = (
            index.isValid() and index.column() == STATUS_COLUMN
            and isinstance(self.run_at(index), BatchGroup)
        )
        self.setCursor(
            Qt.CursorShape.PointingHandCursor if icon >= 0 or over_chevron
            else Qt.CursorShape.ArrowCursor
        )
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        if self.hovered_row.isValid() or self.hovered_icon != -1:
            self.hovered_row, self.hovered_icon = QModelIndex(), -1
            self.viewport().update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        """An icon click must not also start a selection drag underneath it."""
        pos = event.position().toPoint()
        if self._icon_at(self.indexAt(pos), pos) >= 0:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        pos = event.position().toPoint()
        index = self.indexAt(pos)
        icon = self._icon_at(index, pos)
        if icon >= 0 and event.button() == Qt.MouseButton.LeftButton:
            run = self.run_at(index)
            key = ACTIONS[icon][0]
            if run is not None and self._can(run, key):
                self.actionTriggered.emit(key, run)
            event.accept()
            return
        if (event.button() == Qt.MouseButton.LeftButton
                and index.isValid() and index.column() == STATUS_COLUMN
                and isinstance(self.run_at(index), BatchGroup)):
            # The chevron. It is the cell a reader aims at to open a folder, and
            # it is where the verdict dot would be on any other row -- so on a
            # batch that cell is the control, not merely a picture of one.
            self.setExpanded(index, not self.isExpanded(index))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    @staticmethod
    def _can(run: Any, key: str) -> bool:
        """The same rule the delegate paints by, so the two cannot drift."""
        if key == "delete":
            return True
        return run.is_readable and not isinstance(run, BatchGroup)

    def mouseDoubleClickEvent(self, event) -> None:
        """Opening is the default; re-running is the deliberate act.

        An operator arriving at the library far more often wants the answer than
        wants to place corners on the capture again -- and re-running costs a
        full decode of the master to reach a screen they did not ask for.

        On a batch there is no report of a hundred captures to open, so the same
        gesture opens the folder instead -- which is what double-clicking a
        folder means everywhere else.
        """
        pos = event.position().toPoint()
        index = self.indexAt(pos)
        if self._icon_at(index, pos) >= 0:
            # The icon already acted on the first click of the pair.
            event.accept()
            return
        run = self.run_at(index)
        if isinstance(run, BatchGroup):
            self.setExpanded(index, not self.isExpanded(index))
            event.accept()
            return
        if run is not None and run.is_readable:
            self.actionTriggered.emit("open", run)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def viewportEvent(self, event) -> bool:
        """Name the icons on hover. Nothing else does -- they carry no labels."""
        if event.type() == QEvent.Type.ToolTip:
            pos = event.pos()
            index = self.indexAt(pos)
            icon = self._icon_at(index, pos)
            if icon >= 0:
                run = self.run_at(index)
                hover = ACTIONS[icon][2]
                if isinstance(run, BatchGroup) and ACTIONS[icon][0] == "delete":
                    hover = "Move batch to trash"
                QToolTip.showText(event.globalPos(), hover, self)
                return True
            if index.isValid() and index.column() == ACTION_COLUMN:
                QToolTip.hideText()
                return True
        return super().viewportEvent(event)
