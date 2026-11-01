"""The library screen: every saved run, as a table that can be queried.

What a row shows is chosen to answer the question a person actually arrives
with, which is rarely "did it pass". It is "which capture was this, judged
against what, and can I quote the answer" -- so the target, the profile and the
basis of the judgement all sit on the row beside the verdict rather than in a
detail view nobody opens.

The shape changed for a reason worth writing down. This was a list of cards, and
cards are right for the fortnight in which a person has five runs. A department
shooting a few targets a day has a thousand a year, and at that size the library
stops being read and starts being searched: the columns exist so that "every
Golden Thread run that failed this month" is a sort and two words rather than a
scroll. :mod:`deltachrome_ui.run_table` holds the table; this holds the screen
around it and what its actions do.

Deleting is the one action here that could lose something, and it does not
delete: a record holds every per-patch value its run measured and cannot be
regenerated except by finding the master and repeating the placement by hand, so
it moves to the library's own trash folder and the confirmation says so.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from deltachrome_ui.command import command_for_record
from deltachrome_ui.library import (
    BatchGroup,
    RunSummary,
    read_record,
    runs_in,
    scan_library,
    trash_batch,
    trash_run,
)
from deltachrome_ui.paths import remember, remembered, user_run_dir
from deltachrome_ui.run_table import (
    ACTIONS,
    COLUMN_INDEX,
    COLUMNS,
    RunFilterProxy,
    RunTableModel,
    RunTableView,
)
from deltachrome_ui.theme import BAR_HEIGHT, TEXT_DIM
from deltachrome_ui.widgets import (
    IconButton,
    elided_subtitle,
    mark_cell,
    subtitle,
)

#: Where the header's column widths, order and sort are remembered between
#: sessions. A person who widens the capture column has said something about the
#: filenames they use, and being made to say it again every launch is the kind
#: of small insult that makes a tool feel unfinished.
_HEADER_KEY = "library_header_state"


def _column_signature() -> str:
    """Which columns a saved layout was written against."""
    return ",".join(column.key for column in COLUMNS)


class LibraryView(QWidget):
    """The app's home screen."""

    newAnalysisRequested = Signal()
    #: A saved run was chosen as the starting placement for a new analysis.
    reuseRequested = Signal(object)  # RunSummary
    #: A saved run was opened to be read. The ordinary case, and so the default
    #: action: an operator arriving at the library far more often wants the
    #: answer than wants to place corners on the capture again.
    openRequested = Signal(object)  # RunSummary
    #: A saved capture's values are wanted as a file. The shell owns the save
    #: dialog and the folder the last export went to, as it does for the
    #: report's own export square, so both write through one path.
    exportRequested = Signal(object)  # RunSummary

    #: The settings square was pressed. Settings holds the things that are true
    #: between runs rather than about one -- where records and reference files
    #: are kept, and the bank a capture's printed code is paired against.
    settingsRequested = Signal()
    #: A folder of captures to run against one guideline. Emitted with no argument
    #: because choosing the folder is the shell's job, not this screen's -- the
    #: library shows records and does not own a file dialog.
    batchRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.runs: list[RunSummary] = []
        #: What the table shows: the runs above, with each batch's records
        #: gathered under one row that opens.
        self.rows: list = []

        # Two layers, because the header is edge to edge and the rest is not.
        # The bar's action squares are its full height and the last of them is
        # flush against the window, which cannot happen inside a margin.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())

        body = QWidget()
        outer.addWidget(body, 1)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(28, 18, 28, 24)
        layout.setSpacing(12)

        layout.addLayout(self._build_toolbar())

        self.model = RunTableModel(self)
        self.proxy = RunFilterProxy(self)
        self.proxy.setSourceModel(self.model)

        self.table = RunTableView()
        self.table.setModel(self.proxy)
        self.table.actionTriggered.connect(self._on_action)
        self.table.customContextMenuRequested.connect(self._show_menu)
        self.table.sortByColumn(
            COLUMN_INDEX["date"], Qt.SortOrder.DescendingOrder
        )
        self._restore_header()
        # Coalesced. The capture column stretches, so *every* window resize
        # emits sectionResized -- once per pixel while a window edge is being
        # dragged, and writing the settings file that often would be the only
        # disk traffic this screen generates.
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._save_header)
        header = self.table.header()
        header.sectionResized.connect(lambda *_: self._save_timer.start())
        header.sectionMoved.connect(lambda *_: self._save_timer.start())
        header.sortIndicatorChanged.connect(lambda *_: self._save_timer.start())

        # One of three: the table, "nothing saved yet", "nothing matched". The
        # last two are different situations and telling them apart matters --
        # an empty library after a search reads as a lost library.
        self.body = QStackedWidget()
        self.body.addWidget(self.table)
        self.body.addWidget(self._message(
            "No saved runs yet.\n\n"
            "Runs saved from the command line with --save-run appear here, "
            "and so will analyzes started from this window."
        ))
        self.no_match = self._message("")
        self.body.addWidget(self.no_match)
        layout.addWidget(self.body, 1)

        self.count_label = subtitle("")
        layout.addWidget(self.count_label)

        self.reload()

    # -- construction ------------------------------------------------------

    def _build_header(self) -> QWidget:
        """The same bar the report carries, with this screen's own actions.

        Two lines on the left saying where you are and which folder you are
        looking at; the actions on the right as squares the height of the bar,
        butted together with a hairline between and the primary one an accent
        block in the corner. Nothing here is new -- it is
        :meth:`ResultView._build_header`'s shape, and the point of it being the
        same shape is that a reader moving between the two screens does not have
        to find the actions again.

        The title is set at the filename's size rather than the page-title size
        it used to be. A 19 px heading and a path under it do not fit in 52 px,
        and "Run Library" is not competing for attention with anything.
        """
        bar = QWidget()
        bar.setObjectName("HeaderBar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setFixedHeight(BAR_HEIGHT)
        header = QHBoxLayout(bar)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(12)

        # The mark, and only here. The report and the placement screens are
        # about a run; this one is about the application, so it is the only bar
        # where a logo is saying something the screen does not already say. It
        # is also the only surface with anywhere to put it -- the other two
        # spend their left edge on the verdict and the capture's name. In the
        # same cell Settings gives it, so the one logo a person sees on both
        # screens does not move between them.
        header.addWidget(mark_cell())

        head_text = QVBoxLayout()
        head_text.setContentsMargins(0, 0, 0, 0)
        head_text.setSpacing(0)
        name = QLabel("Run Library")
        name.setStyleSheet("font-size: 14px; font-weight: 600;")
        head_text.addWidget(name)
        # Elided, and its ``text()`` is still the whole path: a library can sit
        # anywhere and a bar is not the place to wrap a filesystem path.
        self.path_label = elided_subtitle("")
        head_text.addWidget(self.path_label)
        header.addLayout(head_text, 1)
        header.addSpacing(12)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(0)

        self.refresh_button = IconButton(
            "refresh", "Refresh", size=BAR_HEIGHT)
        self.refresh_button.clicked.connect(self.reload)
        actions.addWidget(self.refresh_button)

        self.settings_button = IconButton(
            "settings", "Settings", size=BAR_HEIGHT)
        self.settings_button.clicked.connect(self.settingsRequested.emit)
        actions.addWidget(self.settings_button)

        # Named rather than drawn, and beside the accent block rather than in it.
        # The icon squares to its left are actions a shape can say -- refresh,
        # settings; "Batch Analysis" is not one of those. It commits something,
        # which is the test the accent block's own actions pass, and a bar has one
        # accent in it, so it takes the same geometry and none of the colour.
        self.batch_button = QPushButton("Batch Analysis")
        self.batch_button.setObjectName("SecondaryBar")
        self.batch_button.setFixedHeight(BAR_HEIGHT)
        self.batch_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.batch_button.setToolTip(
            "Run a folder of captures against one guideline.")
        self.batch_button.clicked.connect(self.batchRequested.emit)
        actions.addWidget(self.batch_button)

        self.new_button = QPushButton("New Analysis")
        self.new_button.setObjectName("PrimaryBar")
        self.new_button.setFixedHeight(BAR_HEIGHT)
        self.new_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.new_button.clicked.connect(self.newAnalysisRequested.emit)
        actions.addWidget(self.new_button)

        header.addLayout(actions)
        return bar

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText(
            "Search — capture, target, guidelines, verdict, warning…"
        )
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._on_search)
        bar.addWidget(self.search, 1)
        return bar

    def _message(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 13px;")
        return label

    # -- loading -----------------------------------------------------------

    def reload(self) -> None:
        directory = user_run_dir()
        self.path_label.setText(str(directory))
        #: The rows the table shows: loose records, and one row per batch folder
        #: that opens to the records inside it.
        self.rows = scan_library(directory)
        #: Every record, batches opened out. What the count at the foot of the
        #: screen reports, because a batch of a hundred is a hundred runs.
        self.runs = runs_in(self.rows)
        self.model.set_runs(self.rows)
        self.table.restore_expansion()
        self._update_state()

    def reveal_batch(self, folder: Path) -> None:
        """Open a batch's row on the way in, for the run that has just finished.

        A hundred records arriving as one shut folder is right when a person
        comes to the library looking for something else, and wrong the moment
        they have pressed "Open in Library" on the run that wrote them: the
        answer they asked for would be behind a chevron.
        """
        self.table.expanded_batches.add(Path(folder))
        self.reload()

    def _on_search(self, text: str) -> None:
        self.proxy.set_query(text)
        self._update_state()

    def _shown(self) -> int:
        """How many records the search has left, batches counted by what is in them.

        The proxy's own row count is the top level only, and a batch is one row
        holding a hundred -- so a search that matched half a batch would report
        that one run matched.
        """
        total = 0
        for row in range(self.proxy.rowCount()):
            index = self.proxy.index(row, 0)
            children = self.proxy.rowCount(index)
            total += children or 1
        return total

    def _update_state(self) -> None:
        shown = self._shown()
        if not self.runs:
            self.body.setCurrentIndex(1)
            self.count_label.setText("")
            return
        if not shown:
            self.no_match.setText(
                f"Nothing in these {len(self.runs)} runs matches "
                f"“{self.search.text()}”."
            )
            self.body.setCurrentIndex(2)
        else:
            self.body.setCurrentIndex(0)
        total = len(self.runs)
        self.count_label.setText(
            f"{total} run{'s' if total != 1 else ''}"
            if shown == total else f"{shown} of {total} runs"
        )

    # -- header state ------------------------------------------------------

    def _restore_header(self) -> None:
        stored = remembered(_HEADER_KEY)
        if not isinstance(stored, dict) or stored.get("columns") != _column_signature():
            # Either nothing saved, or saved against a different set of columns.
            # Qt's own check is on the section *count*, which cannot notice that
            # a column was replaced by another of the same width -- and restoring
            # then puts one column's remembered width on a different column.
            return
        from base64 import b64decode
        from binascii import Error as B64Error

        try:
            self.table.header().restoreState(
                b64decode(str(stored.get("state", "")).encode())
            )
        except (B64Error, ValueError):
            pass  # The defaults are perfectly good; this is not worth an error.
        # A fixed column's width is the code's, not a remembered preference:
        # the actions column widens when an action is added, and a layout saved
        # before then would clip the first icon off its left edge.
        header = self.table.header()
        for i, column in enumerate(COLUMNS):
            if column.resize == "fixed":
                header.resizeSection(i, column.width)

    def _save_header(self) -> None:
        from base64 import b64encode

        raw = bytes(self.table.header().saveState())
        remember(_HEADER_KEY, {
            "columns": _column_signature(),
            "state": b64encode(raw).decode(),
        })

    # -- actions -----------------------------------------------------------

    def _on_action(self, key: str, run) -> None:
        if isinstance(run, BatchGroup):
            # The only one a batch's row is painted with, and the only one that
            # means anything for a folder of records.
            if key == "delete":
                self._delete([run])
            return
        if key == "open":
            self.openRequested.emit(run)
        elif key == "rerun":
            self.reuseRequested.emit(run)
        elif key == "copy":
            self._copy_command(run)
        elif key == "export":
            self.exportRequested.emit(run)
        elif key == "delete":
            self._delete([run])

    def _show_menu(self, point) -> None:
        rows = self.table.selected_rows()
        clicked = self.table.run_at(self.table.indexAt(point))
        if clicked is not None and clicked not in rows:
            rows = [clicked]
        if not rows:
            return
        self.build_menu(rows).exec(self.table.viewport().mapToGlobal(point))

    def build_menu(self, runs: list[RunSummary]) -> QMenu:
        """The same row actions, on whatever is selected.

        The icons act on one row because that is the row they are drawn on. A
        year of records has to be handled in quantity as well as singly, and
        clearing out forty superseded runs one confirmation at a time is not
        handling it -- so the menu is where a multiple selection is acted on.

        Built rather than shown, so that what it offers can be asserted without
        a modal loop.
        """
        menu = QMenu(self)
        # A batch's own row is not a record: there is no report of a hundred
        # captures to open and no single command that made them. It can be
        # trashed, which is the whole reason it is offered a menu at all.
        readable = [r for r in runs
                    if r.is_readable and not isinstance(r, BatchGroup)]
        for key, label, _hover in ACTIONS:
            if key == "delete":
                continue
            action = menu.addAction(label)
            # These act on one run: there is no report of forty runs to open,
            # no single command line that reproduces them, and an export is of
            # one capture, as Save is.
            action.setEnabled(len(readable) == 1)
            if len(readable) == 1:
                action.triggered.connect(
                    lambda _checked=False, k=key, r=readable[0]: self._on_action(k, r)
                )
        # Combining several *captures* from here is withdrawn until it is
        # designed where it belongs -- beside the run rather than scraped back
        # out of a table afterwards. So the only thing a
        # multiple selection is still offered is deleting, which is what the
        # menu existed for before combining was added to it.
        menu.addSeparator()
        delete = menu.addAction(
            "Move to Trash" if len(runs) == 1
            else f"Move {len(runs)} Runs to Trash"
        )
        delete.triggered.connect(lambda _checked=False: self._delete(runs))
        return menu

    def _copy_command(self, run: RunSummary) -> None:
        """The invocation that reproduces this run, from the record itself."""
        from PySide6.QtGui import QGuiApplication

        try:
            command = command_for_record(read_record(run.path))
        except (OSError, ValueError):
            return
        QGuiApplication.clipboard().setText(command)

    def _delete(self, runs: list) -> None:
        """Move records to the library's trash, having said what that means.

        The dialog names the *record* and states that the capture is untouched,
        because the row is named after the image: "Delete 0034.jp2?" reads as an
        offer to delete the master, and on a shared drive that is a sentence
        somebody will act on without reading twice.

        A batch goes as a folder, index and all. The count is said out loud in
        the question, because the row it was pressed on is one row and what it
        holds is a hundred records -- and a batch recovered from the trash has to
        come back as a batch rather than as a hundred loose files.
        """
        if not runs:
            return
        held = sum(len(r.runs) if isinstance(r, BatchGroup) else 1 for r in runs)
        if len(runs) == 1 and isinstance(runs[0], BatchGroup):
            headline = (
                f"Move this whole batch to the trash?\n\n{runs[0].image_name}"
            )
        elif len(runs) == 1:
            headline = f"Move this run record to the trash?\n\n{runs[0].image_name}"
        elif held != len(runs):
            headline = f"Move {held} run records to the trash?"
        else:
            headline = f"Move {len(runs)} run records to the trash?"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Move to Trash")
        box.setText(headline)
        box.setInformativeText(
            "The capture itself is not touched — only Deltachrome's record of "
            "the analysis, which holds every per-patch value the run measured.\n\n"
            "Records are moved to a .trash folder inside the library, so this "
            "can be undone by moving them back."
        )
        box.setStandardButtons(
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes
        )
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        box.button(QMessageBox.StandardButton.Yes).setText("Move to Trash")
        if box.exec() != QMessageBox.StandardButton.Yes:
            return

        failed: list[str] = []
        for run in runs:
            try:
                if isinstance(run, BatchGroup):
                    trash_batch(run.path)
                else:
                    trash_run(run.path)
            except OSError as exc:
                failed.append(f"{run.path.name}: {exc}")
        self.reload()
        if failed:
            QMessageBox.warning(
                self, "Not everything moved",
                "These records are still in the library:\n\n" + "\n".join(failed),
            )

