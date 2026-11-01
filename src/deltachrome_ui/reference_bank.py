"""The bank of per-unit reference files, and what each one answers to.

Reference measurements are *per physical unit*: two ColorChecker SGs look
identical in a photograph and their true values differ by more than the
tolerances being measured, so pointing a capture at the wrong unit's file returns
numbers that look entirely reasonable and mean nothing. The engine has known how
to solve that since :mod:`deltachrome.reference.library` was written -- read the
chart's own printed code off the capture and match it against a folder of files
-- and in the frontend it went unused, because *Pair from a library…* asked for a
folder with a directory dialog on every single run and remembered it nowhere. Two
file-choosing modes doing the same amount of work is one mode.

So the folder is the application's own, it ships empty, and this is where it is
filled. Adding a file **copies** it in. A library of paths into other people's
folders decays -- the measurement a run was judged against has to still be there
in a year, and a path is not a promise -- and the cost, two copies that can
drift, is the right way round: this copy cannot be edited out from under a
record.

**The codes list is the part that earns the screen.** A file answers to its own
serial and to its filename stem, and neither is necessarily what is printed on
the card. The i1Pro measurement of the ColorChecker SG in this repository is the
ordinary case: i1Profiler writes ``DESCRIPTOR "Default Measurement"`` and no
serial at all, so nothing inside it names the unit it describes, and the only
thing that can pair it with a capture is the code somebody binds to it. The
LaserSoft CxF is the other case -- it states ``T1910021`` and the card prints the
same thing, so it pairs on its own and needs nothing here.

The two kinds are shown apart and marked, because one is a fact about the file
and the other is a decision somebody made: a serial or a filename can only be
changed by changing the file, and offering to remove one here would be an offer
this cannot honour.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from deltachrome.errors import DeltachromeError
from deltachrome.reference.library import REFERENCE_SUFFIXES, ReferenceLibrary
from deltachrome_ui.paths import ensure_reference_bank, reference_bank_dir
from deltachrome_ui.theme import BG_INPUT, BORDER, TEXT_DIM
from deltachrome_ui.widgets import divider, field_row, heading, subtitle

#: The file dialog's filter, built from the suffixes the engine actually indexes
#: rather than from a second list that can drift from it.
_FILTER = (
    "Reference data (" + " ".join(f"*{s}" for s in REFERENCE_SUFFIXES) + ");;All files (*)"
)

#: How much of a reference file is shown. A CxF with 864 spectral patches is a
#: megabyte of XML and the panel is for confirming that a file is the one you
#: think it is, which the first screenful answers.
PREVIEW_BYTES = 200_000


class ReferenceBankPanel(QWidget):
    """The files in the bank, and the detail of whichever one is selected.

    A list with ``+`` and ``−`` under it rather than a table: a bank holds a
    department's charts, which is tens of files and not the thousand records a
    year the run library is sized for, and at that size nobody queries -- they
    look.
    """

    #: The bank changed on disk. The placement screen's reference chip reads the
    #: bank to decide whether pairing is even possible, so it has to be told.
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.library: ReferenceLibrary | None = None
        self._error: str | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        layout.addLayout(self._build_list(), 0)
        layout.addLayout(self._build_detail(), 1)
        self.reload()

    # -- construction ------------------------------------------------------

    def _build_list(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        self.list = QListWidget()
        self.list.setObjectName("BankList")
        self.list.setFrameShape(QListWidget.Shape.NoFrame)
        self.list.setFixedWidth(280)
        self.list.currentRowChanged.connect(lambda _row: self._show_selected())
        column.addWidget(self.list, 1)

        # Butted together under the list and flush with its left edge, which is
        # the shape every operating system uses for exactly this control. They
        # are the only two buttons in the application drawn as a pair like this,
        # and that is the point: it says "this list is editable" without a label.
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(0)
        self.add_button = QPushButton("+")
        self.add_button.setObjectName("SegmentLeft")
        self.add_button.setFixedWidth(34)
        self.add_button.setToolTip(
            "Add reference files. They are copied into the bank."
        )
        self.add_button.clicked.connect(self._add)
        buttons.addWidget(self.add_button)

        # Between the two, because it is the other way a file enters the bank and
        # it belongs beside the one that already does. A serial is written on the
        # chart; the target type it has to be looked up under is not, so the
        # dialog asks for the chart and never for that.
        self.fetch_button = QPushButton("↓")
        self.fetch_button.setObjectName("SegmentMiddle")
        self.fetch_button.setFixedWidth(34)
        self.fetch_button.setToolTip(
            "Fetch a measurement by serial number"
        )
        self.fetch_button.clicked.connect(self._fetch_by_serial)
        buttons.addWidget(self.fetch_button)

        self.remove_button = QPushButton("−")
        self.remove_button.setObjectName("SegmentRight")
        self.remove_button.setFixedWidth(34)
        self.remove_button.setEnabled(False)
        self.remove_button.setToolTip(
            "Delete from the bank. The original file is untouched."
        )
        self.remove_button.clicked.connect(self._remove)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        column.addLayout(buttons)
        return column

    def _build_detail(self) -> QVBoxLayout:
        """One container for the whole detail, plus the empty state beside it.

        Hidden wholesale rather than widget by widget. Hiding only the parts that
        hold data left the headings and the rule behind on an empty bank -- two
        section titles about a file that was not there, laid out around a gap.
        """
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.detail = QWidget()
        column = QVBoxLayout(self.detail)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)

        self.detail_name = QLabel("")
        self.detail_name.setStyleSheet("font-size: 14px; font-weight: 600;")
        column.addWidget(self.detail_name)

        self.facts = QVBoxLayout()
        self.facts.setContentsMargins(0, 0, 0, 0)
        self.facts.setSpacing(2)
        column.addLayout(self.facts)

        column.addWidget(divider())
        column.addWidget(heading("Codes That Pair With This File"))
        self.codes_note = subtitle("")
        self.codes_note.setWordWrap(True)
        column.addWidget(self.codes_note)

        self.codes = QListWidget()
        self.codes.setObjectName("BankList")
        self.codes.setFrameShape(QListWidget.Shape.NoFrame)
        self.codes.setMaximumHeight(96)
        self.codes.currentRowChanged.connect(self._on_code_selected)
        column.addWidget(self.codes)

        entry = QHBoxLayout()
        entry.setSpacing(8)
        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("Code printed on the chart…")
        self.code_input.returnPressed.connect(self._bind)
        entry.addWidget(self.code_input, 1)
        self.bind_button = QPushButton("Add Code")
        self.bind_button.setToolTip(
            "Targets printing this code will pair with this file."
        )
        self.bind_button.clicked.connect(self._bind)
        entry.addWidget(self.bind_button)
        self.unbind_button = QPushButton("Remove Code")
        self.unbind_button.setEnabled(False)
        self.unbind_button.clicked.connect(self._unbind)
        entry.addWidget(self.unbind_button)
        column.addLayout(entry)

        column.addWidget(heading("The File"))
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        # A floor rather than a preference: the point of showing the file is that
        # a person can confirm it is the one they think it is, and four lines of
        # a CGATS header is the part every reference file has in common.
        self.preview.setMinimumHeight(150)
        self.preview.setStyleSheet(
            f"QPlainTextEdit {{ background: {BG_INPUT}; border: 1px solid {BORDER};"
            f" border-radius: 4px; color: {TEXT_DIM};"
            " font-family: Menlo, monospace; font-size: 11px; }"
        )
        column.addWidget(self.preview, 1)

        outer.addWidget(self.detail, 1)

        empty = QWidget()
        empty_column = QVBoxLayout(empty)
        empty_column.setContentsMargins(0, 0, 0, 0)
        empty_column.setSpacing(0)
        self.empty = subtitle("")
        self.empty.setWordWrap(True)
        self.empty.setMaximumWidth(520)
        empty_column.addWidget(self.empty)
        empty_column.addStretch(1)
        self.empty_page = empty
        outer.addWidget(empty, 1)
        return outer

    # -- state -------------------------------------------------------------

    @property
    def selected(self) -> str | None:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def reload(self, *, select: str | None = None) -> None:
        """Re-index the bank and redraw. Safe on a folder that does not exist.

        The bank ships empty and is created on first sight, so "no such folder"
        is the ordinary state of a fresh install rather than a fault, and it must
        read as an empty list rather than as an error.
        """
        self._error = None
        try:
            self.library = ReferenceLibrary(ensure_reference_bank())
        except (DeltachromeError, OSError) as exc:
            self.library = None
            self._error = str(exc)

        wanted = select or self.selected
        self.list.blockSignals(True)
        self.list.clear()
        for entry in self.library.entries if self.library else []:
            item = QListWidgetItem(entry.name)
            item.setData(Qt.ItemDataRole.UserRole, entry.name)
            item.setToolTip(entry.describe())
            self.list.addItem(item)
        self.list.blockSignals(False)

        row = next(
            (i for i in range(self.list.count())
             if self.list.item(i).data(Qt.ItemDataRole.UserRole) == wanted),
            0 if self.list.count() else -1,
        )
        self.list.setCurrentRow(row)
        self._show_selected()

    def _entry(self):
        name = self.selected
        if self.library is None or name is None:
            return None
        return next((e for e in self.library.entries if e.name == name), None)

    # -- the detail --------------------------------------------------------

    def _show_selected(self) -> None:
        entry = self._entry()
        self.remove_button.setEnabled(entry is not None)
        self.detail.setVisible(entry is not None)
        self.empty_page.setVisible(entry is None)
        self._clear_facts()

        if entry is None:
            # An empty list, said once. The paragraph that used to be here
            # explained what a reference file is for, which is documentation and
            # not the empty state of a list -- and it appeared exactly when the
            # screen already had nothing else on it to read.
            self.empty.setText(self._error or "No reference files added.")
            return

        self.detail_name.setText(entry.name)
        added = self.library.added_on(entry.name) if self.library else None
        origin = self.library.origin_of(entry.name) if self.library else None
        rows = [
            ("Format", entry.format.upper(), ""),
            ("Serial", entry.serial or "— none stated —",
             "The serial stated inside the file."),
            ("Target", entry.target_type or "— not stated —",
             ""),
            ("Description", entry.description or "— none —", ""),
            ("Added", _added_text(added),
             "When the file was added to the bank, not when it was measured."),
        ]
        # Only for a file this application fetched. A file the operator added --
        # which is most of them -- came from a folder the bank deliberately does
        # not remember, so an empty row here would be a question with no answer
        # rather than a fact about the file.
        note = self.library.note_of(entry.name) if self.library else None
        if note:
            rows.append((
                "Caveat", "Names a different unit",
                note,
            ))
        if origin:
            rows.append((
                "Fetched from", _origin_text(origin),
                f"Downloaded from:\n{origin}",
            ))
        for name, value, tip in rows:
            self.facts.addWidget(field_row(name, value, tooltip=tip))

        self._show_codes(entry)
        self._show_preview(entry.path)

    def _clear_facts(self) -> None:
        while self.facts.count():
            item = self.facts.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _show_codes(self, entry) -> None:
        """Both kinds, marked apart.

        A code read out of the file or off its name is a fact and cannot be
        edited here; one bound in the manifest is a decision somebody made and
        can be undone. Listing them together unmarked would offer to remove
        something this cannot remove.
        """
        self.codes.blockSignals(True)
        self.codes.clear()
        for code in sorted(entry.aliases):
            item = QListWidgetItem(f"{code}      added here")
            item.setData(Qt.ItemDataRole.UserRole, code)
            self.codes.addItem(item)
        stated = sorted(entry.identifiers - {
            _normalized(code) for code in entry.aliases
        })
        for code in stated:
            item = QListWidgetItem(f"{code}      from the file")
            item.setData(Qt.ItemDataRole.UserRole, None)
            self.codes.addItem(item)
        self.codes.blockSignals(False)
        self.codes.setCurrentRow(-1)
        self.unbind_button.setEnabled(False)
        self.unbind_button.setToolTip("")
        self.codes_note.setText(
            "A capture is paired by reading its printed code and matching it "
            "against this list."
            if self.codes.count() else
            "This file answers to nothing, so no capture can pair with it. Add "
            "the code printed on the chart."
        )

    def _on_code_selected(self, row: int) -> None:
        item = self.codes.item(row) if row >= 0 else None
        removable = item is not None and item.data(Qt.ItemDataRole.UserRole)
        self.unbind_button.setEnabled(bool(removable))
        self.unbind_button.setToolTip(
            "Codes stated in the file can't be removed."
            if item is not None and not removable else ""
        )

    def _show_preview(self, path: Path) -> None:
        try:
            raw = path.open("rb").read(PREVIEW_BYTES)
            text = raw.decode("utf-8-sig", errors="replace")
        except OSError as exc:
            self.preview.setPlainText(f"{path.name} could not be read: {exc}")
            return
        if path.stat().st_size > PREVIEW_BYTES:
            text += f"\n\n… truncated at {PREVIEW_BYTES // 1000} kB."
        self.preview.setPlainText(text)

    # -- editing -----------------------------------------------------------

    def _add(self) -> None:
        chosen, _ = QFileDialog.getOpenFileNames(
            self, "Add reference files to the bank", "", _FILTER
        )
        if not chosen:
            return
        ensure_reference_bank()
        self.reload()
        if self.library is None:
            self._warn("The bank could not be opened.", self._error or "")
            return
        added, failed = None, []
        for path in chosen:
            try:
                added = self.library.add_file(path)
            except (DeltachromeError, OSError) as exc:
                failed.append(f"{Path(path).name}: {exc}")
        self.reload(select=added.name if added else None)
        self.changed.emit()
        if failed:
            self._warn(
                "Some files were not added.", "\n\n".join(failed)
            )

    def _fetch_by_serial(self) -> None:
        dialog = FetchBySerialDialog(self)
        if dialog.exec() and dialog.fetched is not None:
            self.reload(select=dialog.fetched)

    def _remove(self) -> None:
        entry = self._entry()
        if entry is None or self.library is None:
            return
        confirm = QMessageBox(self)
        confirm.setIcon(QMessageBox.Icon.Question)
        confirm.setText(f"Delete {entry.name} from the bank?")
        # Said plainly, because the answer is the whole reason this is only a
        # confirmation and not a trash folder: the bank holds a copy.
        confirm.setInformativeText(
            "The bank holds a copy, so the original where it came from is not "
            "touched. Any codes bound to it here are deleted with it."
        )
        confirm.setStandardButtons(
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes
        )
        confirm.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if confirm.exec() != QMessageBox.StandardButton.Yes:
            return
        try:
            self.library.remove_file(entry.name)
        except (DeltachromeError, OSError) as exc:
            self._warn(f"{entry.name} was not deleted.", str(exc))
        self.reload()
        self.changed.emit()

    def _bind(self) -> None:
        entry = self._entry()
        code = self.code_input.text().strip()
        if entry is None or self.library is None or not code:
            return
        try:
            self.library.bind(code, entry.name)
        except (DeltachromeError, OSError) as exc:
            self._warn("That code was not bound.", str(exc))
            return
        self.code_input.clear()
        self.reload(select=entry.name)
        self.changed.emit()

    def _unbind(self) -> None:
        entry = self._entry()
        item = self.codes.currentItem()
        code = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if entry is None or self.library is None or not code:
            return
        try:
            self.library.unbind(code)
        except (DeltachromeError, OSError) as exc:
            self._warn("That binding was not removed.", str(exc))
            return
        self.reload(select=entry.name)
        self.changed.emit()

    def _warn(self, text: str, detail: str) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(text)
        box.setInformativeText(detail)
        box.exec()


def _origin_text(url: str) -> str:
    """Where a fetched file came from, as the host rather than the whole URL.

    A label's URL is around a hundred characters of query string, of which the
    part a reader wants is the host that published the measurement. The whole
    thing is on hover, where it can be read without setting the row's width.
    """
    from urllib.parse import urlsplit

    host = urlsplit(url).netloc
    return host or url


def _added_text(value: str | None) -> str:
    """An ISO timestamp as a date a person reads, or an honest absence.

    A file put in the folder by hand has no arrival date and never will, and
    inventing one from its mtime would state a fact this does not know.
    """
    if not value:
        return "— not recorded —"
    from datetime import datetime

    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y, %H:%M")
    except ValueError:
        return value


def _normalized(code: str) -> str:
    from deltachrome.imaging.codes import normalize_identifier

    return normalize_identifier(code)


def bank_count() -> int:
    """How many reference files the bank holds.

    Read by the placement screen, which offers *Pair from library* and needs two
    things from this: whether pairing can work at all -- zero is a dead end, and
    should say so before a run rather than after a decode -- and a line saying
    what the bank holds, which is more use there than the folder's path.

    Counted by suffix rather than by building a :class:`ReferenceLibrary`, since
    this runs on every change of the reference mode and indexing reads the head
    of every file in the folder.
    """
    directory = reference_bank_dir()
    if not directory.is_dir():
        return 0
    return sum(
        1 for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in REFERENCE_SUFFIXES
    )


class BankPicker(QDialog):
    """Choose a file from the bank by hand, when the code cannot choose it.

    *Pair from library* is meant to open nothing, and normally it does: the
    capture's printed code names its own unit's file and the operator is told
    which, not asked. But three real outcomes leave the code unable to decide --
    the chart prints none, none of the decoders read the one it prints, or the
    code answers to no file in the bank -- and until now every one of them ended
    the same way, in a ``MissingReferenceError`` after the decode.

    So this is the fallback, and it is deliberately *not* the file dialog beside
    it. **Choose a file…** opens the whole filesystem, which is right for a
    measurement that is not in the bank and wrong for one that is: the bank holds
    the copies this application manages, and picking one of those out of a folder
    listing means reading past ``deltachrome-references.json`` and whatever else
    the directory holds. Here a file is a row with what it answers to beside it,
    which is the fact that decides whether it is the right one.

    The codes read from *this* capture are shown at the top when there were any,
    because "the card says T1910021 and nothing in the bank does" is the sentence
    that tells an operator what to fix.

    **And the fix is offered here rather than one screen away.** Where a code was
    read, a checkbox binds it to whichever file is chosen, which is the whole of
    what Settings' *Add Code* does and is the one moment somebody has both facts
    in front of them: the code this capture carries, and the file it belongs to.
    Doing it once means the chart pairs itself from then on. It is a checkbox and
    not automatic because a binding is a claim about a physical object -- that
    this printed code names the unit this file measures -- and the application
    cannot make it.

    The LaserSoft ISO 12641-2 was the case that prompted it -- its barcode reads
    ``1930021`` against ``T1910021`` in its own CxF -- until that turned out to be
    LaserSoft's encoding of the serial, which the bank now applies itself
    (:func:`deltachrome.reference.library.lasersoft_barcode`). What still needs a
    person is a code nothing explains: a file stating no serial, a chart whose
    code is not its serial for reasons nobody has worked out.
    """

    def __init__(self, codes: list[str] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reference Files")
        self.setMinimumWidth(460)
        self._chosen: Path | None = None
        self._codes = [c for c in (codes or []) if c]

        column = QVBoxLayout(self)
        column.setContentsMargins(18, 16, 18, 16)
        column.setSpacing(10)
        column.addWidget(heading("Choose From the Bank"))

        read = [c for c in (codes or []) if c]
        note = subtitle(
            "This capture reads " + ", ".join(read) + ", which no file in the bank "
            "answers to. Choose the one measured from the chart that was in the "
            "frame, and it will pair itself from now on."
            if read else
            "No code was read from this capture, so it cannot pair itself. "
            "Reference data is per physical unit -- choose the file measured "
            "from the chart that was in the frame."
        )
        note.setWordWrap(True)
        column.addWidget(note)

        self.list = QListWidget()
        self.list.setObjectName("BankList")
        self.list.setFrameShape(QListWidget.Shape.NoFrame)
        self.list.setMinimumHeight(220)
        self.list.itemDoubleClicked.connect(lambda _item: self.accept())
        column.addWidget(self.list, 1)

        try:
            library = ReferenceLibrary(ensure_reference_bank())
        except (DeltachromeError, OSError):
            library = None
        for entry in (library.entries if library else []):
            answers = ", ".join(sorted(entry.identifiers)) or "no code"
            item = QListWidgetItem(f"{entry.name}      answers to: {answers}")
            item.setData(Qt.ItemDataRole.UserRole, str(entry.path))
            item.setToolTip(entry.describe())
            self.list.addItem(item)

        self.remember = QCheckBox(
            "Remember " + ", ".join(read) + " for this file"
            if read else ""
        )
        self.remember.setToolTip(
            "This target will pair automatically from now on."
        )
        self.remember.setVisible(bool(read))
        self.remember.setChecked(bool(read))
        column.addWidget(self.remember)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        column.addWidget(buttons)

    def accept(self) -> None:  # noqa: D102 - Qt's name
        item = self.list.currentItem()
        if item is None:
            return
        self._chosen = Path(item.data(Qt.ItemDataRole.UserRole))
        if self._codes and self.remember.isChecked():
            self._bind_codes()
        super().accept()

    def _bind_codes(self) -> None:
        """Write the bindings. A failure here must not lose the choice.

        The file the operator picked is still the right file whether or not the
        manifest could be written -- a read-only bank, a folder on a share that
        went away -- so this reports and returns rather than refusing the dialog.
        """
        try:
            library = ReferenceLibrary(ensure_reference_bank())
            for code in self._codes:
                library.bind(code, self._chosen.name)
        except (DeltachromeError, OSError) as exc:
            QMessageBox.warning(
                self, "Code not remembered",
                f"{self._chosen.name} is still the reference for this run. The "
                f"code could not be bound to it for next time:\n\n{exc}",
            )

    @property
    def chosen(self) -> Path | None:
        return self._chosen


class FetchBySerialDialog(QDialog):
    """Fetch one unit's published measurement by its serial number.

    The other half of pairing. A chart carrying an OpenQualia label says where
    its own measurement lives and the placement screen offers to fetch it; this
    is for the two cases that leaves out, and between them they are most charts
    in most rooms. A Golden Thread made before 2025 prints a serial and no label,
    and there are 3,170 of those published. And a bank is better filled before a
    shoot than during one, when nobody wants to wait on a download.

    **The operator supplies a serial and nothing else.** The manufacturer and the
    target type come from the chosen target's own definition, which is the whole
    reason this could be built: the host offers no listing to pick a target type
    from, matches it case- and whitespace-agnostically but not
    dash-agnostically, and uses different spellings in its web interface from its
    API -- ``OLT-1x`` browses and 404s where ``OLT1x`` resolves. Asking somebody
    to type that would be asking them to guess. A serial is written on the chart.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fetch Measurement")
        # Wide enough for the longest target name this can offer. A combo that
        # elides the chart somebody is choosing is the one control here that
        # must not.
        self.setMinimumWidth(560)
        #: The filename that landed in the bank, so the caller can select it.
        self.fetched: str | None = None
        self._running = False

        from deltachrome.reference.openqualia import publishes_measurements
        from deltachrome.targets.registry import available_targets, load_target
        from deltachrome_ui.widgets import ComboBox

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(10)
        layout.addWidget(heading("Fetch Measurement"))
        layout.addWidget(subtitle(
            "Only charts whose maker publishes per-unit measurements are listed. "
            "The serial is the one printed on the chart."
        ))

        self.target_combo = ComboBox()
        for target_id in available_targets():
            target = load_target(target_id)
            if publishes_measurements(target):
                self.target_combo.addItem(target.display_name, target_id)
        layout.addWidget(self.target_combo)

        self.serial = QLineEdit()
        self.serial.setPlaceholderText("Serial number, e.g. D00011")
        self.serial.textChanged.connect(self._update)
        self.serial.returnPressed.connect(self._fetch)
        layout.addWidget(self.serial)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        layout.addWidget(self.status)

        self.buttons = QDialogButtonBox()
        self.fetch_button = self.buttons.addButton(
            "Fetch", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self._fetch)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.target_combo.currentIndexChanged.connect(lambda _i: self._update())
        self._update()

    # -- state ------------------------------------------------------------

    def _label(self):
        from deltachrome.reference.openqualia import label_for_serial
        from deltachrome.targets.registry import load_target

        target_id = self.target_combo.currentData()
        if not target_id:
            return None
        return label_for_serial(load_target(target_id), self.serial.text())

    def _update(self) -> None:
        label = self._label()
        self.fetch_button.setEnabled(label is not None and not self._running)
        if self._running:
            return
        if label is None:
            self.status.setText(
                "Enter the serial printed on the chart."
                if not self.serial.text().strip()
                else "A serial is letters, digits and dashes only."
            )
        else:
            self.status.setText(f"Will ask {label.host} for {label.describe()}.")

    # -- fetching ---------------------------------------------------------

    def _fetch(self) -> None:
        from deltachrome_ui.workers import MeasurementFetcher, run_in_thread

        label = self._label()
        if label is None or self._running:
            return
        self._running = True
        self.fetch_button.setEnabled(False)
        self.status.setText(f"Asking {label.host}…")
        worker = MeasurementFetcher(label, ensure_reference_bank())
        worker.loaded.connect(self._on_loaded)
        worker.failed.connect(self._on_failed)
        run_in_thread(worker, self)

    def _on_loaded(self, entry, label, caveat=None) -> None:
        self._running = False
        self.fetched = entry.name
        if caveat:
            # Shown before the dialog closes, and kept with the file besides, so
            # it is not a notice that flashes past once.
            QMessageBox.warning(self, "The measurement names a different unit", caveat)
        self.accept()

    def _on_failed(self, failure) -> None:
        self._running = False
        # Reported in the dialog rather than in a box over it: the operator is
        # most likely to have mistyped a serial, and the fix is in the field they
        # are already looking at.
        self.status.setText(failure.detail)
        self.fetch_button.setEnabled(True)
        # A refusal runs to several lines where the ready state is one, so the
        # dialog grows to hold it. Sized to the short text it would clip the long
        # one, and a message an operator cannot read is the same as none.
        self.adjustSize()
