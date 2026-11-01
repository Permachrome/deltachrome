"""The guidelines a site has made, and the editor for whichever is selected.

The same shape as the reference bank next door, and deliberately so: a list on
the left with ``+`` and ``−`` under it, the thing itself on the right. Both hold
a handful of items a person looks at rather than the thousand records a year the
run library is sized to be queried for, and two lists on one screen behaving
differently would be a worse reason to differ than any of the small reasons to.

**What is not here is a list of the published standards.** FADGI's fifty-five
profiles and Metamorfoze's three ship with the application, are not editable, and
are chosen in the run setup by the control built for choosing them. Listing them
here would put a delete button beside a standard.

**A guideline being made has a row of its own**, named *New Guideline* until it is
given a name. That is not decoration. It was represented by *no selection* --
``setCurrentRow(-1)`` -- and `QAbstractItemView` restores the current row to the
first item when a view with none regains focus, which is exactly what a click on
the button beside the list causes. So pressing ``+`` built the new form and Qt
immediately put the first saved guideline back over it: the screen flickered and
nothing happened. With a real row there is never a moment without a current one,
and nothing to restore.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from deltachrome.errors import DeltachromeError
from deltachrome_ui.guideline_builder import (
    GuidelineBuilder,
    delete_guideline,
    local_guidelines,
)
from deltachrome_ui.widgets import subtitle


class GuidelinesPanel(QWidget):
    """A list of local guidelines, and the builder beside it."""

    changed = Signal()

    #: Marks the row for a guideline that does not exist yet.
    CREATING = "\u0000creating"

    def __init__(self) -> None:
        super().__init__()
        self._profiles = []
        self._creating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        layout.addLayout(self._build_list(), 0)
        layout.addLayout(self._build_editor(), 1)
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
        self.list.currentRowChanged.connect(self._on_selected)
        column.addWidget(self.list, 1)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(0)
        self.add_button = QPushButton("+")
        self.add_button.setObjectName("SegmentLeft")
        self.add_button.setFixedWidth(34)
        self.add_button.setToolTip("New guideline")
        self.add_button.clicked.connect(self._new)
        buttons.addWidget(self.add_button)

        self.remove_button = QPushButton("−")
        self.remove_button.setObjectName("SegmentRight")
        self.remove_button.setFixedWidth(34)
        self.remove_button.setEnabled(False)
        self.remove_button.setToolTip(
            "Delete this guideline. Runs already judged against it keep "
            "their verdicts."
        )
        self.remove_button.clicked.connect(self._remove)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        column.addLayout(buttons)
        return column

    def _build_editor(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        self.builder = GuidelineBuilder()
        self.builder.changed.connect(self._on_saved)
        self.builder.name.textChanged.connect(self._rename_new_row)
        column.addWidget(self.builder, 1)

        empty = QWidget()
        empty_column = QVBoxLayout(empty)
        empty_column.setContentsMargins(0, 0, 0, 0)
        empty_column.setSpacing(0)
        self.empty = subtitle("No guidelines added.")
        self.empty.setWordWrap(True)
        self.empty.setMaximumWidth(520)
        empty_column.addWidget(self.empty)
        empty_column.addStretch(1)
        self.empty_page = empty
        column.addWidget(empty, 1)
        return column

    # -- state -------------------------------------------------------------

    @property
    def selected(self) -> str | None:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def reload(self, *, select: str | None = None) -> None:
        """Re-read the folder, keeping whatever was selected -- including a
        guideline that is being made and is not in the folder yet."""
        if select is not None:
            self._creating = False
        wanted = select or self.selected
        try:
            self._profiles = local_guidelines()
        except DeltachromeError:
            self._profiles = []

        self.list.blockSignals(True)
        self.list.clear()
        for profile in self._profiles:
            item = QListWidgetItem(profile.display_name)
            item.setData(Qt.ItemDataRole.UserRole, profile.profile_id)
            # No tooltip. It carried the departure summary, which opens with the
            # parent standard's name -- so hovering a saved guideline named it as
            # somebody else's table with amendments. What this guideline says is
            # the table beside the list, one click away and already open on the
            # selected row.
            self.list.addItem(item)
        if self._creating:
            self.list.addItem(self._new_row_item())
        self.list.blockSignals(False)

        row = next(
            (i for i in range(self.list.count())
             if self.list.item(i).data(Qt.ItemDataRole.UserRole) == wanted),
            0 if self.list.count() else -1,
        )
        self.list.setCurrentRow(row)
        self._on_selected(row)

    def _new_row_item(self) -> QListWidgetItem:
        name = self.builder.name.text().strip()
        item = QListWidgetItem(name or "New Guideline")
        item.setData(Qt.ItemDataRole.UserRole, self.CREATING)
        item.setToolTip("Not saved yet.")
        return item

    def _rename_new_row(self) -> None:
        """Keep the temporary row's label with the name being typed."""
        if not self._creating:
            return
        row = self.list.count() - 1
        item = self.list.item(row)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == self.CREATING:
            item.setText(self.builder.name.text().strip() or "New Guideline")

    def _on_selected(self, row: int) -> None:
        item = self.list.item(row) if row >= 0 else None
        key = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        creating = key == self.CREATING
        profile = self._profiles[row] if 0 <= row < len(self._profiles) else None

        # Moving off the row for an unsaved guideline abandons it, and the row
        # goes with it. Keeping the row would offer a second entry that is not a
        # guideline and cannot be deleted.
        if self._creating and not creating:
            self._creating = False
            self._drop_new_row()
            profile = self._profiles[row] if 0 <= row < len(self._profiles) else None

        self.remove_button.setEnabled(profile is not None)
        self.builder.setVisible(creating or profile is not None)
        self.empty_page.setVisible(not (creating or profile is not None))
        if profile is not None and not creating:
            self.builder.edit(profile)

    def _drop_new_row(self) -> None:
        for row in range(self.list.count() - 1, -1, -1):
            item = self.list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == self.CREATING:
                self.list.blockSignals(True)
                self.list.takeItem(row)
                self.list.blockSignals(False)

    # -- actions -----------------------------------------------------------

    def _new(self) -> None:
        """Start one, and give it a row so the list is never without a selection."""
        if self._creating:
            self._drop_new_row()
        self._creating = True
        self.builder.start_new()
        self.list.blockSignals(True)
        self.list.addItem(self._new_row_item())
        self.list.setCurrentRow(self.list.count() - 1)
        self.list.blockSignals(False)
        self.remove_button.setEnabled(False)
        self.builder.setVisible(True)
        self.empty_page.setVisible(False)

    def _remove(self) -> None:
        row = self.list.currentRow()
        if not 0 <= row < len(self._profiles):
            return
        profile = self._profiles[row]
        confirm = QMessageBox(self)
        confirm.setIcon(QMessageBox.Icon.Question)
        confirm.setText(f"Delete {profile.display_name}?")
        # Said, because it is the consequence nobody thinks of: a run record
        # names the guideline it was judged against, and the verdict in it stands
        # whatever happens here -- but re-scoring against this one will not.
        confirm.setInformativeText(
            "Runs already judged against it keep their verdicts and their "
            "records, and will name a guideline this application can no longer "
            "find. Nothing can be judged against it again."
        )
        confirm.setStandardButtons(
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes
        )
        confirm.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if confirm.exec() != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_guideline(profile)
        except (DeltachromeError, OSError) as exc:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setText(f"{profile.display_name} was not deleted.")
            box.setInformativeText(str(exc))
            box.exec()
        self._creating = False
        self.builder.start_new()
        self.builder.setVisible(False)
        self.reload()
        self.changed.emit()

    def _on_saved(self) -> None:
        saved = self.builder.editing
        self.reload(select=saved.profile_id if saved else None)
        self.changed.emit()
