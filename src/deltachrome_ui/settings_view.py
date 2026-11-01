"""Settings: the things that are true between runs rather than about one.

The line this screen has to hold is the one `options_view.py` already states from
the other side. **Results travel with the settings that produced them**, so
anything that changes an answer belongs beside the run and gets written into the
record. What is left over is everything that changes *where things are* and
*what the application does by default* -- and that had nowhere to go, so it
accumulated: the run folder became a header icon of its own, the reference bank
was a directory dialog on every run, and the settings square emitted a signal
nobody had connected.

**All three folders can be moved, and none of them is the same kind of thing.**
A run record is an archive document: it cannot be regenerated except by finding
the master and placing it again by hand, and a department may quite reasonably
want a project's records to live with the project. The guidelines folder is the
specification a department works to, and a shared drive is a reasonable place for
it. The reference bank holds *copies* this application made and manages, whose
originals are wherever they came from -- so pointing it elsewhere shows a
**different bank** rather than moving anything, which is what that row's own
explanation says. That is why the control can exist here and could not exist
beside the list it would empty.

**And all three default under one visible folder**, ``~/Documents/Deltachrome``
(see :mod:`deltachrome.home`), so an installation nobody has pointed anywhere
keeps a person's work in one place they can find and back up. A folder chosen here is remembered -- but a remembered
folder that is no longer there is *not* used, because a directory is created on
demand wherever these point and a stale path would be silently recreated, leaving
an empty library at an address the operator does not recognise.

``DELTACHROME_RUN_DIR`` and ``DELTACHROME_REFERENCE_DIR`` pin a folder for a
scripted or shared-drive deployment, and where one is set that row's Change
button is disabled with the reason on hover -- which is how every other
conditional control in this application says it, rather than with a chip naming
an environment variable at somebody who either set it themselves or has never
heard of it.

**A left list, not a strip of tabs.** Three sections is where one page stopped
working, and the Guideline Builder is what did it: a parent to fork, a threshold
per metric and the record of where each departs is a screen's worth on its own.
The list is the one the report already uses (``QListWidget#Nav``), and the reason
it is not a tab strip along the top is this application's own icon rule -- a glyph
names *where it goes*, and it has already been found that abstractions do not
survive the size, a gear at 22 px reading as a sun. "Guidelines", "Reference
Files" and "Run Library" are categories with no picture between them, so an
icon-only strip would be three shapes nobody can tell apart, and icon-above-label
wants about 64 px where ``BAR_HEIGHT`` is 52. A row in a left list has room for
both, and the word is the load-bearing half.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from deltachrome.profiles.schema import PROFILE_DIR_ENV, user_profile_dir
from deltachrome.profiling.colprof import ENVIRONMENT_VARIABLE
from deltachrome_ui import paths
from deltachrome_ui.about_panel import AboutPanel
from deltachrome_ui.guidelines_panel import GuidelinesPanel
from deltachrome_ui.reference_bank import ReferenceBankPanel
from deltachrome_ui.targets_panel import TargetsPanel
from deltachrome_ui.theme import BAR_HEIGHT
from deltachrome_ui.widgets import (
    IconButton,
    divider,
    elided_subtitle,
    heading,
    mark_cell,
)

#: The sections, in the order they are listed. Declared here rather than built
#: inline so the nav and the stack cannot fall out of step -- one list, walked
#: twice.
SECTIONS = ("General", "Targets", "Reference Files", "Guidelines", "About")


class SettingsView(QWidget):
    """Where records are kept, owned targets, the bank, the guidelines, and About."""

    backRequested = Signal()
    #: The run library folder changed. The library view has to re-read it, and
    #: only the window knows that both exist.
    runDirChanged = Signal()
    #: The reference bank changed -- a file added, removed, or the folder moved.
    referencesChanged = Signal()
    #: A guideline was saved or deleted. Every profile control in the window was
    #: built from the catalogue when it was constructed, so they have to be told.
    guidelinesChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_bar())
        layout.addWidget(divider())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        nav = self._build_nav()

        # No page scrolls as a whole. Everything on these pages that can outgrow
        # the window scrolls inside itself -- the file list, the file's own text,
        # the metric rows -- and an outer scroll on top of those put the one
        # control that says a list is editable below the fold, which is the same
        # bug the placement panel had with Run Analysis.
        self.pages = QStackedWidget()
        self.pages.addWidget(self._page(self._build_folders()))
        self.pages.addWidget(self._page(self._build_targets()))
        self.pages.addWidget(self._page(self._build_bank()))
        self.pages.addWidget(self._page(self._build_guidelines()))
        self.pages.addWidget(self._page(self._build_about()))
        # The nav is built first and connected last: it drives the stack, which
        # does not exist until the pages are made.
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        body.addWidget(nav)
        body.addWidget(self.pages, 1)
        layout.addLayout(body, 1)
        self.nav.setCurrentRow(0)

    def _page(self, column: QVBoxLayout) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(28, 22, 28, 28)
        outer.setSpacing(0)
        outer.addLayout(column, 1)
        return page

    # -- construction ------------------------------------------------------

    def _build_bar(self) -> QWidget:
        """The mark, the name, and the way out -- and the name is centred.

        The mark is here for the same reason it is on the library and nowhere
        else: those are the two screens about the *application* rather than about
        a run, and the other three spend their left edge on a verdict or a
        capture's name. Nothing here is about a particular capture, so there is
        also no second line -- the bar's usual subtitle is the file being worked
        on, and a settings screen saying what a settings screen is for is a label
        on a label.

        Centred by giving the mark a cell the same width as the action square, so
        the two sides balance exactly rather than by five pixels of luck.
        """
        bar = QWidget()
        bar.setObjectName("HeaderBar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setFixedHeight(BAR_HEIGHT)
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        row.addWidget(mark_cell())

        row.addStretch(1)
        name = QLabel("Settings")
        name.setStyleSheet("font-size: 14px; font-weight: 600;")
        row.addWidget(name, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)

        back = IconButton("back", "Back to the library", size=BAR_HEIGHT)
        back.clicked.connect(self.backRequested.emit)
        row.addWidget(back)
        return bar

    def _build_nav(self) -> QWidget:
        self.nav = QListWidget()
        self.nav.setObjectName("SettingsNav")
        self.nav.setFrameShape(QListWidget.Shape.NoFrame)
        self.nav.setFixedWidth(196)
        for name in SECTIONS:
            self.nav.addItem(QListWidgetItem(name))
        return self.nav

    def _build_folders(self) -> QVBoxLayout:
        """Where each of the three kinds of thing is kept.

        All three together rather than each beside the thing it holds, which is
        what a nav makes possible and what a single page did not: one place to
        look for "where does this live", and the other two pages left to be about
        their contents.

        **The reference bank is changeable again**, which reverses a decision made
        two rounds ago on the operator's asking, and the reasoning holds up. It was
        removed because moving it buys states with no good reading -- point it
        elsewhere and every chart disappears -- and because a Change button beside
        the bank contradicts the one thing that mode is for. Neither of those is
        about the *capability*; both are about where the control sits. Under
        General it is somewhere a person arrives deliberately, next to two other
        folders, rather than beside the list it would empty.
        """
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        column.addWidget(heading("Folders"))
        self.run_row = _FolderRow(
            "Run Library", paths.RUN_DIR_ENV, self._choose_run_dir,
            "Where saved analyses are kept.",
            on_default=self._default_run_dir,
            default=paths.default_run_dir(),
        )
        column.addWidget(self.run_row)
        self.reference_row = _FolderRow(
            "Reference Files", paths.REFERENCE_DIR_ENV, self._choose_reference_dir,
            "Where reference files are kept. Changing it shows a different "
            "bank and moves no files.",
            on_default=self._default_reference_dir,
            default=paths.default_reference_dir(),
        )
        column.addWidget(self.reference_row)
        self.guideline_row = _FolderRow(
            "Guidelines", PROFILE_DIR_ENV, self._choose_guideline_dir,
            "Where your guidelines are kept. Can be a shared drive.",
            on_default=self._default_guideline_dir,
            default=paths.default_guideline_dir(),
        )
        column.addWidget(self.guideline_row)

        column.addSpacing(18)
        column.addWidget(heading("Camera Profiles"))
        self.colprof_row = _FileRow(
            "colprof", ENVIRONMENT_VARIABLE, self._choose_colprof,
            "ArgyllCMS's colprof, which builds camera profiles.",
            on_default=self._default_colprof,
        )
        column.addWidget(self.colprof_row)
        copyright_row = QHBoxLayout()
        copyright_row.setSpacing(14)
        label = QLabel("Copyright")
        label.setFixedWidth(120)
        label.setStyleSheet("font-size: 13px;")
        copyright_row.addWidget(label)
        self.copyright_edit = QLineEdit()
        self.copyright_edit.setPlaceholderText("Copyright written into profiles")
        self.copyright_edit.editingFinished.connect(
            lambda: paths.remember(paths.PROFILE_COPYRIGHT_KEY,
                                   self.copyright_edit.text().strip())
        )
        copyright_row.addWidget(self.copyright_edit, 1)
        column.addLayout(copyright_row)
        column.addStretch(1)
        return column

    def _build_targets(self) -> QVBoxLayout:
        """The targets this operator owns. See ``targets_panel``.

        Here and not beside the run because it changes what is *suggested*, never
        what is measured -- and a run whose chart was named from a narrowed set
        records the set, so it is not an anonymous preference either.
        """
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(heading("Target Selection"))
        self.targets = TargetsPanel()
        column.addWidget(self.targets, 1)
        return column

    def _build_bank(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(heading("Reference Files"))
        self.bank = ReferenceBankPanel()
        self.bank.changed.connect(self.referencesChanged.emit)
        column.addWidget(self.bank, 1)
        return column

    def _build_guidelines(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(heading("Guidelines"))
        self.guidelines = GuidelinesPanel()
        self.guidelines.changed.connect(self.guidelinesChanged.emit)
        column.addWidget(self.guidelines, 1)
        return column

    def _build_about(self) -> QVBoxLayout:
        """Last, being the one section nobody visits to change anything."""
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(heading("About"))
        self.about = AboutPanel()
        column.addWidget(self.about, 1)
        return column

    # -- state -------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read everything this screen shows. Called on the way in.

        The folders can be changed from elsewhere -- an environment variable, or
        the library's own re-read -- and a settings screen showing a stale path
        is worse than one that is slow to open.
        """
        self.run_row.set_path(paths.user_run_dir())
        self.reference_row.set_path(paths.reference_bank_dir())
        self.guideline_row.set_path(user_profile_dir())
        self.colprof_row.set_file(paths.resolve_colprof(),
                                  chosen=paths.remembered_colprof() is not None)
        self.copyright_edit.setText(paths.remembered(paths.PROFILE_COPYRIGHT_KEY, "") or "")
        self.targets.reload()
        self.bank.reload()
        self.guidelines.reload()

    # -- actions -----------------------------------------------------------

    def _choose_colprof(self) -> None:
        current = paths.resolve_colprof()
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Where is colprof?", str(current.parent if current else Path.home())
        )
        if not chosen:
            return
        paths.remember(paths.COLPROF_KEY, chosen)
        self.refresh()

    def _default_colprof(self) -> None:
        paths.remember(paths.COLPROF_KEY, None)
        self.refresh()

    def _choose_run_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Run library folder", str(paths.user_run_dir())
        )
        if not chosen:
            return
        paths.set_run_dir(Path(chosen))
        self.run_row.set_path(paths.user_run_dir())
        self.runDirChanged.emit()

    def _default_run_dir(self) -> None:
        paths.set_run_dir(paths.default_run_dir())
        self.run_row.set_path(paths.user_run_dir())
        self.runDirChanged.emit()

    def _default_reference_dir(self) -> None:
        paths.set_reference_bank(paths.default_reference_dir())
        self.reference_row.set_path(paths.reference_bank_dir())
        self.bank.reload()
        self.referencesChanged.emit()

    def _default_guideline_dir(self) -> None:
        paths.remember_guideline_dir(paths.default_guideline_dir())
        self.guideline_row.set_path(user_profile_dir())
        self.colprof_row.set_file(paths.resolve_colprof(),
                                  chosen=paths.remembered_colprof() is not None)
        self.copyright_edit.setText(paths.remembered(paths.PROFILE_COPYRIGHT_KEY, "") or "")
        self.guidelines.reload()
        self.guidelinesChanged.emit()

    def _choose_reference_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Reference files folder", str(paths.reference_bank_dir())
        )
        if not chosen:
            return
        paths.set_reference_bank(Path(chosen))
        self.reference_row.set_path(paths.reference_bank_dir())
        self.bank.reload()
        self.referencesChanged.emit()

    def _choose_guideline_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Guidelines folder", str(user_profile_dir())
        )
        if not chosen:
            return
        paths.remember_guideline_dir(Path(chosen))
        self.guideline_row.set_path(user_profile_dir())
        self.colprof_row.set_file(paths.resolve_colprof(),
                                  chosen=paths.remembered_colprof() is not None)
        self.copyright_edit.setText(paths.remembered(paths.PROFILE_COPYRIGHT_KEY, "") or "")
        self.guidelines.reload()
        self.guidelinesChanged.emit()


class _FileRow(QWidget):
    """A named program, where it was found, and a way to point at another."""

    def __init__(self, name: str, env_var: str, on_choose, explanation: str,
                 on_default=None) -> None:
        super().__init__()
        self._env_var = env_var
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(14)
        label = QLabel(name)
        label.setFixedWidth(120)
        label.setStyleSheet("font-size: 13px;")
        label.setToolTip(explanation)
        row.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
        self.path_label = elided_subtitle("")
        self.path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        row.addWidget(self.path_label, 1)
        self.reset = QPushButton("Default")
        self.reset.setToolTip(f"{env_var}, then the PATH.")
        if on_default is not None:
            self.reset.clicked.connect(on_default)
        row.addWidget(self.reset, 0, Qt.AlignmentFlag.AlignVCenter)
        self.choose = QPushButton("Change…")
        self.choose.clicked.connect(on_choose)
        row.addWidget(self.choose, 0, Qt.AlignmentFlag.AlignVCenter)

    def set_file(self, path: Path | None, *, chosen: bool) -> None:
        text = str(path) if path is not None else "Not found"
        self.path_label.setText(text)
        self.path_label.setToolTip(text)
        self.reset.setVisible(chosen)


class _FolderRow(QWidget):
    """A named folder, where it currently resolves to, and a way to move it.

    The path is shown elided with the whole of it on hover, the way the library's
    own bar shows it: a folder can sit anywhere and a settings row is not the
    place to wrap a filesystem path.
    """

    def __init__(self, name: str, env_var: str, on_choose, explanation: str,
                 on_default=None, default: Path | None = None) -> None:
        super().__init__()
        self._env_var = env_var
        self._default = default
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(14)

        label = QLabel(name)
        label.setFixedWidth(120)
        label.setStyleSheet("font-size: 13px;")
        label.setToolTip(explanation)
        row.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)

        self.path_label = elided_subtitle("")
        self.path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        row.addWidget(self.path_label, 1)

        # The way back. A folder chosen once is remembered for good, so without
        # this the only route to what the application shipped with is for the
        # operator to know where that was and navigate to it in a file dialog --
        # which means knowing that it is ``~/Documents/Deltachrome`` and that the folder
        # may not exist yet. Shown only while it would do something.
        self.reset = QPushButton("Default")
        self.reset.setToolTip(str(self._default) if self._default else "")
        if on_default is not None:
            self.reset.clicked.connect(on_default)
        row.addWidget(self.reset, 0, Qt.AlignmentFlag.AlignVCenter)

        self.choose = QPushButton("Change…")
        self.choose.clicked.connect(on_choose)
        row.addWidget(self.choose, 0, Qt.AlignmentFlag.AlignVCenter)

    def set_path(self, path: Path) -> None:
        import os

        self.path_label.setText(str(path))
        self.path_label.setToolTip(str(path))
        # Hidden rather than disabled, which is the opposite of what the pinned
        # case below does, and deliberately: a disabled *Change…* is answering
        # "why can I not move this", which is a question the operator is asking.
        # Nobody standing on the default is asking how to get back to it.
        self.reset.setVisible(
            self._default is not None and Path(path) != self._default
            and not os.environ.get(self._env_var)
        )
        # A control that cannot take effect must not look like one that can, and
        # a disabled button with the reason on hover is how this application says
        # that everywhere else. A chip naming the variable was tried and is worse:
        # it appears only for the person who set it, who therefore knows.
        pinned = bool(os.environ.get(self._env_var))
        self.choose.setEnabled(not pinned)
        self.choose.setToolTip(
            f"{self._env_var} is set, so this folder cannot be changed here."
            if pinned else ""
        )
