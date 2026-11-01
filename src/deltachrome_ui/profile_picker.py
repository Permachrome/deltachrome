"""Choosing a profile, in the two places that have to choose one.

Picking a profile is a three-way decision -- which body and edition, which
material category, which performance level -- and it is made twice: once when a
run is set up, and once more on the report, when the answer turns out to have
been judged against the wrong level or the wrong material category.

Two combos rather than one, for the reason
:mod:`deltachrome.profiles.catalogue` gives: a single flat list collapses to a
closed box reading "3-star", which says nothing about three-star *what*. That is
exactly the mistake the report's copy of this control exists to undo, so it would
be a poor idea to make it here.

**Why this is one widget and not two.** The level list only ever holds one
category's profiles, so selecting a profile by id means finding its category
first and rebuilding the level list before the level can be set. That is a small
piece of ordering logic and getting it wrong shows up as a combo box quietly
holding a profile from a category nobody chose.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout, QWidget

from deltachrome.profiles.catalogue import grouped
from deltachrome_ui.widgets import ComboBox


class ProfilePicker(QWidget):
    """Body, edition and material category; then performance level."""

    #: A different profile is selected. Emitted for a change of category too,
    #: because that always lands on a different profile.
    changed = Signal()

    def __init__(self, *, horizontal: bool = False,
                 profile_ids: list[str] | None = None) -> None:
        """``profile_ids`` narrows what may be chosen.

        The Guideline Builder passes the published standards only. A guideline
        derived from another guideline nests its provenance -- the departure list
        would be measured against the intermediate rather than against the
        standard -- and the departures a reader wants named are the ones from the
        published table.
        """
        super().__init__()
        self._ids = list(profile_ids) if profile_ids is not None else None
        self._groups = grouped(self._ids)
        self._horizontal = horizontal

        layout = QHBoxLayout(self) if horizontal else QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.category_combo = ComboBox()
        for key, _ in self._groups:
            self.category_combo.addItem(self._category_label(key))
        self.category_combo.currentIndexChanged.connect(self._on_category_changed)
        # Side by side, nearly all the width goes to the category: a level
        # label is "4-star" and a category label is thirty times that.
        layout.addWidget(self.category_combo, 5 if horizontal else 0)

        self.level_combo = ComboBox()
        self.level_combo.currentIndexChanged.connect(lambda _i: self.changed.emit())
        if horizontal:
            # Sized to the longest level label there is, and no wider. Side by
            # side the two combos were 287 px and 193 px, which is 40% of the
            # control spent on the word "3-star" -- and the 5:1 stretch above was
            # doing nothing about it, because a stretch cannot take a widget
            # below its minimum size hint and ``ComboBox`` asks for twelve
            # characters plus room for an icon it never draws.
            #
            # A fixed width rather than a cap, so the category combo beside it
            # does not change width when the level list is rebuilt under it --
            # a control that resizes when you use it is hard to aim at twice.
            self.level_combo.setMinimumContentsLength(1)
            self.level_combo.setFixedWidth(self._widest_level())
        layout.addWidget(self.level_combo, 0 if horizontal else 0)

        self._fill_levels()

    def reload(self) -> None:
        """Re-read the catalogue, keeping the selection where it still exists.

        The catalogue is read once, at construction, which was right while it was
        a fixed set of published standards. A guideline saved in Settings adds to
        it, and every picker alive in the window was built before that happened --
        the placement panel's lives for the session -- so a guideline would not
        appear until the application was restarted.

        The width is not recomputed. It is measured across every level label in
        the catalogue, and a control that changed width when somebody saved a
        guideline in another screen would be worse than one that is a few pixels
        wider than it needs to be.
        """
        wanted = self.profile_id
        self._groups = grouped(self._ids)
        self.category_combo.blockSignals(True)
        self.category_combo.clear()
        for key, _ in self._groups:
            self.category_combo.addItem(self._category_label(key))
        self.category_combo.blockSignals(False)
        self._fill_levels()
        if wanted:
            self.select_profile(wanted)

    @staticmethod
    def _category_label(key: tuple[str, str, str]) -> str:
        """One group, as a line in the category combo.

        Material category before edition. Both belong in the label and only one of
        them discriminates: every FADGI profile here is the 2023 third edition, and
        which of fifteen material tables is in force is the choice this control
        exists to make -- and the one somebody reading a report is most likely to
        have got wrong. Whichever half comes last is the half a closed combo box
        elides, so the order is not presentation.

        A local guideline carries neither. Its group is ``LOCAL GUIDELINE`` and its
        own name, so the two separators have to be conditional rather than assumed:
        an edition printed unconditionally left a guideline reading "LOCAL
        GUIDELINE — Test3   ·   ", which is a punctuation mark waiting for a fact
        that does not exist.
        """
        body, edition, material = key
        label = body + (f" — {material}" if material else "")
        return label + (f"   ·   {edition}" if edition else "")

    def _widest_level(self) -> int:
        """Room for the longest level label in the catalogue.

        Measured across every profile rather than assumed from the ones on
        screen: the level list is rebuilt whenever the category changes, and a
        box that resized itself as you moved down the category list would be the
        one thing worse than a box that is too wide.

        The width is taken from a **probe combo box** rather than from the font
        alone. What a combo costs around its text is 64 px here and adding up the
        parts the style sheet names gives 43 -- the rest is Qt's own metrics, and
        the two labels that arithmetic clipped were "Single level" and "Extra
        Light", which are the two it was sized for. Asking a real combo what it
        would need is the only version of this that cannot drift from the theme.
        """
        widest = max(
            (self._level_label(summary) for _, group in self._groups
             for summary in group),
            key=QFontMetrics(self.level_combo.font()).horizontalAdvance,
        )
        probe = ComboBox()
        probe.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        probe.setMinimumContentsLength(0)
        probe.addItem(widest)
        probe.ensurePolished()
        return probe.sizeHint().width()

    @staticmethod
    def _level_label(summary) -> str:
        """A level's name, without the body the category combo already states.

        Metamorfoze names its own levels "Metamorfoze Full", "Metamorfoze Light"
        and "Metamorfoze Extra Light", and that is the standard's own wording
        rather than sloppy naming -- so the profile keeps it and the CLI prints
        it, where a level may be read on its own. Here it is read beside a
        category combo that already says Metamorfoze, and the repetition is 82 px
        of the widest label: "Extra Light" needs 101 px against 183.
        """
        label = summary.level_label
        body = summary.group[0]
        prefix = f"{body} "
        return label[len(prefix):] if label.startswith(prefix) else label

    # -- state -------------------------------------------------------------

    @property
    def profile_id(self) -> str | None:
        return self.level_combo.currentData()

    @property
    def summary(self):
        """The catalogue's description of what is selected, or ``None``.

        What a host wants from it is ``flags`` -- the things a person choosing
        from a list needs told at the moment they choose, of which the load
        bearing one is that FADGI's transparency categories specify no color
        accuracy metric at all.
        """
        profile_id = self.profile_id
        return next(
            (s for _, group in self._groups for s in group
             if s.profile_id == profile_id),
            None,
        )

    @property
    def flags(self) -> list[str]:
        summary = self.summary
        return list(summary.flags) if summary else []

    def select_profile(self, profile_id: str) -> bool:
        """Set both combos from a profile id.

        Needed by anything restoring a choice rather than making one -- reusing a
        saved run's settings, or re-scoring one against a different level. The
        category has to be found first, since the level combo only ever holds one
        category's profiles.
        """
        for index, (_, group) in enumerate(self._groups):
            if any(s.profile_id == profile_id for s in group):
                self.category_combo.setCurrentIndex(index)
                self._fill_levels()
                level = self.level_combo.findData(profile_id)
                if level >= 0:
                    self.level_combo.setCurrentIndex(level)
                    return True
        return False

    # -- reactions ---------------------------------------------------------

    def _on_category_changed(self) -> None:
        self._fill_levels()
        self.changed.emit()

    def _fill_levels(self) -> None:
        """Rebuild the level list for the selected category, silently.

        Silently because clearing a combo box emits a selection change per item
        removed, and a host listening to ``changed`` would see a burst of
        profiles nobody chose -- each of which, on the report, is a re-score.
        """
        index = self.category_combo.currentIndex()
        if not 0 <= index < len(self._groups):
            return
        self.level_combo.blockSignals(True)
        self.level_combo.clear()
        for summary in self._groups[index][1]:
            self.level_combo.addItem(self._level_label(summary), summary.profile_id)
        self.level_combo.blockSignals(False)
