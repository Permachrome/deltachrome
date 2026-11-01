"""Which targets this operator owns, and so which a capture is searched for.

**The argument is wrong answers, not speed.** Identification names a chart because
it *fits*, and every target in the search is one more chance for the wrong one to
fit: on the batch's own check, naming two or three targets raised no false alarm in
eight frames and naming all twenty-three raised one. The two pairs that share a
layout -- the IT8s, the two ISA FADGI-ISO 19264 versions -- cannot be told apart by
geometry at all, and are named outright for anybody who owns one of each. What it
saves in time is smaller than the count suggests, because the search is per finder
method and five finders read nothing from the target.

**It only narrows the search.** The placement panel's target list still offers
every target, since choosing one by hand is a deliberate act; a target turned off
here is one the application will not *suggest*. Where a search names nothing, it
says how many targets were left out of it -- a chart borrowed or bought since is the
likeliest reason a chart that is plainly there was not found -- and a run whose
chart was named from a narrowed set says which set, in ``geometry.searched_targets``.

**Grouped by reflective and transmissive, not by maker.** A maker is two different
facts for these charts: the DT Next Generation Target V2 is sold as DT's and its
definition names Avian Rochester, and IT8.7 is a published layout that LaserSoft,
Wolf Faust, Kodak and others have all printed. Any one manufacturer heading files
somebody's chart under the wrong name. Whether a chart is shot by reflected or
transmitted light is how an operator already sorts their own -- prints against
film -- and every definition states it in ``target_type``. Within a group the
names sort alphabetically, and they lead with the brand people recognise.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from deltachrome_ui import paths
from deltachrome_ui.theme import TEXT_DIM


#: The groups, in the order they are listed: the prefix of a definition's
#: ``target_type``, and the heading it is shown under.
GROUPS = (("reflective", "Reflective"), ("transmissive", "Transmissive"))


def _group(target_type: str) -> str:
    """Which heading a target sits under. Anything a later definition declares
    that is neither is listed last rather than dropped."""
    for prefix, heading in GROUPS:
        if target_type.startswith(prefix):
            return heading
    return "Other"


class TargetsPanel(QWidget):
    """A checkbox per target identification can name, reflective then transmissive."""

    #: The set changed. Nothing listens yet -- every search reads the setting as it
    #: starts -- but a screen that holds a stale count would.
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(8)
        self.count = QLabel("")
        self.count.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        top.addWidget(self.count, 1)
        # The way back to the default. An operator who has turned off most of the
        # list and then shoots a borrowed chart wants one press, not twenty.
        self.all_on = QPushButton("Turn All On")
        self.all_on.clicked.connect(self._turn_all_on)
        # Its room is kept while it is hidden. The button is taller than the count
        # beside it, so the row grew when it appeared and pushed the whole list
        # down under the pointer -- on the very click that unticked a box.
        policy = self.all_on.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.all_on.setSizePolicy(policy)
        top.addWidget(self.all_on)
        column.addLayout(top)

        body = QWidget()
        self._rows = QVBoxLayout(body)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(body)
        column.addWidget(scroll, 1)

        self.boxes: dict[str, QCheckBox] = {}
        self._build()

    def _build(self) -> None:
        from deltachrome.imaging.detect import identifiable_targets
        from deltachrome.targets.registry import load_target

        groups: dict[str, list] = {}
        for target_id in identifiable_targets():
            target = load_target(target_id)
            groups.setdefault(_group(target.target_type), []).append(target)

        off = paths.disabled_targets()
        order = [heading for _prefix, heading in GROUPS] + ["Other"]
        for heading in [h for h in order if h in groups]:
            label = QLabel(heading)
            label.setObjectName("SectionHeading")
            # A group heading after the first wants air above it; the first sits
            # directly under the count.
            if self.boxes:
                label.setContentsMargins(0, 10, 0, 0)
            self._rows.addWidget(label)
            for target in sorted(groups[heading], key=lambda t: t.display_name.casefold()):
                # Doubled, or Qt reads the QA-62's "SFR & OEC" as a mnemonic and
                # prints "SFR  OEC".
                box = QCheckBox(target.display_name.replace("&", "&&"))
                box.setChecked(target.target_id not in off)
                box.toggled.connect(self._store)
                self.boxes[target.target_id] = box
                self._rows.addWidget(box)
        self._rows.addStretch(1)
        self._show_count()

    def reload(self) -> None:
        """Re-read the stored set. The settings file is the truth, not the boxes."""
        off = paths.disabled_targets()
        for target_id, box in self.boxes.items():
            box.blockSignals(True)
            box.setChecked(target_id not in off)
            box.blockSignals(False)
        self._show_count()

    def _store(self) -> None:
        # Ids this list does not show -- a target no longer installed -- are kept
        # as they were: removing a definition is not the operator changing their mind.
        hidden = paths.disabled_targets() - set(self.boxes)
        off = {t for t, box in self.boxes.items() if not box.isChecked()}
        paths.set_disabled_targets(hidden | off)
        self._show_count()
        self.changed.emit()

    def _turn_all_on(self) -> None:
        for box in self.boxes.values():
            box.blockSignals(True)
            box.setChecked(True)
            box.blockSignals(False)
        self._store()

    def _show_count(self) -> None:
        on = sum(box.isChecked() for box in self.boxes.values())
        self.count.setText(
            f"{on} target{'' if on == 1 else 's'} selected for analysis"
        )
        self.all_on.setVisible(on < len(self.boxes))
