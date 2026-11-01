"""Per-run settings that change the answer.

Everything here is already an argument to :func:`deltachrome.pipeline.analyze`;
none of it is new capability. The grouping principle is the project's own:
**results must travel with the settings that produced them.** Two runs of one
capture under different parameters are two different measurements, not a
measurement and a repeat.

**What that principle does not license is a settings pane.** Four of the five
controls that used to live here failed the test of belonging to a run:

* *MTF50 unit* is how a frequency is written down. It cannot change what was
  measured -- and now genuinely cannot change the verdict either, since the
  criterion is converted along with the reading -- so it belongs beside the
  result, where a reader can switch units while looking at the number.
* *Fiducial extent* is the measured physical size of one physical chart. That is
  per-unit calibration and a property of the specimen, not of the run; a person
  who has measured their own copy wants it remembered, not retyped.
* *ROI inset* has an unvalidated default and no
  per-run reason to move. Until it has a basis it is a global preference, not a
  per-capture choice.
* *Original size* is real, but it applies to exactly one metric in three
  profiles. Shown unconditionally it asks every operator of every chart to answer
  a question that, on their run, changes nothing at all.

So original size appears only when the chosen profile actually bands a threshold
on it -- read from the profile's own ``pass_condition_by``, not from a list of
profile ids, so a future banded criterion from any body brings its control with
it.

**The fifth has since gone too, and it is the more interesting removal.** The
90th percentile's interpolation method survived the first cut because it is
genuinely a property of the run: it changes the reported figure, and the figure
has to travel with it. What it fails is a different test. **No standard specifies
it.** FADGI and Metamorfoze both say "90th percentile" and neither says which of
the two conventions they mean, so there is no answer to look up, no basis on
which an operator could prefer one, and -- on a 140-patch chart -- a difference
in the third decimal. A control with no correct answer does not become useful by
being offered; it asks every operator to guess, records the guess with the
result, and makes two identical measurements look different.

So the profile's own choice is used, always. It is still a parameter of the
engine, still recorded with every result, and still overridable from the command
line with ``--percentile-method``; what it is not is a question put to somebody
setting up a capture. Where it belongs is a per-profile default in application
settings, which is a thing to change once rather than per run.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from deltachrome_ui.theme import TEXT_DIM
from deltachrome_ui.widgets import heading

#: A tooltip rather than a panel. What is left is the one thing a control
#: cannot say for itself: what happens if it is left alone. The formats it
#: accepts are in the placeholder, and which metric bands is in the note below.
ORIGINAL_SIZE_HINT = (
    "Size of the largest originals. If empty, the metric is measured but not "
    "judged."
)

class OptionsView(QWidget):
    """Run settings, shown only where they apply to the run being set up."""

    def __init__(self) -> None:
        super().__init__()
        self._size_applies = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.title = heading("Run Options")
        layout.addWidget(self.title)

        self.original_size = QLineEdit()
        self.original_size.setPlaceholderText("A2, 500x700, or 450")
        self.size_row = self._row(
            "Original Size", self.original_size, ORIGINAL_SIZE_HINT,
        )
        layout.addWidget(self.size_row)
        #: The band label the profile itself uses, shown when a profile brings
        #: this control with it. Which metric bands is the profile's business.
        self.size_note = QLabel("")
        self.size_note.setWordWrap(True)
        self.size_note.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        self.size_note.setContentsMargins(120, 0, 0, 0)
        layout.addWidget(self.size_note)

        self.set_profile(None)

    def _row(self, name: str, control: QWidget, tooltip: str = "") -> QWidget:
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(6)
        label = QLabel(name)
        label.setFixedWidth(96)
        label.setStyleSheet(f"color: {TEXT_DIM};")
        if tooltip:
            label.setToolTip(tooltip)
        line.addWidget(label)
        line.addSpacing(4)
        line.addWidget(control, 1)
        return row

    # -- what applies ------------------------------------------------------

    def set_profile(self, profile) -> None:
        """Show the controls this profile can actually use.

        A control that cannot affect the run is worse than absent: it invites an
        answer, records it with the result, and changes nothing -- which makes
        two runs look different when they are the same measurement.
        """
        banded = self._banded_metric(profile)
        # Tracked as state rather than read back from setVisible(): a widget whose
        # window has not been shown reports itself invisible whatever it was told,
        # so asking Qt what is on screen would silently drop a set option.
        self._size_applies = banded is not None
        self.size_row.setVisible(banded is not None)
        self.size_note.setVisible(banded is not None)
        if banded is not None:
            self.size_note.setText(
                f"{profile.display_name} bands {banded[0].replace('_', ' ')} on this."
            )
        # The whole section goes, heading included, when nothing in it applies.
        # A heading over an empty space is a section that failed to load; a line
        # saying "this needs no settings" is an answer to a question nobody
        # asked, taking the room the absence already says it better in.
        self.setVisible(self._size_applies)

    @staticmethod
    def _banded_metric(profile) -> tuple[str, str] | None:
        """``(metric_id, parameter)`` for the first metric banded on a run
        parameter, or None. Read from the profile so this is not a list of
        Metamorfoze ids that a new standard would silently fall off."""
        for spec in getattr(profile, "metrics", []) or []:
            banded = getattr(spec, "pass_condition_by", None)
            if banded is not None and getattr(banded, "parameter", None):
                return spec.metric_id, banded.parameter
        return None

    # -- values ------------------------------------------------------------

    def values(self) -> dict:
        """Keyword arguments for :func:`deltachrome.pipeline.analyze`.

        Only what was actually set, and only from a control that is on screen. An
        option left alone is absent rather than passed as its default, so the
        profile's own choice survives instead of being overwritten with a value
        that happens to match today.
        """
        options: dict = {}
        if self._size_applies:
            size = self.original_size.text().strip()
            if size:
                options["original_size"] = size
        return options
