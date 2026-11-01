"""What this capture is being judged as, and against what.

Three choices -- target, profile, reference -- and they used to have a screen of
their own, reached before the placement canvas and left behind on the way to it.
That screen is gone, and this is what is left: a column of controls with no
chrome, embedded in the placement panel so the whole of a run is set up in one
place while the chart it is set up for is on screen beside it.

**Why the separate screen could go.** ``identify_target`` places the chart on the
way to naming it -- the candidate that wins does so partly by its regions landing
on patches -- so by the moment the operator would have seen a setup screen, the
corners are already known. A screen whose job was to precede placement had
nothing left to precede.

**What went with it, and why none of it is missed.** Every removal here was a
paragraph explaining something the screen was already showing:

* *The identification's reasoning* -- "aspect off by 1%, ramp +1.00, 0% flagged"
  -- is a diagnostic for whoever tunes the finder. What an operator needs is that
  a chart was named, and which; the name is in the combo box and the fit is
  visible as ROIs landing on patches.
* *The coverage paragraph* counted a profile's metrics against a target's
  capabilities in prose. The report says the same thing by listing what it
  measured, and says it against actual readings rather than a forecast.
* *"Cannot measure: sfr, geometric_distortion"* named capabilities in the
  engine's own spelling to a reader who chose the chart and knows what it is.
* *The placement radio group* asked which way to place a chart one screen before
  the chart was on screen. Standing on the canvas, the answer is a drag.

What is kept is the one line that changes what the operator has to do next:
whether this chart needs a reference file. Everything else this column says, it
says with a control.

**And the pairing is settled here rather than after the decode.** The chart's
printed code is read on the way to identifying the chart -- a whole-image pass
is a tenth of a second, and only a capture where that finds nothing pays for
tiling the master -- so by the time this column is drawn, *Pair from library*
already knows which file it will use, and says so. It used to find out during the
run, which meant that the case worth knowing about, a code matching nothing in
the bank, was an exception raised after a wait and dismissable only.

The three ways the code can fail to decide are all real -- the chart prints none,
no decoder reads the one it prints, or the bank binds it to nothing -- and none
of them is a dead end any more: the mode offers the bank as a list and the
operator picks. That list is deliberately not the file dialog beside it. *Choose
a file…* opens the whole filesystem, which is right for a measurement that is not
in the bank; the bank is a set of managed copies and each of its files is shown
with the codes it answers to, which is the fact that decides whether it is the
right one.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from deltachrome.profiles.schema import load_profile
from deltachrome.targets.registry import available_targets, load_target
from deltachrome_ui.options_view import OptionsView
from deltachrome_ui.paths import reference_bank_dir
from deltachrome_ui.profile_picker import ProfilePicker
from deltachrome_ui.reference_bank import bank_count
from deltachrome_ui.status import TONE_PASS, TONE_QUALIFIED
from deltachrome_ui.theme import TEXT_DIM
from deltachrome_ui.widgets import Callout, ComboBox, ElidedLabel, Pill, heading

#: Reference modes, in the order they appear.
AUTOMATIC, EXPLICIT_FILE, LIBRARY = "automatic", "file", "library"

#: What the target combo carries while its blank entry is selected.
#:
#: An empty string rather than ``None`` because it is the combo's *data* and Qt
#: returns ``None`` from ``currentData()`` for an item that has none -- so a
#: genuine "nothing chosen" and a missing item would be the same value, and the
#: one place that matters is the guard that decides whether a run may start.
NO_TARGET = ""

#: Said the same way for every chart that needs one, which is the whole point of
#: it: a target either ships values to fall back on or it does not, and that is
#: one bit rather than a paragraph about print tolerance.
REFERENCE_REQUIRED = "Reference file required"

#: The three outcomes of pairing by the chart's own printed code, said as a chip
#: rather than as a sentence. Written out here beside the requirement above,
#: because they are the same slot on the screen and a reader sees exactly one.
REFERENCE_PAIRED = "Reference paired"
REFERENCE_CHOSEN = "Reference chosen"
REFERENCE_UNPAIRED = "Cannot pair this capture"

#: And the two the *Automatic* mode says, because a chart that prints a code has
#: told us something about itself that the mode is not using. One of them is a
#: mistake worth catching -- running on manufacturer values while a measurement
#: of this very unit sits in the bank -- and the other is what the operator
#: asked for: a code was read and nothing answers to it.
REFERENCE_IN_BANK = "Per-unit file available"
REFERENCE_CODE_UNKNOWN = "Target code not in the bank"

#: Said whatever mode is chosen, because it is a fact about the capture and not
#: about the mode: this chart carries nothing machine-readable, so no mode here
#: can pair it by unit.
REFERENCE_NO_CODE = "No code read from this capture"


def _unpaired_detail(resolution, codes) -> str:
    """Why pairing found no file, in the screen's words rather than the engine's.

    The engine's ``reason`` is written for the console and the run's notes: it
    names the bank's full path and remedies an operator does not perform here.
    """
    if resolution is None:
        return "The reference bank could not be read."
    code = ", ".join(codes)
    if resolution.ambiguous:
        count = len({entry.path for entry in resolution.candidates})
        return f"Code {code} matches {count} files in the bank. Pick one from the bank."
    if code:
        return (f"Code {code} isn't in the bank. Choose a file, pick one from the "
                "bank, or bind the code in Settings.")
    return "Choose a file, or pick one from the bank."


def _compares_nothing(target) -> bool:
    """Whether this chart's definition says none of its patches is compared.

    ``source_type: "none"`` is the schema's own spelling of that and the engine
    keys on it in three places; reading it the same way here is what keeps the
    screen from asking for a file the pipeline would decline to read.
    """
    return target.reference_data.get("source_type") == "none"


class ChoicesPanel(QWidget):
    """Target, profile and reference, as controls rather than as prose."""

    #: A different chart was chosen. The placement has to be found again, so this
    #: is separate from the general ``changed`` below.
    targetChanged = Signal(str)
    #: Anything else moved. Nothing downstream has to be recomputed; the run
    #: button's own enablement is all that follows.
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.reference_path: Path | None = None
        self.library_dir: Path | None = None
        #: What the capture's printed code was read to be, or ``None`` where no
        #: scan was made. Set once, by the window, from the identification pass.
        self.scan = None
        #: A bank file the operator picked by hand because the code could not
        #: pick one. It makes the run *explicit*, which is what it is: the file
        #: was named. See :attr:`reference_mode`.
        self.bank_choice: Path | None = None
        #: Whether a measurement download is in flight, so a second press cannot
        #: start a second one -- the button stays visible while it runs and a
        #: repeat would file two copies of one file under two names.
        self._fetching = False
        #: Whether the selected target is still the one the capture was
        #: identified as. False the moment the operator picks a different chart,
        #: so the record does not claim a target nobody chose when somebody did.
        self.identified = False
        self._identified_as: str | None = None
        self._current_target: str | None = None
        #: Set while a whole choice is being restored at once. See :meth:`restore`.
        self._restoring = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self._build(layout)

    # -- construction ------------------------------------------------------

    def _build(self, layout: QVBoxLayout) -> None:
        # -- target
        layout.addWidget(heading("Target"))
        self.target_pill = Pill()
        self.target_pill.hide()
        layout.addWidget(self.target_pill)

        self.target_combo = ComboBox()
        # **An empty first entry, so the screen makes no claim it cannot support.**
        # A combo always holds a value, so a capture nothing could be identified in
        # used to open showing whichever target happens to be first in the list --
        # which reads as an answer, and is the screen assuming a chart on no
        # evidence. Blank reads as a question, and the target is then actively
        # chosen rather than merely left alone.
        self.target_combo.addItem("", NO_TARGET)
        for target_id in available_targets():
            target = load_target(target_id)
            self.target_combo.addItem(target.display_name, target_id)
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        layout.addWidget(self.target_combo)

        # -- profile
        #
        # The same control the report carries, because the report can change this
        # choice after the fact and the two must offer the same one.
        layout.addWidget(heading("Guidelines"))
        self.picker = ProfilePicker()
        self.picker.changed.connect(self._on_selection_changed)
        layout.addWidget(self.picker)
        #: The two combos by their old names, because they are what a test and a
        #: caller reach for and the control moving out of this file is not a
        #: reason to rename them.
        self.category_combo = self.picker.category_combo
        self.profile_combo = self.picker.level_combo

        # The one piece of profile prose that survives, because it is not about
        # this application: FADGI's transparency categories specify no color
        # accuracy metric at all, and a report that never mentions color and
        # never says why is worse than a warning nobody needed. A local guideline
        # built from such a table inherits it without anything having been decided
        # about colour, so the flag is computed for guidelines too -- where what it
        # no longer carries is the guideline's own provenance, which is not a
        # warning about anything.
        self.profile_flags = Callout()
        self.profile_flags.hide()
        layout.addWidget(self.profile_flags)

        # -- reference
        #
        # The whole group hides for a chart that compares none of its patches
        # against anything -- the QA-62 prints a twenty-step ring and ships no aim
        # values for it, and both capabilities it claims are read from its slanted
        # square. A reference file chosen there would be read by nothing, and a
        # control that cannot affect the run invites an answer and records it
        # beside a result it did not change.
        self.reference_heading = heading("Reference Measurements")
        layout.addWidget(self.reference_heading)
        # Its own group, because radio buttons sharing a parent are exclusive
        # with each other whether or not they are about the same question.
        self.mode_group = QButtonGroup(self)
        self.mode_buttons: dict[str, QRadioButton] = {}
        # *Pair from library* opens no dialog, which is the whole change: it used
        # to ask for a folder on every single run and remember it nowhere, so the
        # one mode that can use the printed code was the one that cost the most
        # gestures. The bank is the application's own folder, filled in Settings.
        for mode, label, tip in (
            (AUTOMATIC, "Automatic",
             "The values shipped for this model. Advisory only."),
            (EXPLICIT_FILE, "Choose a file…",
             ""),
            (LIBRARY, "Pair from library",
             "Match the target's printed code to a file in the bank."),
        ):
            button = QRadioButton(label)
            button.setToolTip(tip)
            button.toggled.connect(self._on_mode_changed)
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            layout.addWidget(button)

        picker = QHBoxLayout()
        # Elided on a row of its own, never wrapped beside the buttons: what it
        # names is usually a file, and a filename is one unbroken token, so a
        # wrapping label's minimum width is the whole name -- which widened the
        # panel's contents past its fixed width and pushed every control in it
        # off the right edge. Elided in the middle, since
        # the end of a measurement's filename is where its date and unit are.
        self.path_label = ElidedLabel(mode=Qt.TextElideMode.ElideMiddle)
        self.path_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        self.choose_button = QPushButton("Choose…")
        self.choose_button.clicked.connect(self._choose)
        self.choose_button.hide()
        picker.addWidget(self.choose_button)
        # The fallback for every way a code can fail to decide, and the override
        # for the one case a code decides wrongly. Offered whenever the mode is
        # on rather than only after a failure: an operator who knows the card in
        # front of them is not in the bank's index should not have to watch the
        # pairing fail first.
        self.bank_button = QPushButton("Choose from bank…")
        self.bank_button.clicked.connect(self._choose_from_bank)
        self.bank_button.hide()
        picker.addWidget(self.bank_button)
        # Offered only when the chart's own code says where its measurement is
        # published and the bank does not already hold it. It is a *button*
        # rather than something the identification pass does on its own: a
        # capture is not consent to make an outbound request, and the one thing
        # this application does that reaches somebody else's server should be
        # the thing an operator pressed.
        self.fetch_button = QPushButton("Fetch Measurement")
        self.fetch_button.clicked.connect(self._fetch_measurement)
        self.fetch_button.hide()
        picker.addWidget(self.fetch_button)
        picker.addStretch(1)
        layout.addLayout(picker)
        layout.addWidget(self.path_label)

        self.reference_pill = Pill()
        self.reference_pill.hide()
        layout.addWidget(self.reference_pill)

        self.options = OptionsView()
        layout.addWidget(self.options)

        # Checked only now: toggling fires _on_mode_changed, which touches the
        # picker widgets built just above.
        self.mode_buttons[AUTOMATIC].setChecked(True)
        self._on_selection_changed()

    # -- state -------------------------------------------------------------

    @property
    def target_id(self) -> str:
        """The chosen target, or ``NO_TARGET`` while the blank entry is selected."""
        return self.target_combo.currentData()

    @property
    def has_target(self) -> bool:
        return bool(self.target_id) and self.target_id != NO_TARGET

    def clear_target(self) -> None:
        """Back to the blank entry: nothing here claims to know what this is."""
        index = self.target_combo.findData(NO_TARGET)
        if index >= 0:
            self.target_combo.setCurrentIndex(index)

    @property
    def profile_id(self) -> str | None:
        return self.picker.profile_id

    @property
    def mode(self) -> str:
        return next(m for m, b in self.mode_buttons.items() if b.isChecked())

    @property
    def reference_choice(self) -> Path | None:
        if self.mode == LIBRARY:
            return self.bank_choice or self.library_dir
        return self.reference_path

    @property
    def reference_mode(self) -> str:
        """What the *run* is doing, which is not always what the radio says.

        A bank file picked by hand is a named file, so the run resolves it as one
        -- code verification and all, which is the check worth keeping: if the
        capture does carry a code and it disagrees with the file somebody chose,
        that belongs in the record as a warning rather than being swallowed
        because the choice was deliberate.
        """
        if self.mode == LIBRARY and self.bank_choice is not None:
            return EXPLICIT_FILE
        return self.mode

    def set_scan(self, scan) -> None:
        """The codes read from this capture. Called once, as the screen opens.

        A code that the bank pairs with a file selects *Pair from library*: the
        chart has said which unit it is and the bank holds that unit's
        measurement, so starting on the manufacturer's values would ask the
        operator to notice and correct a choice the capture already made. Only
        then -- a capture with no code, or one the bank does not know, opens on
        *Automatic* as before, since pairing would assert something it cannot.
        Asked whether or not a target is named yet: the code identifies the
        unit, so a chart chosen by hand afterwards finds its reference waiting.
        """
        self.scan = scan
        if self.mode == AUTOMATIC and (not self.has_target or self._offers_reference()):
            resolution = self.pairing()
            if resolution is not None and resolution.path is not None:
                self.mode_buttons[LIBRARY].setEnabled(True)
                self.mode_buttons[LIBRARY].setChecked(True)
        self._on_mode_changed()

    def pairing(self):
        """What *Pair from library* would do with this capture, right now.

        Returns the engine's own :class:`~deltachrome.reference.library.Resolution`
        or ``None`` where there is nothing to resolve. Deliberately the same call
        ``resolve_reference`` makes at run time and against the same scan, so the
        line on this screen and the file the run uses cannot come apart.
        """
        from deltachrome.reference.library import ReferenceLibrary

        # The bank's path, not the mode's: what the bank holds for this capture is
        # worth knowing under *Automatic* too, where the answer is that a per-unit
        # measurement exists and the run is about to ignore it.
        directory = self.library_dir or reference_bank_dir()
        if self.scan is None or directory is None:
            return None
        try:
            return ReferenceLibrary(directory).resolve(
                self.scan.texts, standard_2of5=getattr(self.scan, "standard_2of5", ()))
        except Exception:  # noqa: BLE001 -- an unreadable bank pairs nothing
            return None

    def refresh_bank(self) -> None:
        """Re-read what the bank holds. Called when Settings changes it.

        The bank being empty is the difference between an offer and a dead end,
        and this screen is not necessarily on top when a file is added to it.
        """
        self.library_dir = reference_bank_dir()
        self._on_mode_changed()

    @property
    def ready(self) -> bool:
        """Whether a run made from these choices could reach a reference at all.

        A chart shipping no manufacturer values and paired with nothing has
        nothing to be measured against, and the pipeline says so by raising. That
        is a dialog the operator can only dismiss, arrived at after a wait, so it
        is answered here instead: the run button is off and the chip says what is
        missing.
        """
        # A blank target is not "not ready yet" in the same sense as a missing
        # reference -- there is nothing to look a reference up for -- so it is
        # answered first and by the same gate, since the effect is the same: the
        # run cannot start.
        if not self.has_target:
            return False
        return bool(self.profile_id) and not self._reference_missing()

    def _offers_reference(self) -> bool:
        """Whether a reference file could apply to this capture at all.

        False while no target is chosen, on exactly the same footing as a target
        that compares nothing: there is nothing for a reference to be a reference
        *to*, so a control asking for one would collect an answer it cannot use.
        """
        if not self.has_target:
            return False
        return not _compares_nothing(load_target(self.target_id))

    def _reference_missing(self) -> bool:
        if not self.has_target:
            return False
        target = load_target(self.target_id)
        if _compares_nothing(target):
            return False
        # A pairing that cannot be made stops the run whatever the target ships:
        # ``resolve_reference`` raises when a library was named and no file in it
        # answers to the capture's code, and it is right to -- silently falling
        # back to nominal values for a chart the operator asked to pair by unit
        # would answer a different question in the same report.
        if self._pairing_failed():
            return True
        if target.nominal_reference_path is not None:
            return False
        return self.reference_choice is None

    def _pairing_failed(self) -> bool:
        """Whether *Pair from library* has been asked and cannot deliver."""
        if self.mode != LIBRARY or self.scan is None or self.bank_choice is not None:
            return False
        resolution = self.pairing()
        return resolution is None or resolution.path is None

    def restore(self, *, identification=None, target_id=None,
                profile_id=None) -> None:
        """Set several choices at once, as one choice rather than as a sequence.

        Every setter here is also a user gesture, and a user gesture that changes
        the target means "find this chart in the capture". Restoring a saved run
        touches the target, the profile and sometimes the identification in a row,
        and each of those would ask for a search -- for charts the operator is
        about to replace with the next line of the restore.

        So ``targetChanged`` is held back and the caller places the chart once,
        when the whole choice is settled. Everything else still updates as it
        goes: the run button, the reference chip and the run options are all
        consequences of the state rather than requests for work.
        """
        self._restoring = True
        try:
            if profile_id:
                self.select_profile(profile_id)
            if identification is not None:
                self.show_identification(identification)
            # **An explicit target wins over a fresh identification**, and is
            # applied after it rather than instead of it. Restoring a saved run
            # carries a target somebody already checked; identification runs again
            # on the way in and may decline, which now clears the combo -- so
            # ordering these the other way round lost the restored chart on any
            # capture the search could not name.
            if target_id:
                self.select_target(target_id)
            # The whole reference group belongs to the capture it was set for.
            # Reference data is *per physical unit*, so a file left selected from
            # the last run is the exact failure the pairing exists to prevent:
            # measure the LaserSoft against the ColorChecker's measurement and
            # every number is reasonable and meaningless.
            #
            # **The mode resets too**, which reverses the first attempt at this.
            # Keeping it looked safe, since pairing re-reads this capture's own
            # code -- but it is not what an operator reads. A capture opening on
            # *Pair from library* is the application asserting something about
            # the chart in front of them, and where the chart carries no code it
            # asserts it and then fails, which is how a UTT opened saying it
            # could not be paired.
            self.reference_path = None
            self.bank_choice = None
            self.mode_buttons[AUTOMATIC].setChecked(True)
        finally:
            self._restoring = False
        self._current_target = self.target_id

    def select_target(self, target_id: str) -> bool:
        index = self.target_combo.findData(target_id)
        if index < 0:
            return False
        self.target_combo.setCurrentIndex(index)
        return True

    def select_profile(self, profile_id: str) -> bool:
        return self.picker.select_profile(profile_id)

    def show_identification(self, result) -> None:
        """Select what the capture was identified as, and say only that it was.

        A pill rather than a paragraph. The finder's reasoning -- the aspect
        error, the ramp correlation, the median dE against aim values -- is a
        diagnostic for whoever tunes it; what an operator needs is that a chart
        was named. Which chart is in the combo box directly below, and whether it
        was named *correctly* is answered by the ROIs on the canvas, not by any
        sentence this could write.

        The reasoning is not discarded: it is the pill's tooltip, so it is one
        hover away for the run where the answer surprises somebody.
        """
        self.identified = bool(result.found)
        self._identified_as = result.target_id
        detail = "\n\n".join([result.reason] + list(result.notes))
        if result.found:
            self.select_target(result.target_id)
            self.target_pill.set_state("Target Detected", TONE_PASS, detail=detail)
            return
        # Declining is not an error -- the operator was always going to choose,
        # and this only chooses first. But it does hand the choice back, so it
        # says so rather than leaving a combo box looking like a decision.
        #
        # **And the choice it hands back is blank**, on exactly the argument the
        # reference group below already rests on: a target belongs to the capture
        # it was chosen for. This panel outlives the capture, so the previous
        # one's chart stayed selected -- and the screen then ran a search for it,
        # placed whatever it found, and offered that as a chart of a capture
        # nothing had been identified in. Reported from the running application as
        # a frame holding three targets where two had been asked for.
        self.clear_target()
        self.target_pill.set_state("Choose the Target", TONE_QUALIFIED, detail=detail)

    def show_shortlist(self, target_id: str, reason: str) -> None:
        """The chart could not be named outright; this one leads and wants confirming.

        A third state between "Target Detected" and "Choose the Target", and it earns
        its place because it is a different fact: the chart was found and two targets
        explain it, so there IS a leading answer and it is not certain. Selecting it
        and saying nothing would present a coin toss as a decision; hiding the chip
        and leaving the combo would lose the leader the search did find.
        """
        self.identified = False
        self._identified_as = None
        self.select_target(target_id)
        self.target_pill.set_state("Confirm the Target", TONE_QUALIFIED, detail=reason)

    def settle_chip(self, target_id: str, *, named: bool) -> None:
        """What the chip says about a chart that has a target and no doubt.

        **Named** -- by identification, or by *Search a Region* -- is "Target
        Detected", exactly as a whole-frame identification says it. Otherwise the
        operator chose the chart, which leaves nothing to say: the chip goes, rather
        than going on asking for a choice that has been made.
        """
        if named:
            self.identified = True
            self._identified_as = target_id
            self.target_pill.set_state("Target Detected", TONE_PASS)
        else:
            # Reset, not merely hidden when nothing else is showing: stepping from
            # a chart a region search named to one the operator chose left the
            # first chart's "Target Detected" standing over the second.
            self.identified = False
            self._identified_as = None
            self.target_pill.hide()

    def ask_for_target(self, detail: str) -> None:
        """Nobody named the chart, and deliberately: the operator declares it.

        The same chip identification shows when it declines, for the same reason
        -- the choice is being handed over rather than made -- and reached instead
        of running the search at all. See the caller.
        """
        self.identified = False
        self._identified_as = None
        self.target_pill.set_state("Choose the Target", TONE_QUALIFIED, detail=detail)

    def clear_identification(self) -> None:
        self.identified = False
        self._identified_as = None
        self.target_pill.hide()

    # -- reactions ---------------------------------------------------------

    def _on_target_changed(self) -> None:
        target_id = self.target_id
        if self.identified and self._identified_as != target_id:
            self.identified = False
            self.target_pill.hide()
        elif not self.identified and target_id and not self._restoring:
            # "Choose the Target" and "Confirm the Target" are both answered by
            # choosing one. Held while restoring, because a restore selects the
            # target before it sets the chip that goes with it.
            self.target_pill.hide()
        self._on_selection_changed()
        # After the state above is settled, so a handler asking what is selected
        # gets the answer rather than the previous one.
        if self._restoring:
            return
        if target_id and target_id != self._current_target:
            self._current_target = target_id
            self.targetChanged.emit(target_id)

    def _on_selection_changed(self) -> None:
        self._update_reference_pill()

        profile_id = self.profile_id
        if not profile_id:
            self.options.set_profile(None)
            self.profile_flags.hide()
            self.changed.emit()
            return
        profile = load_profile(profile_id)

        flags = self.picker.flags
        self.profile_flags.set_state(
            ("⚠  " + "; ".join(flags)) if flags else "", TONE_QUALIFIED
        )

        # Run options follow the profile: a control that cannot affect this run
        # should not be on screen inviting an answer.
        self.options.set_profile(profile)
        self.changed.emit()

    def _show_reference_controls(self) -> None:
        """Offer a reference file only where one could be read."""
        wanted = self._offers_reference()
        self.reference_heading.setVisible(wanted)
        for button in self.mode_buttons.values():
            button.setVisible(wanted)
        self.choose_button.setVisible(wanted and self.mode == EXPLICIT_FILE)
        self.path_label.setVisible(wanted)

    def _update_reference_pill(self) -> None:
        """One line, the same for every chart that needs one.

        Shown only while it is actionable. Once a file or a library is chosen the
        requirement is met and the chip goes, which makes its presence mean
        "there is something left to do" rather than "this target is fussy".
        """
        self._show_reference_controls()
        if not self._offers_reference():
            self.reference_pill.hide()
            return
        # Said in every mode, because it is a fact about the capture rather than
        # about the mode: this chart carries no machine-readable identifier, so
        # nothing here or anywhere else can pair it by unit. Saying it only under
        # *Pair from library* made it read as a failure of that mode.
        if self.scan is not None and not self.scan.texts:
            self.reference_pill.set_state(
                REFERENCE_NO_CODE, TONE_QUALIFIED,
                detail="Choose a file, or pick one from the bank.",
            )
            return
        if self.mode == AUTOMATIC and self.scan is not None and self.scan.texts:
            resolution = self.pairing()
            codes = ", ".join(self.scan.texts)
            if resolution is not None and resolution.path is not None:
                self.reference_pill.set_state(
                    REFERENCE_IN_BANK, TONE_QUALIFIED,
                    detail=f"{resolution.path.name} matches code {codes}. Use "
                           "Pair from library for a conformance result.",
                )
            else:
                self.reference_pill.set_state(
                    REFERENCE_CODE_UNKNOWN, TONE_QUALIFIED,
                    detail=f"Code read: {codes}. Add its measurement in "
                           "Settings for a conformance result.",
                )
            return
        if self.mode == LIBRARY and self.scan is not None:
            resolution = self.pairing()
            if self.bank_choice is not None:
                self.reference_pill.set_state(
                    REFERENCE_CHOSEN, TONE_PASS,
                    detail=self.bank_choice.name,
                )
            elif resolution is not None and resolution.path is not None:
                # The tone this application reserves for a conformance claim, and
                # this is the case that earns one: the reference is a measurement
                # of the very unit in the frame, established from the chart's own
                # printed code rather than from anybody's memory.
                self.reference_pill.set_state(
                    REFERENCE_PAIRED, TONE_PASS,
                    detail=f"{resolution.path.name}, matched by code "
                           f"{resolution.matched_code}.",
                )
            else:
                self.reference_pill.set_state(
                    REFERENCE_UNPAIRED, TONE_QUALIFIED,
                    detail=_unpaired_detail(resolution, self.scan.texts),
                )
            return
        if self._reference_missing():
            self.reference_pill.set_state(
                REFERENCE_REQUIRED, TONE_QUALIFIED,
                detail="This target has no built-in reference values.",
            )
        else:
            self.reference_pill.hide()

    def _on_mode_changed(self) -> None:
        wanted = self._offers_reference()
        if self.mode != LIBRARY:
            # A hand-picked bank file belongs to this mode. Leaving it set would
            # make *Automatic* quietly use a file somebody chose two clicks ago.
            self.bank_choice = None
        self.choose_button.setVisible(wanted and self.mode == EXPLICIT_FILE)
        self.choose_button.setText("Choose file…")

        # An empty bank cannot pair anything, and a mode that will certainly fail
        # must say so before the run rather than after the decode. It is disabled
        # rather than hidden, because the fix is a screen away and a control that
        # vanished would not say where.
        empty = bank_count() == 0
        pair = self.mode_buttons[LIBRARY]
        pair.setEnabled(not empty)
        if empty:
            pair.setToolTip(
                "The reference bank is empty. Add files in Settings."
            )
            if self.mode == LIBRARY:
                self.mode_buttons[AUTOMATIC].setChecked(True)

        self.bank_button.setVisible(
            wanted and self.mode == LIBRARY and not empty
        )
        # Offered in both modes that turn on the capture's own code. Under
        # *Automatic* it is the way out of "this chart is measured against
        # manufacturer values because nobody has its file"; under *Pair from
        # library* it is the way out of a pairing that found nothing. Not
        # offered under *Choose file*, where the operator has already said which
        # file to use.
        # Asked once: it resolves against the bank, which is a directory scan.
        label = self.fetchable_label()
        self.fetch_button.setVisible(
            wanted and self.mode in (AUTOMATIC, LIBRARY) and label is not None
        )
        if label is not None:
            # A label and a serial are different claims and must not read alike.
            # A label is the maker's own statement that this unit's measurement
            # exists at that address; a serial is a number this application knows
            # how to look up, and the lookup may well come back empty.
            from deltachrome.reference.openqualia import FROM_CHART

            if label.source == FROM_CHART:
                self.fetch_button.setToolTip(
                    f"Download this target's measurement from {label.host}."
                )
            else:
                self.fetch_button.setToolTip(
                    f"Look up this target's measurement on {label.host}. "
                    f"Not every unit is listed."
                )
        if self.mode == LIBRARY:
            self.library_dir = reference_bank_dir()
            line = self._pairing_line()
            self.path_label.setText(line)
            # The whole line as well as the folder, the line being elided.
            self.path_label.setToolTip(f"{line}\n{self.library_dir}")
        elif self.mode == AUTOMATIC:
            # The code goes beside the chip here too. A chip saying a code is not
            # in the bank, without the code, tells an operator that something is
            # wrong and not what to type into Settings.
            self.path_label.setText(self._code_line())
        elif self.reference_path is not None:
            self.path_label.setText(str(self.reference_path))
        self._update_reference_pill()
        self.changed.emit()

    def _pairing_line(self) -> str:
        """One line saying which file this capture will be measured against.

        It used to say how many files the bank held, which is a fact about the
        bank and not about this capture -- true before the scan existed, and the
        only thing that could be said then. Now the answer is known: the file, or
        the reason there is not one, and in either case **the code that was read**,
        which is the thing an operator needs beside a refusal in order to fix it.
        """
        if self.bank_choice is not None:
            return f"Chosen by hand: {self.bank_choice.name}"
        held = bank_count()
        if self.scan is None:
            return f"{held} reference file{'' if held == 1 else 's'} in the bank"
        resolution = self.pairing()
        if resolution is not None and resolution.path is not None:
            return (f"Paired by code {resolution.matched_code}: "
                    f"{resolution.path.name}")
        codes = ", ".join(self.scan.texts)
        return (f"Code {codes} matches no file in the bank"
                if codes else "This target prints no code this can read")

    def _code_line(self) -> str:
        """What this capture's own code says, for a mode that is not using it."""
        if self.scan is None:
            return ""
        if not self.scan.texts:
            return "This target prints no code this can read"
        codes = ", ".join(self.scan.texts)
        resolution = self.pairing()
        if resolution is not None and resolution.path is not None:
            return f"Code {codes}: {resolution.path.name} is in the bank"
        return f"Code {codes} matches no file in the bank"

    def fetchable_label(self):
        """The capture's OpenQualia label, when fetching it would do something.

        ``None`` in every other case, and the cases are the point. A chart that
        prints no code, a code that is a bare serial rather than a label, and a
        label whose unit the bank already holds are all "nothing to fetch" --
        the last one especially, because offering to download a measurement that
        is already filed invites a second copy of it under a second name.
        """
        if self.scan is None or not self.scan.texts:
            return None
        resolution = self.pairing()
        if resolution is not None and resolution.path is not None:
            return None
        if self.scan.labels:
            # The standard allows one working label per target and says two
            # should be an error, so a second would be a claim this cannot
            # adjudicate anyway.
            return self.scan.labels[0]
        # No label, but the chart may still print a bare serial -- the
        # ColorChecker SG's QR is exactly that -- and the selected target may say
        # where its units are published. Then the two facts needed to look one up
        # are both already on this screen, and it would be odd to make the
        # operator retype the serial in Settings to use them.
        from deltachrome.reference.openqualia import label_for_serial

        if not self.has_target:
            return None
        target = load_target(self.target_id)
        for text in self.scan.texts:
            label = label_for_serial(target, text)
            if label is not None:
                return label
        return None

    def _fetch_measurement(self) -> None:
        from deltachrome_ui.workers import MeasurementFetcher, run_in_thread

        label = self.fetchable_label()
        if label is None or self._fetching:
            return
        self._fetching = True
        self.fetch_button.setEnabled(False)
        self.fetch_button.setText("Fetching…")
        worker = MeasurementFetcher(label, reference_bank_dir())
        worker.loaded.connect(self._on_measurement_fetched)
        worker.failed.connect(self._on_fetch_failed)
        run_in_thread(worker, self)

    def _on_measurement_fetched(self, entry, label, caveat=None) -> None:
        """The file is in the bank; pair with it rather than only saying so.

        Fetching is a means, not an end: an operator who pressed it wants this
        capture measured against that unit, so the mode moves to *Pair from
        library* and the ordinary pairing finds the file by the code that was
        just bound to it. Nothing here selects the file directly -- if the
        pairing would not find it, that is a fault worth seeing now rather than
        one papered over by the button that caused it.
        """
        self._fetching = False
        self.fetch_button.setText("Fetch Measurement")
        self.fetch_button.setEnabled(True)
        self.library_dir = reference_bank_dir()
        self.mode_buttons[LIBRARY].setEnabled(True)
        self.mode_buttons[LIBRARY].setChecked(True)
        self._on_mode_changed()
        # A file that names a different unit than the one asked for is worth
        # stopping the operator over: the pairing will work, and every colour
        # number is wrong if the chart in front of them is not that unit.
        if caveat:
            QMessageBox.warning(self, "The measurement names a different unit", caveat)

    def _on_fetch_failed(self, failure) -> None:
        self._fetching = False
        self.fetch_button.setText("Fetch Measurement")
        self.fetch_button.setEnabled(True)
        QMessageBox.warning(self, failure.headline, failure.detail)

    def _choose_from_bank(self) -> None:
        from deltachrome_ui.reference_bank import BankPicker

        dialog = BankPicker(list(self.scan.texts) if self.scan else [], self)
        if dialog.exec() and dialog.chosen is not None:
            self.bank_choice = dialog.chosen
        self._on_mode_changed()

    def _choose(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Reference Measurements", "",
            "Reference data (*.txt *.cgats *.cie *.cxf *.xml);;All files (*)",
        )
        if chosen:
            self.reference_path = Path(chosen)
            self.path_label.setText(str(chosen))
        self._update_reference_pill()
        self.changed.emit()

    # -- what the run is made of -------------------------------------------

    def values(self) -> dict:
        return self.options.values()
