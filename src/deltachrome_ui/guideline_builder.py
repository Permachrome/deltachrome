"""Building a local specification from a published one.

The case, in the engine's own words: projects are rarely uniformly one star
level. *"Four star for 35mm photographs, except sampling frequency, which we can
set at 2000 ppi"* is a real and reasonable thing to want, and it is **not** FADGI
4-star. `AnalysisProfile.derive` has been able to express that from the beginning
and there has never been a way to reach it except by writing Python, so the one
feature that makes this application fit an actual department's practice has been
sitting behind a programming task.

**Nothing new is computed here.** This screen collects a parent, a set of
thresholds and a set of omissions, hands them to `derive`, and saves what comes
back. The departure list, the direction of each change, the forfeited conformance
claim and the summary sentence are all the engine's, unchanged -- which is what
makes a guideline built here indistinguishable from one written by hand, and why
it appears in the run setup and on the report's own profile control without a
line of code in either.

**What can be edited is deliberately narrow: the numbers.** A threshold's
*operator* is not offered, and that is not laziness. `compare_conditions` can say
`relaxed` or `tightened` only when the two conditions are the same shape; turn a
ceiling into a floor and the honest answer becomes `changed`, which is the least
useful thing a departure list can say about itself. Changing a direction is also
not a thing anybody wants from a standard -- moving where the line sits is. A
criterion this cannot express as numbers, such as a list of permitted file
formats or a threshold banded by original size, shows its condition and offers
only omission.

**Omitting is the other half and is not a weaker edit.** A metric a department
cannot measure with the equipment it has is better dropped from its own
specification than left to report `N/A` forever, and `derive` records the
omission as a departure exactly as it records a moved number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from deltachrome.errors import DeltachromeError, ProfileError
from deltachrome.metrics.conformance import permitted_key
from deltachrome.profiles.catalogue import (
    metric_order,
    metric_templates,
    vocabulary_of,
)
from deltachrome.profiles.schema import (
    DEFINITIONS_DIR,
    AnalysisProfile,
    available_profiles,
    is_local,
    load_profile,
    user_profile_dir,
)
from deltachrome_ui.labels import condition_label, metric_title, unit_label
from deltachrome_ui.status import TONE_ABSENT, TONE_QUALIFIED
from deltachrome_ui.profile_picker import ProfilePicker
from deltachrome_ui.theme import TEXT, TEXT_DIM, tone_colors
from deltachrome_ui.widgets import ComboBox, divider, elided_subtitle, heading

#: ``value <= 3.5``. The engine's own shape, matched here so a condition this
#: cannot take apart is shown rather than mangled.
_ONE_SIDED = re.compile(r"^\s*value\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")
_TWO_SIDED = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*(<=|<)\s*value\s*(<=|<)\s*(-?\d+(?:\.\d+)?)\s*$"
)


def slugify(name: str) -> str:
    """A profile id from a display name.

    Generated rather than asked for. An id is a filename and a key in every
    record that used it, and asking a person to invent one is asking them to make
    a decision whose consequences are invisible at the moment they make it.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "local_specification"


def free_id(base: str) -> str:
    """An id no installed profile answers to.

    Shadowing a shipped id is a real feature -- a site that has decided its own
    ``fadgi_3star_...`` means something particular should get what it decided --
    and it must never happen by *accident*, which is the only way it could happen
    from here.
    """
    taken = set(available_profiles())
    if base not in taken:
        return base
    for n in range(2, 1000):
        candidate = f"{base}_{n}"
        if candidate not in taken:
            return candidate
    return f"{base}_local"


def local_guidelines() -> list[AnalysisProfile]:
    """Every guideline the user owns, most recently named first."""
    found = []
    for profile_id in available_profiles():
        if not is_local(profile_id):
            continue
        try:
            found.append(load_profile(profile_id))
        except DeltachromeError:
            continue
    return sorted(found, key=lambda p: p.display_name.lower())


#: How a metric is treated in this guideline. Three states, and the middle one is
#: the reason this is not a checkbox: a metric can be *reported without being
#: judged*, which is what a department wants for a figure it tracks and does not
#: gate on, and it is a different thing from dropping the metric altogether.
JUDGED, MEASURED, OMITTED = "judged", "measured", "omitted"

STATES = (
    (JUDGED, "Judged",
     "Judged against the value set here."),
    (MEASURED, "Measured",
     "Reported without a threshold. Reads MEASURED, never PASS or FAIL."),
    (OMITTED, "Omitted",
     "Neither measured nor reported."),
)

#: What each state looks like. A **recession**, not a traffic light, and the
#: distinction is the one rule this application will not bend: green means *met
#: the threshold*, and painting "will be judged" green would put a verdict's
#: colour on a row that has not been measured yet. So Judged is the ordinary text
#: colour and the other two step back from it -- Measured in the tone the report
#: already gives a reading with no criterion, Omitted in the one it gives a metric
#: that is not there. A column of plain rows with amber and grey exceptions reads
#: at a glance and cannot be mistaken for a result.
STATE_TONES = {JUDGED: None, MEASURED: TONE_QUALIFIED, OMITTED: TONE_ABSENT}

#: The permitted values a guideline may choose from, per metric. A closed list on
#: purpose: a value nobody's table names is a value no capture will be matched
#: against by accident, and there is no way for a person to tell from this screen
#: whether a string they typed will match the ICC description in their own files.
#: Adding one is a decision to be made from what people actually shoot, and this
#: is the one place it would be made.
#:
#: One spelling per thing. The shipped tables name both `ProPhoto` and `ProPhoto
#: RGB`, and both `ECIRGB_v2` and `eciRGB v2`, because a table lists the spellings
#: a file might carry -- but the metric matches by substring both ways through
#: `permitted_key`, so the two are one value to everything that reads them, and
#: offering both would be two checkboxes that cannot disagree.
KNOWN_ALLOWED = {
    "master_file_format": ["TIFF", "JPEG 2000", "JPEG", "PNG", "PDF", "PDF/A", "DICOM"],
    "bit_depth": ["8", "16"],
    "color_space": [
        "Adobe RGB (1998)", "ECIRGB_v2", "Gray Gamma 2.2", "ProPhoto RGB", "sRGB",
    ],
}

#: The metric table's columns, and the only place their widths are written.
COL_STATE, COL_NAME, COL_STANDARD, COL_VALUE = 124, 186, 130, 200
#: 26 px of control inside a 32 px row. At 24 the theme's own padding clipped the
#: descender off "Judged" and the bottom off every digit -- measured on the render
#: rather than argued about.
FIELD_HEIGHT, ROW_HEIGHT = 26, 32


@dataclass
class _Edit:
    """One metric's row, and what the operator has done to it."""

    spec: object
    #: False where the parent does not declare this metric. The row is still
    #: there -- every guideline shows every metric, so two of them can be read
    #: against each other -- and its baseline is Omitted rather than Judged.
    declared: bool
    state: QComboBox
    fields: list[QLineEdit]
    template: str | None
    #: For a metric whose criterion is a list rather than a number: the button
    #: that opens the list editor, and the list as it currently stands.
    allowed_button: QPushButton | None = None
    allowed: list[str] | None = None
    #: Why this metric's threshold cannot be typed, where it cannot.
    fixed_reason: str | None = None
    #: The two labels that take the row's tone with the state control.
    name_label: object = None
    unit_label: object = None

    @property
    def mode(self) -> str:
        return self.state.currentData()

    def condition(self) -> str | None:
        """The condition these fields describe, or ``None`` if unusable."""
        if self.template is None:
            return None
        numbers = [f.text().strip() for f in self.fields]
        if any(not n for n in numbers):
            return None
        return self.template.format(*numbers)



#: The save button's resting word, and what it says for a moment after a save.
#: Two words rather than a glyph: this button is worded because a shape cannot
#: say which of several things it commits, and the confirmation inherits that.
SAVE_LABEL = "Save Guideline"
SAVED_LABEL = "Saved"
#: Matches `IconButton.flash`, so the two confirmations in this application last
#: the same length of time.
SAVED_MS = 1400

class GuidelineBuilder(QWidget):
    """The editor for one guideline: its name, what it starts from, its metrics."""

    #: A guideline was saved. Every profile control in the window is built from
    #: the catalogue at construction time, so they have to be told.
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.parent_profile: AnalysisProfile | None = None
        self.editing: AnalysisProfile | None = None
        self._edits: list[_Edit] = []
        #: Metrics set to Judged with nothing in the field. See :meth:`_departures`.
        self._incomplete: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(self._build_head())
        layout.addWidget(self._build_metrics(), 1)
        layout.addLayout(self._build_foot())

    # -- construction ------------------------------------------------------

    def _build_head(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)

        # The name, and only the name. The generated id was shown beside it in
        # monospace, and it is not something anybody needs: it is a filename and a
        # key in a record, it cannot be edited, and putting it on screen invites
        # the question of whether it is supposed to be.
        self.name = QLineEdit()
        self.name.setPlaceholderText("Name")
        self.name.textChanged.connect(self._on_name_changed)
        column.addWidget(self.name)

        # What this starts from, and *only while it is new*. Published standards
        # only: a guideline derived from another guideline nests its provenance,
        # and the departures a reader wants named are the ones from the published
        # table. Once saved the control is gone rather than disabled -- a heavily
        # modified guideline is its own specification, and a combo box naming
        # FADGI 4-star above it invites exactly the association it should not have.
        self.picker = ProfilePicker(horizontal=True, profile_ids=published_profiles())
        self.picker.changed.connect(self._on_parent_changed)
        column.addWidget(self.picker)
        return column

    def _build_metrics(self) -> QWidget:
        panel = QVBoxLayout()
        holder = QWidget()
        holder.setLayout(panel)
        panel.setContentsMargins(0, 0, 0, 0)
        panel.setSpacing(0)
        panel.addWidget(self._header_row())
        panel.addWidget(divider())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.metrics_body = QWidget()
        self.metrics_layout = QVBoxLayout(self.metrics_body)
        self.metrics_layout.setContentsMargins(0, 0, 12, 0)
        self.metrics_layout.setSpacing(0)
        scroll.setWidget(self.metrics_body)
        panel.addWidget(scroll, 1)
        return holder

    def _header_row(self) -> QWidget:
        """Column names, in the same widths the rows use.

        A table of five columns with one heading over the lot ("Metrics And Their
        Thresholds") leaves the reader to work out which number is the standard's
        and which is theirs -- which is the one thing this screen has to make
        obvious.
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 12, 6)
        layout.setSpacing(10)
        self.standard_heading = heading("Standard")
        self.standard_heading.setFixedWidth(COL_STANDARD)
        self.standard_heading.setToolTip("The standard's own values.")
        for label, width, tip in (
            (heading(""), COL_STATE, ""),
            (heading("Metric"), COL_NAME, ""),
            (self.standard_heading, COL_STANDARD, ""),
            (heading("Value"), COL_VALUE, ""),
            (heading("Unit"), 0, ""),
        ):
            if width:
                label.setFixedWidth(width)
            if tip:
                label.setToolTip(tip)
            layout.addWidget(label, 0 if width else 1, Qt.AlignmentFlag.AlignVCenter)
        return row

    def _build_foot(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addStretch(1)
        self.save_button = QPushButton(SAVE_LABEL)
        self.save_button.setObjectName("Primary")
        self.save_button.clicked.connect(self._save)
        row.addWidget(self.save_button)
        #: Restores the button's word after a save has been confirmed on it. Not
        #: an `IconButton.flash`, which swaps a glyph -- this button is worded
        #: precisely because a shape could not say what it does.
        self._saved_timer = QTimer(self)
        self._saved_timer.setSingleShot(True)
        self._saved_timer.timeout.connect(
            lambda: self.save_button.setText(SAVE_LABEL)
        )
        column.addLayout(row)
        return column

    # -- state -------------------------------------------------------------

    def start_new(self) -> None:
        """A fresh guideline, on whatever the picker is already showing."""
        self.editing = None
        self.name.setText("")
        self.picker.show()
        self.picker.setEnabled(True)
        self._load_parent()

    def edit(self, profile: AnalysisProfile) -> None:
        """Open a saved guideline for further editing -- **in place**.

        It used to be re-derived from its own parent, so that the departure list
        was recomputed against the standard rather than against the last edit.
        There is no parent any more: a saved guideline records that its criteria
        are local and which of them it never chose, and nothing else. So the
        guideline is its own baseline here, and "what changed" means what changed
        in *this* session.

        That is what ``inherited`` is for, and why it narrows rather than being
        recomputed. A row the operator edits now stops being inherited and does
        not become inherited again by being left alone next time, which is the
        property the old re-derivation was protecting by other means.
        """
        self.editing = profile
        self.name.setText(profile.display_name)
        self.picker.hide()
        # Its own baseline. Every row's "before" is what this guideline says, its
        # own metrics are what may be edited, and what it omits is what may be
        # added.
        self.parent_profile = profile
        self._build_rows(profile)
        self._on_name_changed()

    def _on_parent_changed(self) -> None:
        self.editing = None
        self._load_parent()

    def _on_name_changed(self) -> None:
        self._refresh_summary()

    @property
    def profile_id(self) -> str:
        """The id this would be saved under. Generated, and never shown.

        Fixed for a guideline that already exists: an id is a key in every record
        that used it, so renaming one would orphan them.
        """
        if self.editing is not None:
            return self.editing.profile_id
        text = self.name.text().strip()
        return free_id(slugify(text)) if text else ""

    def _load_parent(self, *, applying: AnalysisProfile | None = None) -> None:
        profile_id = self.picker.profile_id
        self.parent_profile = load_profile(profile_id) if profile_id else None
        self._build_rows(applying)
        self._on_name_changed()

    # -- the metric rows ---------------------------------------------------

    def _build_rows(self, applying: AnalysisProfile | None) -> None:
        while self.metrics_layout.count():
            item = self.metrics_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._edits = []
        if self.parent_profile is None:
            return

        # The standard's own column is shown while a guideline is being *made*,
        # where it is what you are departing from, and not once it is saved. A
        # saved guideline is its own specification -- the same reason its parent
        # is no longer a control on this screen -- and a column of somebody else's
        # numbers beside yours invites the comparison it should not.
        self.standard_heading.setVisible(self.editing is None)

        # Every metric Deltachrome can measure, in one order, whatever the parent
        # declares. A table that showed only what its parent states would give
        # FADGI X-Ray thirteen rows and FADGI Prints eighteen, and a site
        # comparing its two house standards would be comparing two shapes. What
        # differs between guidelines is the *state* of a row, not whether it
        # exists -- and a metric the standard never mentioned is Omitted, which is
        # its baseline and therefore not a departure.
        # A published standard says which body it is. A saved guideline being
        # amended does not -- it has no parent and states no body -- so the
        # vocabulary comes from its own metrics. See `vocabulary_of`: which shape
        # an *added* metric takes changes the measurement, and taking FADGI's
        # white balance into a Metamorfoze-shaped guideline would change it while
        # looking like a tidy-up.
        templates = metric_templates(
            self.parent_profile.body or vocabulary_of(self.parent_profile)
        )
        declared = {spec.metric_id: spec for spec in self.parent_profile.metrics}
        kept = {spec.metric_id: spec for spec in (applying.metrics if applying else [])}

        for index, metric_id in enumerate(
            m for m in metric_order() if m in templates
        ):
            spec = declared.get(metric_id) or templates[metric_id]
            applied = kept.get(metric_id)
            if metric_id in kept:
                mode = MEASURED if _unjudged(applied) else JUDGED
            elif applying is not None or metric_id not in declared:
                mode = OMITTED
            elif _unjudged(spec):
                mode = MEASURED
            else:
                mode = JUDGED
            if index:
                self.metrics_layout.addWidget(divider())
            self.metrics_layout.addWidget(
                self._row(spec, applied, mode, metric_id in declared)
            )
        self.metrics_layout.addStretch(1)

    def _row(self, spec, applied, mode: str, declared: bool) -> QWidget:
        row = QWidget()
        row.setObjectName("MetricRow")
        row.setFixedHeight(ROW_HEIGHT)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        state = ComboBox()
        state.setFixedWidth(COL_STATE)
        state.setFixedHeight(FIELD_HEIGHT)
        for key, label, tip in STATES:
            state.addItem(label, key)
            state.setItemData(
                state.count() - 1, tip, Qt.ItemDataRole.ToolTipRole
            )
        state.setCurrentIndex([k for k, _, _ in STATES].index(mode))
        layout.addWidget(state, 0, Qt.AlignmentFlag.AlignVCenter)

        name = _cell(elided_subtitle(metric_title(spec.metric_id)), COL_NAME)
        name.setStyleSheet("font-size: 12px;")
        layout.addWidget(name, 0, Qt.AlignmentFlag.AlignVCenter)

        # What the standard states, which for three of these is a list of
        # permitted values rather than a threshold. Saying "no threshold" there
        # is true and useless -- the criterion exists, it is just not a number.
        # A metric the parent does not declare has no "standard" to show. Blank
        # rather than a placeholder, which is what the operator asked for and is
        # also the honest answer: the table is silent about it.
        stated = _stated(spec) if declared else ""
        standard = _cell(elided_subtitle(stated), COL_STANDARD)
        standard.setToolTip(
            stated if declared
            else "Not in this standard."
        )
        standard.setVisible(self.editing is None)
        layout.addWidget(standard, 0, Qt.AlignmentFlag.AlignVCenter)

        value = QWidget()
        value.setFixedWidth(COL_VALUE)
        value_row = QHBoxLayout(value)
        value_row.setContentsMargins(0, 0, 0, 0)
        value_row.setSpacing(6)
        template, fields, button, allowed, reason = self._editor_for(
            spec, applied, declared
        )
        for widget in fields if button is None else [button]:
            value_row.addWidget(widget, 0, Qt.AlignmentFlag.AlignVCenter)
        value_row.addStretch(1)
        layout.addWidget(value, 0, Qt.AlignmentFlag.AlignVCenter)

        # Elided, with the whole of it on hover. `% of half-sampling frequency`
        # is 220 px of unit beside SFR50, which is a band and already the widest
        # row -- and a label that reports its full text as a minimum width does
        # not make the window wider, it makes the metric list scroll sideways.
        trailing = elided_subtitle(reason or unit_label(spec.units))
        trailing.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        trailing.setMinimumWidth(40)
        trailing.setToolTip(reason or unit_label(spec.units))
        layout.addWidget(trailing, 1, Qt.AlignmentFlag.AlignVCenter)

        edit = _Edit(
            spec=spec,
            declared=declared,
            state=state,
            fields=[f for f in fields if isinstance(f, QLineEdit)],
            template=template,
            allowed_button=button,
            allowed=allowed,
            fixed_reason=reason,
            name_label=name,
            unit_label=trailing,
        )
        self._edits.append(edit)
        if button is not None:
            button.clicked.connect(lambda _=False, e=edit: self._edit_allowed(e))
        state.currentIndexChanged.connect(
            lambda _index, e=edit: self._on_state_changed(e)
        )
        self._on_state_changed(edit, quiet=True)
        return row

    def _editor_for(self, spec, applied, declared: bool = True):
        """The controls for one metric's own criterion.

        Returns ``(template, fields, button, allowed, reason)``. ``template`` is
        the condition with ``{}`` where each number goes, so rebuilding the string
        is substitution rather than string surgery -- the operator is carried
        through from the standard untouched, which is the whole reason the
        operator is not editable.
        """
        if spec.metric_id in KNOWN_ALLOWED or spec.algorithm.get("allowed"):
            source = applied if applied is not None else spec
            allowed = [str(a) for a in (source.algorithm.get("allowed") or [])]
            if not declared and applied is None:
                allowed = []
            button = QPushButton()
            button.setFixedHeight(FIELD_HEIGHT)
            button.setFixedWidth(COL_VALUE)
            return None, [], button, allowed, None

        current = (applied.pass_condition if applied else None) or spec.pass_condition
        if spec.pass_condition_by is not None:
            return None, [], None, None, "banded by original size"

        # For a metric the parent does not declare, the *shape* comes from the
        # published tables -- which operator, and how many numbers -- and the
        # numbers themselves start blank. There is no value to default to and
        # inventing one would be this application stating a requirement nobody
        # published.
        blank = not declared and applied is None

        one = _ONE_SIDED.match(current or "")
        if one:
            return f"value {one.group(1)} {{}}", [
                self._operator(one.group(1)),
                self._number("" if blank else one.group(2)),
            ], None, None, None

        two = _TWO_SIDED.match(current or "")
        if two:
            between = QLabel("to")
            between.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
            return (
                f"{{}} {two.group(2)} value {two.group(3)} {{}}",
                [self._number("" if blank else two.group(1), width=76), between,
                 self._number("" if blank else two.group(4), width=76)],
                None, None, None,
            )
        return None, [], None, None, "not a number"

    def _operator(self, op: str) -> QLabel:
        label = QLabel(op)
        label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        label.setFixedWidth(20)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return label

    def _number(self, value: str, *, width: int = 88) -> QLineEdit:
        box = QLineEdit(value)
        box.setFixedWidth(width)
        box.setFixedHeight(FIELD_HEIGHT)
        box.setAlignment(Qt.AlignmentFlag.AlignRight)
        box.textChanged.connect(self._refresh_summary)
        return box

    def _on_state_changed(self, edit: _Edit, *, quiet: bool = False) -> None:
        """A criterion that does not apply is disabled rather than hidden.

        Hiding it would shift every column on that row, so a table whose point is
        that two numbers can be compared down a column would stop being one
        whenever a metric was set aside.
        """
        judged = edit.mode == JUDGED
        for widget in edit.fields:
            widget.setEnabled(judged)
        if edit.allowed_button is not None:
            edit.allowed_button.setEnabled(judged)
            self._label_allowed(edit)
        self._tone_row(edit)
        if not quiet:
            self._refresh_summary()

    def _tone_row(self, edit: _Edit) -> None:
        """Colour the state, and take the row with it.

        The metric's name and its unit move too, so an omitted row recedes as a
        whole rather than carrying one grey word in a line of white ones -- which
        is what makes the column readable at a glance instead of needing to be
        read word by word.
        """
        tone = STATE_TONES.get(edit.mode)
        color = tone_colors(tone)[0] if tone else TEXT
        edit.state.setStyleSheet(f"QComboBox {{ color: {color}; }}")
        if edit.name_label is not None:
            edit.name_label.setStyleSheet(f"font-size: 12px; color: {color};")
        if edit.unit_label is not None:
            edit.unit_label.setStyleSheet(
                f"font-size: 11px; color: {color if tone else TEXT_DIM};"
            )

    def _label_allowed(self, edit: _Edit) -> None:
        """The permitted list on the button, elided to fit.

        A push button clips rather than elides, so `TIFF, JPEG 2000, PDF/A` ran off
        the end of the cell with no ellipsis to say it had. The whole list is the
        tooltip.
        """
        values = edit.allowed or []
        full = ", ".join(values) if values else "— none permitted —"
        button = edit.allowed_button
        metrics = QFontMetrics(button.font())
        # Against the column's width rather than the button's own: this runs
        # while the row is being built, before any layout has happened, so
        # ``button.width()`` is still Qt's default and the text came out elided
        # to the wrong length and then centre-clipped at both ends.
        button.setText(metrics.elidedText(
            full, Qt.TextElideMode.ElideRight, COL_VALUE - 28
        ))
        button.setToolTip(full)

    def _edit_allowed(self, edit: _Edit) -> None:
        dialog = AllowedValuesDialog(
            metric_title(edit.spec.metric_id),
            KNOWN_ALLOWED.get(edit.spec.metric_id, []),
            edit.allowed or [],
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        edit.allowed = dialog.values()
        self._label_allowed(edit)
        self._refresh_summary()

    # -- what would be saved -----------------------------------------------

    def _unsaved(self, preview) -> bool:
        """Whether the form says something the saved guideline does not.

        Computed by comparing against what is on disk rather than by tracking a
        dirty flag through every editor, because a flag has to be cleared in as
        many places as it is set and the one that gets missed is a Save button
        that lies. ``_preview`` already builds the profile the form describes,
        and already runs on every edit, so this cannot go stale.

        The name counts as well as the criteria: renaming a saved guideline keeps
        its id -- an id is a key in every record that used it -- but it is still
        an edit, and a Save button that stayed disabled through a rename would
        lose it.
        """
        if self.editing is None or preview is None:
            return True
        return (
            preview.display_name != self.editing.display_name
            or preview.metrics != self.editing.metrics
        )

    def _carried(self) -> list[str] | None:
        """The inherited set to carry forward, or ``None`` on a fresh derivation.

        ``None`` means "everything untouched is inherited", which is exactly right
        the first time: the operator was handed a table and changed some of it.
        Amending is the other case -- the untouched set this time round includes
        rows they chose deliberately in an earlier session, and those are
        decisions. `derive` intersects rather than replaces, so a row leaves the
        inherited set the moment it is edited and never rejoins it.
        """
        return None if self.editing is None else self.editing.inherited

    def _departures(self) -> dict:
        """Everything that differs from the parent, in `derive`'s own vocabulary.

        Four kinds and they are kept apart because the engine keeps them apart: a
        moved threshold, a changed list of permitted values, a metric measured but
        not judged, and a metric left out.
        """
        thresholds: dict[str, str] = {}
        algorithms: dict[str, dict] = {}
        omit: list[str] = []
        unjudge: list[str] = []
        add: dict[str, object] = {}
        self._incomplete = []
        for edit in self._edits:
            metric_id = edit.spec.metric_id
            if not edit.declared:
                # Its baseline is absence, so Omitted is no departure at all --
                # saying "average_color_accuracy omitted" of a table that never
                # mentioned it would be a departure from something that was never
                # there.
                if edit.mode == OMITTED:
                    continue
                spec = self._added_spec(edit)
                if spec is None:
                    self._incomplete.append(metric_id)
                else:
                    add[metric_id] = spec
                continue
            if edit.mode == OMITTED:
                omit.append(metric_id)
                continue
            if edit.mode == MEASURED:
                # Only a departure where the parent judged it. A metric the
                # standard already leaves unjudged is unchanged, not changed to
                # what it already was.
                if not _unjudged(edit.spec):
                    unjudge.append(metric_id)
                continue
            if edit.allowed is not None:
                if edit.allowed != [str(a) for a in edit.spec.algorithm.get("allowed", [])]:
                    algorithms[metric_id] = {"allowed": edit.allowed}
                continue
            condition = edit.condition()
            if condition is None and edit.template is not None:
                self._incomplete.append(metric_id)
            elif condition and condition != edit.spec.pass_condition:
                thresholds[metric_id] = condition
        return {
            "thresholds": thresholds,
            "algorithms": algorithms,
            "omit": omit,
            "unjudge": unjudge,
            "add": add,
        }

    def _added_spec(self, edit: _Edit):
        """The full spec for a metric this guideline is adding, or ``None``.

        ``None`` means the operator asked for it and has not said what it should
        be -- Judged with an empty field. Refused rather than quietly dropped: a
        row set to Judged and then ignored is the control lying about what it did.
        """
        from dataclasses import replace

        spec = edit.spec
        if edit.mode == MEASURED:
            algorithm = dict(spec.algorithm)
            algorithm.pop("allowed", None)
            return replace(
                spec, pass_condition=None, pass_condition_by=None, algorithm=algorithm
            )
        if edit.allowed is not None:
            if not edit.allowed:
                return None
            return replace(spec, algorithm={**spec.algorithm, "allowed": edit.allowed})
        condition = edit.condition()
        if condition is None:
            return None if spec.pass_condition_by is None else spec
        return replace(spec, pass_condition=condition)

    def _preview(self) -> AnalysisProfile | None:
        """What `derive` would return, or ``None`` if it would refuse.

        The engine is asked rather than imitated. A summary composed here would be
        a second implementation of the sentence that goes into the record, and the
        two would drift.
        """
        if self.parent_profile is None:
            return None
        departures = self._departures()
        if self._incomplete:
            return None
        try:
            return self.parent_profile.derive(
                self.profile_id or "preview",
                display_name=self.name.text().strip() or None,
                allow_identical=True,
                inherited=self._carried(),
                **departures,
            )
        except ProfileError:
            return None

    def _refresh_summary(self) -> None:
        """What the save button can and cannot do, and why.

        The departure sentence used to be shown here as a coloured callout above
        the button. It is gone: the table already shows every departure, column
        against column, and an amber block is the shape this application uses for
        a *warning* -- which is the wrong signal for something the operator has
        just deliberately done. The sentence itself is unchanged and is still
        composed by the engine, written into the record, and shown on the report
        of every run judged against this guideline.
        """
        preview = self._preview()
        named = bool(self.name.text().strip())
        unsaved = self._unsaved(preview)
        self.save_button.setEnabled(preview is not None and named and unsaved)
        # Only two things can stop a save: a metric asked to be judged with nothing
        # to judge it by, and no name. Departing from nothing is not one of them --
        # "our standard is FADGI 3-star, for now" is a real thing to record, and
        # the guideline is where the site's later decisions will go.
        if preview is None:
            self.save_button.setToolTip(
                "Set a value for "
                + ", ".join(metric_title(m) for m in self._incomplete)
                + ", or set it to Omitted."
            )
            return
        if not named:
            self.save_button.setToolTip("Name the guideline first.")
            return
        # The departure sentence, and only while the guideline is new -- the same
        # rule the Standard column and the parent combo follow. It opens with the
        # parent standard's name and reads "with noise relaxed from ... to ...",
        # which is exactly the connection a saved guideline is meant to have shed:
        # once saved it is its own specification, and the table below states it.
        # "Nothing outstanding" is the state the button spends most of its life in
        # once a guideline exists, and the one it could never previously express:
        # a Save button that stays live forever cannot answer "is there anything
        # of mine not written down", which is the question an editor is actually
        # asked. It goes live again on the next edit.
        #
        # The conformance sentence stays either way. It is a fact about the
        # guideline rather than about the button, so it must not come and go with
        # whether there is anything to press.
        state = "" if unsaved else "No unsaved changes."
        self.save_button.setToolTip(
            "\n\n".join(part for part in (
                state,
                "Runs judged against this guideline are not conformance results.",
            ) if part)
        )

    # -- saving ------------------------------------------------------------

    def _save(self) -> None:
        preview = self._preview()
        if preview is None or self.parent_profile is None:
            return
        profile_id = self.profile_id
        if not profile_id:
            return
        # A shipped id can be shadowed on purpose and must never be shadowed by
        # accident, which is the only way it could happen from here.
        shipped = DEFINITIONS_DIR / f"{profile_id}.json"
        if self.editing is None and shipped.exists():
            self._warn(
                "That name is already a published standard.",
                f"{profile_id} names a profile that ships with Deltachrome. "
                "Choose another name.",
            )
            return
        try:
            saved = self.parent_profile.derive(
                profile_id,
                display_name=self.name.text().strip(),
                allow_identical=True,
                inherited=self._carried(),
                **self._departures(),
            )
            saved.save()
        except (DeltachromeError, OSError) as exc:
            self._warn("This guideline was not saved.", str(exc))
            return
        self.editing = saved
        self.picker.hide()
        self.changed.emit()
        # Said on the button, after `changed` rather than before: the panel
        # reloads on that signal and reopens this guideline, which rebuilds the
        # form and would otherwise put the resting word straight back.
        #
        # It is worth saying at all because amending a guideline changes nothing
        # on screen -- the name is already in the field, the parent combo is
        # already gone, the row in the list already reads what it read. Creating
        # one does visibly change the form, but a confirmation that only appears
        # the first time is one nobody learns to trust.
        self.save_button.setText(SAVED_LABEL)
        self._saved_timer.start(SAVED_MS)

    def _warn(self, text: str, detail: str) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(text)
        box.setInformativeText(detail)
        box.exec()


def published_profiles() -> list[str]:
    """The standards a guideline may be built from: everything that ships.

    A guideline derived from another guideline nests its provenance -- the
    departure list would be measured against the intermediate rather than against
    the published table -- and the departures a reader wants named are the ones
    from the table. It is also how a guideline saved in an earlier session ended
    up offered as a parent, which is not a thing anybody meant to ask for.
    """
    return [pid for pid in available_profiles() if not is_local(pid)]


def _cell(label, width: int):
    """An elided label pinned to a column width.

    ``ElidedLabel`` asks for :attr:`QSizePolicy.Policy.Ignored` horizontally, which
    is what lets it shrink instead of forcing a panel wide -- and which also lets a
    layout shrink it *past* a fixed width, so the Standard column collapsed to ten
    pixels showing `TI` and `>:`. Fixed both ways here: the width is the column's,
    and the elision happens inside it.
    """
    label.setFixedWidth(width)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
    return label


def _unjudged(spec) -> bool:
    """Whether this metric carries no criterion of any kind.

    Three shapes count as one: a threshold, a threshold banded by a run input,
    and the list of permitted values the file-level metrics carry. A metric with
    none of them runs and reports and is not judged.
    """
    return not (
        spec.pass_condition
        or spec.pass_condition_by is not None
        or spec.algorithm.get("allowed")
    )


def _stated(spec) -> str:
    """What the published table says about a metric, in one cell."""
    allowed = spec.algorithm.get("allowed")
    if allowed:
        return ", ".join(str(a) for a in allowed)
    if spec.pass_condition_by is not None:
        return "by original size"
    return condition_label(spec.pass_condition) or "— not judged —"


class AllowedValuesDialog(QDialog):
    """Which values a guideline permits, for a criterion that is a list.

    Three metrics carry one -- the master's container, its bit depth and its
    colour space -- and they are the tables' own requirements rather than
    decoration: a level that permits TIFF and JPEG 2000 is not met by a JPEG,
    however good the colour is. A site restricting its own specification to one
    container, or adding the working space it actually uses, is the same kind of
    edit as moving a threshold.

    **The list is closed**, and that is a shipping decision rather than a
    limitation of the control. A colour space is matched against the description
    string inside an ICC profile, and nothing on this screen could tell somebody
    whether a name they typed will match the profile their own files carry -- so a
    free-text box would offer a guideline that silently fails every capture.
    Values are added to :data:`KNOWN_ALLOWED` when there is a reason to, from what
    people actually shoot.

    A value a guideline already carries is still listed even where no table names
    it, because a profile written by hand is entitled to say what it says and this
    must not silently drop it.
    """

    def __init__(self, title: str, known: list[str], current: list[str],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        layout.addWidget(heading("Permitted Values"))

        self._boxes: list[QCheckBox] = []
        self._list = QVBoxLayout()
        self._list.setContentsMargins(0, 0, 0, 0)
        self._list.setSpacing(4)
        layout.addLayout(self._list)

        chosen = {permitted_key(v): v for v in current}
        for value in known:
            self._add(value, permitted_key(value) in chosen)
        for key, value in chosen.items():
            if key not in {permitted_key(v) for v in known}:
                self._add(value, True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add(self, value: str, checked: bool) -> None:
        box = QCheckBox(value)
        box.setChecked(checked)
        self._boxes.append(box)
        self._list.addWidget(box)

    def values(self) -> list[str]:
        return [box.text() for box in self._boxes if box.isChecked()]


def delete_guideline(profile: AnalysisProfile) -> None:
    """Remove a local guideline's file.

    Not moved aside the way a run record is. A record cannot be regenerated
    except by finding the master and placing it again by hand; a guideline is a
    handful of numbers and its own summary says which ones.
    """
    path = profile.source_path
    if path is None or path.parent != user_profile_dir():
        raise ProfileError(
            f"{profile.profile_id} is not one of your own guidelines."
        )
    path.unlink(missing_ok=True)
    from deltachrome.profiles.schema import _index

    _index.cache_clear()
