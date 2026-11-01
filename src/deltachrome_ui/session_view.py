"""Several saved runs, read together.

Almost no chart measures everything a standard asks for. A ColorChecker SG has no
slanted edges; an ISA transmissive target has no colour patches at all. So a site
shoots two or three targets in one sitting, and the question it is actually asking
is whether *the system* met the specification -- not whether any one chart did.
The engine has been able to answer that since `analyze_session`; no screen reached
it, which meant the only way to ask was a hand-written JSON file and a terminal.

**It is assembled from the library, not from a second capture pipeline.** That is
the whole design decision here, and it follows from the record being the archive:
every number a session reduces was written down when each run was made, so
combining them needs no pixels, no placement and no re-measurement. Selecting three
saved runs is instant where re-analyzing three masters is a minute of decoding to
recompute numbers that have not changed. It also matches how the work is actually
done -- analyze each chart as it comes, ask the combined question at the end --
rather than demanding the operator know at the start that a sitting was a session.

**The screen is arranged around coverage, not around captures.** Which chart
supplied which reading is the detail underneath; whether anything reached each
metric at all is the point, because that is the one thing a session can say that a
single run cannot. A lone chart's blind spots are inherent. A session's are a
choice about which charts were shot, and can be closed by shooting another.

**Where two captures disagree gets its own section.** Two charts measuring one
system should agree about it, and where they do not the question is which is wrong
-- a misplaced quad, the wrong reference file, a chart that is not what it claims
-- not what their average is. Metrics that describe each *file* rather than the
system are excluded from that: two captures differ in bit depth or sampling rate by
design.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from deltachrome_ui.frame_map import FrameBlock, FrameInfo
from deltachrome_ui.labels import condition_label, metric_title
from deltachrome_ui.status import metric_status, overall_status
from deltachrome_ui.tables import CENTER, RIGHT, Col, DataTable
from deltachrome_ui.widgets import caveat, divider, heading, subtitle
from deltachrome_ui.theme import BAR_HEIGHT, TEXT_DIM, VERDICT_WIDTH



def _row(metric) -> dict[str, Any]:
    """One metric's readings across the session, flattened for the table."""
    summary = metric.summary()
    return {
        "metric_id": metric.metric_id,
        "status": metric.status,
        "covered_by": summary["covered_by"],
        "not_measurable_by": summary["not_measurable_by"],
        "values": summary["values"],
        # Three metrics read as a word or a whole number rather than as a
        # figure -- the master's container, its colour space, its bit depth --
        # and the engine says which by recording ``value_text``. Carried here so
        # this table shows ``JPEG 2000`` where it used to show a dash: the
        # numeric path had nothing to format and fell through.
        "texts": [r.value_text for _, r in metric.measured if r.value_text],
        "per_capture": summary["per_capture"],
        "governing": metric.governing,
    }


def _value_text(row: dict[str, Any]) -> str:
    """One value where the targets agree, a range where they do not.

    Three decimals and the body face, which is how a chart's own report writes
    a reading -- the two tables are read one after the other, and a figure that
    changes precision or typeface between them reads as a different quantity.

    A range is also the whole of what this screen says about disagreement now.
    There was a Spread column and a section below the table spelling out each
    metric's readings capture by capture, and both said what ``0.209 – 0.254``
    already says: they do not agree, and by about this much. Which of them is
    wrong is a question for that target's own report, which is one click away.
    """
    texts = row["texts"]
    if texts:
        # Every target read the same file, so these agree by construction --
        # but say so rather than assume it.
        unique = list(dict.fromkeys(texts))
        return unique[0] if len(unique) == 1 else ", ".join(unique)
    values = list(row["values"].values())
    if not values:
        return "—"
    low, high = min(values), max(values)
    if high - low < 5e-4:
        return f"{low:.3f}"
    return f"{low:.3f} – {high:.3f}"


def conformance_caveat(session) -> str | None:
    """Why this capture is not a conformance result, in one line, or ``None``.

    Composed by the engine's own :func:`advisory_line` from the two facts it
    always rests on -- what the readings were compared against, and whether the
    criteria were the standard's own -- so the line a capture shows here and the
    line each of its targets shows on its own report are built the same way and
    cannot drift into saying different things about one run.

    The **worst** basis across the targets governs. A capture whose five targets
    include one judged on nominal values is a capture whose verdict rests partly
    on nominal values, and the session's own verdict already takes the same view.
    """
    from deltachrome.profiles.evaluator import advisory_line
    from deltachrome.reference.cgats import PER_UNIT_MEASUREMENT, POPULATION_MEAN

    # Worst first: anything unrecognised is treated as the weakest evidence,
    # which is the safe direction and the one `advisory_line` already takes.
    bases = {c.reference_basis for c in session.captures}
    if bases - {PER_UNIT_MEASUREMENT, POPULATION_MEAN}:
        worst = next(iter(bases - {PER_UNIT_MEASUREMENT, POPULATION_MEAN}))
    elif POPULATION_MEAN in bases:
        worst = POPULATION_MEAN
    else:
        worst = PER_UNIT_MEASUREMENT
    return advisory_line(
        reference_basis=worst,
        local_criteria=bool(getattr(session.profile, "local_criteria", False)),
    )


def _condition(row: dict[str, Any]) -> str:
    governing = row["governing"]
    if governing is None:
        return "— not measured —"
    # Written as the report writes it: ">= 190" rather than "value >= 190".
    return condition_label(governing[1].condition) or "— no threshold —"


class SessionView(QWidget):
    """The combined report for a set of saved runs."""

    backRequested = Signal()
    #: This screen is a *capture's* own, and saving it means saving the capture
    #: -- one record holding every target, named after the image, exactly as a
    #: single-target run is.
    #:
    #: There is no second, session-shaped save. There was: it wrote a record
    #: with no ``run_id``, no ``image`` and no ``charts``, which the library
    #: listed as a row called ``session-a1b2c3-d4e5f6`` and opened as a report
    #: of nothing. Combining several *captures* is withdrawn entirely until it
    #: is designed where it belongs -- beside the run rather than scraped back
    #: out of the library afterwards.
    saveCaptureRequested = Signal()
    #: A capture's own run was asked for. A session never replaces the individual
    #: reports -- it combines them without altering any -- so getting back to one
    #: has to be a click rather than a navigation back through the library.
    captureRequested = Signal(str)  # run_id
    #: Every target of this capture should be written out as text, CSV or XML --
    #: the same export a target's own report offers, which writes the whole
    #: capture from either screen.
    exportRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.session = None
        self.export_button = None
        #: The capture behind this session, when its captures are one frame.
        #: Supplied by the shell: a session assembled from three separate files
        #: has no such frame and this stays ``None``.
        self.frame: FrameInfo | None = None
        self.frame_block: FrameBlock | None = None
        #: What this screen's primary action commits: ``capture`` writes one
        #: record for the image and every target in it, ``none`` offers nothing
        #: because what is on screen is already in the library.
        #:
        #: ``none`` is the default because the safe answer to "which record
        #: should this write" is not to guess one. The shell always says which
        #: it means.
        self.save_mode = "none"


        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header_slot = QVBoxLayout()
        self._header_slot.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._header_slot)
        layout.addWidget(divider())

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        layout.addWidget(self._scroll, 1)

    # -- content -----------------------------------------------------------

    def set_session(self, session, frame: FrameInfo | None = None,
                    save_mode: str = "none") -> None:
        self.session = session
        self.frame = frame if frame is not None and frame.usable else None
        self.frame_block = None
        self.save_mode = save_mode

        while self._header_slot.count():
            item = self._header_slot.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._header_slot.addWidget(self._build_header(session))

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(28, 20, 28, 28)
        column.setSpacing(18)
        caveat_block = self._build_verdict(session)
        if caveat_block is not None:
            column.addWidget(caveat_block)
        # The frame, where there is one, stands in for the captures list: on a
        # board of one model that list is four identical filenames, and the
        # outlines say the same thing and say which is which.
        captures = self._build_frame(session) or self._build_captures(session)
        column.addWidget(captures)
        column.addWidget(self._build_metrics(session))
        # Two sections stood below the table and both were the table read aloud.
        # *Where the Captures Disagree* listed each metric's readings target by
        # target, which the Values column's own range already says -- and which
        # of them is wrong is a question for that target's report, one click
        # away on its outline. *Reached by No Capture* listed the metrics
        # nothing measured, which are the rows already sitting in the table
        # marked N/A. Neither is deleted from the record; both are still in it,
        # and `deltachrome analyze` still prints them.
        column.addStretch(1)
        self._scroll.setWidget(body)

    # -- header ------------------------------------------------------------

    def _build_header(self, session) -> QWidget:
        """The bar every screen wears, at the height every screen wears it.

        It was a padded row with a rounded ``← Library`` button at one end and a
        rounded ``Save Session`` at the other, which is the shape this
        application stopped using everywhere else: 52 px, the verdict and the
        name on the left, the actions on the right as squares of the bar's own
        height butted together, and the primary action as an accent block flush
        into the corner.
        """
        bar = QWidget()
        bar.setObjectName("HeaderBar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setFixedHeight(BAR_HEIGHT)
        layout = QHBoxLayout(bar)
        # No right margin and none top or bottom: that is what puts the actions
        # against the edge of the window.
        layout.setContentsMargins(20, 0, 0, 0)
        layout.setSpacing(16)

        # A bare layout, never a QWidget holding one. The bar paints itself
        # through ``#HeaderBar``, and a plain child widget inside it paints the
        # *base* background over that -- a lighter rectangle around the name and
        # the filename, boxing them off from the bar they sit in. The report's
        # own header has always added this as a layout; wrapping it was the
        # accident. It also clipped the filename's first letter, the wrapper
        # being one pixel narrower than the text it held.
        stack = QVBoxLayout()
        stack.setContentsMargins(0, 0, 0, 0)
        # No gap: the two line boxes already separate them, and a gap here is
        # height the bar pays for twice.
        stack.setSpacing(0)
        # This screen is a capture's own, so it says where you are -- *All
        # Targets*, the place a chart's report steps up to -- and puts the
        # capture's name on the second line, which is what every other screen in
        # this application has in that slot.
        headline = "All Targets"
        # The capture's filename, never the session's capture *labels* --
        # those read "chart 1 (Applied Image QA-62)", which names neither
        # the image nor anything a reader would recognise it by.
        capture = self.frame.filename if self.frame is not None else ""
        below = (f"{capture}   ·   {session.profile.display_name}"
                 if capture else session.profile.display_name)
        name = QLabel(headline)
        name.setStyleSheet("font-size: 14px; font-weight: 600;")
        stack.addWidget(name)
        stack.addWidget(subtitle(below))


        from deltachrome_ui.widgets import IconButton, VerdictStripe

        # The verdict leads the bar, as it does on a report, and takes the same
        # floor width: a reader comparing two captures is comparing exactly the
        # ones whose verdicts differ, so the name beside it must not shift by a
        # hundred pixels according to what the answer turned out to be.
        status = overall_status(
            session.overall, session.conformance_claim,
            evidence_sound=getattr(session, "evidence_is_sound", None),
        )
        stripe = VerdictStripe(status)
        stripe.setMinimumWidth(VERDICT_WIDTH)
        stripe.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)
        layout.addWidget(stripe, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(stack, 1)
        layout.addSpacing(8)

        # Butted together, each carrying its own hairline on its left edge, so
        # the group reads as a strip of cells rather than a row of objects.
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(0)
        back = IconButton("back", "Back to the library", size=BAR_HEIGHT)
        back.clicked.connect(self.backRequested.emit)
        actions.addWidget(back)
        # Always: this screen only ever shows one capture's targets, combining
        # separate captures having been withdrawn from the frontend.
        export = IconButton(
            "export", "Export values as TXT, CSV or XML", size=BAR_HEIGHT,
        )
        export.clicked.connect(self.exportRequested.emit)
        actions.addWidget(export)
        self.export_button = export

        if self.save_mode == "capture":
            # The same action, the same words and the same block as the one on a
            # chart's report: a capture is saved once, whichever of its screens
            # the operator happens to be looking at.
            save = QPushButton("Save to Library")
            save.setObjectName("PrimaryBar")
            save.setFixedHeight(BAR_HEIGHT)
            save.setCursor(Qt.CursorShape.PointingHandCursor)
            save.setToolTip(
                "Saves all targets as one record."
            )
            save.clicked.connect(self.saveCaptureRequested.emit)
            actions.addWidget(save)
        layout.addLayout(actions)
        return bar

    # -- sections ----------------------------------------------------------

    def _build_verdict(self, session) -> QWidget | None:
        block = QWidget()
        column = QVBoxLayout(block)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        # What was here: a bold coverage count, a sentence saying the guidelines
        # were not fully covered, and a paragraph naming every metric that
        # failed and every metric nothing reached. All three are the table read
        # aloud -- it has a row per metric with a verdict on each, which is the
        # same facts in the form a reader can sort, search and scan. The table
        # is what this screen is for; a paragraph above it restating it is one a
        # reader learns to skip, and it taught them to skip the line that is not
        # a restatement.
        #
        # That line is the caveat, and it survives because nothing else says it:
        # a verdict reached on nominal values is not a conformance result, and no
        # column of the table carries that. It is composed from the same two
        # facts the record's long sentence is composed from rather than by
        # shortening it, so a reopened capture produces the identical line.
        line = conformance_caveat(session)
        if line is None:
            return None
        column.addWidget(caveat(line))
        return block

    def _build_frame(self, session) -> QWidget | None:
        """The capture with every chart outlined where it was measured.

        Only when the shell supplied one, which it does for a capture holding
        several charts and never for a session assembled from separate files --
        three charts shot in one sitting are three frames, and a picture of one
        of them would be a picture of the wrong thing.
        """
        if self.frame is None:
            return None
        block = FrameBlock(self.frame)
        block.chartActivated.connect(self.captureRequested.emit)
        self.frame_block = block
        return block

    def _build_captures(self, session) -> QWidget:
        block = QWidget()
        column = QVBoxLayout(block)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addWidget(heading("Captures"))
        for capture in session.captures:
            column.addWidget(self._capture_row(capture))
        return block

    def _capture_row(self, capture) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        name = QLabel(capture.name)
        name.setStyleSheet("font-size: 13px;")
        layout.addWidget(name, 0)

        detail = capture.target_name
        if capture.reference_basis != "per_unit_measurement":
            detail += "   ·   nominal reference"
        note = QLabel(detail)
        note.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        layout.addWidget(note, 1)

        from deltachrome_ui.widgets import StatusBadge

        layout.addWidget(
            StatusBadge(overall_status(
                capture.overall, capture.conformance_claim,
                evidence_sound=capture.reference_basis == "per_unit_measurement",
            )),
            0, Qt.AlignmentFlag.AlignVCenter,
        )
        if capture.run_id:
            open_run = QPushButton("Open")
            open_run.clicked.connect(
                lambda _checked=False, rid=capture.run_id: self.captureRequested.emit(rid)
            )
            layout.addWidget(open_run, 0, Qt.AlignmentFlag.AlignVCenter)
        return row

    def _build_metrics(self, session) -> QWidget:
        block = QWidget()
        column = QVBoxLayout(block)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        # Five targets on one board are not five captures, and this screen is
        # the one place that distinction is visible. The word follows what the
        # screen is about; the engine's own `coverage_line` still says
        # "captures" and is still what the CLI prints, which is correct there --
        # a session combining separate images really is several captures, and
        # `deltachrome session` still does that.
        column.addWidget(heading("Metrics Across Targets"))

        rows = [_row(m) for m in session.metrics]
        table = DataTable(self._columns(), rows)
        table.setMinimumHeight(min(460, 34 + 26 * max(len(rows), 1)))
        column.addWidget(table)
        # The footnote that stood here explained the Spread column's "per
        # capture" marker. The column has gone, and with it the thing that
        # needed explaining.
        return block

    def _columns(self) -> list[Col]:
        return [
            Col("", lambda r: "", tone=lambda r: metric_status(r["status"]).tone,
                width=32, resize="fixed",
                tooltip=lambda r: metric_status(r["status"]).hint,
                sort=lambda r: metric_status(r["status"]).rank),
            Col("Metric", lambda r: metric_title(r["metric_id"]), width=232,
                tooltip=lambda r: (
                    "Describes the file, so each target is judged on its own."
                    if r["per_capture"] else ""
                )),
            # Wide enough for a range: "8.000 – 16.000" is what two disagreeing
            # targets look like, and it is the case the column exists for. In
            # the body face and at three decimals, exactly as a chart's own
            # report writes a reading.
            Col("Values", _value_text, align=RIGHT, width=164,
                sort=lambda r: (not r["values"],
                                max(r["values"].values()) if r["values"] else 0.0),
                tooltip=lambda r: ", ".join(
                    f"{name} = {value:.3f}" for name, value in r["values"].items()
                ) or "No value."),
            Col("Measured By",
                lambda r: ", ".join(r["covered_by"]) or "—",
                width=220, dim=True, resize="stretch",
                tooltip=lambda r: _coverage_hint(r)),
            Col("Condition", _condition, width=192, dim=True),
            Col("Verdict", lambda r: metric_status(r["status"]).label,
                align=CENTER, width=118,
                sort=lambda r: metric_status(r["status"]).rank,
                tooltip=lambda r: metric_status(r["status"]).hint),
        ]

def _coverage_hint(row: dict[str, Any]) -> str:
    parts = []
    if not row["covered_by"]:
        parts.append("No target measured this.")
    if row["not_measurable_by"]:
        parts.append(", ".join(row["not_measurable_by"]) + " can't measure this.")
    return "  ".join(parts)
