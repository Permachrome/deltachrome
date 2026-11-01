"""One screen: the capture, where the target sits on it, and what to judge it as.

There were two. A setup screen chose the target, the profile and the reference
and then handed over to this one, which placed the chart. The split made sense
while placement was a click-and-drag job that needed to know which chart it was
placing -- but automatic identification places the chart on the way to naming it,
so the corners are known before the first screen would have appeared, and a
screen whose job was to come first had nothing left to come before.

So the choices moved into this panel (:mod:`deltachrome_ui.choices`) and the
screen they were on is gone. Two consequences worth stating, because both are
improvements the merge gets for free rather than costs it pays:

* **Changing the target re-finds the chart, in place.** It always did; it used to
  do it behind a full-screen wait after the operator had left the screen where
  they made the choice. Now the canvas stays on screen and the panel says what is
  happening, so the connection between the choice and its effect is visible.
* **The reference requirement is answered before the run, not by it.** A chart
  with no manufacturer values and nothing paired to it cannot be measured at all,
  which used to be a dialog arrived at after the analysis started.

**What the panel says, and what it no longer says.** Four blocks of standing
prose came off this screen: how to drag a handle, how automatic placement works,
what the finder keyed on, and what the preview is not. Every one of them was true
on every run and therefore read past by anyone who had read it once. They went
behind an icon, and then the icon went too -- they are documentation now
(:mod:`deltachrome_ui.explanations`), written apart from the application so that
a beta can find out which of these controls a person in the trade can read
unaided.

Three things stay in the open, and the test for each is that it is a fact about
*this* run rather than about the application:

* **The corner readout**, in image pixels and in the CLI's own ``--corners``
  syntax, so a placement made here is reproducible on the command line.
* **The ordering check.** The engine warns when the placed quadrilateral's aspect
  disagrees with the chart's real proportions, which catches a great deal -- but
  not a correctly-proportioned quad whose corners are in the wrong order. That
  homography is wrong without looking wrong, and here it can be caught while the
  user still has a hand on the mouse.
* **The target's own placement instruction**, which is shown when it is needed
  and not before. The ISA Golden Thread's corners are the centers of the white
  fields inside its corner *modules*, nowhere near the corners of the card --
  nobody guesses that, and getting it wrong is a placement that looks fine and
  measures the wrong pixels. It is furniture while the chart is being found
  automatically and essential the moment a human takes over, so that is exactly
  when it appears: on a refusal, or on the first drag.
"""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from deltachrome_ui.canvas import CORNER_LABELS, PlacementCanvas
from deltachrome_ui.choices import ChoicesPanel
from deltachrome_ui.imageproxy import ImageProxy
from deltachrome_ui.status import TONE_PASS, TONE_QUALIFIED, TONE_UNKNOWN
from deltachrome_ui.theme import BAR_HEIGHT, BORDER
from deltachrome_ui.workers import SampleRegionFinder, run_in_thread
from deltachrome_ui.widgets import (
    BusyStrip,
    ComboBox,
    ElidedLabel,
    IconButton,
    Pill,
    divider,
    elided_subtitle,
    figures_font,
    heading,
)

@dataclass(frozen=True)
class ChartOffer:
    """One target on the strip: whether it will be run, and where it came from.

    Held by the shell and handed here to render. The view never decides which
    charts exist -- that is the search's answer and the operator's edit of it --
    and keeping the decision out of the widget is what lets the same strip show a
    board of four found automatically and a second chart somebody added by hand.
    """

    added: bool = False


#: Wider than it was. The panel now carries the whole setup as well as the
#: placement, and at 330 the two combo boxes naming a FADGI category elided into
#: uselessness.
PANEL_WIDTH = 380

#: How long the corners must be still before the sampled regions are redrawn.
#: Sampling every region costs 0.15 to 0.5 s on the captures in this repository,
#: which is fine once and hopeless per mouse-move, so a drag clears the regions
#: and this decides when they come back.
SAMPLE_REFRESH_DELAY_MS = 250


def sample_regions(image, target, corners):
    """Every region a run on these corners would read, of both kinds.

    Patches *and* slanted edges. An edge ROI has the same way of being silently
    wrong as a patch ROI does -- it can straddle the wrong transition, or catch a
    neighbouring printed feature, and either returns a confident number -- and on
    a chart that measures only sharpness the edges are the whole of what there is
    to look at. Which regions exist is read from the target rather than assumed,
    so such a chart needs nothing added here to be checkable.

    A free function because two callers need it and neither should own it: the
    worker that computes it off the UI thread, and any caller that wants the
    answer without a thread. It touches no widget, which is what makes running it
    on another thread safe.
    """
    from deltachrome.imaging.edges import extract_edges
    from deltachrome.imaging.geometry import place_chart
    from deltachrome.imaging.sampling import sample_patches

    placement = place_chart(
        corners,
        image_size=(image.width, image.height),
        corner_positions_relative=target.fiducials.corner_positions or None,
    )
    polygons: list = []
    flagged: set[str] = set()
    if target.grid_regions:
        # Noise is a per-patch statistic the report will compute; drawing an
        # outline does not need it, and it is the expensive half.
        sampling = sample_patches(image, placement, target, measure_noise=False)
        polygons = [
            (patch_id, sample.polygon_px)
            for patch_id, sample in sampling.samples.items()
            if sample.usable
        ]
        # A flagged region is still drawn distinctly, which is where a clipped or
        # contaminated patch can actually be looked at. What is gone is the amber
        # summary that used to sit under the button: it announced a count before
        # anything had been measured, and the report says the same per patch with
        # the reading beside it.
        flagged = {pid for pid, s in sampling.samples.items() if s.flags}

    edges: list = []
    if any(r.kind == "slanted_edge" for r in target.regions):
        # The ROI is cut with the refinement the measurement uses, not a cheaper
        # approximation of it. A picture of a region that is not the region being
        # read is worse than no picture: it would confirm a placement the run does
        # not use. It is also the expensive half -- 818 of the 898 ms this costs
        # on the ISA transmissive 4x5 -- and the reason it runs off the UI thread.
        edges = [
            (roi.edge_id, roi.bounds_px, bool(roi.notes))
            for roi in extract_edges(image, placement, target)
        ]
    return polygons, flagged, edges


class PlacementView(QWidget):
    """The canvas, the choices, and the panel that keeps both honest."""

    backRequested = Signal()
    #: Emitted with the four corners when the run is started.
    continueRequested = Signal(list)
    #: A different chart was chosen, so its placement has to be found again. The
    #: shell owns the decoded capture and the worker threads, so it does the
    #: finding; this only says that it is wanted.
    targetChanged = Signal(str)
    #: Which of several charts found in one frame should be shown. Emitted with
    #: its index. The shell holds the placements, for the same reason it holds
    #: the capture: this screen shows one placement at a time and does not own
    #: the search that produced them.
    chartSelected = Signal(int)
    #: A chart the search did not find. The frame holds one the operator can see
    #: and the finder could not -- two charts too close to separate is the case
    #: this exists for -- so it is added by hand and placed by dragging, which is
    #: what this screen has always been able to do for one chart.
    chartAdded = Signal()
    #: Take the shown chart out of the run, or put it back. Emitted with its index.
    chartDropped = Signal(int)
    #: A rectangle the operator dragged round the chart, as ``(x0, y0, x1, y1)``
    #: in image pixels. The shell owns the capture and the worker threads, so it
    #: does the searching; this only says where.
    regionSearched = Signal(tuple)
    #: Build a camera profile from the chart on screen, emitted with its corners.
    profileRequested = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.proxy: ImageProxy | None = None
        self.target = None
        self.image_path: Path | None = None
        #: Whether these corners came out of a refinement. Part of what produced
        #: the placement, so it belongs in the command that reproduces it.
        self._refined = False
        #: Whether these corners were found rather than placed. Kept so a hand
        #: moved corner stops claiming to be a detection, for the same reason it
        #: stops claiming to be a refinement.
        self._detected = False
        #: How many charts the run will actually cover. One until a frame is found
        #: to hold several, which is every capture but a board.
        self._charts_running = 1
        #: How many captures the folder holds, when this placement is being made
        #: for a batch rather than for the capture on screen. Zero otherwise, which
        #: is every ordinary run.
        self._batch_total = 0
        #: Whether the placed quad is usable at all. Held rather than read back
        #: off the button, because two separate conditions gate the run now and
        #: whichever is checked second would otherwise overwrite the first.
        self._placement_valid = True
        #: Whether a search is running. The panel is inert while one is, since
        #: every control on it would change the answer being computed.
        self._busy = False
        #: Whether the sampled regions are drawn. **On by default**: the regions
        #: are the one thing on this screen that says whether the placement is
        #: good enough to measure from, and a button offering to reveal them
        #: asks the operator to suspect a problem before showing them one. It is
        #: a preference rather than per-capture state, so turning them off stays
        #: off for the next chart.
        self._show_sampling = True
        #: Whether this chart declares anything to draw. A frame-filling white
        #: sheet declares neither patches nor edges, and asking the sampler for
        #: patches raises rather than returning nothing.
        self._has_regions = False
        self._has_edges = False
        self._sample_timer = QTimer(self)
        self._sample_timer.setSingleShot(True)
        self._sample_timer.timeout.connect(self._refresh_sample_regions)
        #: One sampling job at a time, with a request that arrived while it ran
        #: remembered rather than queued -- see :meth:`_refresh_sample_regions`.
        self._sample_running = False
        self._sample_pending = False
        #: Whether anybody has said which chart this is. **Nothing is drawn over
        #: the capture until somebody has**, because the combo always holds a
        #: value and a capture nothing could be identified in therefore opens on
        #: whichever target happens to be first in the list. Drawing that chart's
        #: hundred and forty ROIs over a capture of something else is not a
        #: neutral default: it is the screen asserting a chart it has no evidence
        #: for, in the one place the operator is meant to be checking evidence.
        self._target_settled = False
        #: Whether these corners are a placement rather than a starting rectangle.
        #: A chart **added by hand** begins on the canvas's default inset box, in
        #: the middle of the frame and over pixels that are almost certainly not a
        #: chart -- so drawing its ROIs there commits the same fault as drawing an
        #: unidentified chart's: a picture of something nobody has claimed. It is a
        #: separate flag from the one above because the two unknowns clear
        #: differently. *Which chart* is settled by naming it; *where it is* is
        #: settled by somebody putting it somewhere.
        self._placement_settled = True
        #: Whether the capture carries no ICC profile. It can be placed and a
        #: camera profile built from it, and nothing else: every QC measurement
        #: reads the capture through its profile.
        self._untagged = False

        # A bar across the top, then the canvas and its panel under it -- the
        # same shape the report and the library carry. What moved up into it is
        # what those two put there: where you are on the left, what you can do
        # on the right, and the primary action as a block in the corner.
        #
        # It is not only consistency. The panel scrolls, and ``Run Analysis``
        # was the last thing in it: at 1320x800 the one action this screen
        # exists to reach was below the fold. Taking four controls out of the
        # panel is what puts the rest of it on screen at once.
        shell = QVBoxLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addWidget(self._build_header())

        row = QWidget()
        shell.addWidget(row, 1)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.canvas = PlacementCanvas()
        self.canvas.cornersChanged.connect(self._on_corners_changed)
        self.canvas.cornersDragged.connect(self._on_corners_dragged)
        self.canvas.validityChanged.connect(self._on_validity_changed)
        self.canvas.regionDrawn.connect(self._on_region_drawn)
        self.canvas.regionAbandoned.connect(self._on_region_abandoned)
        layout.addWidget(self.canvas, 1)

        # The panel scrolls. It holds a variable number of controls -- run
        # options appear only where a profile can use them -- and a fixed panel
        # silently truncates whichever happens to be last.
        panel = QScrollArea()
        panel.setFixedWidth(PANEL_WIDTH)
        panel.setWidgetResizable(True)
        panel.setFrameShape(QScrollArea.Shape.NoFrame)
        panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel.setStyleSheet(f"QScrollArea {{ border-left: 1px solid {BORDER}; }}")
        layout.addWidget(panel)

        contents = QWidget()
        panel.setWidget(contents)
        self.panel_layout = QVBoxLayout(contents)
        self.panel_layout.setContentsMargins(20, 20, 20, 20)
        self.panel_layout.setSpacing(12)
        self._build_panel()

    # -- panel -------------------------------------------------------------

    def _build_header(self) -> QWidget:
        """The capture on the left, what can be done with it on the right.

        The two labels were the panel's own title and subtitle, and they say the
        same thing the report's header says in the same place: which file this
        is, and what the file itself is. The title drops to the filename size
        the other bars use -- a 19 px heading and a description under it do not
        fit in 52 px.
        """
        bar = QWidget()
        bar.setObjectName("HeaderBar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setFixedHeight(BAR_HEIGHT)
        header = QHBoxLayout(bar)
        header.setContentsMargins(20, 0, 0, 0)
        header.setSpacing(16)

        stack = QVBoxLayout()
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(0)
        self.title_label = ElidedLabel()
        self.title_label.setStyleSheet("font-size: 14px; font-weight: 600;")
        stack.addWidget(self.title_label)
        self.subtitle_label = elided_subtitle()
        stack.addWidget(self.subtitle_label)
        header.addLayout(stack, 1)
        header.addSpacing(8)

        # Which of several targets in this frame is being placed. Hidden on the
        # ordinary capture, which holds one -- a control offering a choice of one
        # asks the operator to consider an arrangement they have not got. It sits
        # with the actions rather than with the title because it changes what is
        # on screen, which is what the report's own header combos do.
        self.chart_combo = ComboBox()
        self.chart_combo.setFixedHeight(24)
        self.chart_combo.currentIndexChanged.connect(self._on_chart_chosen)
        self.chart_combo.hide()
        header.addWidget(self.chart_combo)
        header.addSpacing(12)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(0)

        back = IconButton("back", "Back to the library", size=BAR_HEIGHT)
        back.clicked.connect(self.backRequested.emit)
        actions.addWidget(back)

        self.copy_button = IconButton(
            "command",
            "Copy command",
            size=BAR_HEIGHT,
        )
        self.copy_button.clicked.connect(self._copy_command)
        actions.addWidget(self.copy_button)

        #: Beside the run rather than in the panel: it leaves this screen for
        #: another, as Run Analysis does, and the panel is about the placement.
        self.profile_button = QPushButton("Build Profile")
        self.profile_button.setObjectName("SecondaryBar")
        self.profile_button.setFixedHeight(BAR_HEIGHT)
        self.profile_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.profile_button.clicked.connect(
            lambda: self.profileRequested.emit(self.canvas.corners())
        )
        actions.addWidget(self.profile_button)

        self.continue_button = QPushButton("Run Analysis")
        self.continue_button.setObjectName("PrimaryBar")
        self.continue_button.setFixedHeight(BAR_HEIGHT)
        self.continue_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.continue_button.clicked.connect(
            lambda: self.continueRequested.emit(self.canvas.corners())
        )
        actions.addWidget(self.continue_button)

        header.addLayout(actions)
        return bar

    def _build_panel(self) -> None:
        # -- what this is being judged as, and against what
        self.choices = ChoicesPanel()
        self.choices.targetChanged.connect(self.targetChanged.emit)
        self.choices.changed.connect(self._update_run_button)
        self.panel_layout.addWidget(self.choices)
        self.panel_layout.addWidget(divider())

        # -- placement
        self.panel_layout.addWidget(heading("Placement"))

        found_row = QHBoxLayout()
        found_row.setSpacing(6)
        self.detection_pill = Pill()
        self.detection_pill.hide()
        found_row.addWidget(self.detection_pill)
        found_row.addStretch(1)
        self.panel_layout.addLayout(found_row)

        #: The wait for a search, shown where the answer will be. Narrower than
        #: the strip's own default because the panel is 320 px wide and this one
        #: sits inside its margins.
        self.busy_strip = BusyStrip(width=PANEL_WIDTH - 56, font_px=12)
        self.busy_strip.hide()
        self.panel_layout.addWidget(self.busy_strip, 0, Qt.AlignmentFlag.AlignHCenter)

        self.corner_labels: list[QLabel] = []
        for name in CORNER_LABELS:
            row = QHBoxLayout()
            row.setSpacing(10)
            key = QLabel(name)
            key.setObjectName("Subtitle")
            key.setFixedWidth(34)
            key.setAlignment(Qt.AlignmentFlag.AlignRight)
            value = QLabel("—")
            # Sized by the style sheet, which would otherwise override a font's
            # size; the face's tabular figures survive it.
            value.setFont(figures_font(12.0, value.font()))
            value.setStyleSheet("font-size: 12px;")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            row.addWidget(key)
            row.addWidget(value, 1)
            self.corner_labels.append(value)
            self.panel_layout.addLayout(row)

        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(
            "color: #ff6050; background: #3a1f1c; border: 1px solid #ff605044;"
            "border-radius: 6px; padding: 8px 10px; font-size: 12px;"
        )
        self.warning.hide()
        self.panel_layout.addWidget(self.warning)

        self.panel_layout.addWidget(heading("Tools"))

        # One strip rather than three buttons. They are three answers to one
        # question -- how should the canvas be framed -- and butting them
        # together says so, the way the header's action squares do. They keep
        # their words: "Fit" and "1:1" are shorter as text than any icon is as a
        # picture, and "1:1" has no icon at all.
        buttons = QHBoxLayout()
        buttons.setSpacing(0)
        for name, label, slot in (
            ("SegmentLeft", "Fit", lambda: self.canvas.fit()),
            ("SegmentMiddle", "1:1", lambda: self.canvas.zoom_to(1.0)),
            ("SegmentRight", "Reset", lambda: self.canvas.seed_default()),
        ):
            button = QPushButton(label)
            button.setObjectName(name)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        self.panel_layout.addLayout(buttons)

        #: Placed above *Refine Placement* because that is the order the two run
        #: in: this finds the chart, refinement snaps what was found onto the
        #: printed features.
        self.region_button = QPushButton("Search a Region\u2026")
        self.region_button.setCheckable(True)
        self.region_button.clicked.connect(self._toggle_region_search)
        #: **Round the whole card, not round its patches**, which is the part an
        #: operator gets wrong and the one that costs the search. Measured on the
        #: Golden Thread Object Level strip in a Smithsonian Odonata frame: a
        #: rectangle on the card itself finds the chart at every padding from 2%
        #: inside it to 80% outside, landing within 2.5 px of the answer; a
        #: rectangle on the patch row -- which is what "the chart" looks like on a
        #: strip -- finds nothing at all until it has been padded by 40%, which is
        #: only an indirect way of reaching the card's own height. Every size gate
        #: in a finder is a fraction of the region's short side, and on a strip
        #: that side is the patches' own height until the printed rules and the
        #: value table are inside the box.
        self.region_button.setToolTip(
            "Drag a rectangle around the whole target including a margin."
        )
        self.panel_layout.addWidget(self.region_button)

        self.refine_button = QPushButton("Refine Placement")
        self.refine_button.clicked.connect(self._refine)
        self.panel_layout.addWidget(self.refine_button)

        #: What the last refinement did. A pill and not the paragraph it was: a
        #: refinement moves every measurement that follows it, so it cannot be
        #: silent -- but "moved 12-31 px onto 10 printed features, which agree
        #: with the result to 0.2 px" is four lines of a panel for a fact the
        #: corner readout above already shows having changed. The sentence is on
        #: hover, where the run that surprises somebody can still reach it.
        self.refine_pill = Pill()
        self.refine_pill.hide()
        self.panel_layout.addWidget(self.refine_pill)

        self.check_button = QPushButton("Hide Sampling")
        self.check_button.clicked.connect(self._toggle_sample_regions)
        self.panel_layout.addWidget(self.check_button)

        # The last two tools act on the *list* of targets rather than on the one
        # being placed, so they sit at the end of the group. With words rather
        # than icons: "add" and "ignore" as glyphs at 22 px are a plus and a
        # cross, and a cross beside a target selector reads as "close this"
        # rather than "leave this one out".
        self.add_chart_button = QPushButton("Add a Target")
        self.add_chart_button.clicked.connect(self.chartAdded.emit)
        self.panel_layout.addWidget(self.add_chart_button)

        self.drop_chart_button = QPushButton("Remove This Target")
        self.drop_chart_button.clicked.connect(
            lambda: self.chartDropped.emit(self.chart_combo.currentIndex())
        )
        self.drop_chart_button.hide()
        self.panel_layout.addWidget(self.drop_chart_button)

        self.panel_layout.addStretch(1)
        self._update_run_button()

    # -- the run button ----------------------------------------------------

    def set_batch(self, total: int) -> None:
        """Place for a folder rather than for this capture.

        The screen is unchanged -- it is still one capture with the chart on it,
        and checking that placement is still the whole job. What changes is what
        the primary action starts, and the button has to say so: pressing it runs
        a hundred captures, and a button reading "Run Analysis" would be the one
        place in this application where the label understates what it commits.
        """
        self._batch_total = int(total)
        self._update_run_button()

    def update_readiness(self) -> None:
        """Re-ask whether a run could start. Public because the shell changes the
        answer -- choosing the blank target entry is the case."""
        self._update_run_button()

    def _update_run_button(self) -> None:
        """Three separate conditions, and the button says which one is unmet.

        A crossed quadrilateral has no homography worth computing, a target with
        no reference has nothing to be measured against, and a frame with no
        target left in it has nothing to run. None is a reason to let the run
        start and fail; the first two were, once, in different ways.

        Its label counts the targets, because since removal became outright the
        list *is* what will run -- which is the plainest form of the rule this
        button has always followed, that it may not say "Run 5 Targets" and run
        four.
        """
        running = self._charts_running
        if self._batch_total:
            # Placing for a batch. The chart on screen is the first of a folder and
            # the button says what pressing it starts, which is not this capture.
            self.continue_button.setText(f"Run Batch of {self._batch_total}")
        else:
            self.continue_button.setText(
                f"Run {running} Targets" if running > 1 else "Run Analysis"
            )
        placeable = self._placement_valid and not self._busy
        ready = self.choices.ready
        self.continue_button.setEnabled(
            placeable and ready and running > 0 and not self._untagged
        )
        self._update_profile_button(placeable)

        # **Adding a target needs a target.** With the blank entry selected there
        # is no model to add -- a chart with no definition has no patches, no
        # edges and nothing to measure, and building one crashed the selection
        # handler on the first press. The offer on a capture nothing was named in
        # is *Search a Region*, which is what this says.
        self.add_chart_button.setEnabled(self.choices.has_target)
        self.add_chart_button.setToolTip(
            "" if self.choices.has_target else
            "Choose a target first, or use Search a Region."
        )

        if not self.choices.has_target:
            self.continue_button.setToolTip(
                "Choose a target, or use Search a Region."
            )
            return
        if self._untagged:
            self.continue_button.setToolTip(
                "This capture has no embedded ICC profile. Build a profile from it, "
                "or embed one to analyze it."
            )
            return
        if running == 0:
            self.continue_button.setToolTip(
                "No targets left. Use Add a Target."
            )
        elif not ready:
            self.continue_button.setToolTip(
                "This target needs a reference file."
            )
        elif not self._placement_valid:
            self.continue_button.setToolTip(
                "The corners don't form a usable shape."
            )
        else:
            self.continue_button.setToolTip("")

    def _update_profile_button(self, placeable: bool) -> None:
        """Offered for any placed chart with colour patches and reference values."""
        from deltachrome.profiling import profilable

        # A batch's placement is the folder's seed; a profile is of one capture.
        self.profile_button.setVisible(not self._batch_total)
        why = ""
        if self.target is None or not self.choices.has_target:
            why = "Choose a target first."
        else:
            ok, why = profilable(self.target)
            if ok and not self.drawing_regions:
                why = "Place the target first."
            elif ok and not placeable:
                why = "The corners don't form a usable shape."
        self.profile_button.setEnabled(not why and not self._busy)
        self.profile_button.setToolTip(why)

    def set_untagged(self, untagged: bool) -> None:
        """The capture carries no ICC profile, so only a profile can be built."""
        self._untagged = bool(untagged)
        self._update_run_button()

    # -- population --------------------------------------------------------

    def settle_target(self) -> None:
        """Somebody has said which chart this is, so the regions may be drawn.

        Three things say it and they are all the same claim: identification named
        the chart, the operator chose it from the combo, or a region search named
        one. What does *not* say it is a capture opening on whatever the combo
        happens to hold.
        """
        if self._target_settled:
            return
        self._target_settled = True
        self._update_run_button()
        # The handles come back with it. While nothing is named there is nothing
        # for them to be a placement *of*, and showing them invites the gesture
        # this screen wants reached for last.
        self.canvas.set_placement_visible(True)
        self._schedule_sample_refresh()

    def unsettle_target(self) -> None:
        """Nothing here is known to be the chart in the frame. Draw nothing.

        Not the regions and not the handles either. The canvas is left clean, so
        the offer on the screen is *Search a Region* rather than four reticles in
        the middle of the frame -- which is the order these are meant to be
        reached in: found automatically, then searched for in a region somebody
        drew, and placed by hand only when both have failed.
        """
        self._target_settled = False
        self._update_run_button()
        self._sample_timer.stop()
        self.canvas.clear_sample_regions()
        self.canvas.set_placement_visible(False)

    @property
    def target_settled(self) -> bool:
        return self._target_settled

    def settle_placement(self) -> None:
        """These corners are somebody's placement now, not a starting rectangle."""
        if self._placement_settled:
            return
        self._placement_settled = True
        self._update_run_button()
        self._schedule_sample_refresh()

    def unsettle_placement(self) -> None:
        """A chart with nowhere yet to be. Draw nothing over the capture."""
        self._placement_settled = False
        self._update_run_button()
        self._sample_timer.stop()
        self.canvas.clear_sample_regions()

    @property
    def placement_settled(self) -> bool:
        return self._placement_settled

    @property
    def drawing_regions(self) -> bool:
        """Whether enough is known to draw anything. Both halves must hold."""
        return self._target_settled and self._placement_settled

    # -- searching a region the operator drew -------------------------------

    def _toggle_region_search(self) -> None:
        """Arm or disarm the next drag.

        A checkable button rather than a momentary one because the canvas is in a
        different mode until the gesture happens, and a mode nothing shows is a
        mode the operator discovers by finding that panning has stopped working.
        """
        if self.region_button.isChecked():
            self.canvas.begin_region_search()
        else:
            self.canvas.cancel_region_search()

    def _on_region_drawn(self, region) -> None:
        """Hand the rectangle up, and say that a search is running.

        **The rectangle is a search region and never a placement.** It would be
        far easier to treat the four dragged corners as the chart's own and be
        done, and it would be wrong in the way this project cares most about: a
        hand-drawn box is out by tens of pixels, which reads every ROI off the
        edge of its patch and returns a full set of confident numbers. What it is
        good for is telling the finders where to look, which is a question it only
        has to answer to within a few percent.
        """
        self.region_button.setChecked(False)
        self.regionSearched.emit(region)

    def _on_region_abandoned(self) -> None:
        """A click, or a drag too small to be a rectangle. Put the button back."""
        self.region_button.setChecked(False)

    def _refine(self) -> None:
        """Snap to the printed features, and say exactly what changed.

        Never silent and never automatic. A refinement moves every measurement
        that follows it, so the operator sees how far the corners went and on how
        much evidence, and can undo it by placing them again.
        """
        if self.proxy is None or self.target is None:
            return
        from deltachrome.imaging.geometry import place_chart
        from deltachrome.imaging.refine import refine_placement

        placement = place_chart(
            self.canvas.corners(),
            image_size=(self.proxy.width, self.proxy.height),
            corner_positions_relative=self.target.fiducials.corner_positions or None,
        )
        result = refine_placement(self.proxy.image, placement, self.target)

        # A refusal is not an error: the placement given is still usable, and
        # some charts print nothing this can snap to at all.
        if not result.applied:
            self._show_refine("declined", "Not Refined", result.reason)
            return

        self.canvas.set_corners(result.corners)
        self.settle_placement()
        self._refined = True
        if result.max_shift_px < 1.0:
            self._show_refine(
                "confirmed", "Already Refined",
                f"Already on the printed features — {result.features_used} of them "
                f"agree with this placement to {result.residual_px:.1f} px. "
                "Nothing moved.",
            )
            return
        self._show_refine(
            "moved",
            f"Refined · {max(result.moved_px):.0f} px",
            f"Corners moved {min(result.moved_px):.0f}–{max(result.moved_px):.0f} px "
            f"onto {result.features_used} printed features, which agree with the "
            f"result to {result.residual_px:.1f} px. Every measurement that follows "
            "uses the new placement.",
        )

    #: Which tone each refinement outcome reads in. A refusal is amber rather
    #: than red for the reason above, and "nothing moved" is neutral rather than
    #: green: confirming a placement is not an improvement to it.
    _REFINE_TONE = {
        "moved": TONE_PASS, "confirmed": TONE_UNKNOWN, "declined": TONE_QUALIFIED,
    }

    def _show_refine(self, outcome: str, label: str, detail: str) -> None:
        self.refine_pill.set_state(label, self._REFINE_TONE[outcome], detail=detail)

    # -- sampling ----------------------------------------------------------

    def _toggle_sample_regions(self) -> None:
        """Flip whether the sampled regions are drawn, and say which way it is.

        The button used to name the *action* while its label was set from
        whichever branch of the toggle last ran, and the two came apart the
        moment this view was reused for a second capture: it read "Hide
        Sampling" over a canvas with nothing on it, so the first press appeared
        to do the opposite of what it said.

        The label is now derived from the state and nowhere else, in
        :meth:`_sync_check_button`, so there is no branch that can set one
        without the other.
        """
        self._show_sampling = not self._show_sampling
        self._sync_check_button()
        if self._show_sampling:
            self._refresh_sample_regions()
        else:
            self.canvas.clear_sample_regions()

    def _sync_check_button(self) -> None:
        showing = self._show_sampling and self._has_regions
        self.check_button.setEnabled(self._has_regions)
        self.check_button.setText("Hide Sampling" if showing else "Check Sampling")
        self.check_button.setToolTip(
            "" if self._has_regions
            else ("No regions to sample on this target."
                  if self.target else "")
        )

    def _schedule_sample_refresh(self) -> None:
        """Redraw once the corners stop moving, rather than on every pixel.

        Sampling every region costs 163 to 898 ms across the committed placements
        -- the worst being the ISA transmissive 4x5, where 818 of those go on
        snapping ten quarter discs -- which is fine once and hopeless per mouse
        move. So the regions are cleared the instant a corner moves, because what
        is drawn is then a picture of a placement that no longer exists, and
        recomputed when the movement stops.

        **The delay is not what made this bearable and never could be.** While
        the work ran on the UI thread, waiting longer only moved a near-second
        freeze; it is off the thread now, and this timer is back to doing the one
        thing it is good for, which is not computing an answer nobody asked to
        see. The figure quoted here used to be "0.15 to 0.5 s" and was measured
        before the edge ROIs were cut with the refinement the run uses.
        """
        self.canvas.clear_sample_regions()
        self._sample_timer.start(SAMPLE_REFRESH_DELAY_MS)

    def _refresh_sample_regions(self) -> None:
        """Ask for every region this run would read, and draw it when it arrives.

        **Off the UI thread**, because the picture is deliberately the real thing.
        The edge ROIs drawn are the ones the measurement cuts, refinement and all
        -- see :func:`sample_regions` -- and that is 898 ms on the ISA transmissive
        4x5. Run inline it landed squarely on the gesture this screen exists for:
        the overlay rebuilds 250 ms after the corners stop moving, so placing a
        corner *carefully* -- nudge, look, nudge -- put a near-second freeze
        between every nudge and the next, with the mouse events queueing behind it
        and the handle jumping to catch up when it returned. Working more
        precisely made it worse.

        **One job at a time.** A drag outruns this by a wide margin, so starting a
        job per request would pile up threads all but one of whose answers is
        already stale before it is computed. A request arriving while one runs
        sets ``_sample_pending`` instead, and the result that comes back is drawn
        only if the corners have not moved under it.
        """
        self._sample_timer.stop()
        if not (self._show_sampling and self._has_regions and self.drawing_regions):
            return
        if self.proxy is None or self.target is None:
            return
        token = self._sample_token()
        if self._sample_running:
            self._sample_pending = True
            return
        self._sample_running = True
        self._sample_pending = False
        finder = SampleRegionFinder(
            image=self.proxy.image, target=self.target,
            corners=self.canvas.corners(), token=token,
        )
        finder.finished.connect(self._on_sample_regions)
        run_in_thread(finder, self)

    def _sample_token(self):
        """What a set of regions belongs to: this capture, this chart, these corners.

        Identity for the first two rather than equality, because a target reloaded
        from the registry is a different object describing the same chart and the
        regions computed for either are equally good. What must match exactly is
        the corners, which is the thing a drag changes.
        """
        return (
            id(self.proxy.image) if self.proxy is not None else None,
            id(self.target),
            tuple(self.canvas.corners()),
        )

    def _on_sample_regions(self, result) -> None:
        """Draw what came back, if it is still about what is on screen."""
        self._sample_running = False
        if not self.drawing_regions:
            return
        if self._sample_pending or result.token != self._sample_token():
            # The corners moved while this was computed, so it is a picture of a
            # placement that no longer exists -- which is the same reason the
            # overlay is cleared the instant a handle moves.
            self._sample_pending = False
            self._refresh_sample_regions()
            return
        if not (self._show_sampling and self._has_regions):
            return
        self.canvas.show_sample_regions(
            result.polygons, result.flagged, result.edges
        )

    def set_busy(self, message: str) -> None:
        """A search is running. Say so here rather than on a screen of its own.

        Changing the target used to leave this screen for a full-window wait and
        come back to it, which severed the choice from its effect: the operator
        picked a chart from a list and the next thing they saw was a different
        screen. The canvas stays put now and the pill carries the wait.
        """
        self._busy = True
        # **The bar says it, so the pill does not.** This set a chip reading
        # "Searching the Region..." directly above a bar reading "Searching the
        # Region..." -- the same sentence twice, one of which was also wearing the
        # tone this screen uses for a placement that needs attention. The chip is
        # for what the search *found*; while it is running there is nothing found
        # to report.
        self.detection_pill.hide()
        # The same indeterminate bar the loading screen and the capture panel
        # use. A pill reading "Searching the Region..." says a search started and
        # nothing after that -- and a region search on a hundred-megapixel master
        # runs for seconds, over a canvas that does not change, which is
        # indistinguishable from a press that did nothing.
        self.busy_strip.set_message(message)
        self.busy_strip.show()
        self.choices.setEnabled(False)
        self._update_run_button()

    def _clear_busy(self) -> None:
        self._busy = False
        self.busy_strip.hide()
        self.choices.setEnabled(True)
        self._update_run_button()

    #: Outcome styling for an automatic placement. A refusal is not an error --
    #: the operator places the chart by hand, which is what they did before this
    #: existed -- and a success is not a result either, since nothing has been
    #: measured yet.
    _DETECTION_TONE = {True: TONE_PASS, False: TONE_QUALIFIED}

    def offer_charts(self, charts: "list[ChartOffer]", index: int = 0, *,
                     emptied: bool = False) -> None:
        """The targets in this frame, which one is showing, and what may be done.

        Takes the state of each rather than a count, because three controls read
        from it and they need different things: the selector needs a label per
        target, the run button needs how many will actually run, and the ignore
        button needs to know whether the shown target is one the operator added.

        The selector is shown only when there is more than one -- a control
        offering a choice of one asks the operator to consider an arrangement they
        have not got -- but *Add a Target* is always there, because the frame that
        needs it most is the one where the search found a single target and the
        operator can see two.
        """
        count = len(charts)
        self.chart_combo.blockSignals(True)
        self.chart_combo.clear()
        if count > 1:
            self.chart_combo.addItems([
                f"Target {n} of {count}" for n in range(1, count + 1)
            ])
            # Every entry visible at once. Qt shows ten and scrolls the rest, and
            # a popup that scrolls is the gesture that has already caused trouble
            # on this screen -- a flick runs the list under the pointer. A frame
            # holds a handful of charts, so the whole list fits and there is
            # nothing to scroll past.
            self.chart_combo.setMaxVisibleItems(count)
            self.chart_combo.setCurrentIndex(max(0, min(index, count - 1)))
        self.chart_combo.blockSignals(False)
        self.chart_combo.setVisible(count > 1)

        # An empty list means two different things and only the caller knows
        # which. Before any search has run there are no offers and exactly one
        # implicit target -- the one on the canvas -- which is every ordinary
        # single-chart run. After the operator has removed the last target there
        # are none, and the run button must say so rather than offering to
        # measure a frame nobody has placed anything in.
        self._charts_running = 0 if emptied else (count or 1)
        shown = charts[index] if 0 <= index < count else None
        # One action either way now. Keeping a *found* target on screen, greyed,
        # was the earlier answer and it was answering the archive's question with
        # the screen: the commonest reason to press this is that the search
        # returned a region that is not a chart at all, and a non-chart left in
        # the selector is still something the operator has to step through. The
        # record keeps the evidence -- it says the capture held more than was
        # measured -- and the wording differs only in what it can promise about
        # putting one back.
        self.drop_chart_button.setVisible(shown is not None and count > 0)
        if shown is not None:
            if shown.added:
                self.drop_chart_button.setText("Remove This Target")
                self.drop_chart_button.setToolTip("")
            else:
                self.drop_chart_button.setText("Remove This Target")
                self.drop_chart_button.setToolTip(
                    "Not measured. The record notes it was found."
                )
        self._update_run_button()

    def _on_chart_chosen(self, index: int) -> None:
        if index >= 0 and self.chart_combo.isVisible():
            self.chartSelected.emit(index)

    def show_detection(self, detection) -> None:
        """Say what the automatic placement did, or why it would not.

        Never silent, in either direction, but no longer a paragraph. A placement
        the operator did not make is the one thing on this screen they have no
        memory of doing, and what they need from that is two words and a color:
        it was placed, so glance at it. What it keyed on and how far refinement
        moved the corners is the tooltip -- available, and not spending four lines
        of a panel on every run.

        A refusal leaves the corners exactly where they were and says so, which
        is the only handover that leaves the operator better off than not having
        tried. What *this* chart is placed on -- and the Golden Thread's corners
        are inside its corner modules, nowhere near the corners of the card -- is
        a fact about the target rather than about the run, so it belongs in the
        target documentation and not on this panel.
        """
        self._clear_busy()
        if detection.found:
            self.settle_placement()
        self._detected = bool(detection.found)
        self._refined = bool(detection.found and detection.refinement
                             and detection.refinement.applied)
        detail = "\n\n".join([detection.reason] + list(detection.notes))
        if detection.found and (detection.diagnostics or {}).get("restored"):
            # A saved run's placement, checked when that run was made. Not
            # "Placed Automatically", which would claim a search nobody ran, and
            # recorded as it was placed then.
            self._detected = detection.diagnostics.get("placed_as") != "by_hand"
            self._set_detection_state(
                "Placement Restored", self._DETECTION_TONE[True], detail=detail,
            )
            return
        self._set_detection_state(
            "Placed Automatically" if detection.found else "Place the Target by Hand",
            self._DETECTION_TONE[bool(detection.found)],
            detail=detail,
        )

    @property
    def placed_automatically(self) -> bool:
        """Whether the corners about to be run are still the ones that were found.

        False the moment a handle is dragged. The record must not say a placement
        was automatic when half of it was not, and this is the only place that
        knows.
        """
        return self._detected

    def _set_detection_state(self, text: str, tone: str, *, detail: str = "") -> None:
        """What the search decided, as a chip with its reasoning on hover."""
        self.detection_pill.set_state(text, tone, detail=detail)

    def clear_detection(self) -> None:
        self._detected = False
        self.detection_pill.hide()

    def set_target(self, target) -> None:
        """Name the handles and set up what this chart can and cannot be offered.

        Refinement and sampling are both properties of the definition rather than
        of the capture, so whether to offer them is knowable without touching a
        pixel -- and offering an action that will refuse is worse than not
        offering it, since the refusal arrives after the click.

        **The refinement rule is the engine's and is asked for, never restated.**
        Counting the features here was a second statement of it and drifted both
        ways: rotated squares only, so the three ISA transmissive charts were shown
        a dead button claiming they print no marks, and both Object Level Targets --
        whose one edge is declared as ``straight_edges`` and yields no centre at all
        -- were shown a live one that refused on every press.
        """
        from deltachrome.imaging.refine import refinable

        self.target = target
        self.refine_pill.hide()
        self.clear_detection()
        # **No target is a real state**, not a caller's mistake: the combo opens
        # blank on a capture nothing could be identified in, and everything this
        # method sets up is a property of a definition there is not one of. It has
        # to be said explicitly rather than left stale, because a panel outlives
        # the capture and the previous chart's answers are worse than none.
        if target is None:
            self._has_edges = False
            self._has_regions = False
            self.refine_button.setEnabled(False)
            self.refine_button.setToolTip(
                "Choose a target first."
            )
            self._sync_check_button()
            self.canvas.set_corner_labels(CORNER_LABELS)
            return
        can_refine, why_not = refinable(target)
        self.refine_button.setEnabled(can_refine)
        self.refine_button.setToolTip(why_not)
        # A frame-filling white sheet has neither patches nor edges, and asking
        # the sampler for patches raises rather than returning nothing -- so the
        # button is disabled instead of failing on every press. Both kinds are
        # counted, so a target that measures only sharpness is drawable on its
        # edges alone.
        self._has_edges = any(r.kind == "slanted_edge" for r in target.regions)
        self._has_regions = bool(target.grid_regions) or self._has_edges
        self._sync_check_button()
        self.canvas.set_corner_labels(target.fiducials.corner_ids)
        self._update_run_button()
        # A different chart samples different regions, and the ones on screen
        # belong to the previous one. Cleared here rather than left for the next
        # corner change, which on a target swap may never come.
        self._schedule_sample_refresh()

    def set_proxy(self, proxy: ImageProxy, path: Path) -> None:
        self.proxy = proxy
        self.canvas.set_proxy(proxy)
        self.image_path = path
        self._schedule_sample_refresh()

    def set_image(self, path: Path, description: str) -> None:
        """The capture's name and what the file itself is.

        Carried over from the setup screen, which was the only place the bit
        depth and the embedded profile were ever stated. They belong on whichever
        screen precedes the run, and now there is only one.
        """
        self.title_label.setText(path.name)
        self.subtitle_label.setText(description)

    def set_corners(self, corners) -> None:
        self.canvas.set_corners(corners)

    # -- reactions ---------------------------------------------------------

    def _on_corners_changed(self, corners) -> None:
        """The readout, for a placement from any source.

        Fires for a detection and a refinement as well as for a drag, which is
        right for the numbers and wrong for everything below -- see
        :meth:`_on_corners_dragged`.
        """
        for label, (x, y) in zip(self.corner_labels, corners):
            label.setText(f"{x:8.1f},{y:8.1f}")
        self._schedule_sample_refresh()

    def _on_corners_dragged(self) -> None:
        """A person moved a handle, which is a different event from the corners
        moving.

        A hand-moved corner is no longer a refined placement, and a command
        claiming otherwise would refine these corners a second time. It is no
        longer an automatic placement either -- what is on screen is now partly
        the operator's, and the record must not say the machine placed it.

        Listening to the corners themselves cannot express that: a detection
        arrives *as* a corner change, so the distinction reduced to calling
        ``set_corners`` before ``show_detection`` and never the other way round
        -- a rule nothing states and nothing checks. The canvas knows which it
        is, so the canvas says.
        """
        self._refined = False
        self.refine_pill.hide()
        # A hand on a handle is exactly what an added chart was waiting for: it
        # began on a default rectangle nobody chose, and now somebody is putting
        # it somewhere. The regions come back as the placement is made, which is
        # what they are for.
        self.settle_placement()
        if self._detected:
            self._detected = False
            self._set_detection_state("Placed by Hand", TONE_UNKNOWN)

    #: Error and caution look different because they are different: one placement
    #: is impossible, the other is merely unusual and might be exactly right.
    _SEVERITY_STYLE = {
        "error": ("#ff6050", "#3a1f1c"),
        "caution": ("#ffc44d", "#3a2f18"),
    }

    def _on_validity_changed(self, severity: str, reason: str) -> None:
        # A crossed quadrilateral has no homography worth computing, so the run
        # is blocked rather than allowed to produce a confident wrong answer.
        self._placement_valid = severity != "error"
        self._update_run_button()
        self.warning.setVisible(severity != "ok")
        if severity == "ok":
            # Cleared, not merely hidden. A stale message behind a hidden widget
            # is one styling slip away from being shown against a placement it
            # was never about.
            self.warning.clear()
            return
        fg, bg = self._SEVERITY_STYLE[severity]
        self.warning.setStyleSheet(
            f"color: {fg}; background: {bg}; border: 1px solid {fg}44;"
            "border-radius: 6px; padding: 8px 10px; font-size: 12px;"
        )
        self.warning.setText(reason)

    def _copy_command(self) -> None:
        from deltachrome_ui.command import analyze_command

        QGuiApplication.clipboard().setText(analyze_command(
            image_path=self.image_path,
            target_id=self.choices.target_id,
            profile_id=self.choices.profile_id,
            corners=self.canvas.corners(),
            reference_path=self.choices.reference_choice,
            options=self.choices.values(),
            refined=self._refined,
            save_run=True,
        ))
        self.copy_button.flash()
