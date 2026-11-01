"""The report.

The engine is careful never to state a number it cannot justify, and this is the
screen where that care is either preserved or thrown away. Three rules follow
from it, and none of them changed when the layout did.

**The verdict carries its caveats in the same breath.** ``conformance_caveats``
sits with the badge, never behind a click, because a run scored against
manufacturer nominal values measured something real but did not test the thing
the word PASS appears to claim.

**Every metric shows its condition beside its value.** A number with no threshold
next to it invites the reader to supply one from memory, and the thresholds differ
by star level, by material category and by standard.

**MEASURED is visually distinct from PASS.** A profile that sets no threshold
produced a reading and judged nothing. It is not a success and must not look like
one -- :mod:`deltachrome_ui.status` gives it its own tone and this screen uses
what it is given.

**The shape: one metric at a time.** The report was compartments -- a family of
metrics on a page, chosen from a tree, with a strip of tiles above and a tab bar
below. Every part of that was defensible on its own and together they were four
controls for one question, with the answer, a summary of the answer and a map to
the answer all on screen at once.

What is here now is three things and no more:

* **Overview**, and it is where a run opens. Every metric as a row in the same
  table the library uses: dot, name, value, unit, what it was judged against,
  verdict. Fifteen metrics, one screen.
* **One metric**, reached from the flat list at the left. A line saying what it
  read, whatever qualifies that reading immediately underneath it, and then the
  picture. Nothing repeats the line.
* **The readings**, as a table. The only tab a metric ever has besides its
  picture, because a list of measurements is a list and belongs in the same
  table as every other list in the application -- sorted on any column, since
  "which patch is worst" and "which record is soft" are orderings of one list.

**What the picture is, is decided by the data.** A metric whose readings are
patches gets the chart; a ramp gets its curve; edges get their MTF overlay. Not
by the family's name -- the same rule the engine uses for capabilities, one
layer up.

**Where a reading physically sits is answered on the capture, not on a diagram.**
There used to be a Field view: marks at their chart positions, sized by value.
It was a drawing of the target, shown beside a panel already displaying the
target, and the two competed to be believed. The size ramp moved to the capture
panel, where the marks sit on the pixels they were measured from.

**The profile is a control, not a caption.** The commonest thing wrong with a
report is not the measurement: it is that the wrong performance level or the
wrong material category was chosen, and the report is where that becomes
visible. So the header's profile is two combo boxes wherever the reading is
still in memory, and choosing another judges the same reading again --
:func:`deltachrome.pipeline.rescore`, which is 0.21 s on the UTT's JPEG 2000
capture against 16.4 s to read it. Save writes whichever judgement is on screen,
because the button is rebuilt with the result and there is nowhere for the two
to disagree. Where the measurement is gone -- a run reopened from the library
holds every number and no pixels -- it is a label again, in the same place,
saying the same thing.

**And beside that label, where the record can supply what is missing, is
Change Guidelines…** -- named for why it is pressed, where it was *Measure
Again*, named for what it does first. The record already carries
the corners, the target, the reference and every run option, so reading the
capture again is ``analyze`` with nothing asked of the operator. It is a worded
button rather than a third state of the picker, because the two controls cost
two different things -- picking a profile is a fifth of a second and this is a
decode -- and a control that looked the same for both would let one click mean
either. Pressing it hands back a fresh measurement, after which the label above
is a picker like any other run's.

**And there is a way back to the placement.** One step back rather than out: the
setup screen is still holding this capture, its target and its four corners, so
a placement the report has just shown to be wrong can be corrected without
starting again. Offered only on a fresh run, because that screen holds whatever
this session last placed and a run out of the library was placed in another one.

**Nothing on this screen explains itself.** There were ⓘ buttons beside the
metric name, the charts and the capture panel, opening a column of standing
prose. It has all gone to documentation, for a reason that is about finding out
what the screen communicates rather than about tidiness: a control that answers
its own question can never be shown to be unclear. What stayed is every sentence
that qualifies *this* run -- a metric's ``reason``, the conformance caveat, a
warning -- because those are part of the answer rather than an explanation of it,
and they were never behind the icon.

**There is no Provenance page.** It listed the reference basis, the ICC profile
and transform, the illuminant, the ROI inset and how the sampling rate was
established -- all of it true, all of it in the record, and none of it a question
a person opens a report to ask. Copy Command puts the whole invocation on the
clipboard, which is the same facts in the form somebody would actually use them.

Views are built the first time they are looked at, because a report can carry a
hundred and forty patches and twenty edge curves and nobody opens all of them.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from deltachrome_ui.capture_dock import CaptureDock
from deltachrome_ui.charts import ChartSelection, group_styles
from deltachrome_ui.detail_view import (
    EdgeReadings,
    PatchDetail,
    PatchTable,
    RegistrationReadings,
    edge_readings_for,
    pair_readings_for,
    placed_samples,
    sample_regions,
)
from deltachrome_ui.families import families
from deltachrome_ui.labels import condition_label, metric_title, unit_label
from deltachrome_ui.metric_charts import chart_for, has_readings
from deltachrome_ui.nav import METRIC, OVERVIEW, PAGE, ReportNav
from deltachrome_ui.profile_picker import ProfilePicker
from deltachrome_ui.status import metric_status, overall_status
from deltachrome_ui.tables import CENTER, RIGHT, Col, DataTable
from deltachrome_ui.theme import (
    BAR_HEIGHT, BORDER, TEXT_DIM, VERDICT_WIDTH, tone_colors,
)
from deltachrome_ui.widgets import (
    Callout,
    IconButton,
    StatusBadge,
    Stepper,
    VerdictStripe,
    caveat,
    divider,
    elided_subtitle,
    subtitle,
    title,
)


#: How much of the header the profile control may take. The category combo is
#: what needs the room: its label is body, edition and material category, and
#: the material category is both the last of the three and the one somebody is
#: most likely to have got wrong. Bounded so the filename above keeps a readable
#: share of a narrow window.
#:
#: The longest material label in the catalogue needs 561 px and the level box
#: beside it 131, so the cap is what those two come to. At 640 the category
#: stopped growing at 503 on a window with room to spare, which elided the label
#: for no reason other than the constant.
PICKER_WIDTH = 700

#: And how much the chart's name beside it may take, before it elides.
#:
#: Less than it had, and the two things sharing this row are not equals. The
#: chart's name is a **fact**, it is stated again on the overview, and it is
#: already elided at any width -- the longest of the seventeen wants 372 px. The
#: category combo is a **control**, and one that cannot show its own value is
#: worse than a name that stops early: at 1180 px it was down to 209 px, which
#: reads "FADGI -- Documents (Ur" on the field this control exists to correct.
TARGET_NAME_WIDTH = 240

# VERDICT_WIDTH lives in theme.py: two screens now set a verdict to it.


def _scrolled(widget: QWidget) -> QWidget:
    """A view in a scroll area that never scrolls sideways.

    Content reflows to the width available instead of running off the edge. A
    horizontal scrollbar under a column of prose means the prose was cut off,
    not that the reader wanted to pan.

    The minimum width matters as much as the scrolling: without it a page's
    contents dictate the window's smallest size, and a report with a wide plot
    in it silently grew the application larger than the screen it opened on.
    """
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(widget)
    scroll.setMinimumWidth(320)
    return scroll


def metric_name(metric: dict) -> str:
    """Title Case. A measurement is a proper noun in this domain."""
    return metric_title(metric.get("metric_id", ""))


def metric_value(metric: dict) -> str:
    """The reading, in the form the metric itself said to write it in.

    Three of them are not numbers: the master's container is ``JPEG 2000``, its
    colour space is the name of an ICC profile, and its bit depth is ``8`` --
    a count, where ``8.000`` claims a precision it does not have. The engine
    decides that and records it as ``value_text``; nothing here switches on a
    metric's id to work it out.
    """
    text = metric.get("value_text")
    if text:
        return str(text)
    value = metric.get("value")
    return "—" if value is None else f"{value:.3f}"


def metric_reading(metric: dict) -> str:
    if metric.get("value_text") is None and metric.get("value") is None:
        return "—"
    return f"{metric_value(metric)} {unit_label(metric.get('units'))}".strip()


class LazyTab(QWidget):
    """A tab whose contents are built the first time it is looked at."""

    def __init__(self, build) -> None:
        super().__init__()
        self._build = build
        self._built = False
        self._layout = QVBoxLayout(self)
        # A gap under the Charts / Values tabs, set once for every view. Each
        # chart began flush under the tab bar, so a plot's own heading -- TONE
        # RESPONSE, a registration grid's G-B -- sat six pixels from the tab's
        # underline and read as part of the bar.
        self._layout.setContentsMargins(0, 14, 0, 0)

    def ensure(self) -> None:
        if self._built:
            return
        self._built = True
        self._layout.addWidget(self._build())


class ViewTabs(QTabWidget):
    """Representations of one metric. Only the visible one is built."""

    def __init__(self) -> None:
        super().__init__()
        self.setDocumentMode(True)
        self.setElideMode(Qt.TextElideMode.ElideNone)
        # Without this the bar shrinks to whatever the pane's minimum allows and
        # elides its own labels -- "Ch...", "Readin..." -- which is the one
        # place in the window where a control cannot say what it does.
        self.tabBar().setUsesScrollButtons(False)
        self.tabBar().setExpanding(False)
        self.tabBar().setMinimumWidth(0)
        self.currentChanged.connect(self._on_changed)

    def add_view(self, name: str, build) -> None:
        self.addTab(LazyTab(build), name)

    def _on_changed(self, index: int) -> None:
        widget = self.widget(index)
        if isinstance(widget, LazyTab):
            widget.ensure()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().showEvent(event)
        self._on_changed(self.currentIndex())


class MetricPage(QWidget):
    """One metric: what it read, what qualifies it, and the picture.

    The reading is a *line*, not a tile. A tile made sense when several sat side
    by side and were compared; on a page about one metric it was a box around
    the only thing on the page, and the box was larger than what it held.
    """

    def __init__(self, family, metric: dict, run, proxy=None, selection=None) -> None:
        super().__init__()
        self.family = family
        self.record = metric
        self.run = run

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 12)
        layout.setSpacing(10)
        layout.addWidget(self._reading_line(metric))

        # A metric this target cannot carry has nothing to qualify and nothing to
        # draw, so the page says so once, where a picture would be. The record's
        # own reason named the chart and the capability's identifier --
        # "... does not declare the capability 'lightness_uniformity'" -- and it
        # stays in the record and the CLI; the page composes its own line from
        # the metric's name, so a reopened old record reads the same.
        if metric.get("outcome") == "not_applicable":
            layout.addWidget(self._nothing_to_draw(
                f"{metric_name(metric)} not capable for this target."
            ), 1)
            self.views = ViewTabs()
            return

        # The same for a metric this run could not finish and that read nothing:
        # its reason *is* the page, centred where the picture would be, rather
        # than a line at the top over a second line saying there is nothing to
        # draw. Not for the edge families, whose picture comes from the edges.
        if (metric.get("outcome") == "incomplete" and not has_readings(metric)
                and family.key not in ("sfr", "color_registration")):
            layout.addWidget(self._nothing_to_draw(
                metric.get("reason") or f"{metric_name(metric)} could not be measured."
            ), 1)
            self.views = ViewTabs()
            return

        # Qualification, immediately and always visible. Why a metric could not
        # be finished, or what its number does not cover, is part of the result
        # rather than an explanation of it -- so it is never a tab, and it did
        # not leave with the standing prose.
        for text in self._qualifications(metric):
            layout.addWidget(caveat(text))

        self.views = ViewTabs()
        for name, build in available_views(
            family, run, metric, proxy=proxy, selection=selection,
        ):
            self.views.add_view(name, build)
        if self.views.count():
            # A single view needs no switch. The tab bar is a control for
            # choosing, and offering a choice of one is furniture.
            self.views.tabBar().setVisible(self.views.count() > 1)
            layout.addWidget(self.views, 1)
        else:
            layout.addWidget(self._nothing_to_draw(), 1)

    def _reading_line(self, metric: dict) -> QWidget:
        """Name, reading, verdict. Nothing else.

        The condition was here and is not any more: the limits are on the chart
        below, drawn as the band the reading sits in, which answers "how much
        room is left" that a printed threshold never did. The metric's own prose
        -- several hundred words on some of them -- is documentation now.

        What is *not* documentation is ``reason``: why this run could not finish
        a measurement is part of its answer rather than an explanation of it,
        and it is on the page directly under this line.
        """
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(10)

        name = QLabel(metric_name(metric))
        name.setStyleSheet("font-size: 17px; font-weight: 600;")
        line.addWidget(name, 0)

        # A rule between the name and the number. They were running together --
        # "sfr10 sampling efficiency 132.121 % of Nyquist" reads as one phrase
        # and the eye has to find where the name stops.
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.VLine)
        rule.setFixedWidth(1)
        rule.setStyleSheet(f"background: {BORDER}; border: none;")
        line.addWidget(rule, 0)

        # The body face, as on the overview's Value column. The mono face was
        # there to line decimal points up down a column, and this is one number
        # with nothing under it to line up with -- so all it did was set the
        # loudest thing on the page in a typeface nothing else on the page uses.
        reading = QLabel(metric_value(metric))
        reading.setStyleSheet("font-size: 17px; font-weight: 600;")
        line.addWidget(reading, 0)

        units = QLabel(unit_label(metric.get("units")))
        units.setStyleSheet(f"color: {TEXT_DIM}; font-size: 13px;")
        line.addWidget(units, 1, Qt.AlignmentFlag.AlignVCenter)

        status = metric_status(metric.get("status", ""))
        badge = StatusBadge(status)
        condition = metric.get("condition")
        badge.setToolTip(
            f"{status.hint}\n\nThreshold: {condition_label(condition)}"
            if condition
            else status.hint
        )
        line.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)
        return row

    @staticmethod
    def _qualifications(metric: dict) -> list[str]:
        """What is wrong with *this* run's answer. Never the standing prose."""
        reason = metric.get("reason")
        return [reason] if reason else []

    def _nothing_to_draw(self, text: str = "No further analysis.") -> QWidget:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        return label

    # -- driven from outside -----------------------------------------------

    def current_metric(self) -> dict:
        return self.record

    def show_view(self, name: str) -> None:
        for index in range(self.views.count()):
            if self.views.tabText(index) == name:
                self.views.setCurrentIndex(index)
                return


def available_views(family, run, metric: dict, proxy=None, selection=None):
    """Which representations exist for one metric, and how to build each.

    Two at most, and the names are the same for every metric. **Charts** is
    whatever picture the metric gets -- curves for the edges, bars for a field
    of patches, the OECF for tone -- because a reader choosing between two tabs
    is choosing between "show me" and "list it", not between a curve and a bar
    chart. **Values** is the readings as a table.

    The grid of swatches that used to be a third tab is gone: colour accuracy is
    now drawn in the patches' own colours, so the chart of the errors carries
    the identity the grid was there to supply.
    """
    views: list[tuple[str, object]] = []
    if chart_for(family, run, metric, selection) is not None:
        views.append((
            "Charts",
            lambda: _scrolled(chart_for(family, run, metric, selection)),
        ))
    if family.key == "color_registration" and (
        (getattr(run, "registration_summary", None) or {}).get("positions")
    ):
        # Its own readings rather than the edge table it used to borrow. The
        # rows are what the grid above draws -- position, axis, channel pair --
        # and they carry the antisymmetric term, which is the quantity pairing
        # takes *out* of the verdict and which was three sentences of prose.
        views.append((
            "Values", lambda: _scrolled(RegistrationReadings(run, selection)),
        ))
    elif family.key in ("sfr", "color_registration") and getattr(run, "edges", None):
        views.append(("Values", lambda: _scrolled(EdgeReadings(run, selection))))
    elif [p for p in metric.get("per_patch", []) if p.get("value") is not None]:
        # One reading is still the measurement. Gain modulation highlight has
        # exactly one pair by definition, and hiding its value behind a
        # "more than one" test made the metric look unfinished.
        views.append(("Values", lambda: _scrolled(PatchTable(metric, selection))))
    return views


def _swatches(run, readings: dict) -> dict:
    """Aim and measured colour per patch, and their Lab, for the panel to show.

    Only where the reading *is* a colour patch. A flat-field spot, a printed
    rule and a slanted edge were never aimed at a colour, and inventing one for
    them would be a swatch that means nothing.
    """
    from deltachrome_ui.swatches import lab_to_display_rgb

    out: dict = {}
    for patch in getattr(run, "patches", None) or []:
        name = str(getattr(patch, "patch_id", ""))
        if name not in readings:
            continue
        reference = getattr(patch, "reference_lab", None)
        measured = getattr(patch, "measured_lab", None)
        if reference is None and measured is None:
            continue
        try:
            out[name] = (
                lab_to_display_rgb(reference) if reference is not None else None,
                lab_to_display_rgb(measured) if measured is not None else None,
                # The numbers behind the two colours, for the panel to print.
                tuple(float(v) for v in reference) if reference is not None else None,
                tuple(float(v) for v in measured) if measured is not None else None,
            )
        except Exception:  # noqa: BLE001 -- a swatch is never worth a crash
            continue
    return out


class ResultView(QWidget):
    """Verdict, the run's metrics, and one metric at a time."""

    backRequested = Signal()
    saveRequested = Signal(object)  # AnalysisResult
    newAnalysisRequested = Signal()
    #: A saved run should be opened for placement again.
    rerunRequested = Signal(object)
    #: A different chart of the same capture. Emitted with its index; the shell
    #: re-reads that chart from the record, which is a read rather than a run.
    chartRequested = Signal(int)
    #: The capture should be decoded so the panel can show pixels.
    captureRequested = Signal()
    #: Back to the screen this run was set up on. Only offered where that screen
    #: still holds the capture, which is a fact the shell owns and not this one.
    placementRequested = Signal()
    #: Judge the same reading against another profile, named by its id.
    rescoreRequested = Signal(str)
    #: This run's capture should be read again, from the record's own placement,
    #: so it stops being numbers with no pixels behind them and the profile
    #: picker above stops being a label.
    establishRequested = Signal()
    #: Every chart of this capture, read together. Offered only where there are
    #: several, and never as the way IN to them: a board is shot to measure a
    #: system and this is that answer, with the charts' own reports the detail
    #: underneath it.
    sessionRequested = Signal()
    #: The capture this report belongs to should be written out as text, CSV or
    #: XML. The shell answers it, being what knows whether the capture is a
    #: record in the library or charts just measured -- and a board's export
    #: holds every chart, not the one on screen, as Save does.
    exportRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.result = None
        #: The bar's export square, so the shell can confirm on it once the file
        #: is written. Rebuilt with the header.
        self.export_button = None
        self.saved = False
        self.placed_here = True
        self.back_button = None
        self._back_place: str | None = None
        #: The loaded capture, when there is one. Held so a view can show the
        #: pixels behind a mark; the report is complete without it.
        self.proxy = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # A widget rather than a layout, replaced whole on every run. The
        # header used to be cleared by walking its layout and deleting the
        # widgets -- which silently skipped the nested layout holding the
        # filename, so every new result drew its title on top of the last one's.
        self.header = QWidget()
        self.header.setObjectName("HeaderBar")
        # What makes the theme's HeaderBar rule actually paint. A plain QWidget
        # ignores a style sheet's background and border without it, so the header
        # has been rendering in the base colour with no stroke under it for as
        # long as the rule has existed.
        self.header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        header_shell = QVBoxLayout(self.header)
        header_shell.setContentsMargins(0, 0, 0, 0)
        self._header_slot = header_shell
        outer.addWidget(self.header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        outer.addLayout(body, 1)

        self.nav = ReportNav()
        self.nav.navigate.connect(self._on_navigate)
        body.addWidget(self.nav)
        #: Target tuple -> index in the page stack.
        self._page_index: dict[tuple, int] = {}

        self.pages = QStackedWidget()
        self.pages.setMinimumWidth(340)
        body.addWidget(self.pages, 1)
        #: Which page is open, so a re-judgement can put the reader back on the
        #: metric they were reading. Changing the level while looking at tone
        #: response is a question about tone response.
        self._current_target: tuple | None = None
        #: The header's profile control, where the run can be judged again.
        self.picker = None
        #: The "Change Guidelines…" button, where a saved run's record can supply
        #: its own placement -- ``None`` when this run has nothing to establish.
        self._establish_button = None

        #: One selection for the whole report. The mark on the capture, the
        #: swatch on the chart and the row in a table are renderings of one
        #: reading, so which reading is being asked about has to be one fact.
        self.selection = ChartSelection(self)
        self.dock = CaptureDock(self.selection)
        self.dock.loadRequested.connect(self.captureRequested.emit)
        body.addWidget(self.dock)

    # -- population --------------------------------------------------------

    def set_proxy(self, proxy) -> None:
        """The capture arrived, or went away.

        ``None`` has to tear the panel's pixel views down rather than being
        ignored. Opening a saved run passes it, and a panel that kept the last
        run's decoded frame showed one capture's pixels beside another
        capture's numbers -- with nothing on screen saying so.
        """
        self.proxy = proxy
        self.dock.set_proxy(proxy)

    def _copy_command(self, run, button=None) -> None:
        """Copy the invocation, and say so.

        The button is a shape and cannot do what a worded one did and read
        "Copied", so it shows a tick for a moment instead. Without it the only
        way to find out whether the press registered is to paste somewhere and
        look -- and the thing copied is eight hundred characters that change
        nothing on screen.
        """
        from PySide6.QtGui import QGuiApplication

        from deltachrome_ui.command import command_for_record

        QGuiApplication.clipboard().setText(command_for_record(run.to_dict()))
        if button is not None:
            button.flash()

    def set_back_destination(self, place: str | None) -> None:
        """Say where Back goes: ``None`` for the library, or the screen's name.

        A chart's report reached by clicking a target on All Targets goes back
        there, and its tooltip said "Back to the library" regardless. Held, so a
        bar rebuilt by stepping to another chart keeps saying it.
        """
        self._back_place = place
        if self.back_button is not None:
            self.back_button.setToolTip(f"Back to {place or 'the library'}")

    @property
    def current_page(self) -> tuple | None:
        """Which page is open, as a navigation target.

        Public because the shell hands it back when a re-judgement returns: the
        same capture judged again is the same list of metrics, and dropping the
        reader on the overview after every change would make them find their
        place each time.
        """
        return self._current_target

    @property
    def can_rescore(self) -> bool:
        """Whether this run can be judged again without re-reading the capture.

        Read off the result rather than off ``saved``, because it is a different
        question with a different answer: what re-scoring needs is the
        measurement, and a run rebuilt from a saved record has every number and
        no pixels. The engine refuses that case; this keeps the control off the
        screen so nobody has to be refused.
        """
        return getattr(self.result, "measurement", None) is not None

    @property
    def can_establish(self) -> bool:
        """Whether this run's own record carries enough to be read again.

        A run already live in this session (``can_rescore``) has nothing to
        establish, and a record written before a target or profile id was
        carried at all -- or one whose placement never reached disk -- has
        nothing this can supply either. Everything else can be read again with
        the operator asked nothing: what it takes is in the record.
        """
        if self.result is None or self.can_rescore:
            return False
        from deltachrome_ui.library import establish_kwargs

        return establish_kwargs(self.result.run.to_dict()) is not None

    def set_result(self, result, *, saved: bool = False, focus: tuple | None = None,
                   chart: tuple[int, int] = (0, 1), placed_here: bool = True,
                   keep_view: bool = False) -> None:
        """``saved`` distinguishes a run already in the library from a fresh one.

        Only the actions differ. A saved run cannot be saved again -- offering it
        would write a second record of the same analysis under a new id -- so it
        offers to re-run the capture and to copy its command instead.

        ``focus`` is the page to open on, and exists for re-judging: the same
        capture judged again is the same fifteen metrics, so returning the reader
        to the overview would make them find their place after every change.

        ``chart`` is which chart of its capture this is, and of how many. Passed in
        rather than read off the result, because the two kinds of result reach here
        by different routes -- a saved capture knows its own position, a freshly
        measured chart is one of a list the shell is holding -- and sniffing for an
        attribute meant the fresh path silently reported one chart and hid the
        selector. What the screen is told, it shows.
        """
        self.result = result
        self.saved = saved
        #: Whether this run was placed on this session's placement screen. A
        #: saved run read again by *Change Guidelines…* is live and was not, so the way
        #: to its placement is Re-run rather than "back" to whatever that screen
        #: holds.
        self.placed_here = placed_here
        # How many charts the capture this came from holds, and which of them this
        # is. Both default to one chart, so a live run and every record written
        # before a capture could hold several behave exactly as they did.
        self.chart_count = max(1, int(chart[1]))
        self.chart_index = max(0, min(int(chart[0]), self.chart_count - 1))
        self.selection.selected = None
        self.selection.hovered = None
        self.selection.isolating = False
        self.selection.muted.clear()
        # ``keep_view`` is another chart of the same capture, or the same chart
        # judged again: the SFR chips stay as the reader left them. Anything
        # else is a different capture and opens on its own defaults.
        if not keep_view:
            self.selection.edge_view = {}

        while self._header_slot.count():
            item = self._header_slot.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.nav.clear()
        self._page_index.clear()
        while self.pages.count():
            page = self.pages.widget(0)
            self.pages.removeWidget(page)
            page.setParent(None)
            page.deleteLater()
        self.dock.reset()
        # ``reset`` clears the panel of the *previous* run, and then this run's
        # capture -- if the shell handed one over before calling this, which is
        # what a fresh analysis does -- is handed back. Without it a run whose
        # master was already decoded in memory showed "Load Capture" and, on
        # being asked, decoded the same file a second time.
        self.dock.set_proxy(self.proxy)

        self._header_slot.addWidget(self._build_header(result))
        self._build_pages(result)
        target = focus if focus in self._page_index else (PAGE, OVERVIEW)
        self.nav.show_target(target)
        self._on_navigate(target)

    def _build_header(self, result) -> QWidget:
        """The verdict, what was measured, and what can be done about it.

        In that order, left to right, and the order is the whole of the change.

        **The verdict is first and is not a chip.** It was a filled badge sitting
        in a row of filled buttons, which made a statement look like a control --
        and a person who thinks a verdict is pressable does not read it as a
        finding. It is a coloured rule and a word now, alone on the left, with
        ADVISORY beside the word rather than inside it.

        **Every action is on the right, and all but one is an icon.** Four worded
        buttons is most of a header's width spent on things a shape says as well,
        and the width is needed by the middle: the profile control has a FADGI
        material category to show. Save keeps its name -- it is the primary
        action and the only one that writes anything, and an icon for "commit
        this to the archive" is the icon people hesitate over.

        **Save is inside that group rather than beside it.** It sat in the bar's
        own layout, which carries 16 px of spacing, so a gap opened between the
        last icon and the accent block and the strip stopped reading as one run
        of cells against the window's edge. Being the primary action is a reason
        for it to be worded and coloured, not for it to be detached.

        **The middle is a layout, not a widget.** A plain ``QWidget`` paints the
        base background, which over a raised header is a dark box around the
        filename sized by whatever the layout gave it.
        """
        run = result.run
        evaluation = result.evaluation

        bar = QWidget()
        # Styled here as well as on the container that holds it. The container
        # is the thing the outer layout positions, and this is the thing that
        # covers it -- a plain QWidget paints the base background, so without
        # this the raised tint and the stroke were painted and then painted over.
        bar.setObjectName("HeaderBar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setFixedHeight(BAR_HEIGHT)
        layout = QHBoxLayout(bar)
        # No margin on the right and none top or bottom. That is what puts the
        # actions against the edge of the window: they are the bar's full height
        # and the last of them has nothing beyond it. The left keeps its inset,
        # because the verdict is being read rather than aimed at.
        layout.setContentsMargins(20, 0, 0, 0)
        layout.setSpacing(16)

        status = overall_status(
            evaluation.overall, evaluation.conformance_claim,
            evidence_sound=getattr(evaluation, "evidence_is_sound", None),
        )
        stripe = VerdictStripe(status)
        # Its own policy is Fixed, which is right where it stands alone. Here it
        # is a column, and a column has a width of its own -- see VERDICT_WIDTH.
        stripe.setMinimumWidth(VERDICT_WIDTH)
        stripe.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Maximum)
        layout.addWidget(stripe, 0, Qt.AlignmentFlag.AlignVCenter)
        # No divider after it. The stripe's own colour is what separates the
        # verdict from the filename, and a hairline there would make the left of
        # the bar look like the right, where the hairlines mean "another button".

        stack = QVBoxLayout()
        stack.setContentsMargins(0, 0, 0, 0)
        # No gap. The filename's own line box and the combo's padding already
        # separate them, and a gap here is height the bar pays for twice.
        stack.setSpacing(0)
        # Centred, with the slack above and below. Without the stretches the
        # stack took the bar's whole height and gave the slack to the filename,
        # so the second line -- and the worded button on it -- sat flush on the
        # bar's bottom edge.
        stack.addStretch(1)
        name = QLabel(run.image.get("filename", "—"))
        name.setStyleSheet("font-size: 14px; font-weight: 600;")
        stack.addWidget(name)
        stack.addLayout(self._judged_as(run))
        stack.addStretch(1)
        layout.addLayout(stack, 1)
        layout.addSpacing(8)

        # Which chart of this capture, where the capture holds several. It sits with
        # the profile combos above the actions because it is the same kind of control
        # as they are -- it changes what this screen is showing, and it costs nothing
        # to change: every chart's readings were measured when the capture was and are
        # in the record beside this one. Switching is a redraw, not a re-analysis.
        #
        # **A stepper rather than a combo box**, because this is now the whole of
        # how a reader moves between the charts of a board. A multi-chart run used
        # to land on the session and be entered through a list of its captures, so
        # the control here was for jumping about inside something already reached;
        # the run lands on this report instead, and a drop-down that has to be
        # opened to say whether there is another chart is the wrong shape for that.
        if self.chart_count > 1:
            charts = Stepper(self.chart_index, self.chart_count, noun="Target")
            charts.stepped.connect(self.chartRequested.emit)
            layout.addWidget(charts, 0, Qt.AlignmentFlag.AlignVCenter)
            layout.addSpacing(12)

        # Butted together: no spacing at all, and each carries its own hairline
        # on its left edge, so the group reads as a strip of cells rather than as
        # a row of separate objects.
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(0)

        back = IconButton(
            "back", f"Back to {self._back_place or 'the library'}", size=BAR_HEIGHT,
        )
        back.clicked.connect(self.backRequested.emit)
        actions.addWidget(back)
        self.back_button = back

        # The middle slot, on either kind of run, goes to the screen where the
        # chart is placed -- and it is the same icon in both, because it is the
        # same destination. Which of the two actions gets it depends on what is
        # still in memory, and only the wording and the signal differ.
        if not self.saved and self.placed_here:
            # One step back rather than out. The placement screen is still
            # holding this capture, its target and its four corners, so this
            # costs nothing to offer and is the only way to correct a placement
            # the report has just shown to be wrong.
            replace = IconButton(
                "placement", "Back to the placement", size=BAR_HEIGHT,
            )
            replace.clicked.connect(self.placementRequested.emit)
            actions.addWidget(replace)
        else:
            rerun = IconButton(
                "rerun", "Re-run", size=BAR_HEIGHT,
            )
            rerun.clicked.connect(lambda: self.rerunRequested.emit(result))
            actions.addWidget(rerun)

        copy = IconButton(
            "command",
            "Copy command",
            size=BAR_HEIGHT,
        )
        copy.clicked.connect(lambda: self._copy_command(run, copy))
        actions.addWidget(copy)

        export = IconButton(
            "export", "Export values as TXT, CSV or XML", size=BAR_HEIGHT,
        )
        export.clicked.connect(self.exportRequested.emit)
        actions.addWidget(export)
        self.export_button = export

        if self.chart_count > 1:
            # Named rather than drawn, on the same argument the library's own
            # "Batch Analysis" rests on: a shape cannot say this, and what it
            # opens is a different *question* rather than a different view of
            # the one on screen. It is where the board's own verdict lives --
            # four cards measure one system, and whether that system passed at
            # every position is not any one card's answer.
            #
            # It was *Combined Verdict*, which described the answer rather than
            # the place, and it now has to name a place: a capture holding
            # several targets lands on that screen, so this button is the way
            # back **up** to it rather than a way of asking for something new.
            # "Target" is this application's word for the card in front of the
            # camera, which is what the screen holds several of.
            combined = QPushButton("All Targets")
            combined.setObjectName("SecondaryBar")
            combined.setFixedHeight(BAR_HEIGHT)
            combined.setCursor(Qt.CursorShape.PointingHandCursor)
            combined.clicked.connect(self.sessionRequested.emit)
            actions.addWidget(combined)

        if not self.saved:
            # In the same layout as the icons, whose spacing is zero, so it is
            # butted against the last of them exactly as they are against each
            # other.
            save = QPushButton("Save to Library")
            save.setObjectName("PrimaryBar")
            save.setFixedHeight(BAR_HEIGHT)
            save.setCursor(Qt.CursorShape.PointingHandCursor)
            save.clicked.connect(lambda: self.saveRequested.emit(result))
            actions.addWidget(save)
        layout.addLayout(actions)
        return bar

    def _judged_as(self, run) -> QHBoxLayout:
        """What this capture is, and what it was judged against.

        The second half is a **control** wherever the reading is still in memory,
        because the commonest thing wrong with a report is not the measurement --
        it is that the wrong level or the wrong material category was chosen, and
        finding that out is what reading the report is for. Judging the same
        reading again costs a fifth of a second where re-analyzing the capture
        costs a decode, so the choice belongs on the screen that reveals the
        mistake rather than two screens back.

        Where it cannot be changed it is a label, in the same place, saying the
        same thing.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        target_name = run.target.get("display_name", "—")

        if not self.can_rescore:
            # One elided line. It was a wrapping label, and a FADGI profile's
            # display name beside a chart's took three lines of a header built
            # for two -- which is what pushed the filename against the top edge.
            self.picker = None
            row.addWidget(elided_subtitle(
                f"{target_name}   ·   {run.profile.get('display_name', '—')}"
            ), 1)
            # Distinct from the picker on purpose. "Change the
            # profile" and "decode a 300 MB master" must not be the same click
            # without saying so, so this is a worded button naming the cost
            # rather than a control that looks like the one above it.
            if self.can_establish:
                establish = QPushButton("Change Guidelines…")
                establish.setObjectName("HeaderButton")
                establish.setCursor(Qt.CursorShape.PointingHandCursor)
                establish.setToolTip(
                    "Judge this run against other guidelines. The capture is read "
                    "again first."
                )
                establish.clicked.connect(self.establishRequested.emit)
                self._establish_button = establish
                row.addWidget(establish, 0)
            else:
                self._establish_button = None
            return row

        # One line, elided. A chart's display name reaches fifty characters --
        # "Image Engineering Universal Test Target (UTT), A3" -- and wrapped
        # beside the control it doubles the height of the header on the charts
        # with the longest names.
        # A fixed width, and the size policy said again with it. An elided label
        # is deliberately ``Ignored`` horizontally -- that is how it survives a
        # panel narrower than its text -- and a layout reads that as "claims no
        # space", so it allocated nothing and the control beside it was drawn on
        # top of the name.
        name = elided_subtitle(f"{target_name}   ·")
        name.setFixedWidth(TARGET_NAME_WIDTH)
        name.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        row.addWidget(name, 0)
        picker = ProfilePicker(horizontal=True)
        picker.setMaximumWidth(PICKER_WIDTH)
        # Selected before the signal is connected. Setting a profile moves the
        # category combo first and the level list is rebuilt under it, so a
        # listener attached earlier would see a profile nobody chose -- and here
        # every one of those is a re-judgement of the run being drawn.
        picker.select_profile(str(run.profile.get("profile_id") or ""))
        picker.changed.connect(self._on_profile_chosen)
        self.picker = picker
        # No trailing stretch. The control takes the room the header has left,
        # up to its own cap, because what it has to show is a FADGI material
        # category and blank space beside a truncated one helps nobody.
        row.addWidget(picker, 1)
        return row

    def _on_profile_chosen(self) -> None:
        """Ask for a re-judgement, unless nothing was actually chosen.

        Re-selecting what is already selected happens on the way through the
        category list, and answering it would re-run every metric to arrive at
        the report already on screen.
        """
        if self.picker is None or self.result is None:
            return
        profile_id = self.picker.profile_id
        if not profile_id or profile_id == self.result.run.profile.get("profile_id"):
            return
        self.rescoreRequested.emit(profile_id)

    def set_rescoring(self, busy: bool) -> None:
        """Hold the profile control still while a re-judgement runs, and after a
        failed one put it back to what the report actually says.

        Inert while it runs for the same reason the placement panel is inert
        while a chart is being found: every position of the control would change
        the answer being computed.

        Released, it is re-synchronised rather than merely re-enabled. A
        re-judgement that failed leaves a combo box naming a profile and a report
        underneath it judged against a different one, which is the one state this
        control must never be in -- and the whole reason the profile was made a
        control was that a report should say what it was judged against.
        """
        if self.picker is None:
            return
        self.picker.setEnabled(not busy)
        if busy or self.result is None:
            return
        self.picker.blockSignals(True)
        self.picker.select_profile(str(self.result.run.profile.get("profile_id") or ""))
        self.picker.blockSignals(False)

    def set_establishing(self, busy: bool) -> None:
        """Say that the capture is being read again, and hold the button still.

        A disabled button that still reads "Change Guidelines…" invites a second
        press, which is exactly the double-decode the capture panel's own
        ``BusyStrip`` exists to prevent -- this is the same failure in a
        smaller control, so it gets the same fix: the label says what is
        happening rather than merely refusing to be pressed again.
        """
        if self._establish_button is None:
            return
        self._establish_button.setEnabled(not busy)
        self._establish_button.setText("Reading…" if busy else "Change Guidelines…")

    def _add_page(self, target: tuple, page: QWidget) -> None:
        self._page_index[tuple(target)] = self.pages.count()
        self.pages.addWidget(page)

    def _build_pages(self, result) -> None:
        run = result.run
        self._add_page((PAGE, OVERVIEW), self._overview(result))

        grouped = families(run.metrics, run.profile.get("profile_id"))
        for family in grouped:
            for metric in family.metrics:
                metric_id = metric.get("metric_id", "")
                self._add_page(
                    (METRIC, family.key, metric_id),
                    MetricPage(
                        family, metric, run,
                        proxy=self.proxy, selection=self.selection,
                    ),
                )

        extras: list[tuple[str, str, tuple]] = []
        if run.patches:
            self._add_page((PAGE, "patches"), self._titled(
                "Per-Patch Values", _scrolled(PatchDetail(run))
            ))
            extras.append((
                "Per-Patch Values", str(len(run.patches)), (PAGE, "patches"),
            ))
        warnings = run.warnings or []
        if warnings:
            self._add_page((PAGE, "warnings"), self._titled(
                f"Warnings  ·  {len(warnings)}", self._warnings(warnings)
            ))
            extras.append(("Warnings", str(len(warnings)), (PAGE, "warnings")))

        self.nav.build(run, grouped, extras)

    def _on_navigate(self, target: tuple) -> None:
        index = self._page_index.get(tuple(target))
        if index is None:
            return
        self._current_target = tuple(target)
        # An isolation is a request to see less of *this* chart, and the next
        # metric is a different population. Carried across, it landed the reader
        # on a chart that opened dimmed for a mark they pinned two pages ago --
        # which is the state this prevents. The pin itself survives, because
        # it is what the capture panel is pointing at.
        self.selection.isolating = False
        self.pages.setCurrentIndex(index)
        page = self.pages.widget(index)
        # The panel shows the readings of whatever is being looked at. A page
        # with none -- overview, warnings -- leaves it empty rather
        # than showing the last metric's, which would be a lie about what is on
        # screen.
        self._point_dock_at(
            page.current_metric() if isinstance(page, MetricPage) else None
        )

    def show_metric(self, family_key: str, metric_id: str) -> None:
        """Navigate as if the row had been clicked. For tests and for links."""
        target = (METRIC, family_key, metric_id)
        self.nav.show_target(target)
        self._on_navigate(target)

    def _point_dock_at(self, metric: dict | None) -> None:
        if self.result is None:
            return
        if not metric:
            self.dock.set_regions({}, {}, "", {})
            return
        run = self.result.run
        placed = placed_samples(metric)
        roles = sorted({str(p.get("role") or "") for p in placed})
        regions = sample_regions(run, metric, group_styles(roles))
        units = unit_label(metric.get("units"))
        readings = {
            str(p.get("patch_id")): (p["value"], units, p.get("flags") or [])
            for p in placed if p.get("value") is not None
        }
        if not regions:
            # Edge readings carry their bounds on the edge records rather than as
            # placed samples. Only a metric whose readings ARE edges gets them:
            # this used to hand the slanted edges, with their SFR50, to every
            # metric that placed nothing -- gain modulation, a metric the target
            # cannot carry, the file's bit depth.
            regions, edge_readings = edge_readings_for(run, metric, units)
            readings = edge_readings or readings
        if not regions:
            regions, pair_readings = pair_readings_for(
                run, metric, units, group_styles(roles)
            )
            readings = pair_readings or readings
        self.dock.set_regions(
            regions, readings, metric_name(metric), _swatches(run, readings)
        )

    # -- pages -------------------------------------------------------------

    def _overview(self, result) -> QWidget:
        """The verdict, its caveats, and every metric as a row.

        A table rather than a wall of tiles. Fifteen metrics is where a run
        starts on the FADGI profiles, and fifteen boxes is a page to scroll
        rather than a thing to read -- while the same fifteen as rows fit one
        screen and line up at the decimal point.
        """
        run = result.run
        evaluation = result.evaluation
        # Not inside a scroll area. The table scrolls on its own, and nesting one
        # scrolling thing inside another gives a page with two scrollbars where
        # the wheel does different things an inch apart.
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        # No prose summary of the table. It said "All 12 judged metrics met
        # their thresholds" directly above twelve rows each showing that it did,
        # and a paragraph restating what is under it is a paragraph a reader
        # learns to skip -- along with, eventually, the one line here that is
        # not a restatement.
        #
        # The caveats are that line. They stay in the open, never behind a
        # control: a reader fluent enough to stop opening things is exactly the
        # reader who must still see that this run cannot be quoted. One line,
        # composed from the same two facts the record's own sentences are
        # composed from, with those sentences on hover.
        advisory = evaluation.advisory_line
        if advisory:
            note = Callout(advisory, "qualified")
            layout.addWidget(note)

        line = run.profile.get("coverage_line") or ""
        if line:
            layout.addWidget(caveat(line))
        layout.addWidget(divider())

        table = DataTable(self._metric_columns(), list(run.metrics))
        table.doubleClicked.connect(self._open_metric_row)
        table.setToolTip("Double-click a metric to open it.")
        self.metric_table = table
        layout.addWidget(table, 1)
        return holder

    @staticmethod
    def _metric_columns() -> list[Col]:
        # The value and its unit are separate columns. Together they elide --
        # "1.190 % (std …" -- and a column of readings that cannot be read is
        # the one column this table exists for. Apart, the numbers line up at
        # the decimal point and the unit takes the room it needs.
        #
        # Name and unit are both ``fit``: their content is longer than any width
        # guessed in pixels, and both had rows eliding while the condition
        # column -- which absorbs the window -- sat half empty.
        return [
            Col("", lambda m: "", tone=lambda m: metric_status(
                m.get("status", "")).tone, width=32, resize="fixed",
                tooltip=lambda m: metric_status(m.get("status", "")).hint,
                sort=lambda m: metric_status(m.get("status", "")).rank),
            Col("Metric", metric_name, width=212, fit=True),
            # "Value", and set in the same face as the rest of the row. The mono
            # face was there to line the decimal points up, and it lines up a
            # column that now also holds "JPEG 2000" and "Adobe RGB (1998)" --
            # a second typeface for one column of five, to align numbers that
            # are one per metric and are never added up.
            Col("Value", metric_value, align=RIGHT, width=96,
                sort=lambda m: (m.get("value") is None, m.get("value") or 0.0)),
            Col("Units", lambda m: unit_label(m.get("units")), width=118,
                dim=True, fit=True),
            # "Condition" rather than "Judged Against": the header is
            # uppercased and letter-spaced, and the longer label was the one
            # thing on the row that could not fit its own column.
            Col("Condition",
                lambda m: condition_label(m.get("condition"))
                or "— no threshold —",
                width=168, dim=True, resize="stretch"),
            Col("Verdict",
                lambda m: metric_status(m.get("status", "")).label,
                align=CENTER, width=98,
                sort=lambda m: metric_status(m.get("status", "")).rank,
                tooltip=lambda m: m.get("reason") or metric_status(
                    m.get("status", "")).hint),
        ]

    def _open_metric_row(self, index) -> None:
        from deltachrome_ui.tables import ROW_ROLE

        metric = index.data(ROW_ROLE)
        if not metric or self.result is None:
            return
        metric_id = metric.get("metric_id", "")
        for family in families(
            self.result.run.metrics, self.result.run.profile.get("profile_id")
        ):
            if any(m.get("metric_id") == metric_id for m in family.metrics):
                self.show_metric(family.key, metric_id)
                return

    def _titled(self, name: str, body: QWidget) -> QWidget:
        """A page that carries its own name.

        A metric page says what it is by its reading line. These have nothing of
        the sort, and identifying a pane only by the row that reached it asks the
        reader to remember where they clicked.
        """
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        layout.addWidget(title(name))
        layout.addWidget(body, 1)
        return holder

    #: How wide a warning is allowed to run. A paragraph set across a 1400 px
    #: window is one the eye loses its place in on every line; the usual measure
    #: for body text is sixty to eighty characters, and these are paragraphs
    #: rather than labels.
    WARNING_MEASURE = 720

    def _warnings(self, warnings: list[str]) -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        for text in warnings:
            line = caveat(f"·  {text}")
            line.setMaximumWidth(self.WARNING_MEASURE)
            layout.addWidget(line)
        layout.addStretch(1)
        return _scrolled(holder)
