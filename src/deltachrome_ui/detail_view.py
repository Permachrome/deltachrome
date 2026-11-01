"""Getting from a verdict to a cause.

A failing aggregate says a system is out of specification; it does not say why,
and the two most common causes look completely different in the raw data. A
color cast moves every patch a little. Dust or a specular hit moves one patch a
lot. The mean cannot tell them apart, and the 90th percentile only half can --
which is the entire reason the engine persists per-patch values rather than
aggregates.

So this shows the distribution rather than the extremes: every patch, every
component, in a table that reorders on any column. The same argument applies to
a field of edges, where the differences between edges -- not their average --
are what point at astigmatism, field curvature or a tilted plane.

Nothing here computes anything. Every number was produced by the engine and is
read out of the saved record.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from deltachrome_ui.charts import (
    CHANNEL_COLORS,
    ChartSelection,
    FilterChips,
    GroupLegend,
    LinePlot,
    Series,
    group_color,
    group_styles,
)
from deltachrome_ui.labels import (
    field_order,
    flag_labels,
    reading_unit,
    region_name,
    sfr_direction,
    sfr_direction_hint,
    sfr_direction_label,
)
from deltachrome_ui.tables import RIGHT, ROW_ROLE, Col, DataTable
from deltachrome_ui.theme import TEXT, TEXT_DIM
from deltachrome_ui.widgets import caveat

def target_aspect(run) -> float | None:
    """The proportions of the extent the fiducials bound, if the run says.

    Relative positions live in that space, not in the card's, so it is the shape
    a field map must be drawn at -- a field stretched to fill its widget puts
    distance from the center, which is what separates field curvature from a
    tilted plane, at the wrong length.
    """
    extent = (getattr(run, "geometry", None) or {}).get("sampling_rate_reference_mm")
    if not extent or len(extent) != 2 or not extent[1]:
        return None
    return float(extent[0]) / float(extent[1])


class PatchTable(QWidget):
    """Per-patch values for one metric, as a table.

    Was a hand-laid grid of monospace labels, truncated at twelve with a
    sentence apologising for the rest. It is a list of measurements and it is
    now the same table every other list uses, with all hundred and forty rows
    in it. It opens worst first -- a failing aggregate is a question about the
    tail -- and the header reorders it from there, which is the whole reason a
    hundred and forty readings belong in a table: "which patch", "how bad", and
    "were the flagged ones the failing ones" are three orderings of one list.

    Hovering a row lights that reading on the capture panel, which is the whole
    reason a table is better here than a picture of one: the row *is* the mark.
    """

    def __init__(self, metric: dict, selection: ChartSelection | None = None) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        per_patch = [
            p for p in metric.get("per_patch", []) if p.get("value") is not None
        ]
        if not per_patch:
            layout.addWidget(caveat("This metric reports no per-patch values."))
            return

        # The unit heads the column instead of trailing every cell in it. A
        # hundred and forty rows each ending "% (σ / mean L*)" is the same nine
        # words printed a hundred and forty times, and it pushed the numbers --
        # the thing the column is for -- left of where the eye looks for them.
        table = SelectingTable([
            Col("Patch", lambda p: str(p.get("patch_id", "—")), width=110,
                fit=True),
            Col(reading_unit(metric) or "Value", lambda p: f"{p['value']:.3f}",
                sort=lambda p: p["value"], align=RIGHT, width=110, fit=True),
            # No Role column. A metric selects the patches it measures by role,
            # so every row on this page carries the same one -- and where it
            # does not, the difference is between "achromatic ramp" and
            # "chromatic", which is not what a reader has come to this table to
            # find out. It is on the patch's own record either way.
            Col("Warnings", lambda p: ", ".join(flag_labels(p.get("flags"))),
                width=200, resize="stretch"),
        ], per_patch, selection, key=lambda p: str(p.get("patch_id", "")))
        # Opens worst first, because a failing aggregate is a question about the
        # tail. The header reorders it from there.
        table.order_by(1, descending=True)
        layout.addWidget(table, 1)


class SelectingTable(DataTable):
    """A readings table joined to the report's one selection.

    Pointing at a row lights the reading it names on the capture panel, and
    clicking pins it. Without that the table is the one place in the report
    where "which reading is this" has to be answered by reading, while the panel
    beside it answers the same question by pointing.
    """

    def __init__(self, columns, rows, selection, key) -> None:
        super().__init__(columns, rows)
        self._selection = selection
        self._key = key
        self.setMouseTracking(True)
        if selection is not None:
            selection.changed.connect(self._follow)
        self.clicked.connect(self._on_clicked)

    def _name_at(self, index) -> str | None:
        row = index.data(ROW_ROLE) if index.isValid() else None
        return self._key(row) if row is not None else None

    def mouseMoveEvent(self, event) -> None:
        if self._selection is not None:
            self._selection.set_hovered(
                self._name_at(self.indexAt(event.position().toPoint()))
            )
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        if self._selection is not None:
            self._selection.set_hovered(None)
        super().leaveEvent(event)

    def _on_clicked(self, index) -> None:
        if self._selection is not None:
            self._selection.toggle_selected(self._name_at(index))

    def _follow(self) -> None:
        """Scroll to whatever was focused elsewhere, without echoing back."""
        if self._selection is None:
            return
        name = self._selection.primary
        if not name:
            return
        model = self.model()
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            if self._name_at(index) == name:
                self.scrollTo(index)
                return


def placed_samples(metric: dict) -> list[dict]:
    """The readings of one metric that know where they were taken.

    A metric may place some of its readings and not others -- a patch whose grid
    cannot give a center, a registration point measured before a placement was
    available -- so this is a filter, not a yes/no about the metric.
    """
    return [
        sample for sample in (metric.get("per_patch") or [])
        if sample.get("value") is not None
        and sample.get("position_relative")
        and len(sample["position_relative"]) == 2
    ]


class PatchDetail(QWidget):
    """What was actually measured on each patch, before any metric ran.

    **Every patch, and each L\\*a\\*b\\* component in its own column.** It was
    twelve rows of a hand-laid monospace grid -- the worst twelve by separation
    -- under a sentence saying how many there were and how many of those were
    flagged. Three things were wrong with that and they are one thing: the page
    is called Per-Patch Values and it was not showing the per-patch values. A
    hundred and forty readings truncated to twelve is a summary, the sentence
    above it was the page apologising for being a summary, and neither could be
    reordered to ask a different question of the same list.

    So it is the table, sorted and searchable by every column like the rest of
    them. L\\*, a\\* and b\\* are six columns rather than two cells of packed
    text, because the question is always about one of them -- a cast is a\\* and
    b\\*, a tone error is L\\* -- and a column of six-figure strings cannot be
    sorted on any of them.
    """

    def __init__(self, run, limit: int | None = None) -> None:  # noqa: ARG002
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        patches = [p for p in run.patches if p.measured_lab and p.reference_lab]
        if not patches:
            layout.addWidget(caveat("This target carries no patches to compare."))
            return

        def measured(index: int):
            return lambda p: float(p.measured_lab[index])

        def reference(index: int):
            return lambda p: float(p.reference_lab[index])

        def number(read):
            return lambda p: f"{read(p):.2f}"

        columns = [Col("Patch", lambda p: p.patch_id, width=84, fit=True)]
        for name, read in (
            ("L*", measured(0)), ("a*", measured(1)), ("b*", measured(2)),
        ):
            columns.append(Col(name, number(read), sort=read,
                               align=RIGHT, width=74))
        for name, read in (
            ("Aim L*", reference(0)), ("Aim a*", reference(1)),
            ("Aim b*", reference(2)),
        ):
            columns.append(Col(name, number(read), sort=read,
                               align=RIGHT, width=80, dim=True))
        columns.append(Col(
            "Warnings", lambda p: ", ".join(flag_labels(p.flags)), width=160,
            resize="stretch",
        ))

        table = DataTable(columns, patches)
        # Darkest first, which is the order the chart prints a ramp in and the
        # order a reader scans a list of neutrals. Any other question the table
        # answers is one click on a heading.
        table.order_by(1, descending=True)
        layout.addWidget(table, 1)


#: Half the sampling frequency, in cycles per pixel. Every SFR threshold in the
#: shipped profiles is a fraction of it, which is what lets four metrics share
#: one plot and differ only in where their aim sits on it.
NYQUIST = 0.5

def _at(region: str | None) -> str:
    """A field position as a mute key, namespaced away from the record keys."""
    return f"at:{region}" if region else ""


def sfr_aim(metric: dict | None) -> tuple[list[tuple], list[tuple]]:
    """Where on the MTF curve this metric's requirement lives.

    ``(regions, hlines)``. All four SFR metrics are conditions on the *same*
    curve at different places, which is why they share a plot and why the
    picture changes when the metric does:

    * **SFR50** -- the curve must cross MTF 0.5 inside a window of frequency.
      Two coordinates, not one, so it is a box and not a limit line. This is the
      shape OpenDICE draws and it is the right one.
    * **SFR10 sampling efficiency** -- the curve must still be at MTF 0.1 out
      beyond some fraction of Nyquist.
    * **Response at Nyquist** -- a ceiling on the modulation at one frequency.
    * **Sharpening** -- a ceiling on the peak, wherever the peak is, so a line
      across the whole width.

    Returns nothing for a metric with no threshold or none this can place, which
    draws no aim rather than a guessed one.
    """
    if not metric:
        return [], []
    from deltachrome_ui.labels import parse_condition

    metric_id = str(metric.get("metric_id", ""))
    bounds = parse_condition(metric.get("condition"))
    if not bounds:
        return [], []
    if metric_id == "sfr50":
        low = (bounds.low or 0.0) / 100.0 * NYQUIST
        high = (bounds.high or 100.0) / 100.0 * NYQUIST
        return [(low, high, 0.40, 0.60, "MTF50 aim")], []
    if metric_id == "sfr10_sampling_efficiency":
        low = (bounds.low or 0.0) / 100.0 * NYQUIST
        return [(low, 1.05, 0.05, 0.15, "MTF10 aim")], []
    if metric_id == "sfr_response_at_nyquist":
        high = bounds.high if bounds.high is not None else 1.0
        return [(NYQUIST - 0.015, NYQUIST + 0.015, 0.0, high, "at Nyquist")], []
    if metric_id == "sharpening" and bounds.high is not None:
        return [], [(bounds.high, "peak modulation limit")]
    return [], []


class EdgeCurves(QWidget):
    """Every edge's MTF, with controls for looking at fewer of them.

    Twenty edges in three colour records is eighty curves, and eighty curves is
    a texture rather than a plot. OpenDICE avoids that by never drawing more
    than four at once -- one field position, one orientation, R, G, B and
    luminance -- at the cost of never being able to see the field's spread,
    which is the finding the aggregate cannot give you.

    So both, from one control. It opens on every edge's luminance curve, which
    is the spread; the chips above it narrow to one corner, one orientation and
    whichever colour records are wanted, and a reader who lands on Centre,
    Horizontal, all records has exactly OpenDICE's four lines.

    Narrowing is muting rather than rebuilding, so the plot, the values table
    and the capture panel cannot come to disagree about which edges are in play:
    all three read the same :class:`~deltachrome_ui.charts.ChartSelection`.

    The aim box comes from whichever metric is on screen. Four metrics share
    this curve and each is a requirement at a different place on it, so the
    picture is the same and the box moves -- which is the argument for showing
    them here at all rather than as four numbers.

    **And the field can be reduced instead of hidden.** Narrowing answers "look
    at fewer edges"; it does not answer "is this corner softer than that one",
    which is the question a field of edges is shot to ask and which twenty
    overlaid curves cannot be read for. So the fourth control averages what is
    currently shown, on one of the two keys worth averaging on:

    *Per Position* gives one curve for each corner and the centre -- the
    comparison across the field, with the individual edges' scatter traded for
    being able to see it. *Per Direction* gives one curve for each direction the
    response was measured in, which with the Edge Location chips narrowed to a
    single corner is the other useful reduction: the horizontals against the
    verticals at one place, which is astigmatism if they differ there and plane
    tilt if they differ everywhere.

    **The average is of whatever is on screen**, not of a fixed set, which is
    what makes two controls do the work of six: every narrowing already
    available composes with it, and a muted record or a hidden orientation is
    out of the mean as well as off the plot. The cost is that the curves have to
    be rebuilt when a mute changes rather than merely re-drawn, which is why
    this widget listens to the selection at all.
    """

    #: The colour records, and what they are called. Off by default: three more
    #: curves per edge is what made the plot unreadable, and a reader who wants
    #: them is a reader who has already narrowed to one position.
    RECORDS = (("r", "R"), ("g", "G"), ("b", "B"))

    def __init__(self, run, selection: ChartSelection | None = None,
                 metric: dict | None = None) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        # Off the tab bar above it. At zero the first chip row sat against the
        # rule under Charts/Values and read as part of it.
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(10)

        edges = list(run.edges or [])
        if not edges:
            layout.addWidget(caveat(
                "This target carries no slanted edges, so no resolution field was "
                "measured. That is a property of the target, not a fault."
            ))
            return

        self.selection = selection or ChartSelection(self)
        # Field positions are namespaced before they become mute keys. They and
        # the colour records share one namespace in ChartSelection, and a target
        # whose region happened to be called "r" would otherwise be hidden the
        # moment the red record was turned off -- which is exactly what a
        # fixture named region "r" did.
        # Reading order across the field, rather than the order the detector
        # happened to find them in.
        found = {e.region for e in edges if e.region}
        self.regions = sorted(found, key=field_order)
        # Read before anything below mutes a record: every mute emits, and the
        # previous chart's page -- alive until the report drops it -- writes the
        # selection's state back on every emission, defaults included.
        carried = dict(self.selection.edge_view)
        for record, _ in self.RECORDS:
            self.selection.set_muted(record, True)
        with_channels = [e for e in edges if getattr(e, "channels", None)]
        present_orientations = {e.orientation for e in edges}
        self._carry_view(carried, present_orientations, bool(with_channels))

        # Listed in the order their *displayed* names read, which is the
        # opposite of the record's -- a near-vertical edge reports the
        # horizontal response, so the chip row would otherwise open with
        # "Vertical".
        present = [
            orientation for orientation in ("vertical", "horizontal")
            if any(e.orientation == orientation for e in edges)
        ]
        styles = group_styles(present)

        self._region_styles = group_styles(list(self.regions))
        if len(self.regions) > 1:
            # A legend rather than a filter, which is the operator's correction:
            # a filter answers "which one corner", and the question a field of
            # nine modules is usually asking is "these three against those
            # three". The mute keys are the same ones ``_choose_region`` always
            # set, so nothing downstream changed -- what changed is that more
            # than one can be on at once, and that the chips now say truthfully
            # which are.
            self._position_chips = GroupLegend(
                self.selection,
                # White until a per-location mean colours them -- see
                # ``_apply_average``: no curve wears a location's colour before.
                [(_at(r), QColor(TEXT), region_name(r)) for r in self.regions],
                all_label="All Locations",
            )
            layout.addWidget(self._chip_row("Edge Location", self._position_chips))
        # The word is the *direction the response was measured in*, not the way
        # the edge runs -- see ``labels.sfr_direction``. The mute key is still
        # the record's own orientation, so nothing about the plot moves.
        layout.addWidget(self._chip_row("Orientation", GroupLegend(
            self.selection,
            [
                (orientation, styles[orientation][0],
                 sfr_direction_label(orientation))
                for orientation in present
            ],
            hints={o: sfr_direction_hint(o) for o in present},
        )))
        if with_channels:
            layout.addWidget(self._chip_row("Channel", GroupLegend(
                self.selection,
                # White: a luminance curve wears its orientation's or its
                # location's colour, never one of its own, so this square is a
                # switch and not a key.
                [("lum", QColor(TEXT), "Luminance")]
                + [(key, CHANNEL_COLORS[key], label)
                   for key, label in self.RECORDS],
            )))

        self._edges = edges
        self._with_channels = with_channels
        # **Averaged per location by default**, which reverses how this opened
        # when it was built. The spread of every edge is a real finding and it is
        # the one a reader can go to; what they arrive asking is whether one part
        # of the field is soft, and twenty overlaid curves is a texture that
        # cannot be read for it. So the plot opens on the answer and *Every Edge*
        # is one press away. A frame with one field position has nothing to
        # average across and opens on its edges, as before.
        self._average = "region" if len(self.regions) > 1 else ""
        if len(edges) > 1:
            entries = []
            if len(self.regions) > 1:
                entries.append(("region", "Per Location"))
            if len(present) > 1:
                entries.append(("orientation", "Per Direction"))
            # Fewest lines first, most last, which is the order a reader opens
            # them in: the summary, then the summary split, then everything.
            entries.append(("", "Every Edge"))
            average = carried.get("average")
            if average is not None and average in {key for key, _ in entries}:
                self._average = average
                self.selection.edge_view["average"] = average
            if len(entries) > 1:
                self._average_chips = FilterChips(entries, current=self._average)
                self._average_chips.chosen.connect(self._choose_average)
                layout.addWidget(self._chip_row("Average", self._average_chips))

        plot = LinePlot(
            x_label="cycles / pixel", y_label="MTF", y_range=(0.0, 1.15),
            selection=self.selection,
        )
        plot.add_marker(NYQUIST, "Nyquist")
        boxes, hlines = sfr_aim(metric)
        for box in boxes:
            plot.add_region(*box)
        for level, label in hlines:
            plot.add_hline(level, label)

        self._plot = plot
        # What the last rebuild was for. A hover emits ``changed`` too, and
        # rebuilding a hundred and forty curves on every mouse move is a cost
        # with nothing to show for it.
        self._series_key: tuple | None = None
        self.selection.changed.connect(self._refresh_series)
        self.selection.changed.connect(self._remember_view)
        # Through the same path a press takes, so the colours the chips wear and
        # the curves the plot draws are set in one place rather than two.
        self._apply_average(self._average)
        # The plot takes the height, not a trailing stretch. With one the plot
        # sat at its minimum and the empty space went under the prose.
        layout.addWidget(plot, 1)
        # No sentence under the plot. It used to say how far apart the colour
        # records' SFR50 readings were and whether that grew towards the corners:
        # a diagnostic nothing is judged on, and the values table carries each
        # record's own reading, which is the figure to act on.

    # -- what is drawn ------------------------------------------------------

    def _per_edge_series(self) -> list[Series]:
        """Every edge, in luminance and in whichever records are on."""
        series = [
            Series(
                x=list(edge.mtf_curve["frequency_cyc_px"]),
                y=list(edge.mtf_curve["mtf"]),
                color=group_color(edge.orientation),
                name=edge.edge_id,
                group=(edge.orientation, "lum", _at(edge.region)),
            )
            for edge in self._edges if edge.mtf_curve
        ]
        series += [
            Series(
                x=list(edge.mtf_curve["frequency_cyc_px"]),
                y=list(entry["mtf"]),
                color=CHANNEL_COLORS[record],
                name=edge.edge_id,
                group=(edge.orientation, record, _at(edge.region)),
                width=1.0,
            )
            for edge in self._with_channels if edge.mtf_curve
            for record, entry in sorted(edge.channels.items())
            if record in CHANNEL_COLORS and entry.get("mtf")
        ]
        return series

    def _shown(self, edge) -> bool:
        """Whether this edge survives the narrowing chips, records aside.

        Position and orientation only. Which colour *record* is on is a separate
        question, answered once per record rather than once per edge, and asking
        it here would drop an edge from the luminance mean because its red curve
        was switched off.
        """
        muted = self.selection.muted
        return (bool(edge.mtf_curve) and edge.orientation not in muted
                and _at(edge.region) not in muted)

    def _averaged_series(self, key: str) -> list[Series]:
        """One curve per group, over the edges currently on screen.

        Every curve in a run is sampled on the same frequency grid --
        ``build_edge_record`` writes a fixed one -- so a mean is the mean of the
        columns and needs no resampling. That is worth stating because it is the
        only reason this is four lines rather than an interpolation: if the grid
        were ever per-edge, averaging would have to say what it did about the
        ends.
        """
        shown = [e for e in self._edges if self._shown(e)]
        if not shown:
            return []
        records: list[tuple[str, QColor | None]] = []
        if "lum" not in self.selection.muted:
            records.append(("lum", None))
        records += [
            (record, CHANNEL_COLORS[record]) for record, _ in self.RECORDS
            if record not in self.selection.muted
            and any(e.channels.get(record, {}).get("mtf") for e in shown)
        ]

        series: list[Series] = []
        for record, record_color in records:
            groups: dict[str, list] = {}
            for edge in shown:
                groups.setdefault(
                    edge.region if key == "region" else edge.orientation, []
                ).append(edge)
            for name, members in sorted(
                groups.items(),
                key=lambda pair: field_order(pair[0]) if key == "region" else pair[0],
            ):
                curves = [
                    edge.mtf_curve["mtf"] if record == "lum"
                    else (edge.channels.get(record) or {}).get("mtf")
                    for edge in members
                ]
                curves = [c for c in curves if c]
                if not curves:
                    continue
                width = min(len(c) for c in curves)
                mean = [
                    sum(c[i] for c in curves) / len(curves) for i in range(width)
                ]
                label = (region_name(name) if key == "region"
                         else sfr_direction_label(name))
                if key == "region":
                    color = record_color or self._region_styles[name][0]
                    group = (record, _at(name))
                else:
                    color = record_color or group_color(name)
                    group = (name, record)
                series.append(Series(
                    x=list(members[0].mtf_curve["frequency_cyc_px"][:width]),
                    y=mean,
                    color=color,
                    # Named for what it is and how much of it there is: a mean of
                    # one edge is not a mean, and a reader has to be able to see
                    # that from the readout rather than from the chip row.
                    name=f"{label} mean of {len(curves)}"
                         + ("" if record == "lum" else f"  ·  {record.upper()}"),
                    group=group,
                    width=2.2,
                ))
        return series

    def _refresh_series(self) -> None:
        """Rebuild the curves, but only when the answer would change."""
        key = (frozenset(self.selection.muted), self._average)
        if key == self._series_key:
            return
        self._series_key = key
        self._plot.set_series(
            self._averaged_series(self._average) if self._average
            else self._per_edge_series()
        )

    #: The chips whose state is carried from one chart of a capture to the next.
    #: Not the locations: each target's field positions are its own.
    VIEW_KEYS = ("vertical", "horizontal", "lum", "r", "g", "b")

    def _carry_view(self, view: dict, orientations: set, channels: bool) -> None:
        """Turn the orientations and channels the way the last chart had them.

        Two guards, both so a carried view cannot leave the plot empty: a chart
        offering only the orientations somebody turned off shows them anyway, and
        on a chart with no per-record curves luminance is never off, there being
        no chip to turn it back on.
        """
        carried = view.get("muted")
        if carried is None:
            return
        for key in self.VIEW_KEYS:
            self.selection.set_muted(key, key in carried)
        if orientations and orientations <= self.selection.muted:
            for key in orientations:
                self.selection.set_muted(key, False)
        if not channels:
            self.selection.set_muted("lum", False)

    def _remember_view(self) -> None:
        self.selection.edge_view["muted"] = {
            key for key in self.VIEW_KEYS if key in self.selection.muted
        }

    def _choose_average(self, key: str) -> None:
        """A press: remembered for the next chart, then applied."""
        self.selection.edge_view["average"] = key
        self._apply_average(key)

    def _apply_average(self, key: str) -> None:
        self._average = key
        chips = getattr(self, "_position_chips", None)
        if chips is not None:
            # The chips are the only key on screen for a per-location mean, and
            # nothing else colours by location -- so they carry those colours
            # exactly while that is what is drawn. See
            # :meth:`GroupLegend.set_colors`.
            chips.set_colors(
                {_at(r): self._region_styles[r][0] for r in self.regions}
                if key == "region" else None
            )
        self._refresh_series()



    def _chip_row(self, label: str, chips: QWidget) -> QWidget:
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(10)
        name = QLabel(label.upper())
        # Wide enough for EDGE LOCATION, which is the longest of the three and
        # the one that must not wrap: a two-line key beside a row of chips
        # pushes the chips down and makes the group look like two.
        name.setFixedWidth(104)
        name.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; font-weight: 700;"
            "letter-spacing: 0.8px;"
        )
        line.addWidget(name, 0, Qt.AlignmentFlag.AlignTop)
        line.addWidget(chips, 1)
        return row

    def _choose_region(self, region: str) -> None:
        """Narrow to exactly one field position, or to all of them.

        Kept as a method although the chips no longer call it: it is how another
        part of the report, and the tests, say "look only here", and expressing
        that as mutes is what makes the values table and the capture panel narrow
        with the plot. Three renderings of one set of edges, and a control that
        moved only the picture would leave the other two describing edges that
        are no longer on screen. The legend follows the selection rather than
        only driving it, so it redraws itself.
        """
        for name in self.regions:
            self.selection.set_muted(_at(name), bool(region) and name != region)


class RegistrationReadings(QWidget):
    """Every position, axis and channel pair as a row.

    The same population the six-plot grid draws -- three pairs across, two axes
    down, one column of bars per field position -- so the chart and the table
    are two views of one list rather than two different subsets. The metric's
    own ``per_patch`` is the *worst* pair at each position and axis, which is
    the right thing to judge on and the wrong thing to read a field from: it
    cannot say which record went where.

    **The antisymmetric term is a column here rather than a paragraph.** It is
    the quantity that pairing opposite edges *removes* from the verdict -- a
    displacement moves both edges of a square the same way, and only a change in
    the feature's apparent size moves them oppositely -- so it is the one
    diagnostic on this page that qualifies a number the report actually states.
    It used to be three sentences ending "Worst here: G-B", which is a column
    sorted descending, read aloud. No standard asks for it: FADGI 2.4.7 and ISO
    19264-1 both state the criterion in pixels of dislocation and neither
    mentions it. It is here because on the ISA Golden Thread capture it runs
    0.036 px against a translation of 0.025 -- larger than the thing being
    judged -- and a reader who cannot see that cannot tell whether pairing
    earned its keep.
    """

    def __init__(self, run, selection: ChartSelection | None = None) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        positions = (getattr(run, "registration_summary", None) or {}).get("positions")
        if not positions:
            layout.addWidget(caveat(
                "This run reports no per-position registration readings."
            ))
            return

        rows = []
        for entry in positions:
            translation = entry.get("translation_px") or {}
            antisymmetric = entry.get("antisymmetric_px") or {}
            for pair in sorted(translation):
                rows.append({
                    "region": str(entry.get("region") or ""),
                    "axis": str(entry.get("axis") or ""),
                    "pair": pair,
                    "translation": translation.get(pair),
                    "antisymmetric": antisymmetric.get(pair),
                    "error": entry.get("standard_error_px"),
                })
        if not rows:
            layout.addWidget(caveat(
                "This run reports no per-position registration readings."
            ))
            return

        def signed(key: str, places: int = 4):
            def text(row):
                value = row.get(key)
                # Signed, and the sign is the measurement: two displacements of
                # equal size and opposite direction are opposite findings.
                return "—" if value is None else f"{value:+.{places}f}"
            return text

        def magnitude(key: str):
            # Sorted on size rather than on sign, because "which is worst" is
            # the question a reader brings to a column of signed numbers.
            return lambda row: (
                -abs(row[key]) if row.get(key) is not None else 1.0
            )

        columns = [
            Col("Position", lambda r: region_name(r["region"]),
                sort=lambda r: field_order(r["region"]), width=130, fit=True),
            # The direction the displacement was measured in, named as the grid
            # above names its rows -- an ``x`` translation is measured across a
            # vertical edge and is a horizontal movement.
            Col("Direction", lambda r: "Horizontal" if r["axis"] == "x" else "Vertical",
                sort=lambda r: r["axis"], width=104, dim=True),
            Col("Pair", lambda r: r["pair"], width=76),
            Col("Translation", signed("translation"), sort=magnitude("translation"),
                align=RIGHT, width=118, fit=True,
                tooltip=lambda r: "Mean of the two opposite edges. Used for the verdict."),
            Col("Antisymmetric", signed("antisymmetric"),
                sort=magnitude("antisymmetric"), align=RIGHT, width=126, fit=True,
                dim=True,
                tooltip=lambda r: (
                    "Half the difference between the two opposite edges: a size "
                    "difference, not a shift. Not used for the verdict."
                )),
            Col("Fit error", lambda r: (
                "—" if r.get("error") is None else f"{r['error']:.4f}"),
                sort=lambda r: r.get("error") if r.get("error") is not None else -1.0,
                align=RIGHT, width=100, dim=True, resize="stretch",
                tooltip=lambda r: (
                    "Standard error of the edge fits. No shift smaller than this "
                    "can be resolved."
                )),
        ]
        table = DataTable(columns, rows)
        # Worst translation first, which is the row the verdict came from.
        table.order_by(3)
        layout.addWidget(table, 1)


class EdgeReadings(QWidget):
    """Every edge as a row, joined to the curves by one selection.

    **Orientation is the direction the response was measured in.** The record
    holds the edge's own geometry, which is what it is; ISO 12233 names the
    direction -- horizontal resolution is measured across a near-vertical edge
    -- and so does the registration grid on the next page. Two pages of one
    report using one word for opposite things is worse than either convention.

    **Where an edge sits is not a column any more.** It was the fiducial-bounded
    relative position, printed to two decimals, in a report that already draws
    every edge on the capture beside this table and groups them by field
    position in the chips above it. Two coordinates are the one form of that
    fact nobody can read.

    **The color records are three columns rather than one spread.** A single
    "how far apart are they" number says an edge has a chromatic problem without
    saying which record is soft, which is the thing to act on -- and it is what
    OpenDICE prints: SFR50 for luminance, red, green and blue at each position.
    Sorted on the blue column, a lens focusing blue short is one click.
    """

    def __init__(self, run, selection: ChartSelection | None = None) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        edges = list(run.edges or [])
        if not edges:
            layout.addWidget(caveat("This run measured no slanted edges."))
            return

        def record(key: str):
            """One color record's SFR50, or ``None`` where it was not measured.

            Grayscale captures carry one record, so this is empty on them and
            the column shows a dash rather than an invented number.
            """
            def read(edge):
                entry = (getattr(edge, "channels", None) or {}).get(key) or {}
                value = entry.get("sfr50_pct")
                return None if value is None else float(value)
            return read

        def number(read, places: int = 1):
            return lambda e: "—" if read(e) is None else f"{read(e):.{places}f}"

        columns = [
            # The id is the row's identity, so it gets the room it needs and
            # the table scrolls sideways rather than eliding every row to
            # "module_…". A table may scroll sideways; prose may not.
            Col("Edge", lambda e: e.edge_id, width=190, fit=True),
            Col("Orientation", lambda e: sfr_direction_label(e.orientation),
                sort=lambda e: sfr_direction(e.orientation), width=104, dim=True,
                tooltip=lambda e: sfr_direction_hint(e.orientation)),
            # Absolute. The sign is which way the edge leans, and the
            # measurement is indifferent to it -- what matters is that the tilt
            # is far enough from zero to sample every sub-pixel phase.
            Col("Angle°", lambda e: f"{abs(e.edge_angle_deg):.2f}",
                sort=lambda e: abs(e.edge_angle_deg), align=RIGHT, width=76,
                tooltip=lambda e: "Tilt from the axis."),
            Col("SFR50", number(lambda e: e.sfr50_pct),
                sort=lambda e: e.sfr50_pct, align=RIGHT, width=76,
                tooltip=lambda e: "Percent of half the sampling frequency."),
            # The reading a digitisation department actually quotes for
            # sharpness, and it was measured on every edge and shown on none of
            # them: the metric page reports the field's aggregate, and which
            # *position* the system is failing at is the question a field of
            # edges exists to answer.
            Col("Efficiency", number(lambda e: e.sampling_efficiency_pct),
                sort=lambda e: e.sampling_efficiency_pct, align=RIGHT, width=94,
                tooltip=lambda e: "SFR10 sampling efficiency, as a percent of "
                                  "half the sampling frequency."),
            Col("Nyquist", lambda e: f"{e.mtf_at_nyquist:.3f}",
                sort=lambda e: e.mtf_at_nyquist, align=RIGHT, width=82,
                tooltip=lambda e: "Modulation at half the sampling frequency."),
            # The quantity the sharpening metric is judged on: the highest
            # modulation anywhere on this edge's curve. A clean edge reads
            # 1.000 and anything above it is the system putting back contrast
            # the optics did not deliver.
            Col("Max", lambda e: f"{e.peak_modulation:.3f}",
                sort=lambda e: e.peak_modulation, align=RIGHT, width=76,
                tooltip=lambda e: "Peak modulation."),
        ]
        if any(getattr(e, "channels", None) for e in edges):
            for key, name, label in (("r", "R", "red"), ("g", "G", "green"),
                                     ("b", "B", "blue")):
                read = record(key)
                columns.append(Col(
                    name, number(read), sort=read, align=RIGHT, width=68,
                    tooltip=lambda e, n=label: f"SFR50 in the {n} channel. Not "
                                               "used for the verdict.",
                ))
        # Last, and asking for the slack. Nine columns of readings already exceed
        # a 1360 px window, so this is also the column that says whether the
        # table can scroll sideways at all -- see ``DataTable.setModel``.
        columns.append(Col("Warnings", lambda e: ", ".join(e.warnings), width=240,
                           fit=True, resize="stretch"))

        table = SelectingTable(columns, edges, selection, key=lambda e: e.edge_id)
        # Opens on SFR50, softest edge first: the same question as the readings
        # table, asked of a field of edges.
        table.order_by(3)
        layout.addWidget(table, 1)


@dataclass
class CaptureRegion:
    """One reading's place in the capture: what to read, and what to outline."""

    #: The four corners, in image pixels, of the region the reading was taken
    #: from. A quadrilateral rather than a rectangle, because under perspective
    #: a patch is one.
    outline: list[list[float]]
    color: QColor
    #: Where the reading is centerd, when that is a meaningful point of its own.
    #: An edge has one; a patch's center is just the middle of its outline and
    #: marking it says nothing.
    center_px: tuple[float, float] | None = None
    #: Further outlines belonging to the same reading. Gain modulation reads a
    #: *pair* of ramp patches, often with others between them, so its region
    #: is two patches rather than one box round the stretch that holds both.
    others: tuple = ()
    #: Drawn dashed: a slanted edge or a position measured on edges, where a
    #: solid outline is a patch. The capture panel draws every region in one
    #: faint ink, so the line is what tells the two kinds apart.
    dashed: bool = False

    @property
    def shapes(self) -> list:
        """Every outline of this reading, the first being ``outline``."""
        return [self.outline, *self.others]

    def bounds(self, pad: float) -> tuple[int, int, int, int]:
        """The rectangle to read, padded by a fraction of the region's size.

        Padded because the question being asked is usually about the *border* of
        the region -- a speck on the edge of a patch, a rule band sitting off the
        printed scale. Showing exactly the sampled area answers "is the middle
        clean", which is rarely what was in doubt. The outline is drawn on top,
        so what was measured stays unambiguous.
        """
        xs = [p[0] for shape in self.shapes for p in shape]
        ys = [p[1] for shape in self.shapes for p in shape]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        margin_x, margin_y = (x1 - x0) * pad, (y1 - y0) * pad
        return (
            int(x0 - margin_x), int(y0 - margin_y),
            int(x1 + margin_x) + 1, int(y1 + margin_y) + 1,
        )


def edge_regions(edges) -> dict[str, CaptureRegion]:
    """Capture regions for a field of slanted edges."""
    regions = {}
    for edge in edges:
        if not edge.bounds_px or len(edge.bounds_px) != 4:
            continue
        x0, y0, x1, y1 = (float(v) for v in edge.bounds_px)
        regions[edge.edge_id] = CaptureRegion(
            outline=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            color=group_color(edge.orientation),
            center_px=(float(edge.center_px[0]), float(edge.center_px[1])),
            dashed=True,
        )
    return regions


def pair_readings_for(run, metric: dict, units: str, styles=None):
    """Regions and readings for a metric read on *pairs* of patches.

    Gain modulation compares two ramp steps, recording the reading under
    ``"<first>-<second>"``. Each half is matched against the run's own patch
    ids rather than split on the hyphen, so a patch id that carries one cannot
    be cut in the wrong place. Two empty dicts where the readings are not pairs.
    """
    by_patch = {
        str(p.patch_id): p for p in (getattr(run, "patches", None) or [])
        if getattr(p, "roi_px", None) and len(p.roi_px) >= 3
    }
    regions: dict[str, CaptureRegion] = {}
    readings: dict = {}
    for sample in metric.get("per_patch") or []:
        key = str(sample.get("patch_id") or "")
        if sample.get("value") is None or "-" not in key:
            continue
        halves = next((
            (key[:i], key[i + 1:]) for i, ch in enumerate(key)
            if ch == "-" and key[:i] in by_patch and key[i + 1:] in by_patch
        ), None)
        if halves is None:
            continue
        first, second = (by_patch[h].roi_px for h in halves)
        color, _ = (styles or {}).get(str(sample.get("role") or ""), (None, None))
        regions[key] = CaptureRegion(
            outline=[[float(x), float(y)] for x, y in first],
            color=color or QColor(TEXT_DIM),
            others=([[float(x), float(y)] for x, y in second],),
        )
        readings[key] = (sample["value"], units, sample.get("flags") or [])
    return regions, readings


def edge_readings_for(run, metric: dict, units: str):
    """Regions and readings for a metric read on slanted edges, or two empty dicts.

    Each SFR metric records its own value per edge, keyed by the edge's id --
    SFR50 its SFR50, sampling efficiency its efficiency -- so the panel shows the
    reading of the metric on screen rather than SFR50 for all four. Registration
    reads one value per field position and axis, keyed ``position:axis``; its
    region is the box round that position's edges, both axes sharing it.
    """
    edges = list(getattr(run, "edges", None) or [])
    values = [
        p for p in (metric.get("per_patch") or []) if p.get("value") is not None
    ]
    if not edges or not values:
        return {}, {}
    by_id = {e.edge_id: e for e in edges}
    readings = {
        str(p.get("patch_id")): (p["value"], units, p.get("flags") or [])
        for p in values
    }
    if all(key in by_id for key in readings):
        return edge_regions([by_id[key] for key in readings]), readings

    by_position: dict[str, list] = {}
    for edge in edges:
        if edge.region and edge.bounds_px and len(edge.bounds_px) == 4:
            by_position.setdefault(edge.region, []).append(edge)
    regions: dict[str, CaptureRegion] = {}
    for key in readings:
        position, _, _axis = key.partition(":")
        held = by_position.get(position)
        if not held:
            continue
        x0 = min(float(e.bounds_px[0]) for e in held)
        y0 = min(float(e.bounds_px[1]) for e in held)
        x1 = max(float(e.bounds_px[2]) for e in held)
        y1 = max(float(e.bounds_px[3]) for e in held)
        regions[key] = CaptureRegion(
            outline=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            color=QColor(TEXT_DIM),
            center_px=((x0 + x1) / 2, (y0 + y1) / 2),
            dashed=True,
        )
    if not regions:
        return {}, {}
    return regions, {key: value for key, value in readings.items() if key in regions}


def sample_regions(run, metric: dict, styles) -> dict[str, CaptureRegion]:
    """Capture regions for one metric's readings.

    Two sources, for one reason. A patch's sampled quadrilateral lives on its own
    record and is joined here by id, because the metrics repeat their per-patch
    lists and storing the corners once per metric would write the same numbers
    out half a dozen times. Everything else -- a flat-field spot, a rule band, a
    registration feature -- carries its own, since each appears in one metric and
    they number tens rather than hundreds.
    """
    by_patch = {p.patch_id: p for p in (run.patches or [])}
    regions: dict[str, CaptureRegion] = {}
    for sample in placed_samples(metric):
        patch_id = str(sample.get("patch_id") or "")
        outline = sample.get("roi_px")
        if not outline:
            record = by_patch.get(patch_id)
            outline = getattr(record, "roi_px", None) if record else None
        if not outline or len(outline) < 3:
            continue
        color, _ = styles.get(str(sample.get("role") or ""), (None, None))
        regions[patch_id] = CaptureRegion(
            outline=[[float(x), float(y)] for x, y in outline],
            color=color or QColor(TEXT_DIM),
        )
    return regions


def ramp_patches(run) -> list:
    """The neutral ramp steps that carry both a measurement and an aim.

    Selected by role rather than by patch id: which patches form the ramp is a
    property of the target and is already resolved once, at analysis time, by
    :mod:`deltachrome.targets.roles`. Re-deriving it here from lightness would be
    a second opinion that could disagree with the one the metrics used.
    """
    return [
        patch for patch in (run.patches or [])
        if patch.role == "achromatic_ramp"
        and patch.measured_lab and patch.reference_lab
    ]
