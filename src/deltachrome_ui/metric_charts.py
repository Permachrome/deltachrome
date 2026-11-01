"""What each metric is drawn as.

One function decides, and it decides from the metric rather than from the family
it sits in. That distinction was a live bug: the tone-response curve was built
for `family.key == "tonal_response"`, so noise, gain modulation highlight and
gain modulation other -- all of which are in that family and none of which
measure tone reproduction -- were each shown the OECF curve. Four metrics, one
picture, three of them wrong.

**Every chart carries the profile's own limits.** The band comes from the
metric's `pass_condition`, parsed in :mod:`deltachrome_ui.labels`, and it is the
single change that does most for readability: "did it pass" and "by how much"
become one glance instead of a number and a remembered threshold. The band is
neutral-coloured on purpose -- it says where the profile asked the reading to
be, and tinting it as success would read as a per-reading verdict, which these
thresholds, being conditions on the aggregate, never form.

**Colour accuracy is drawn in the chart's own colours.** OpenDICE plots a stem
per patch and leaves the reader to look up which patch is which; a patch's own
reference colour is an identity the eye reads without a lookup. So each bar is
the colour the patch is supposed to be, and the grid of swatches that used to
carry that information is not needed beside it.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from deltachrome_ui.charts import Bar, BarPlot, LinePlot, Series
from deltachrome_ui.detail_view import ramp_patches
from deltachrome_ui.labels import (
    aggregation_label,
    field_code,
    field_order,
    parse_condition,
    unit_label,
)
from deltachrome_ui.swatches import lab_to_display_rgb
from deltachrome_ui.theme import TEXT, TEXT_DIM
from deltachrome_ui.widgets import caveat, heading

#: What a reading with no colour of its own is drawn in.
NEUTRAL = QColor(150, 150, 162)


def _readings(metric: dict) -> list[dict]:
    return [p for p in metric.get("per_patch", []) if p.get("value") is not None]


def has_readings(metric: dict) -> bool:
    """Whether a metric recorded any per-reading value at all."""
    return bool(_readings(metric))


def _patch_colors(run) -> dict[str, QColor]:
    """Each patch's reference colour, for drawing its reading in.

    The reference rather than the measurement: the bar says *which patch* this
    is, and a patch whose reproduction is badly wrong would otherwise be drawn
    in the wrong colour precisely when the reader most needs to identify it.
    """
    colors: dict[str, QColor] = {}
    for patch in getattr(run, "patches", None) or []:
        lab = getattr(patch, "reference_lab", None) or getattr(
            patch, "measured_lab", None)
        if lab is None:
            continue
        try:
            colors[str(patch.patch_id)] = lab_to_display_rgb(lab)
        except Exception:  # noqa: BLE001 -- a swatch is never worth a crash
            continue
    return colors


def _reference_lightness(run) -> dict[str, float]:
    out: dict[str, float] = {}
    for patch in getattr(run, "patches", None) or []:
        lab = getattr(patch, "reference_lab", None)
        if lab is not None:
            out[str(patch.patch_id)] = float(lab[0])
    return out


#: Aggregations under which the profile's threshold applies to *every* reading,
#: so drawing it across a field of them is honest. Everything else -- a mean, a
#: percentile, a spread -- is a condition on the population, and shading a band
#: behind the individual readings would show a per-reading verdict the profile
#: never formed. That is the same rule that keeps marks from being coloured by
#: pass or fail, applied to the other axis.
PER_READING = {"max", "max_per_patch", "min", "worst", "worst_case", "worst_edge"}


def _bar_plot(metric: dict, selection, *, y_label: str, x_label: str) -> BarPlot:
    plot = BarPlot(y_label=y_label, x_label=x_label, selection=selection)
    bounds = parse_condition(metric.get("condition"))
    if bounds and str(metric.get("aggregation") or "") in PER_READING:
        plot.set_band(bounds.low, bounds.high)
    return plot


def _titled(*widgets: QWidget) -> QWidget:
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    for widget in widgets:
        layout.addWidget(widget)
    return holder


def _nothing(text: str) -> QWidget:
    label = caveat(text)
    label.setWordWrap(True)
    # Where the picture would be, as the page's own empty states are.
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


# -- the charts ---------------------------------------------------------------


def patch_bars(run, metric: dict, selection, *, percentile: float | None = None,
               order_by_lightness: bool = False) -> QWidget:
    """A reading per patch, drawn in the patch's own colour.

    ``percentile`` dims the readings the aggregate did *not* form on, which is
    how the 90th-percentile metric says which patches it is about: nine tenths
    of the field is context and the top tenth is the answer, and a chart that
    drew them alike would be the average-accuracy chart with a different title.
    """
    readings = _readings(metric)
    if not readings:
        return _nothing("This metric records no per-patch readings to draw.")

    colors = _patch_colors(run)
    lightness = _reference_lightness(run)
    if order_by_lightness and lightness:
        readings = sorted(
            readings,
            key=lambda p: -lightness.get(str(p.get("patch_id")), 0.0),
        )

    cut = None
    if percentile is not None and len(readings) > 1:
        ordered = sorted(float(p["value"]) for p in readings)
        index = int(round(percentile / 100.0 * (len(ordered) - 1)))
        cut = ordered[max(0, min(index, len(ordered) - 1))]

    units = unit_label(metric.get("units"))
    plot = _bar_plot(
        metric, selection,
        y_label=units or "reading",
        x_label="patch",
    )
    plot.set_bars([
        Bar(
            label=str(p.get("patch_id", "")),
            value=float(p["value"]),
            color=colors.get(str(p.get("patch_id")), NEUTRAL),
            group=str(p.get("role") or ""),
            flagged=bool(p.get("flags")),
            highlighted=cut is None or float(p["value"]) >= cut,
        )
        for p in readings
    ])
    value = metric.get("value")
    if value is not None and len(readings) > 1:
        # The statistic's name, not the record's spelling of it. The line was
        # labelled ``percentile_90``, which is a key in a JSON file.
        plot.add_hline(float(value), aggregation_label(metric.get("aggregation")))
        # Where the threshold is a condition on the aggregate rather than on
        # each reading, it belongs beside the aggregate's own line and nowhere
        # else -- as a line, not as a band the readings sit inside.
        bounds = parse_condition(metric.get("condition"))
        if bounds and str(metric.get("aggregation") or "") not in PER_READING:
            for level, _ in bounds.lines:
                plot.add_hline(level, "limit")
    return _titled(plot)


def _density_of(L: float) -> float:
    """Optical density from a reference lightness. Ordering, not a measurement."""
    import numpy as np

    from deltachrome.imaging.color import lab_to_xyz, luminance_factor_to_density

    Y = float(np.asarray(lab_to_xyz(np.asarray([[L, 0.0, 0.0]])))[0][1])
    value = float(np.asarray(luminance_factor_to_density(np.asarray([Y])))[0])
    return value if np.isfinite(value) else 3.0


def white_balance_chart(run, metric: dict, selection) -> QWidget:
    """Neutral balance per step, ordered by density.

    Bars rather than a line, and the reason is what the metric is: white balance
    aggregates ``max_per_patch``, so each step stands on its own against the
    threshold and there is no reading *between* two neutral steps for a line to
    interpolate. A line drawn through them says the steps are a curve, and they
    are ten separate answers to the same question.

    Density stays as the ordering because the fault is described in it -- a cast
    that grows in the shadows is a statement about the dark end of the ladder --
    and it comes from each patch's *reference* lightness, so the axis does not
    move when the capture does.
    """
    readings = _readings(metric)
    lightness = _reference_lightness(run)
    placed = [
        (p, lightness[str(p.get("patch_id"))])
        for p in readings if str(p.get("patch_id")) in lightness
    ]
    if len(placed) < 2:
        return patch_bars(run, metric, selection)
    placed.sort(key=lambda pair: _density_of(pair[1]))

    colors = _patch_colors(run)
    plot = _bar_plot(
        metric, selection,
        y_label=unit_label(metric.get("units")) or "reading",
        x_label="neutral patch",
    )
    plot.set_bars([
        Bar(
            label=str(p.get("patch_id", "")), value=float(p["value"]),
            color=colors.get(str(p.get("patch_id")), NEUTRAL),
            flagged=bool(p.get("flags")),
        )
        for p, _ in placed
    ])
    return _titled(plot)


def tone_response(run, metric: dict, selection) -> QWidget:
    """The OECF, and the judged departure from it.

    Two plots and not three. The signed residual in plain L* used to sit under
    the curve carrying no limit line, deliberately: the residual is a lightness
    difference and the threshold is a ΔL*2000, and drawing one against the other
    would put a limit on a quantity it was not written for. The answer to
    "where are the limits" is therefore to plot the quantity that *is* judged --
    the engine's own per-step values -- which carries the band honestly and
    leaves the curve above it to say which direction the ramp went.
    """
    ramp = ramp_patches(run)
    if len(ramp) < 2:
        return _nothing(
            "This target carries no neutral ramp, so there is no tone response "
            "curve to plot. That is a property of the chart, not a fault."
        )
    ramp = sorted(ramp, key=lambda p: float(p.reference_lab[0]))
    reference = [float(p.reference_lab[0]) for p in ramp]
    measured = [float(p.measured_lab[0]) for p in ramp]
    labels = [p.patch_id for p in ramp]

    curve = LinePlot(
        x_label="reference L*",
        y_label="measured L*",
        y_range=(0.0, 100.0), x_range=(0.0, 100.0), selection=selection,
    )
    curve.set_series([
        Series(x=[0.0, 100.0], y=[0.0, 100.0], color=QColor(TEXT_DIM),
               name="aim", dashed=True, width=1.0),
        # The brightest neutral, not the accent. A plot carrying one series
        # needs no hue: the reading is the subject, and the accent it used to
        # wear is within a unit of green of the aim band drawn behind it.
        Series(x=reference, y=measured, color=QColor(TEXT), name="measured",
               markers=True, width=1.6, point_labels=labels),
    ])

    # Two headings, each naming the quantity its own plot draws and nothing
    # else. They used to name the comparison being made -- "Measured Against
    # Aim", "Departure Per Step" -- which is what a plot with two labelled axes
    # already says, and left the reader to work out that the second one is in a
    # different unit from the first.
    widgets: list[QWidget] = [heading("Tone Response"), curve]
    if _readings(metric):
        widgets.append(heading(unit_label(metric.get("units")) or "Departure"))
        widgets.append(_step_bars(run, metric, selection, ramp))
    return _titled(*widgets)


def _step_bars(run, metric: dict, selection, ramp) -> BarPlot:
    order = {str(p.patch_id): float(p.reference_lab[0]) for p in ramp}
    readings = sorted(
        _readings(metric),
        key=lambda p: -order.get(str(p.get("patch_id")), 0.0),
    )
    colors = _patch_colors(run)
    plot = _bar_plot(
        metric, selection,
        y_label=unit_label(metric.get("units")) or "reading",
        x_label="neutral patch",
    )
    plot.set_bars([
        Bar(
            label=str(p.get("patch_id", "")), value=float(p["value"]),
            color=colors.get(str(p.get("patch_id")), NEUTRAL),
            flagged=bool(p.get("flags")),
        )
        for p in readings
    ])
    return plot


def uniformity_bars(run, metric: dict, selection) -> QWidget:
    """Each repeated neutral's departure from the family mean.

    **No threshold band, deliberately.** The metric is a coefficient of
    variation -- the standard deviation of the measured lightnesses over their
    mean, as a percentage -- and it has no lower bound at all: zero is perfectly
    uniform and there is nothing below it. The readings underneath it are
    something else again, signed lightness differences from the family's own
    mean, and putting the aggregate's "< 1 %" across them drew a limit in one
    unit over readings in another and implied a floor that does not exist.

    What is drawn instead is the mean, at zero, which is what these deviations
    are measured from and the only reference they have.
    """
    readings = _readings(metric)
    if not readings:
        return _nothing("This metric records no per-sample readings to draw.")
    colors = _patch_colors(run)
    plot = BarPlot(
        y_label="L* from family mean", x_label="uniformity sample",
        selection=selection,
    )
    plot.add_hline(0.0, "family mean")
    plot.set_bars([
        Bar(label=str(p.get("patch_id", "")), value=float(p["value"]),
            color=colors.get(str(p.get("patch_id")), NEUTRAL),
            group=str(p.get("role") or ""), flagged=bool(p.get("flags")))
        for p in readings
    ])
    # No line under the plot restating the aggregate and its condition. Both are
    # on the metric's own reading line at the top of the page, which is where a
    # reader has just read them.
    return _titled(plot)


def pair_bars(run, metric: dict, selection) -> QWidget:
    """Readings that name a pair of patches rather than one.

    Gain modulation is the slope between two neighbouring steps, so its
    readings are ``E5-F5`` and have no patch, no position and no colour of
    their own. They are still a list of numbers against a band, which is what
    this draws -- and the *highlight* metric has exactly one of them, which is
    the whole measurement rather than a truncated one.
    """
    readings = _readings(metric)
    if not readings:
        return _nothing("This metric records no per-reading values to draw.")
    plot = _bar_plot(
        metric, selection,
        y_label=unit_label(metric.get("units")) or "reading",
        x_label="neutral pair",
    )
    plot.set_bars([
        # As in the tone response curve: these readings have no colour of
        # their own -- a gain pair is ``E5-F5`` and names no patch -- so they
        # take the brightest neutral rather than the accent the band wore.
        Bar(label=str(p.get("patch_id", "")), value=float(p["value"]),
            color=QColor(TEXT), flagged=bool(p.get("flags")))
        for p in readings
    ])
    return _titled(plot)


#: OpenDICE's layout, and the reason for it: the three pairs a three-record
#: system can be out of register in, each measured on both axes. Which pair is
#: adrift says *which* record moved, and one worst-case magnitude cannot.
_PAIRS = ("G-B", "G-R", "B-R")
#: The rows. ``x`` is a displacement measured across a vertical edge, which is a
#: horizontal displacement -- the label is what moved, not what it was measured
#: on.
_AXES = (("x", "Horizontal"), ("y", "Vertical"))

class _SideLabel(QWidget):
    """A row label turned on its side.

    Six plots across a pane leave no width for a word beside them, and a
    horizontal label clipped to "rizon" is worse than none. Rotated it costs
    fourteen pixels, which is the reason OpenDICE writes its row labels the same
    way.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text
        self.setFixedWidth(16)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name  # noqa: ARG002
        from PySide6.QtGui import QPainter, QPen

        painter = QPainter(self)
        painter.setPen(QPen(QColor(TEXT_DIM)))
        font = painter.font()
        font.setPointSizeF(9.5)
        font.setBold(True)
        painter.setFont(font)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(-90)
        painter.drawText(
            QRectF(-self.height() / 2, -9, self.height(), 18),
            int(Qt.AlignmentFlag.AlignCenter), self.text,
        )
        painter.end()


def registration_grid(run, metric: dict, selection) -> QWidget:
    """Six small plots: three channel pairs, two axes, five field positions.

    The shape OpenDICE uses, and the record already had everything for it --
    ``registration_summary["positions"]`` carries a signed translation per pair
    per axis per position, which is exactly the grid. What the report was
    drawing instead was the metric's per-reading list, which is one number per
    position per axis for the *worst* pair only: it can say the corner is out
    without saying which record went there.

    The values are **signed**, so the band is symmetric about zero even though
    the profile's condition is on the magnitude. A displacement of −0.3 px and
    one of +0.3 px are equally far out and opposite in meaning, and folding them
    together loses the one thing the sign is for -- arrows pointing the same way
    everywhere are a bodily offset between the records, and only the sign says
    so.
    """
    positions = (getattr(run, "registration_summary", None) or {}).get("positions")
    if not positions:
        return pair_bars(run, metric, selection)

    bounds = parse_condition(metric.get("condition"))
    limit = bounds.high if bounds.high is not None else None

    places: list[str] = []
    for entry in positions:
        region = str(entry.get("region") or "")
        if region and region not in places:
            places.append(region)
    places.sort(key=field_order)

    holder = QWidget()
    grid = QGridLayout(holder)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(14)
    grid.setVerticalSpacing(10)

    for column, pair in enumerate(_PAIRS):
        title = QLabel(pair)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 12px; font-weight: 600;")
        grid.addWidget(title, 0, column + 1)

    for row, (axis, axis_label) in enumerate(_AXES):
        grid.addWidget(_SideLabel(axis_label), row + 1, 0)
        for column, pair in enumerate(_PAIRS):
            plot = BarPlot(
                y_label="px" if column == 0 else "",
                x_label="field position" if row == len(_AXES) - 1 else "",
                selection=selection,
            )
            plot.setMinimumHeight(190)
            if limit is not None:
                plot.set_band(-limit, limit)
            plot.add_hline(0.0, "")
            bars = []
            for place in places:
                entry = next(
                    (e for e in positions
                     if e.get("region") == place and e.get("axis") == axis),
                    None,
                )
                if entry is None:
                    continue
                value = (entry.get("translation_px") or {}).get(pair)
                if value is None:
                    continue
                bars.append(Bar(
                    # Named as the metric names its readings, so pointing at a
                    # bar lights that position on the capture panel too.
                    label=f"{place}:{axis}",
                    value=float(value), color=NEUTRAL,
                ))
            plot.set_bars(bars)
            # One or two letters. Six plots across a pane leave about 200 px
            # for an axis, and a chart with nine field positions -- the UTT has
            # a full three by three -- put nine words on it: "Module Cc" beside
            # "Module Cr" beside "Module Tc", overlapping into a grey smear.
            plot.tick_labels = [field_code(p) for p in places]
            grid.addWidget(plot, row + 1, column + 1)
    for column in range(len(_PAIRS)):
        grid.setColumnStretch(column + 1, 1)
    return holder


#: Metrics drawn by something other than the default patch bars, keyed by the
#: metric id the profiles use. Read from the metric and not from its family:
#: noise and gain modulation live in the tone family and measure nothing about
#: tone reproduction.
_BY_METRIC = {
    "tone_response_oecf": tone_response,
    "white_balance": white_balance_chart,
    "gain_modulation_highlight": pair_bars,
    "gain_modulation_other": pair_bars,
    "lightness_uniformity": uniformity_bars,
    "color_channel_misregistration": registration_grid,
}


def chart_for(family, run, metric: dict, selection) -> QWidget | None:
    """The graphical view for one metric, or ``None`` when there is nothing to draw."""
    metric_id = str(metric.get("metric_id", ""))
    builder = _BY_METRIC.get(metric_id)
    if builder is not None:
        return builder(run, metric, selection)

    if family.key in ("sfr", "color_registration") and getattr(run, "edges", None):
        from deltachrome_ui.detail_view import EdgeCurves

        # The metric decides which aim box is drawn on the shared curve.
        return EdgeCurves(run, selection=selection, metric=metric)

    if not _readings(metric):
        return None
    if metric_id == "color_accuracy_90th_percentile":
        return patch_bars(run, metric, selection, percentile=90.0)
    if metric_id == "noise":
        return patch_bars(run, metric, selection, order_by_lightness=True)
    if any(p.get("position_relative") or p.get("roi_px") for p in _readings(metric)):
        return patch_bars(run, metric, selection)
    return pair_bars(run, metric, selection)
