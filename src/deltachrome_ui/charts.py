"""Plots, drawn directly, and the shared idea of which mark is being looked at.

Hand-painted rather than pulled from a charting library, for three reasons: the
plots are simple, they need to follow the application's own palette in a way a
general-purpose library fights, and the field map is not a chart type any library
offers -- it is edge positions in target space with orientation and reading
encoded together.

**What these are for.** The aggregate SFR figure is the least informative thing an
edge field produces. What points at astigmatism, field curvature or a tilted plane
is the *difference between edges* -- which orientation, at which position, reading
what -- so the field map plots each edge where it physically sits on the chart and
the curve plot overlays every edge's MTF so an outlier is visible as an outlier
rather than as a moved average.

**A plot never changes a value.** It changes what a reader can see. The choices
that *do* change a number -- edge aggregation, percentile method -- belong to the
engine and are reported alongside the reading, which is exactly why a chart is
free to hide, re-order and re-scale as much as is useful: none of it can reach the
verdict.

**Why selection is an object and not per-widget state.** Twenty overlaid curves are
unreadable without being able to ask which one is which, and the answer is only
worth anything if it arrives everywhere at once -- the mark on the field map and
the curve in the overlay are two renderings of one edge, and pointing at either
should light both. :class:`ChartSelection` holds that one identity; the widgets
only render it.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
        QLayout,
    QLayoutItem,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from deltachrome_ui.theme import ACCENT, BORDER, TEXT, TEXT_DIM

_AXIS = QColor(BORDER)
_LABEL = QColor(TEXT_DIM)
_GRID = QColor(58, 58, 66)

#: A field map plots whatever population has positions, and its marks are split
#: by *group*: orientation for a field of edges, role for a field of patches.
#: Orientation keeps fixed colors because it is the comparison that matters most
#: and because the two are named throughout the report.
#:
#: Teal and violet, and neither is an accident. The three color records draw in
#: their own hues -- red, green, blue -- and the accent is amber, so an
#: orientation wearing blue put a luminance curve within a few code values of
#: the blue record's curve on the same plot, and an orientation wearing amber
#: put a chip within one unit of green of the accent every other chip on the
#: row uses for "chosen". Both survivors sit well clear of all four.
ORIENTATION_COLORS = {
    "horizontal": QColor(96, 208, 214),
    "vertical": QColor(196, 150, 255),
}
#: Everything else draws from here by a stable hash of the group's name, so a role
#: keeps its color between two runs and between two metrics of one run.
_GROUP_PALETTE = (
    QColor(143, 184, 255), QColor(255, 196, 77), QColor(126, 224, 138),
    QColor(214, 152, 255), QColor(255, 138, 128), QColor(120, 214, 214),
)
_OTHER = QColor(150, 150, 160)

#: Shape carries the group too, so a field map stays readable to a reader who
#: cannot separate the hues.
ORIENTATION_GLYPHS = {"horizontal": "○", "vertical": "□"}
_GLYPHS = ("○", "□", "◇", "△")

#: The color records draw in their own colors, which is the one place in this
#: palette where the hue is the meaning rather than a label. Muted well down from
#: full primaries so three overlaid curves stay legible on the dark ground.
CHANNEL_COLORS = {
    "r": QColor(255, 122, 122),
    "g": QColor(126, 224, 138),
    "b": QColor(130, 168, 255),
}

#: What a series that is not in focus fades to. Far enough down that the focused
#: one reads immediately, not so far that the field it sits in disappears -- the
#: point of highlighting one edge is to see it *against* the others.
DIM = 0.24


def axis_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round tick values spanning a range, with the step they were built on.

    Cutting a range into equal fractions is what an axis label wants to be and
    almost never is: a residual running to -4.7765 gives ticks like 0.251958,
    which are unreadable and imply a precision the axis does not have. Steps are
    chosen from 1, 2, 2.5 and 5 times a power of ten, which is the set that reads
    as a number rather than as an artefact of the data's extremes.
    """
    span = high - low
    if not math.isfinite(span) or span <= 0:
        return [low]
    rough = span / max(1, count)
    magnitude = 10.0 ** math.floor(math.log10(rough))
    step = magnitude * 10.0
    for multiple in (1.0, 2.0, 2.5, 5.0):
        if rough <= multiple * magnitude:
            step = multiple * magnitude
            break
    ticks: list[float] = []
    value = math.ceil(low / step) * step
    while value <= high + step * 1e-6:
        ticks.append(0.0 if abs(value) < step * 1e-9 else value)
        value += step
    return ticks or [low, high]


def tick_label(value: float, step: float) -> str:
    """A tick's text at the precision its own step justifies."""
    decimals = max(0, int(math.ceil(-math.log10(step))) + 1) if step > 0 else 2
    return f"{round(value, decimals):g}"


def group_color(group: str) -> QColor:
    if group in ORIENTATION_COLORS:
        return ORIENTATION_COLORS[group]
    if not group:
        return _OTHER
    # crc32 rather than hash(): Python randomises string hashing per process, and
    # a role that changes color when the application restarts is a role a reader
    # cannot learn.
    return _GROUP_PALETTE[zlib.crc32(group.encode()) % len(_GROUP_PALETTE)]


def group_glyph(group: str) -> str:
    return ORIENTATION_GLYPHS.get(group, "◇")


def group_styles(groups: list[str]) -> dict[str, tuple[QColor, str]]:
    """A color and a shape per group, distinct within the view that asks.

    Assigned by position in this view's own group list rather than by hashing
    each name, because within one map distinctness is the property that matters:
    two groups sharing both a color and a shape is a broken key, and a hash
    cannot promise they will not. Orientation keeps its established colors and
    shapes wherever it appears, since horizontal and vertical are named all
    through the report and should not change appearance between screens.
    """
    styles: dict[str, tuple[QColor, str]] = {}
    spare = [c for c in _GROUP_PALETTE if c not in ORIENTATION_COLORS.values()]
    # More groups than the palette holds is not a rare case: a three-by-three
    # field of slanted-edge modules is nine, and the palette is six, so a key
    # built by taking colours modulo its length gave the lower row the upper
    # row's colours -- which is worse than no key, because it reads as one.
    # Evenly spaced hues at the palette's own saturation and value keep the
    # family and cannot collide.
    wanted = sum(1 for g in groups if g not in ORIENTATION_COLORS)
    if wanted > len(spare):
        spare = [
            QColor.fromHsvF(i / wanted, 0.58, 0.95)
            for i in range(wanted)
        ]
    index = 0
    for group in groups:
        if group in ORIENTATION_COLORS:
            styles[group] = (ORIENTATION_COLORS[group], ORIENTATION_GLYPHS[group])
            continue
        styles[group] = (
            spare[index % len(spare)] if spare else _OTHER,
            _GLYPHS[index % len(_GLYPHS)],
        )
        index += 1
    return styles


def _faded(color: QColor, alpha: float) -> QColor:
    faded = QColor(color)
    faded.setAlphaF(max(0.0, min(1.0, alpha)))
    return faded


#: Below this many readings a size ramp spans the whole range; at or above it,
#: the middle 90%. One contaminated patch reading eight where the rest read one
#: would otherwise compress every other mark into the same size -- the same
#: failure, drawn, that makes a mean useless when one sample is dust.
CLIP_ABOVE = 12


def value_extent(values: list[float]) -> tuple[float, float]:
    """The range a size ramp should span, given the readings it must show.

    Shared with the capture panel, which draws the same ramp on the pixels a
    reading came from rather than on a diagram of the chart.
    """
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return (0.0, 1.0)
    if len(ordered) < CLIP_ABOVE:
        return (ordered[0], ordered[-1])
    last = len(ordered) - 1
    low, high = ordered[int(0.05 * last)], ordered[int(round(0.95 * last))]
    return (low, high) if high > low else (ordered[0], ordered[-1])


def radius_for(
    value: float | None, low: float, high: float,
    smallest: float = 4.0, largest: float = 11.0,
) -> float:
    """Where one reading sits on the size ramp.

    Clamped: the range is set from the middle of the distribution, so a reading
    outside it is drawn at the end of the ramp rather than off it. The outlier
    does not disappear -- it is the largest or smallest mark, it carries its
    flag, and its value is in the table beside it.
    """
    if value is None or high - low < 1e-9:
        return (smallest + largest) / 2.0
    fraction = min(1.0, max(0.0, (value - low) / (high - low)))
    return smallest + (largest - smallest) * fraction


class ChartSelection(QObject):
    """Which mark is being pointed at, which is pinned, and which are hidden.

    Shared by every plot showing the same population, so identity is one fact
    rather than one per widget. Hovering and pinning are deliberately *both*
    focus: pin the edge you are suspicious of, then hover the others to compare
    against it without losing it.

    **Pointing at a mark does not dim the others**, and that is
    :attr:`isolating`. It used to: any focus at all -- including a *hover* --
    took every other reading down to :data:`DIM`, so running the cursor down a
    readings table strobed the chart beside it, and clicking a row left a
    hundred and thirty-nine patches greyed out until something cleared it. The
    reader had asked "which one is GS_R_4", and the chart answered by throwing
    away the population that made the question worth asking.

    Dimming is a real and useful thing to want -- one MTF curve among eighty is
    invisible without it -- but it is a *request to see less*, so it is made
    where seeing less is the point: by clicking a mark on the plot itself.
    Everything else -- a hover, a row in the table, a region on the capture
    panel -- lights its mark and leaves the field alone, which is enough,
    because a lit mark already carries a white cap and its own readout.
    """

    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.hovered: str | None = None
        self.selected: str | None = None
        self.muted: set[str] = set()
        #: Set only by a click on a plot. See the class docstring.
        self.isolating: bool = False
        #: How the SFR pages were left -- which orientations and channels are
        #: off, and the average -- so stepping to another chart of the capture
        #: shows the same view of it. Kept by the report across a chart switch
        #: and cleared when another capture opens. See ``EdgeCurves``.
        self.edge_view: dict = {}

    # -- state -------------------------------------------------------------

    def set_hovered(self, name: str | None) -> None:
        if name != self.hovered:
            self.hovered = name
            self.changed.emit()

    def toggle_selected(self, name: str | None, *, isolate: bool = False) -> None:
        """Pin ``name``, or unpin it if it is already pinned.

        ``isolate`` is the caller saying *this gesture happened on the plot*.
        Only a plot passes it, and unpinning always clears it: a field that
        stays dimmed after the mark it was dimmed for has gone is the state
        nobody can get out of.
        """
        self.selected = None if name == self.selected else name
        self.isolating = bool(isolate) and self.selected is not None
        self.changed.emit()

    def set_muted(self, group: str, muted: bool) -> None:
        if muted:
            self.muted.add(group)
        else:
            self.muted.discard(group)
        self.changed.emit()

    # -- queries -----------------------------------------------------------

    def focus(self) -> set[str]:
        return {name for name in (self.selected, self.hovered) if name}

    @property
    def primary(self) -> str | None:
        """The single mark being asked about, where only one will do.

        Focus is deliberately plural -- pin one, hover another, compare -- but
        anything showing *one* thing has to break the tie. Hovering wins: it is
        the more recent intent, and letting go of the cursor falls back to the
        pin rather than to nothing, which is what makes pinning worth doing.
        """
        return self.hovered or self.selected

    def is_focus(self, name: str) -> bool:
        focus = self.focus()
        return bool(name) and name in focus

    def alpha(self, name: str, group: str | tuple[str, ...] = "") -> float:
        """How strongly to draw this mark. Zero means do not draw it at all.

        A mark may belong to more than one group and is hidden if *any* of them
        is muted. A per-channel MTF curve is both an orientation and a color
        record, and hiding the verticals has to take their red curve with it --
        otherwise the two keys disagree about what is on screen.
        """
        groups = (group,) if isinstance(group, str) else tuple(group)
        if any(g and g in self.muted for g in groups):
            return 0.0
        if not self.isolating:
            return 1.0
        focus = self.focus()
        if not focus or (name and name in focus):
            return 1.0
        return DIM


@dataclass(frozen=True)
class PlotArea:
    """The data-to-pixel mapping for one plot, and its inverse.

    Extracted so that hit-testing and painting cannot disagree about where a
    point is: they ask the same object. Everything interactive in this module
    depends on being able to run the mapping backwards, which is the one thing
    a transform computed inline inside ``paintEvent`` can never do.
    """

    rect: QRectF
    x0: float
    x1: float
    y0: float
    y1: float

    def to_px(self, x: float, y: float) -> QPointF:
        return QPointF(
            self.rect.left() + (x - self.x0) / (self.x1 - self.x0) * self.rect.width(),
            self.rect.bottom() - (y - self.y0) / (self.y1 - self.y0) * self.rect.height(),
        )

    def to_data(self, point: QPointF) -> tuple[float, float]:
        return (
            self.x0 + (point.x() - self.rect.left()) / self.rect.width() * (self.x1 - self.x0),
            self.y0 + (self.rect.bottom() - point.y()) / self.rect.height() * (self.y1 - self.y0),
        )


@dataclass
class Series:
    """One line: x values, y values, a color, and what it is.

    ``name`` is the identity the selection speaks in, and ``group`` is what a
    legend chip mutes -- for an edge field, its orientation. ``point_labels``
    names the individual points where they have identities of their own: a curve
    is one edge throughout, but a tone ramp is a different patch at every step,
    and "which patch is that" is the question being asked of it.
    """

    x: list[float]
    y: list[float]
    color: QColor
    name: str = ""
    #: One group, or several. A curve belonging to several is hidden when any of
    #: them is muted.
    group: str | tuple[str, ...] = ""
    width: float = 1.2
    dashed: bool = False
    markers: bool = False
    point_labels: list[str] = field(default_factory=list)


#: The aim band's fill and its edge. Deliberately neutral rather than green:
#: the band says where the profile asked the reading to be, and colouring it as
#: success would make every mark inside it read as a per-reading pass -- a
#: verdict these thresholds, which are conditions on the aggregate, never form.
#:
#: Neutral in the *value* as well as in the docstring, which it had not been.
#: These were ``QColor(255, 214, 0, ...)`` -- :data:`~deltachrome_ui.theme.ACCENT`
#: to within a unit of green -- while the measured series was drawn in the accent
#: itself, so a reading and the limit it was being judged against were one hue
#: separated only by alpha. Telling those two apart is the entire reason a band
#: is drawn.
#:
#: What separates them now is brightness and weight rather than hue: the reading
#: is the subject and takes the brightest neutral, and the band is an annotation
#: that recedes to a faint wash under a dim dashed rule. Colour goes back to
#: meaning category or status, as it does everywhere else in this application --
#: which is also what lets a band sit behind a hundred and forty patches drawn in
#: their own colours without competing with any of them.
BAND_FILL = QColor(154, 154, 164, 16)
BAND_EDGE = QColor(154, 154, 164, 140)


def paint_band(painter: QPainter, area: "PlotArea", band) -> None:
    """Shade the range a profile asks a reading to stay inside.

    The thing OpenDICE does that makes its plots readable without arithmetic:
    the limits are on the picture, so "did it pass" and "by how much" are the
    same glance. Half a band -- a ceiling with no floor -- shades from the
    ceiling to the edge of the plot, so the *out of aim* side is what is tinted
    and the reader is never asked which side of the line is good.
    """
    if not band:
        return
    low, high = band
    if low is None and high is None:
        return
    top, bottom = max(area.y0, area.y1), min(area.y0, area.y1)
    lower = low if low is not None else bottom
    upper = high if high is not None else top
    lower, upper = max(lower, bottom), min(upper, top)
    if upper <= lower:
        return
    corner_a = area.to_px(area.x0, lower)
    corner_b = area.to_px(area.x1, upper)
    painter.fillRect(QRectF(corner_a, corner_b), BAND_FILL)
    painter.setPen(QPen(BAND_EDGE, 1, Qt.PenStyle.DashLine))
    for level in (low, high):
        if level is None or not bottom <= level <= top:
            continue
        painter.drawLine(area.to_px(area.x0, level), area.to_px(area.x1, level))


#: Height and vertical gap of an ``hline`` name chip. Declared because the
#: stacking arithmetic below needs both, and a chip whose height is written
#: twice is a chip that eventually overlaps its neighbour by a pixel.
_CHIP_HEIGHT = 14.0
_CHIP_GAP = 2.0


def _visible_hlines(area: "PlotArea", hlines) -> list[tuple[float, str]]:
    top, bottom = max(area.y0, area.y1), min(area.y0, area.y1)
    return [(level, label) for level, label in hlines if bottom <= level <= top]


def paint_hlines(painter: QPainter, area: "PlotArea", hlines,
                 pen: QPen) -> None:
    """The reference levels themselves. Drawn *under* the data."""
    painter.setPen(pen)
    for level, _ in _visible_hlines(area, hlines):
        painter.drawLine(area.to_px(area.x0, level), area.to_px(area.x1, level))


def paint_hline_labels(painter: QPainter, area: "PlotArea", hlines) -> None:
    """Their names. Drawn *over* the data, and never over each other.

    Two things were wrong and they are separate. A name drawn before the bars
    is a name behind them -- on a colour-accuracy chart the mean and the limit
    sit low in a field of a hundred and forty readings, so both chips were
    buried under whichever patch happened to be under them. And a mean and a
    limit are *near each other by construction* on a capture that nearly
    passes, which is the interesting case: at 1.95 and 2.05 their two chips
    occupied the same fourteen pixels and printed one word on top of the other.

    So the lines stay under the data, where a reference line belongs, and the
    names come back on top afterwards -- stacked downward from the topmost when
    they would collide, which keeps each one beside its own line and none of
    them on another's.
    """
    levels = sorted(_visible_hlines(area, hlines), key=lambda pair: -pair[0])
    metrics = painter.fontMetrics()
    floor: float | None = None
    for level, label in levels:
        if not label:
            continue
        width = metrics.horizontalAdvance(label) + 10
        point = area.to_px(area.x1, level)
        top = point.y() - _CHIP_HEIGHT - 1
        if floor is not None and top < floor:
            top = floor
        floor = top + _CHIP_HEIGHT + _CHIP_GAP
        box = QRectF(point.x() - width, top, width, _CHIP_HEIGHT)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(24, 24, 28, 236))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(QPen(_LABEL, 1))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, label)
        painter.setBrush(Qt.BrushStyle.NoBrush)


@dataclass
class Bar:
    """One reading in a bar chart: what it is, what it read, what colour it is.

    ``color`` is the patch's own reference colour where it has one. That is the
    change this chart is for: OpenDICE plots a stem per patch and leaves the
    reader to look up which patch is which, and the patch's own colour is an
    identity the eye reads without a lookup -- so the chart of the chart's
    errors is drawn in the chart's colours.
    """

    label: str
    value: float
    color: QColor
    group: str = ""
    flagged: bool = False
    #: Draws at full weight while the rest are dimmed. What the 90th-percentile
    #: metric uses to say which readings the aggregate was formed on.
    highlighted: bool = True


class BarPlot(QWidget):
    """A reading per patch, as bars, joined to the report's one selection.

    Pointing at a bar lights that patch on the capture panel and in the values
    table, because they are three renderings of one reading. That is why this
    is a widget in this module rather than a picture: a chart that cannot be
    pointed at is a chart the rest of the report cannot agree with.
    """

    MARGINS = (52, 12, 14, 34)
    #: Below this many bars they are drawn as bars; above it, as stems. A
    #: hundred and forty bars at two pixels each is a solid block.
    STEM_ABOVE = 60

    def __init__(
        self,
        *,
        y_label: str = "",
        x_label: str = "",
        selection: ChartSelection | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.bars: list[Bar] = []
        #: Optional text under the bars. Used where the x positions are places
        #: rather than an index -- five field positions read as UL/UR/C/LL/LR
        #: and as nothing at all when they are 0 to 4.
        self.tick_labels: list[str] = []
        self.y_label = y_label
        self.x_label = x_label
        self.band: tuple[float | None, float | None] | None = None
        self.hlines: list[tuple[float, str]] = []
        self.selection = selection or ChartSelection(self)
        self.selection.changed.connect(self.update)
        self.setMouseTracking(True)
        self.setMinimumHeight(260)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding
        )

    def set_bars(self, bars: list[Bar]) -> None:
        self.bars = list(bars)
        self.update()

    def set_band(self, low: float | None, high: float | None) -> None:
        self.band = (low, high)
        self.update()

    def add_hline(self, y: float, label: str) -> None:
        self.hlines.append((y, label))

    # -- geometry ----------------------------------------------------------

    def _bounds(self) -> tuple[float, float, float, float]:
        values = [b.value for b in self.bars] or [0.0, 1.0]
        edges = list(values) + [0.0]
        if self.band:
            edges += [v for v in self.band if v is not None]
        edges += [level for level, _ in self.hlines]
        interval = bool(self.band and self.band[0] is not None
                        and self.band[1] is not None)
        if interval:
            # A ratio judged between two limits -- gain modulation, sampling
            # efficiency -- has no meaningful zero, and forcing one squeezes the
            # whole aim band into a sliver at the top of the plot. The range is
            # the band and the readings, and the bar says where in the band it
            # sits.
            edges = [v for v in self.band if v is not None] + values
            y0, y1 = min(edges), max(edges)
            pad = (y1 - y0) * 0.35 or 0.5
            return -0.5, max(0.5, len(self.bars) - 0.5), y0 - pad, y1 + pad
        y0, y1 = min(edges), max(edges)
        pad = (y1 - y0) * 0.08 or 0.5
        # Zero stays the floor when nothing is negative: a bar chart whose
        # baseline floats makes small readings look large.
        y0 = y0 - pad if y0 < 0 else 0.0
        return -0.5, max(0.5, len(self.bars) - 0.5), y0, y1 + pad

    def _area(self) -> PlotArea:
        left, right, top, bottom = self.MARGINS
        rect = QRectF(
            left, top,
            max(1.0, self.width() - left - right),
            max(1.0, self.height() - top - bottom),
        )
        return PlotArea(rect, *self._bounds())

    def _at(self, position: QPointF) -> Bar | None:
        if not self.bars:
            return None
        area = self._area()
        if not area.rect.adjusted(-4, -4, 4, 4).contains(position):
            return None
        index = int(round(area.to_data(position)[0]))
        if 0 <= index < len(self.bars):
            return self.bars[index]
        return None

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt's name
        bar = self._at(event.position())
        self.selection.set_hovered(bar.label if bar else None)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt's name  # noqa: ARG002
        self.selection.set_hovered(None)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt's name
        bar = self._at(event.position())
        self.selection.toggle_selected(bar.label if bar else None, isolate=True)

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont()
        font.setPointSizeF(9.0)
        painter.setFont(font)

        area = self._area()
        plot, x0, x1, y0, y1 = area.rect, area.x0, area.x1, area.y0, area.y1

        y_ticks = axis_ticks(min(y0, y1), max(y0, y1))
        y_step = (y_ticks[1] - y_ticks[0]) if len(y_ticks) > 1 else abs(y1 - y0)
        painter.setPen(QPen(_GRID, 1, Qt.PenStyle.DotLine))
        for y in y_ticks:
            painter.drawLine(area.to_px(x0, y), area.to_px(x1, y))
        painter.setPen(QPen(_LABEL, 1))
        for y in y_ticks:
            point = area.to_px(x0, y)
            painter.drawText(
                QRectF(0, point.y() - 8, plot.left() - 6, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                tick_label(y, y_step),
            )

        paint_band(painter, area, self.band)

        painter.setPen(QPen(_AXIS, 1))
        baseline = max(min(0.0, y1), y0)
        painter.drawLine(area.to_px(x0, baseline), area.to_px(x1, baseline))
        painter.drawLine(area.to_px(x0, y0), area.to_px(x0, y1))

        paint_hlines(painter, area, self.hlines,
                     QPen(QColor(230, 230, 234, 150), 1))

        step = plot.width() / max(1, len(self.bars))
        width = max(1.4, min(18.0, step * 0.62))
        stems = len(self.bars) > self.STEM_ABOVE
        focus = self.selection.focus()
        for index, bar in enumerate(self.bars):
            alpha = self.selection.alpha(bar.label, bar.group)
            if alpha <= 0.0:
                continue
            if not bar.highlighted:
                alpha *= 0.28
            lit = bar.label in focus
            top = area.to_px(index, bar.value)
            foot = area.to_px(index, baseline)
            color = _faded(bar.color, 1.0 if lit else alpha)
            if stems:
                pen = QPen(color, 2.4 if lit else max(1.0, width))
                pen.setCapStyle(Qt.PenCapStyle.FlatCap)
                painter.setPen(pen)
                painter.drawLine(top, foot)
            else:
                # The outline contrasts with the *bar*, not with the ground. A
                # dark neutral patch drawn in its own colour on a dark plot is
                # nearly invisible, and a dark outline round it made it worse.
                edge = (QColor(210, 210, 216, 150) if color.lightnessF() < 0.32
                        else QColor(20, 20, 22, 160))
                painter.setPen(QPen(edge, 0.8))
                painter.setBrush(color)
                painter.drawRect(QRectF(
                    top.x() - width / 2, min(top.y(), foot.y()),
                    width, abs(foot.y() - top.y()),
                ))
                painter.setBrush(Qt.BrushStyle.NoBrush)
            if lit:
                painter.setPen(QPen(QColor(255, 255, 255), 1.2))
                painter.drawLine(
                    QPointF(top.x() - width / 2 - 2, top.y()),
                    QPointF(top.x() + width / 2 + 2, top.y()),
                )
            if bar.flagged:
                # A sampling flag is a fact about how the reading was taken,
                # not a judgement of it -- so it is marked, and marked without
                # spending a colour that means something else.
                painter.setPen(QPen(QColor(255, 196, 77), 1.2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(top.x(), top.y() - 6), 2.2, 2.2)

        if self.tick_labels:
            painter.setPen(QPen(_LABEL, 1))
            for index, text in enumerate(self.tick_labels[:len(self.bars)]):
                point = area.to_px(index, baseline)
                painter.drawText(
                    QRectF(point.x() - 32, plot.bottom() + 3, 64, 13),
                    int(Qt.AlignmentFlag.AlignHCenter), text,
                )

        # After the bars: a limit nobody can read is a limit that is not on the
        # picture. See :func:`paint_hline_labels`.
        paint_hline_labels(painter, area, self.hlines)

        hovered = self.selection.primary
        named = next((b for b in self.bars if b.label == hovered), None)
        if named is not None:
            index = self.bars.index(named)
            self._paint_readout(painter, area, index, named)

        painter.setPen(QPen(_LABEL, 1))
        if self.x_label:
            painter.drawText(
                QRectF(plot.left(), self.height() - 16, plot.width(), 14),
                Qt.AlignmentFlag.AlignHCenter, self.x_label,
            )
        if self.y_label:
            painter.save()
            painter.translate(12, plot.center().y())
            painter.rotate(-90)
            painter.drawText(QRectF(-90, -10, 180, 14),
                             Qt.AlignmentFlag.AlignHCenter, self.y_label)
            painter.restore()

    def _paint_readout(
        self, painter: QPainter, area: PlotArea, index: int, bar: Bar
    ) -> None:
        text = f"{bar.label}   {bar.value:.3f}"
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 14
        point = area.to_px(index, bar.value)
        left = min(max(area.rect.left(), point.x() - width / 2),
                   area.rect.right() - width)
        box = QRectF(left, max(area.rect.top(), point.y() - 26), width, 18)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(24, 24, 28, 232))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(QPen(QColor(235, 235, 240)))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)


class LinePlot(QWidget):
    """Overlaid line series with labeled axes, identifiable by pointing at them."""

    #: How near, in pixels, the cursor must come to a curve to pick it out. Wide
    #: enough to catch a curve without chasing it, narrow enough that two curves
    #: crossing do not both answer.
    HIT_RADIUS = 14.0
    #: left, right, top, bottom
    MARGINS = (52, 12, 12, 34)

    def __init__(
        self,
        *,
        x_label: str = "",
        y_label: str = "",
        y_range: tuple[float, float] | None = (0.0, 1.05),
        x_range: tuple[float, float] | None = None,
        selection: ChartSelection | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.series: list[Series] = []
        self.x_label = x_label
        self.y_label = y_label
        self.y_range = y_range
        self.x_range = x_range
        self.markers: list[tuple[float, str]] = []
        self.hlines: list[tuple[float, str]] = []
        #: The aim band, from the profile's own pass condition. Shaded rather
        #: than drawn as two lines because the question a reader has is "how
        #: much room is left", and a band answers it without arithmetic.
        self.band: tuple[float | None, float | None] | None = None
        #: ``(x0, x1, y0, y1, label)`` -- where a curve is required to pass.
        self.regions: list[tuple[float, float, float, float, str]] = []
        self.selection = selection or ChartSelection(self)
        self.selection.changed.connect(self.update)
        self._hover: tuple[Series, float, float, int] | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(230)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding
        )

    def set_series(self, series: list[Series]) -> None:
        self.series = series
        self._hover = None
        self.update()

    def add_marker(self, x: float, label: str) -> None:
        """A vertical reference line -- Nyquist, or a threshold."""
        self.markers.append((x, label))

    def add_hline(self, y: float, label: str) -> None:
        """A horizontal reference line -- zero on a residual, or an aim."""
        self.hlines.append((y, label))

    def set_band(self, low: float | None, high: float | None) -> None:
        """The range the profile asks the reading to stay inside."""
        self.band = (low, high)
        self.update()

    def add_region(
        self, x0: float, x1: float, y0: float, y1: float, label: str = ""
    ) -> None:
        """A box the curve is required to pass through.

        The shape an MTF requirement actually has. "MTF50 between 45% and 65% of
        half-sampling" is not a ceiling on the curve, it is a window the curve
        must cross the 0.5 line inside -- two coordinates, not one -- and a
        horizontal limit line cannot say that. OpenDICE draws these as dashed
        boxes and it is the right picture.
        """
        self.regions.append((x0, x1, y0, y1, label))
        self.update()

    # -- geometry ----------------------------------------------------------

    def _bounds(self) -> tuple[float, float, float, float]:
        drawable = [s for s in self.series if self.selection.alpha(s.name, s.group) > 0]
        xs = [v for s in drawable for v in s.x] or [0.0, 1.0]
        ys = [v for s in drawable for v in s.y] or [0.0, 1.0]
        x0, x1 = (self.x_range if self.x_range is not None else (min(xs), max(xs)))
        if self.y_range is not None:
            y0, y1 = self.y_range
        else:
            # The band counts towards the range. A limit drawn off the top of
            # the plot is a limit the reader cannot see they are inside of,
            # which is the whole reason for drawing it.
            edges = list(ys)
            if self.band:
                edges += [v for v in self.band if v is not None]
            y0, y1 = min(edges), max(edges)
            pad = (y1 - y0) * 0.08 or 0.5
            y0, y1 = y0 - pad, y1 + pad
        if x1 - x0 < 1e-12:
            x1 = x0 + 1.0
        if y1 - y0 < 1e-12:
            y1 = y0 + 1.0
        return x0, x1, y0, y1

    def _area(self) -> PlotArea:
        left, right, top, bottom = self.MARGINS
        rect = QRectF(
            left, top,
            max(1.0, self.width() - left - right),
            max(1.0, self.height() - top - bottom),
        )
        return PlotArea(rect, *self._bounds())

    # -- pointing ----------------------------------------------------------

    def _nearest(self, position: QPointF) -> tuple[Series, float, float, int] | None:
        area = self._area()
        best: tuple[float, Series, float, float, int] | None = None
        for line in self.series:
            if self.selection.alpha(line.name, line.group) <= 0.0:
                continue
            for index, (x, y) in enumerate(zip(line.x, line.y)):
                point = area.to_px(x, y)
                distance = (point.x() - position.x()) ** 2 + (point.y() - position.y()) ** 2
                if best is None or distance < best[0]:
                    best = (distance, line, x, y, index)
        if best is None or best[0] > self.HIT_RADIUS ** 2:
            return None
        _, line, x, y, index = best
        return line, x, y, index

    def mouseMoveEvent(self, event) -> None:
        hit = self._nearest(event.position())
        self._hover = hit
        self.selection.set_hovered(hit[0].name if hit and hit[0].name else None)
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: ARG002
        self._hover = None
        self.selection.set_hovered(None)
        self.update()

    def mousePressEvent(self, event) -> None:
        hit = self._nearest(event.position())
        # Clicking empty space clears the pin, which is the only way back to
        # seeing the whole field once one curve is pinned.
        self.selection.toggle_selected(
            hit[0].name if hit and hit[0].name else None, isolate=True)

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont()
        font.setPointSizeF(9.0)
        painter.setFont(font)

        area = self._area()
        plot, x0, x1, y0, y1 = area.rect, area.x0, area.x1, area.y0, area.y1

        # Grid and axis labels, on rounded values rather than on fifths of
        # whatever range the data happened to occupy.
        y_ticks = axis_ticks(min(y0, y1), max(y0, y1))
        x_ticks = axis_ticks(x0, x1)
        y_step = (y_ticks[1] - y_ticks[0]) if len(y_ticks) > 1 else abs(y1 - y0)
        x_step = (x_ticks[1] - x_ticks[0]) if len(x_ticks) > 1 else abs(x1 - x0)

        painter.setPen(QPen(_GRID, 1, Qt.PenStyle.DotLine))
        for y in y_ticks:
            painter.drawLine(area.to_px(x0, y), area.to_px(x1, y))
        painter.setPen(QPen(_LABEL, 1))
        for y in y_ticks:
            point = area.to_px(x0, y)
            painter.drawText(
                QRectF(0, point.y() - 8, plot.left() - 6, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                tick_label(y, y_step),
            )
        for x in x_ticks:
            point = area.to_px(x, y0)
            painter.drawText(
                QRectF(point.x() - 30, plot.bottom() + 4, 60, 14),
                Qt.AlignmentFlag.AlignHCenter, tick_label(x, x_step),
            )

        painter.setPen(QPen(_AXIS, 1))
        painter.drawLine(area.to_px(x0, y0), area.to_px(x1, y0))
        painter.drawLine(area.to_px(x0, y0), area.to_px(x0, y1))

        paint_band(painter, area, self.band)

        for rx0, rx1, ry0, ry1, label in self.regions:
            corner_a = area.to_px(max(rx0, x0), max(ry0, min(y0, y1)))
            corner_b = area.to_px(min(rx1, x1), min(ry1, max(y0, y1)))
            box = QRectF(corner_a, corner_b).normalized()
            painter.fillRect(box, BAND_FILL)
            painter.setPen(QPen(BAND_EDGE, 1.2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(box)
            if label:
                painter.setPen(QPen(_LABEL, 1))
                painter.drawText(
                    QRectF(box.left(), box.top() - 15, max(80.0, box.width()), 14),
                    int(Qt.AlignmentFlag.AlignLeft), label,
                )

        for position, label in self.markers:
            if not x0 <= position <= x1:
                continue
            painter.setPen(QPen(QColor(200, 200, 210, 110), 1, Qt.PenStyle.DashLine))
            painter.drawLine(area.to_px(position, y0), area.to_px(position, y1))
            painter.setPen(QPen(_LABEL, 1))
            painter.drawText(area.to_px(position, y1) + QPointF(4, 10), label)

        paint_hlines(painter, area, self.hlines,
                     QPen(QColor(200, 200, 210, 110), 1, Qt.PenStyle.DashLine))

        # Focused series last, so the one being asked about is never buried
        # under the twenty it is being compared with.
        drawable = [
            (line, self.selection.alpha(line.name, line.group))
            for line in self.series
        ]
        for line, alpha in sorted(drawable, key=lambda pair: pair[1]):
            if alpha <= 0.0 or len(line.x) < 2:
                continue
            focused = self.selection.is_focus(line.name)
            pen = QPen(_faded(line.color, alpha), line.width * (2.0 if focused else 1.0))
            if line.dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            points = [area.to_px(x, y) for x, y in zip(line.x, line.y)]
            painter.drawPolyline(QPolygonF(points))
            if line.markers:
                painter.setBrush(_faded(line.color, alpha))
                for point in points:
                    painter.drawEllipse(point, 2.6, 2.6)
                painter.setBrush(Qt.BrushStyle.NoBrush)

        paint_hline_labels(painter, area, self.hlines)

        if self._hover is not None:
            self._paint_readout(painter, area, *self._hover)

        painter.setPen(QPen(_LABEL, 1))
        if self.x_label:
            painter.drawText(
                QRectF(plot.left(), self.height() - 16, plot.width(), 14),
                Qt.AlignmentFlag.AlignHCenter, self.x_label,
            )
        if self.y_label:
            painter.save()
            painter.translate(12, plot.center().y())
            painter.rotate(-90)
            painter.drawText(QRectF(-90, -10, 180, 14),
                             Qt.AlignmentFlag.AlignHCenter, self.y_label)
            painter.restore()

    def _paint_readout(
        self, painter: QPainter, area: PlotArea,
        line: Series, x: float, y: float, index: int,
    ) -> None:
        point = area.to_px(x, y)
        painter.setPen(QPen(line.color, 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(point, 4.5, 4.5)
        if 0 <= index < len(line.point_labels):
            identity = line.point_labels[index]
        else:
            identity = line.name
        text = (f"{identity}   " if identity else "") + f"({x:g}, {y:.3f})"
        painter.setPen(QPen(line.color, 1))
        painter.drawText(
            QRectF(area.rect.left(), area.rect.top() - 1, area.rect.width(), 14),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, text,
        )


@dataclass
class FieldPoint:
    """One measurement, where it sits on the chart and what it read.

    An edge, a patch, a flat-field spot or a printed rule -- whatever carries a
    position. ``group`` is what splits the marks and what a legend chip mutes:
    orientation for edges, role for patches.
    """

    x: float  # 0..1 across the target
    y: float  # 0..1 down the target
    value: float | None
    group: str
    label: str = ""
    #: A sampling fact already recorded against this measurement -- clipping, ROI
    #: contamination. Drawn as a dashed outline rather than a color, because the
    #: colors are spoken for and a dashed mark survives being color-blind.
    flagged: bool = False
    #: A direction, where the reading has one, in whatever unit the caller uses
    #: consistently. Drawn as an arrow scaled against the largest in the field,
    #: because the quantities that carry direction here are sub-pixel on a
    #: master tens of thousands of pixels wide: at true scale every arrow would
    #: be shorter than the mark it starts from. What the longest arrow is worth
    #: is stated instead of a multiplier.
    vector: tuple[float, float] | None = None


class FieldMap(QWidget):
    """Edges plotted where they physically sit on the chart.

    A pooled figure cannot show that the left side is soft or that horizontals
    differ from verticals. Position is the whole point, so the aspect ratio is
    the chart's and the readings are written next to the marks.
    """

    HIT_SLACK = 4.0
    #: Above this many visible marks the per-mark captions are dropped.
    CAPTION_LIMIT = 24

    def __init__(
        self,
        *,
        selection: ChartSelection | None = None,
        show_key: bool = True,
        aspect: float | None = None,
        styles: dict[str, tuple[QColor, str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.points: list[FieldPoint] = []
        self.units = ""
        self.show_key = show_key
        self.aspect = aspect
        self.styles = styles or {}
        self.selection = selection or ChartSelection(self)
        self.selection.changed.connect(self.update)
        self.setMouseTracking(True)
        self.setMinimumHeight(300)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding
        )

    def set_points(self, points: list[FieldPoint], units: str = "") -> None:
        self.points = points
        self.units = units
        self.update()

    # -- geometry ----------------------------------------------------------

    def _area(self) -> PlotArea:
        margin = 26
        rect = QRectF(
            margin, margin,
            max(1.0, self.width() - 2 * margin),
            max(1.0, self.height() - 2 * margin),
        )
        # Drawn at the chart's own proportions where they are known. A field
        # stretched to fill the widget is a field where distance from center --
        # the quantity that separates field curvature from a tilted plane -- is
        # not the distance the reader sees.
        if self.aspect:
            if rect.width() / rect.height() > self.aspect:
                width = rect.height() * self.aspect
                rect = QRectF(
                    rect.center().x() - width / 2, rect.top(), width, rect.height()
                )
            else:
                height = rect.width() / self.aspect
                rect = QRectF(
                    rect.left(), rect.center().y() - height / 2, rect.width(), height
                )
        # y runs downward: relative 0 is the top of the chart, which is the top
        # of the rectangle, so the range is given inverted rather than flipped
        # by hand at every call site.
        return PlotArea(rect, 0.0, 1.0, 1.0, 0.0)

    def _scale(self, area: PlotArea) -> tuple[float, float]:
        """Smallest and largest mark radius the plot has room for.

        Nine edges and a hundred and forty patches are both fields, and a mark
        sized for one is unreadable as the other: at 140 the marks merge, at 9
        they vanish. So the range follows the room each mark actually has.
        """
        shown = sum(
            1 for p in self.points if self.selection.alpha(p.label, p.group) > 0
        )
        if shown < 1:
            return 4.0, 11.0
        spacing = math.sqrt(area.rect.width() * area.rect.height() / shown)
        largest = max(3.0, min(11.0, spacing * 0.42))
        return max(1.5, largest * 0.36), largest

    def _style(self, group: str) -> tuple[QColor, str]:
        return self.styles.get(group) or (group_color(group), group_glyph(group))

    def _radius(
        self, point: FieldPoint, low: float, high: float,
        scale: tuple[float, float] = (4.0, 11.0),
    ) -> float:
        # Size carries the reading, so a soft corner is visible as a small mark
        # before anyone reads a number. Clamped, because the size range is set
        # from the middle of the distribution and readings outside it are drawn
        # at the end of the ramp rather than off it.
        smallest, largest = scale
        if point.value is None or high - low < 1e-9:
            return (smallest + largest) / 2.0
        fraction = min(1.0, max(0.0, (point.value - low) / (high - low)))
        return smallest + (largest - smallest) * fraction

    #: Below this many readings the size ramp spans the whole range; at or above
    #: it, the middle 90%. See :meth:`_extent`.
    CLIP_ABOVE = 12

    def _extent(self) -> tuple[float, float]:
        """The value range the mark sizes span.

        Taken from the middle 90% once there are enough readings to have a
        middle. One contaminated patch reading eight where the rest read one
        would otherwise compress all hundred and thirty-nine others into
        indistinguishable marks -- which is the same failure, in a picture, that
        makes a mean useless when one sample is dust, and the failure this whole
        detail view exists to expose rather than repeat.

        The outlier does not disappear: it is drawn at the top of the ramp, it
        carries its flag, and its value is in the table beside it.
        """
        values = sorted(
            p.value for p in self.points
            if p.value is not None and self.selection.alpha(p.label, p.group) > 0
        )
        if not values:
            return (0.0, 1.0)
        if len(values) < self.CLIP_ABOVE:
            return (values[0], values[-1])
        last = len(values) - 1
        low, high = values[int(0.05 * last)], values[int(round(0.95 * last))]
        return (low, high) if high > low else (values[0], values[-1])

    @property
    def vector_extent(self) -> float:
        """Length of the longest arrow currently shown, in the caller's unit."""
        return max(
            (
                math.hypot(*p.vector) for p in self.points
                if p.vector and self.selection.alpha(p.label, p.group) > 0
            ),
            default=0.0,
        )

    #: Longest arrow, as a fraction of the shorter side of the map.
    ARROW_SPAN = 0.16

    @property
    def clips_extremes(self) -> bool:
        return sum(
            1 for p in self.points
            if p.value is not None and self.selection.alpha(p.label, p.group) > 0
        ) >= self.CLIP_ABOVE

    # -- pointing ----------------------------------------------------------

    def _nearest(self, position: QPointF) -> FieldPoint | None:
        area = self._area()
        low, high = self._extent()
        scale = self._scale(area)
        best: tuple[float, FieldPoint] | None = None
        for point in self.points:
            if self.selection.alpha(point.label, point.group) <= 0.0:
                continue
            center = area.to_px(point.x, point.y)
            distance = (
                (center.x() - position.x()) ** 2 + (center.y() - position.y()) ** 2
            ) ** 0.5
            if distance <= self._radius(point, low, high, scale) + self.HIT_SLACK:
                if best is None or distance < best[0]:
                    best = (distance, point)
        return None if best is None else best[1]

    def mouseMoveEvent(self, event) -> None:
        hit = self._nearest(event.position())
        self.selection.set_hovered(hit.label if hit and hit.label else None)

    def leaveEvent(self, event) -> None:  # noqa: ARG002
        self.selection.set_hovered(None)

    def mousePressEvent(self, event) -> None:
        hit = self._nearest(event.position())
        self.selection.toggle_selected(
            hit.label if hit and hit.label else None, isolate=True)

    # -- painting ----------------------------------------------------------

    #: Offsets tried for a caption, in order: below the mark, above it, then
    #: further out on each side. Two readings at one place is the normal case,
    #: not the corner case -- a square feature presents two orientations, a
    #: region is measured on two axes -- so a caption that simply overprints its
    #: neighbor turns both into an unreadable smudge.
    CAPTION_OFFSETS = (1, -14, 14, -27, 27, -40)

    def _caption_rect(
        self, center: QPointF, radius: float, point: FieldPoint,
        placed: list[QRectF],
    ) -> QRectF | None:
        """Where this mark's caption can go without landing on another.

        Returns nothing when every offset is taken, which drops the caption
        rather than adding to the pile. The value is still in the metric's table
        and still named on hover.
        """
        # A vertical mark prefers to sit above, which keeps the two orientations
        # of one region apart before any collision test is needed.
        offsets = self.CAPTION_OFFSETS
        if point.group == "vertical":
            offsets = (-14, 1, 14, -27, 27, -40)
        for offset in offsets:
            top = center.y() + (radius + offset if offset > 0 else -radius + offset)
            rect = QRectF(center.x() - 44, top, 88, 13)
            if not any(rect.intersects(taken) for taken in placed):
                return rect
        return None

    def _draw_arrows(self, painter: QPainter, area: PlotArea) -> None:
        """Directions, scaled together so their *relative* lengths stay true.

        Drawn after the marks so an arrow is never hidden under one, and scaled
        against the longest in the field rather than by a fixed factor, so the
        picture answers "do these point the same way, or outward" -- which is the
        question direction exists to answer -- without implying a length anyone
        should measure off the screen.
        """
        extent = self.vector_extent
        if extent <= 0:
            return
        span = min(area.rect.width(), area.rect.height()) * self.ARROW_SPAN
        for point in self.points:
            alpha = self.selection.alpha(point.label, point.group)
            if alpha <= 0.0 or not point.vector:
                continue
            color, _ = self._style(point.group)
            start = area.to_px(point.x, point.y)
            length = span / extent
            end = QPointF(
                start.x() + point.vector[0] * length,
                start.y() + point.vector[1] * length,
            )
            wide = self.selection.is_focus(point.label)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(_faded(color, alpha), 2.2 if wide else 1.4))
            painter.drawLine(start, end)

            # A head, so the arrow reads as pointing rather than as a whisker.
            dx, dy = end.x() - start.x(), end.y() - start.y()
            size = math.hypot(dx, dy)
            if size < 3.0:
                continue
            ux, uy = dx / size, dy / size
            head = min(7.0, size * 0.45)
            painter.setBrush(_faded(color, alpha))
            painter.drawPolygon(QPolygonF([
                end,
                QPointF(end.x() - head * (ux + uy * 0.5),
                        end.y() - head * (uy - ux * 0.5)),
                QPointF(end.x() - head * (ux - uy * 0.5),
                        end.y() - head * (uy + ux * 0.5)),
            ]))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    @staticmethod
    def _draw_mark(
        painter: QPainter, center: QPointF, radius: float, glyph: str
    ) -> None:
        """Shape follows the group, so the split survives being color-blind."""
        if glyph == "□":
            painter.drawRect(QRectF(
                center.x() - radius, center.y() - radius, radius * 2, radius * 2
            ))
        elif glyph == "◇":
            painter.drawPolygon(QPolygonF([
                QPointF(center.x(), center.y() - radius),
                QPointF(center.x() + radius, center.y()),
                QPointF(center.x(), center.y() + radius),
                QPointF(center.x() - radius, center.y()),
            ]))
        elif glyph == "△":
            painter.drawPolygon(QPolygonF([
                QPointF(center.x(), center.y() - radius),
                QPointF(center.x() + radius * 0.92, center.y() + radius * 0.7),
                QPointF(center.x() - radius * 0.92, center.y() + radius * 0.7),
            ]))
        else:
            painter.drawEllipse(center, radius, radius)

    def paintEvent(self, event) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont()
        font.setPointSizeF(8.5)
        painter.setFont(font)

        area = self._area()
        rect = area.rect
        painter.setPen(QPen(_AXIS, 1))
        painter.drawRect(rect)
        painter.setPen(QPen(_GRID, 1, Qt.PenStyle.DotLine))
        painter.drawLine(
            QPointF(rect.center().x(), rect.top()),
            QPointF(rect.center().x(), rect.bottom()),
        )
        painter.drawLine(
            QPointF(rect.left(), rect.center().y()),
            QPointF(rect.right(), rect.center().y()),
        )

        low, high = self._extent()
        scale = self._scale(area)
        shown = sum(
            1 for p in self.points if self.selection.alpha(p.label, p.group) > 0
        )
        # Above this many marks the captions collide into a gray wash and the map
        # stops showing the pattern it exists to show. The numbers are still on
        # the metric's own table, and hovering still names any one of them.
        caption_all = shown <= self.CAPTION_LIMIT
        placed: list[QRectF] = []

        for point in self.points:
            alpha = self.selection.alpha(point.label, point.group)
            if alpha <= 0.0:
                continue
            center = area.to_px(point.x, point.y)
            color, glyph = self._style(point.group)
            focused = self.selection.is_focus(point.label)
            pen = QPen(_faded(color, alpha), 2.4 if focused else 1.4)
            if point.flagged:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(_faded(color, alpha * (0.45 if focused else 0.24)))
            if point.value is None and point.vector is not None:
                # A reading that is only a direction. The mark is a tail for the
                # arrow, not a quantity: sizing it from a value it does not have
                # would put a mark on the size ramp that means nothing.
                radius = 2.6
                painter.drawEllipse(center, radius, radius)
            else:
                radius = self._radius(point, low, high, scale)
                self._draw_mark(painter, center, radius, glyph)
            if point.value is not None and (caption_all or focused):
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(
                    _faded(color if focused else _LABEL, alpha), 1
                ))
                caption = (
                    f"{point.label}  {point.value:.3g}" if focused
                    else f"{point.value:.3g}"
                )
                where = self._caption_rect(center, radius, point, placed)
                if where is not None:
                    placed.append(where)
                    painter.drawText(
                        where, Qt.AlignmentFlag.AlignHCenter, caption
                    )

        self._draw_arrows(painter, area)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(_LABEL, 1))
        key = "○ horizontal   □ vertical" if self.show_key else ""
        footer = "   ·   ".join(part for part in (key, self.units) if part)
        if footer:
            painter.drawText(
                QRectF(0, self.height() - 18, self.width(), 16),
                Qt.AlignmentFlag.AlignHCenter, footer,
            )


#: How square a chip is. The row used to be pills -- a 10 px radius on a control
#: 22 px tall is a capsule -- and nothing else in this application is shaped that
#: way: every button, combo and input is a 4 px rectangle. A shape used in one
#: place reads as a different kind of object rather than as the same kind styled
#: differently.
CHIP_RADIUS = 4


#: How far two chips in one strip overlap, so that the border each draws lands
#: on its neighbour's and the pair reads as one hairline rather than two lines
#: with a gap. The same seam :class:`~deltachrome_ui.widgets.IconButton` gets
#: from ``border-left`` in the bars, reached differently because these chips
#: carry a border all the way round and the bar's squares do not.
CHIP_OVERLAP = -1



def chip_style(
    color: QColor, *, resting: bool = False,
    first: bool = True, last: bool = True,
) -> str:
    """One chip, unchecked in the quiet ink and checked in its own colour.

    Shared by the filter and the legend because they are the same object with
    different selection rules, and their styling drifting apart is what made a
    row of three groups read as three unrelated controls.

    ``resting`` colours the *unchecked* chip too, and exists for the one case
    where a filter has to double as a key. Normally it must not: colour on an
    unchecked chip is this row's way of saying *chosen*, and spending it on
    something else would make a row of nine look like nine selections. But a
    :class:`FilterChips` has exactly one chip checked at a time, so in the
    ordinary style a colour assigned to a key is invisible on the eight chips
    that would need it -- which is no key at all.

    ``first`` and ``last`` say where this chip sits in its strip, and only the
    strip's outer corners are rounded. These rows used to be separate 4 px
    rectangles with 8 px between them, which is the one shape in this
    application that is not the bars': every action there is a butted cell with
    a hairline between it and its neighbour and no radius at all. A row of six
    separate rectangles reads as six objects, and what it is is one control with
    six cells. Both ends default true, so a chip standing on its own -- the
    "all" chip, which is not one of the groups -- keeps the old shape without
    asking for it.
    """
    radius = "; ".join(filter(None, (
        f"border-top-left-radius: {CHIP_RADIUS if first else 0}px",
        f"border-bottom-left-radius: {CHIP_RADIUS if first else 0}px",
        f"border-top-right-radius: {CHIP_RADIUS if last else 0}px",
        f"border-bottom-right-radius: {CHIP_RADIUS if last else 0}px",
    )))
    return (
        "QPushButton {"
        f"  background: transparent;"
        f"  border: 1px solid {f'rgba({color.red()},{color.green()},{color.blue()},0.45)' if resting else BORDER};"
        f"  color: {color.name() if resting else TEXT_DIM};"
        f"  {radius};"
        "   padding: 3px 11px; font-size: 11px; font-weight: 600;"
        "}"
        f"QPushButton:hover {{ color: {TEXT_DIM}; border-color: "
        f"rgba({color.red()},{color.green()},{color.blue()},0.40); }}"
        "QPushButton:checked {"
        f"  color: {color.name()};"
        f"  border-color: rgba({color.red()},{color.green()},{color.blue()},0.55);"
        f"  background: rgba({color.red()},{color.green()},{color.blue()},0.10);"
        "}"
    )


def segment_style(*, first: bool = True, last: bool = True) -> str:
    """A cell of a pick-one strip, chosen in a raised grey rather than a colour.

    The strip used to wear the accent -- a tinted fill, a coloured border and
    coloured text all at once -- and a row of them read as a neon sign. Which
    cell is chosen is said by the raised cell and white text alone; colour on
    these rows is kept for the one job only colour can do, being a key.
    """
    radius = "; ".join((
        f"border-top-left-radius: {CHIP_RADIUS if first else 0}px",
        f"border-bottom-left-radius: {CHIP_RADIUS if first else 0}px",
        f"border-top-right-radius: {CHIP_RADIUS if last else 0}px",
        f"border-bottom-right-radius: {CHIP_RADIUS if last else 0}px",
    ))
    return (
        "QPushButton {"
        f"  background: transparent; border: 1px solid {BORDER};"
        f"  color: {TEXT_DIM}; {radius};"
        "   padding: 3px 11px; font-size: 11px; font-weight: 600;"
        "}"
        f"QPushButton:hover {{ color: {TEXT}; }}"
        f"QPushButton:checked {{ color: {TEXT}; background: {SEGMENT_ON}; }}"
    )


#: A chosen segment: a step above the panel's own raised grey, so it reads as
#: pressed in rather than lit up.
SEGMENT_ON = "#2f2f35"
#: A key that is off: text and an empty square, a step below the dim text.
KEY_OFF = "#6b6b74"


class LegendKey(QPushButton):
    """One entry of a legend that is also a switch: a square and a word.

    No border and no fill. A legend's entries were segmented chips that lit up
    in their own colour when on -- three rows of them, each a strip of glowing
    cells -- and what they are is a key that can be turned off, so they are drawn
    as one: the square filled in the group's colour when it is shown and empty
    when it is not, the word bright or dim to match. The square carries the
    colour because the plot draws that group in it, which is the only reason
    the row has colour at all.
    """

    SQUARE = 9

    def __init__(self, label: str, color: QColor) -> None:
        super().__init__(label)
        self.color = QColor(color)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QPushButton {"
            "  background: transparent; border: none; text-align: left;"
            f"  padding: 3px 0px 3px {self.SQUARE + 6}px;"
            f"  font-size: 11px; font-weight: 600; color: {KEY_OFF};"
            "}"
            f"QPushButton:hover {{ color: {TEXT_DIM}; }}"
            f"QPushButton:checked {{ color: {TEXT}; }}"
        )

    def set_color(self, color: QColor) -> None:
        self.color = QColor(color)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        top = (self.height() - self.SQUARE) / 2
        box = QRectF(0.75, top + 0.75, self.SQUARE - 1.5, self.SQUARE - 1.5)
        if self.isChecked():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.color)
            painter.drawRoundedRect(QRectF(0, top, self.SQUARE, self.SQUARE), 2, 2)
        else:
            off = QColor(TEXT_DIM if self.underMouse() else KEY_OFF)
            painter.setPen(QPen(off, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(box, 2, 2)
        painter.end()


#: The space between two keys of a legend. Wide, because without a border the
#: gap is what says where one entry ends and the next begins.
KEY_SPACING = 18


class FilterChips(QWidget):
    """Pick one of several. A *filter*, which is not what a legend does.

    :class:`GroupLegend` mutes groups -- any combination of them can be off at
    once, which is right for comparing orientations. Choosing which corner of
    the field to look at is a different gesture: exactly one answer at a time,
    with an "all" that means no narrowing. Sharing one control for both would
    make "everything hidden" a reachable state, which is a blank plot the reader
    has to work out how to escape from.
    """

    #: The chosen key, or ``""`` for all.
    chosen = Signal(str)

    def __init__(
        self,
        entries: list[tuple[str, str]],
        *,
        current: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.current = current
        self.chips: dict[str, QPushButton] = {}
        self._order = [key for key, _ in entries]
        # One control with cells in it, not several controls in a row: butted,
        # overlapping by a pixel so each pair of borders reads as one seam, and
        # rounded only at the two ends. Exactly what a single-choice control is.
        layout = FlowLayout(self, CHIP_OVERLAP, vertical_spacing=6)
        for index, (key, label) in enumerate(entries):
            chip = QPushButton(label)
            chip.setCheckable(True)
            chip.setChecked(key == current)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setStyleSheet(segment_style(**self._ends(index)))
            chip.clicked.connect(lambda _checked=False, k=key: self._choose(k))
            self.chips[key] = chip
            layout.addWidget(chip)

    def _ends(self, index: int) -> dict[str, bool]:
        return {"first": index == 0, "last": index == len(self._order) - 1}

    def set_colors(self, colors: dict[str, QColor] | None) -> None:
        """Wear the colours the plot is drawing these keys in, or the accent.

        A filter is not a legend and normally must not pretend to be one. The
        exception is a plot that has *started* colouring by the very keys this
        row lists -- averaging by field position draws one curve per position,
        and then the chips are the only thing on the screen that can say which
        colour is which corner. Called with ``None`` the moment that stops being
        true, so the row never claims a key that is not in force.
        """
        for index, (key, chip) in enumerate(self.chips.items()):
            color = (colors or {}).get(key)
            ends = self._ends(index)
            chip.setStyleSheet(
                chip_style(color, resting=True, **ends) if color
                else segment_style(**ends)
            )

    def _choose(self, key: str) -> None:
        self.current = key
        for name, chip in self.chips.items():
            chip.blockSignals(True)
            chip.setChecked(name == key)
            chip.blockSignals(False)
        self.chosen.emit(key)


class GroupLegend(QWidget):
    """A key whose entries are switches.

    A legend that only names the colors answers a question nobody asked. Muting
    a group is how the comparison gets made: turning the verticals off is how you
    see whether one side of the field is soft *regardless* of orientation, which
    is the difference between a tilted plane and astigmatism.
    """

    def __init__(
        self,
        selection: ChartSelection,
        entries: list[tuple[str, QColor, str]],
        parent: QWidget | None = None,
        *,
        hints: dict[str, str] | None = None,
        all_label: str = "",
    ) -> None:
        super().__init__(parent)
        self.selection = selection
        self.chips: dict[str, QPushButton] = {}
        self._groups = [group for group, _, _ in entries]
        hints = hints or {}
        # Wrapping, so a legend never dictates how wide the pane holding it has
        # to be -- see :class:`FlowLayout`. Butted, overlapping by a pixel so
        # each pair of borders reads as one seam: the groups are cells of one
        # control, not a row of separate rectangles. Wrapped rows keep a real
        # gap, which is why the two spacings are not the same number.
        layout = FlowLayout(self, KEY_SPACING, vertical_spacing=6)
        for group, color, label in entries:
            chip = LegendKey(label, color)
            chip.setChecked(group not in selection.muted)
            chip.setToolTip(hints.get(group) or "")
            chip.toggled.connect(
                lambda shown, name=group: self.selection.set_muted(name, not shown)
            )
            self.chips[group] = chip
            layout.addWidget(chip)

        self.all_chip: QPushButton | None = None
        if all_label:
            # A way back, for a legend long enough to need one. Orientation has
            # two entries and channel four, so turning everything on again is one
            # or two presses and this would be clutter; a field of nine module
            # positions is eight presses, which is a control missing rather than a
            # convenience.
            #
            # **At the end of the row and outlined**, so it cannot be read as the
            # row's heading or as one more location. The keys have no border and
            # the heading is caps beside them; this is the one thing on the row
            # drawn as a button, and it says what pressing it does -- "Clear"
            # while everything is shown, "Show All" otherwise. Checked still
            # means "nothing is muted", which is what the toggle reads.
            self.all_chip = QPushButton(all_label)
            self.all_chip.setCheckable(True)
            self.all_chip.setCursor(Qt.CursorShape.PointingHandCursor)
            self.all_chip.setStyleSheet(
                "QPushButton {"
                f"  background: transparent; border: 1px solid {BORDER};"
                f"  border-radius: {CHIP_RADIUS}px; padding: 2px 9px;"
                f"  font-size: 11px; font-weight: 600; color: {TEXT_DIM};"
                "}"
                f"QPushButton:hover {{ color: {TEXT}; border-color: #4a4a55; }}"
            )
            self.all_chip.clicked.connect(self._toggle_all)
            layout.addWidget(self.all_chip)

        # A key that can disagree with what is drawn is worse than no key: it
        # would name a hidden group as shown. So the chips follow the selection
        # rather than only driving it.
        selection.changed.connect(self._sync)
        self._sync()

    def _toggle_all(self) -> None:
        """Everything on, or everything off, from the state it now shows.

        Both directions, because both are gestures. Turning a field of nine on
        again is what this exists for; turning them all off is how a reader gets
        to *one* of them -- clear the row, then pick, which is two presses where
        muting eight is eight. It leaves an empty plot reachable, which a
        single-choice filter deliberately did not, and that is the trade: an
        empty plot is one press from full and says nothing false, where eight
        presses to answer a question is a control being in the way.
        """
        self._show_all(self.all_chip.isChecked())

    def _show_all(self, shown: bool = True) -> None:
        for group in self._groups:
            self.selection.set_muted(group, not shown)

    def set_colors(self, colors: dict[str, QColor] | None) -> None:
        """Wear the colours the plot is drawing these groups in, or the accent.

        A legend normally *is* its colours, and for orientation and channel they
        never change. Field position is the exception: the curves are coloured by
        orientation until the plot is asked to average per location, at which
        point one curve per location is drawn and this row becomes the only thing
        on the screen that can say which colour is which corner. Called with
        ``None`` the moment that stops being true, so the row never claims a key
        that is not in force.
        """
        for group, chip in self.chips.items():
            color = (colors or {}).get(group)
            chip.set_color(color if color else QColor(TEXT))

    def _sync(self) -> None:
        if self.all_chip is not None:
            everything = not any(g in self.selection.muted for g in self._groups)
            self.all_chip.blockSignals(True)
            self.all_chip.setChecked(everything)
            self.all_chip.blockSignals(False)
            self.all_chip.setText("Clear" if everything else "Show All")
        for group, chip in self.chips.items():
            shown = group not in self.selection.muted
            if chip.isChecked() != shown:
                chip.blockSignals(True)
                chip.setChecked(shown)
                chip.blockSignals(False)


class FlowLayout(QLayout):
    """A row of items that wraps to the next line when it runs out of width.

    Qt ships no such layout. The legend needs one: its chips are laid out in a
    row, and a row that cannot wrap reports a minimum width equal to all of them
    side by side. That minimum propagates up through every container holding the
    legend, so three chips were enough to force the whole detail pane wider than
    the space it had -- and the prose beside them was cut off at the edge rather
    than wrapped, because the container it was in had already been made too wide
    to fit.

    Standard implementation: measure by laying out for real against a trial
    width, which is also how the height-for-width answer is produced.
    """

    def __init__(self, parent: QWidget | None = None, spacing: int = 8,
                 *, vertical_spacing: int | None = None) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._spacing = spacing
        # Separate, because a segmented strip overlaps its cells horizontally by
        # a pixel so two borders collapse into one seam -- and a row that wrapped
        # with the same -1 between the rows would overlap those too, which is a
        # legend sitting on top of itself. Defaults to the horizontal gap, so a
        # caller that does not care is unaffected.
        self._vertical_spacing = (
            spacing if vertical_spacing is None else vertical_spacing
        )
        self.setContentsMargins(0, 0, 0, 0)
        if parent is not None:
            # A parent layout only *asks* for height-for-width when the item's
            # size policy says it has one, and a plain QWidget's does not. Without
            # this the host is handed whatever vertical space is going -- a
            # one-row legend was being stretched to 181px, which put a gap the
            # height of a paragraph between it and the plot below.
            policy = parent.sizePolicy()
            policy.setHeightForWidth(True)
            policy.setVerticalPolicy(QSizePolicy.Policy.Minimum)
            parent.setSizePolicy(policy)

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt's name
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802 - Qt's name
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802 - Qt's name
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802 - Qt's name
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt's name
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt's name
        return self._lay_out(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt's name
        super().setGeometry(rect)
        self._lay_out(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt's name
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt's name
        # The widest single item, not the sum: that is the whole point. A legend
        # must be allowed to become one chip per line rather than force its
        # container to fit them all on one.
        width = max((i.minimumSize().width() for i in self._items), default=0)
        height = max((i.minimumSize().height() for i in self._items), default=0)
        return QSize(width, height)

    def _lay_out(self, rect: QRect, *, apply: bool) -> int:
        x, y, line_height = rect.x(), rect.y(), 0
        for item in self._items:
            size = item.sizeHint()
            if x > rect.x() and x + size.width() > rect.right():
                x = rect.x()
                y += line_height + self._vertical_spacing
                line_height = 0
            if apply:
                item.setGeometry(QRect(x, y, size.width(), size.height()))
            x += size.width() + self._spacing
            line_height = max(line_height, size.height())
        return y + line_height - rect.y()
