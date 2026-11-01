"""Reading a target's own printed ruler, to measure the sampling rate.

A chart that prints a scale carries its own dimensional reference. Measuring the
tick pitch turns a capture into pixels per millimetre without anyone having to
know, declare or trust the chart's overall size -- which matters because a
target's physical extent is a per-batch, per-cut property that a model-level
definition cannot honestly assert, and because a file's header resolution is
metadata that can say anything at all.

The measurement is a period estimate, and it is done as one rather than as a run
of independent gaps. Ticks are located to sub-pixel precision, assigned integer
division numbers, and a single line is fitted through the lot: the pitch then
comes from the full span rather than from any one gap, so it improves with the
length of the ruler instead of being limited by how sharply one tick can be
located. On the ISA 4x5 that gives a pitch known to about 0.02 px out of 79.

Two things the fit has to survive, both present on real rulers:

* **Ticks of different length.** Every fifth or tenth tick is drawn longer. That
  changes a tick's weight in the projected profile but not its position, so
  indices are assigned from the estimated period rather than from a running count
  -- a missed short tick then leaves a gap in the numbering instead of shifting
  every subsequent division by one.
* **A baseline.** The rule itself is a line running the length of the band. It
  contributes a constant to the projection and is removed by detrending, not by
  trying to crop it out.
* **Either polarity of ink.** Every reflective chart here prints dark ticks on a
  light ground, and a *positive* transmissive chart prints the opposite. That is not
  a refusal when it is got wrong: looking for dark ticks in a band of bright ones
  finds the noise between them and fits a confident period to it. So the polarity is
  decided per band, from which end departs further from its own baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.signal import find_peaks

from deltachrome.imaging.geometry import ChartPlacement
from deltachrome.imaging.loader import LoadedImage, to_luma
from deltachrome.targets.schema import TargetDefinition

MM_PER_INCH = 25.4

#: Divisions a ruler must yield before its pitch is reported. Fewer than this and
#: the fit is not averaging over enough of the scale to beat the noise on any one
#: tick, which is the whole reason for reading a ruler rather than two marks.
MIN_DIVISIONS = 8
#: Fit residual, **as a fraction of the pitch**, above which the located ticks are
#: not a regular comb -- usually the band is off the rule, or has caught text or a
#: neighboring feature.
#:
#: Relative, not absolute, because what is being judged is whether the ticks form a
#: regular comb and a coarse rule tolerates more absolute scatter for the same
#: regularity. An earlier 1.5 px ceiling got this backwards: it accepted the Golden
#: Thread's cm rule at 2.5% of a 24 px pitch while rejecting the 35mm's metric rule
#: at 1.5% of a 157 px one -- stricter on the better reading. Real rules here run
#: 0.6% to 2.5%; the misreads that prompted the check ran 9% and 18%.
MAX_RESIDUAL_FRACTION = 0.04
#: How far apart two rulers on one chart may read before the disagreement is worth
#: reporting. They measure the same capture through different printed scales, so a
#: real difference means one of them was misread.
CROSS_CHECK_TOLERANCE = 0.01
#: How many standard errors two *unit families* must differ by before the difference
#: is called real rather than reading noise.
#:
#: A separate check from :data:`CROSS_CHECK_TOLERANCE`, and a much more sensitive
#: one, because it is a different claim. That tolerance asks whether any single rule
#: is wildly out -- a scale whose declared division does not match what the chart
#: prints reads by a factor, not by a fraction of a percent. This asks whether the
#: chart's *metric* and *imperial* engravings agree with each other, which is a
#: property of the printing and is systematic: every rule in a family shifts
#: together, so it averages down where field variation does not. Judging it against
#: the readings' own standard errors rather than a fixed percentage means the
#: threshold tightens as the rules get easier to read, which is the right direction.
UNIT_AGREEMENT_FLOOR_MULTIPLE = 3.0
#: How many divisions a ruler may put between its long ticks. Rules mark every 2nd,
#: 4th, 5th or 10th; this bounds the search for a fundamental beneath the major-tick
#: period rather than letting it run down into noise.
MAX_TICKS_PER_MAJOR = 10
#: Below this a tick pitch is not resolved well enough to locate centroids in, and a
#: "period" that small is far more likely to be noise than a scale.
MIN_PERIOD_PX = 4.0
#: Percentile taken down the band to build the tick profile. See :func:`measure_ruler`.
#: Applied from whichever end the ticks are: a chart may print them dark on a light
#: ground or light on a dark one, and both are here.
PROJECTION_PERCENTILE = 10.0
#: How much stronger one polarity's excursion must be before it is believed over the
#: other. Well above 1: a band whose two polarities are within a few percent of each
#: other is not a rule with faint ticks, it is a band that missed the rule, and
#: picking the marginally stronger of two noise floors would turn that into a number.
POLARITY_MARGIN = 1.2
#: Fraction of the strongest autocorrelation peak a candidate period must reach to
#: count as present. Low enough to find a fundamental that its own harmonics
#: outshine, high enough to ignore noise between them.
FUNDAMENTAL_THRESHOLD = 0.3

# -- searching for the band, rather than being told where it is ---------------
#
# A declared ``bounds_relative`` is a **search region**, not the band that gets
# read. It was the band, and that made every reading depend on a height somebody
# tuned against the specimens they happened to have: on the Golden Thread Object
# Level Target the window where all three known units read is 0.02 quad-heights
# wide -- two or three pixels -- because the quad is ONE PATCH TALL there, so the
# v axis is thirty-four times more sensitive than the u axis. A third specimen
# missed it and the chart reported no scale at all.
#
# Three printed things bound where a band may sit, and only the last is obvious:
# the rule's own dark border above it, the short ticks ending partway down (below
# that the pitch doubles, because only the long ticks remain), and the numerals,
# which on that chart overlap the lower half of the tick zone rather than sitting
# under it. A band has to thread between them, and where the gap is a few pixels
# no declared number will survive the next specimen.
#
#: Offsets and heights the search tries, as multiples of the declared band's own
#: height -- reaching two band-heights either side.
#:
#: **The step has to be small against the window, and the window here is tiny.**
#: A first cut stepped an eighth of a band height, which on the Object Level
#: Target is 0.02 of the quad, and the window where that chart's rule reads is
#: *also* about 0.02 -- so the sweep stepped straight over the answer and reported
#: the same refusal as before. A sixteenth lands in it.
SEARCH_OFFSETS = tuple(round(-2.0 + 0.125 * i, 4) for i in range(33))
SEARCH_HEIGHTS = (0.4, 0.5, 0.75, 1.0)
#: How far the band's **ends** are pulled in, as a fraction of the declared
#: length, and the reason this is searched rather than fixed.
#:
#: A rule's ends are where its label and the chart's own edge are, and the three
#: known Object Level units disagree completely about how much room that needs:
#: the older one reads best with no inset at all and reads worse as it is pulled
#: in, while a third specimen reads **nothing** until a tenth is taken off each
#: end. Whatever is at those ends differs per unit, so no declared figure is
#: right for the next one.
SEARCH_INSETS = (0.0, 0.05, 0.10, 0.15)
#: Bands agreeing before the search stops looking. The insets are tried in order
#: and the cheapest one that yields this many wins, so a rule that reads without
#: its ends trimmed never pays for the passes that trim them.
ENOUGH_BANDS = 8
#: How far a candidate's pitch may sit from the pitch the target's own declared
#: physical extent predicts, as a ratio either way.
#:
#: **This is the one check the regularity test cannot make.** A millimetre comb
#: read as centimetres is perfectly regular -- it fits a beautiful straight line
#: through its own ticks and reports a rate wrong by a factor of ten. Nothing in
#: the located ticks can say so; only something outside them can. Measured across
#: three units and both of the Object Level Target's rules, an honest reading sits
#: within 0.6% of the predicted pitch, so a quarter either way is two decades of
#: room while still catching a factor of two.
#:
#: Available only where the target declares ``fiducial_extent_mm`` -- five of the
#: twelve that declare rules. The other seven declare between four and twenty
#: rules each, where agreement *between* rules already makes the same check.
PREDICTED_PITCH_TOLERANCE = 0.25
# -- a scale printed as blocks rather than as ticks ---------------------------
#
# The Golden Thread charts print a two-row **checkerboard** above each rule,
# alternating light and dark at the rule's own major interval -- one centimetre on
# the metric half, one inch on the imperial. It is the same measurement as the
# ticks and a far better feature to make it on: the boundaries are long straight
# transitions between large filled areas rather than marks three pixels wide, so
# the band can be averaged down its whole height instead of taking an extreme
# percentile, and grain averages away instead of moving a centroid.
#
# Measured against three units whose rate is known independently, a block fit is
# about **thirty times more regular** than a tick fit on the same chart -- 0.001
# to 0.004 of the pitch against 0.028 -- and lands within 0.09% to 0.41% of the
# rate. That is what makes it worth a second reader.
#
#: Fraction of the strongest gradient a boundary must reach. Blocks are high
#: contrast by construction, so this can be strict: what it excludes is the
#: texture inside a cell and the faint edge of whatever sits beside the pattern.
BLOCK_EDGE_THRESHOLD = 0.30
#: Regularity a BLOCK fit must reach, and it is far stricter than the tick gate.
#:
#: A block boundary is a transition between two large filled areas, located from
#: a gradient peak on a band averaged down its whole height -- so when it is
#: being read properly it fits at 0.1% to 0.5% of the pitch, where a good tick
#: fit runs 0.6% to 2.5%. Holding blocks to the tick gate therefore admits
#: readings an order of magnitude worse than a block fit ought to be: on one of
#: the three units the declared band scraped in at 3.9%, half a percent off the
#: rate, and dragged that chart's pooled reading with it. Rejecting it sends the
#: band to the search instead, which is the right thing to do with a feature
#: this clean that is not reading cleanly.
MAX_BLOCK_RESIDUAL_FRACTION = 0.01
#: Fewest boundaries a block pattern must show. One lower than the tick floor
#: because a cell is a whole major division: nine centimetre cells is the most
#: the Object Level Target's metric half prints, where its millimetre comb offers
#: ninety-odd ticks in the same span.
MIN_BLOCK_EDGES = 6
#: How far the bands of ONE rule may disagree before the reading is worth a note.
#: They are readings of the same ticks at different heights, so they should agree
#: far more closely than two different rules need to.
BAND_SPREAD_TOLERANCE = 0.01


@dataclass
class RulerReading:
    """One printed scale, measured."""

    region_id: str
    unit: str
    #: Which relative axis the scale runs along. Carried because a reading only
    #: describes the magnification *in its own direction*, and comparing the two
    #: directions is what geometric distortion is.
    axis: str
    division_mm: float
    divisions_used: int
    span_px: float
    pitch_px: float
    residual_px: float
    warnings: list[str] = field(default_factory=list)
    #: Which band this reading actually came from. Not the declared region: that
    #: is the area searched, and this is where inside it the ticks were read.
    bounds_used: tuple[float, float, float, float] | None = None
    #: How many bands of the search region produced a reading, and how far apart
    #: they were. The agreement is the evidence -- the same argument the disc
    #: finder rests on, that a mark staying put while the setting changes is what
    #: says the mark is real -- and one band agreeing with itself says nothing.
    bands_read: int = 1
    band_spread: float = 0.0

    @property
    def px_per_mm(self) -> float:
        return self.pitch_px / self.division_mm

    @property
    def ppi(self) -> float:
        return self.px_per_mm * MM_PER_INCH

    @property
    def standard_error_px(self) -> float:
        """Standard error of the fitted pitch, in pixels per division.

        From the residual scatter and the span the fit covers: a longer ruler pins
        the period harder, which is the point of fitting one line through all of it.
        """
        n = max(self.divisions_used, 3)
        return self.residual_px / max(self.pitch_px, 1e-9) * np.sqrt(12.0 / (n * (n * n - 1))) * self.pitch_px


def read_rulers(
    image: LoadedImage, placement: ChartPlacement, target: TargetDefinition
) -> tuple[list[RulerReading], list[str]]:
    """Measure every ruler the target declares with geometry. Returns readings and notes."""
    readings: list[RulerReading] = []
    notes: list[str] = []
    declared: list[tuple[Any, dict, tuple, float]] = []
    for region in target.regions:
        if region.kind != "scale":
            continue
        geometry = region.geometry or {}
        bounds = geometry.get("bounds_relative")
        division_mm = geometry.get("division_mm")
        if bounds is None or division_mm is None:
            continue
        declared.append((region, geometry, tuple(bounds), float(division_mm)))

    # PASS ONE: every rule at the band its own definition declares, which is what
    # this has always done and costs exactly what it always did.
    failed: list[tuple[Any, dict, tuple, float]] = []
    for region, geometry, bounds, division_mm in declared:
        axis = str(geometry.get("axis", "u"))
        try:
            reading = measure_ruler(
                image, placement, bounds_relative=bounds, division_mm=division_mm,
                axis=axis, region_id=region.region_id,
                unit=str(geometry.get("unit", "mm")),
                polarity=str(geometry.get("polarity", "auto")),
                pattern=str(geometry.get("pattern", "ticks")),
                expected_pitch_px=predicted_pitch(
                    placement, target, division_mm, axis
                ),
            )
        except RulerError:
            failed.append((region, geometry, bounds, division_mm))
            continue
        if not _pitch_is_plausible(
            reading.pitch_px, predicted_pitch(placement, target, division_mm, axis)
        ):
            # It read something perfectly regular and a factor away from what the
            # chart's own size says a division spans. A millimetre comb read as
            # centimetres looks exactly like this, and nothing in the located
            # ticks can tell -- so it goes to the search rather than being
            # reported.
            failed.append((region, geometry, bounds, division_mm))
            continue
        reading.bounds_used = bounds
        readings.append(reading)

    # PASS TWO: search for the ones that did not read -- but only where there is
    # something to hold a candidate against.
    #
    # **An ungated search invents readings, and that is measured rather than
    # feared.** Run without a gate on a deliberately degraded placement of the
    # Device Level Target, it rescued six of twenty rules at 201, 243, 280, 291,
    # 351 and 422 ppi on a chart whose real rate is 609: six perfectly regular
    # combs found in whatever the band happened to cross. A refusal is a fact an
    # operator can act on; a plausible wrong number is the failure this whole
    # project is built to avoid.
    #
    # Two gates are available and either will do. The chart's own declared extent
    # predicts the pitch directly. Failing that, the rules that DID read give a
    # rate measured on this very capture, which is stronger -- eleven of the
    # twelve targets that declare rules declare more than one, and the only two
    # that declare a single rule both declare an extent. Where neither exists,
    # nothing is searched and the refusals stand exactly as before.
    consensus = (
        float(np.median([r.px_per_mm for r in readings])) if readings else None
    )
    for region, geometry, bounds, division_mm in failed:
        axis = str(geometry.get("axis", "u"))
        expected = (
            consensus * division_mm if consensus
            else predicted_pitch(placement, target, division_mm, axis)
        )
        try:
            if expected is None:
                raise RulerError(
                    "no reading could be taken at the band this rule declares, and "
                    "nothing on this chart or in its definition says what a "
                    "division should span -- so the band is not searched, because "
                    "an unconstrained search finds a regular comb in almost "
                    "anything"
                )
            reading = search_ruler(
                image, placement, bounds_relative=bounds, division_mm=division_mm,
                axis=axis, region_id=region.region_id,
                unit=str(geometry.get("unit", "mm")),
                polarity=str(geometry.get("polarity", "auto")),
                pattern=str(geometry.get("pattern", "ticks")),
                expected_pitch_px=expected,
            )
        except RulerError as exc:
            notes.append(f"{region.region_id}: {exc}")
            continue
        readings.append(reading)
    return readings, notes


def predicted_pitch(
    placement: ChartPlacement,
    target: TargetDefinition,
    division_mm: float,
    axis: str,
) -> float | None:
    """How many pixels one division should span, from the target's own geometry.

    ``fiducial_extent_mm`` says how large the placed quad is in millimetres, so
    the quad's length in pixels gives a scale, and one division of the declared
    size follows. ``None`` where the target does not declare an extent.

    **This is a sanity gate and never a measurement.** The extent is a
    hand-measured or published figure for one specimen of a model, which is
    precisely why the rule is read at all -- but it is accurate to well under a
    percent, and what it is asked here is only whether a candidate reading is out
    by a *factor*. That is the failure a regular comb cannot reveal.
    """
    extent = getattr(target, "fiducial_extent_mm", None)
    if not extent:
        return None
    try:
        millimetres = float(extent[0] if axis == "u" else extent[1])
    except (IndexError, TypeError, ValueError):
        return None
    if millimetres <= 0:
        return None
    corners = np.asarray(
        placement.to_pixel(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])),
        dtype=np.float64,
    )
    edge = corners[1] - corners[0] if axis == "u" else corners[2] - corners[0]
    span = float(np.linalg.norm(edge))
    if not np.isfinite(span) or span <= 0:
        return None
    return division_mm * span / millimetres


def _candidate_bands(
    bounds: tuple[float, float, float, float],
    inset: float,
    offsets: Sequence[float],
    heights: Sequence[float],
) -> list[tuple[float, float, float, float]]:
    """Bands to try at one end-inset, as multiples of the declared region's height.

    Its own height rather than a fixed fraction of the quad, because that is the
    scale of the thing being looked for. A tenth of the quad is most of the card
    on a chart whose quad is one patch tall and invisible on one whose quad is the
    whole chart, and both are here.
    """
    u0, v0, u1, v1 = bounds
    height = abs(v1 - v0)
    if height <= 0:
        return [bounds]
    trim = (u1 - u0) * inset
    left, right = u0 + trim, u1 - trim
    out: list[tuple[float, float, float, float]] = []
    for scale in heights:
        for offset in offsets:
            top = v0 + offset * height
            out.append((left, top, right, top + height * scale))
    return out


def _pitch_is_plausible(pitch_px: float, expected_px: float | None) -> bool:
    if not expected_px or expected_px <= 0 or pitch_px <= 0:
        return True
    ratio = pitch_px / expected_px
    return 1.0 / (1.0 + PREDICTED_PITCH_TOLERANCE) <= ratio <= 1.0 + PREDICTED_PITCH_TOLERANCE


def search_ruler(
    image: LoadedImage,
    placement: ChartPlacement,
    *,
    bounds_relative: tuple[float, float, float, float],
    division_mm: float,
    axis: str = "u",
    region_id: str = "ruler",
    unit: str = "mm",
    polarity: str = "auto",
    pattern: str = "ticks",
    expected_pitch_px: float | None = None,
) -> RulerReading:
    """Read a rule anywhere in its declared region, and report the agreement.

    **The median of every band that reads, not the best of them.** Ranking the
    bands was the obvious thing and it is biased: where the short ticks end
    partway down the rule, a band lower in the zone reads *fewer* ticks, and
    fewer ticks fit a straight line more easily -- so "lowest residual" prefers
    the band that read least of the scale. Measured on a capture whose rate is
    known independently from four other charts in the same frame, ranking picked
    a 20-tick band over a 26-tick one. The median picks neither and beat both.

    It is also what makes one band reading a factor out harmless: on one of the
    two committed units, one of the twenty-one bands that read does exactly that,
    and the median does not notice.

    Falls back to the declared region itself when nothing in the search reads, so
    the caller gets that band's own refusal rather than a vaguer one about a
    search it never asked for.
    """
    def read(bounds: tuple[float, float, float, float]) -> RulerReading:
        return measure_ruler(
            image, placement, bounds_relative=bounds, division_mm=division_mm,
            axis=axis, region_id=region_id, unit=unit, polarity=polarity,
            pattern=pattern, expected_pitch_px=expected_pitch_px,
        )

    declared = tuple(bounds_relative)
    # The declared band first, and alone. A chart whose region was already right
    # pays nothing for this at all, which is what makes the search affordable:
    # the Device Level Target declares TWENTY rules, and sweeping every one of
    # them costs ten seconds to re-derive twenty answers that were already right.
    # What the declared band does not get any more is the benefit of the doubt --
    # it must also survive the predicted-pitch gate, so a band that locks a
    # millimetre comb while the region says centimetres now searches instead of
    # reporting a rate ten times out.
    try:
        first = read(declared)
    except RulerError:
        first = None
    if first is not None and _pitch_is_plausible(first.pitch_px, expected_pitch_px):
        first.bounds_used = declared
        return first

    readings: list[RulerReading] = []
    for inset in SEARCH_INSETS:
        for bounds in _candidate_bands(
            declared, inset, SEARCH_OFFSETS, SEARCH_HEIGHTS
        ):
            try:
                reading = read(bounds)
            except RulerError:
                continue
            if not _pitch_is_plausible(reading.pitch_px, expected_pitch_px):
                continue
            reading.bounds_used = bounds
            readings.append(reading)
        if len(readings) >= ENOUGH_BANDS:
            break
    if not readings:
        if first is not None:
            # The declared band did read -- it was the pitch gate that rejected
            # it, and no band in the region reads a plausible one either. Saying
            # so beats handing back the rejected reading, which is what the
            # fallback below did until a test caught it: a comb this regular and
            # this far out is a *different scale* being read, most often a finer
            # comb printed inside the one the region declares.
            raise RulerError(
                f"the declared band reads a regular pitch of {first.pitch_px:.1f} px "
                f"where this chart's own size says a division spans "
                f"{expected_pitch_px:.1f} px, and no band in the region reads a "
                "plausible one. A comb that regular and that far out is a different "
                "scale, not a misread of this one"
            )
        # Nothing read anywhere in the region. Read the declared band so that the
        # reason reaching the operator is about the place they declared.
        return read(declared)

    rates = np.array([r.px_per_mm for r in readings], dtype=np.float64)
    median = float(np.median(rates))
    chosen = min(readings, key=lambda r: abs(r.px_per_mm - median))
    chosen.bands_read = len(readings)
    chosen.band_spread = (
        float(rates.max() - rates.min()) / median if median > 0 else 0.0
    )
    if chosen.bands_read > 1 and chosen.band_spread > BAND_SPREAD_TOLERANCE:
        chosen.warnings.append(
            f"The {chosen.bands_read} bands that read this rule disagree by "
            f"{chosen.band_spread:.2%}. They are readings of the same ticks at "
            "different heights, so they should agree far more closely than two "
            "different rules need to -- a spread this wide means some of them are "
            "not reading the scale cleanly."
        )
    return chosen


class RulerError(ValueError):
    """The ruler could not be read."""


def measure_ruler(
    image: LoadedImage,
    placement: ChartPlacement,
    *,
    bounds_relative: tuple[float, float, float, float],
    division_mm: float,
    axis: str = "u",
    region_id: str = "ruler",
    unit: str = "mm",
    polarity: str = "auto",
    pattern: str = "ticks",
    expected_pitch_px: float | None = None,
) -> RulerReading:
    """Measure one ruler's tick pitch, in pixels per division.

    ``pattern`` is ``"ticks"`` or ``"blocks"``. Blocks are the Golden Thread
    charts' checkerboard, which marks the same divisions as the rule beneath it
    with filled cells instead of thin marks -- see :func:`_block_edges`. The two
    differ only in how the divisions are located; everything after that, the comb
    fit and the regularity gate included, is shared.

    ``bounds_relative`` is ``(u0, v0, u1, v1)`` around the tick band, and ``axis``
    says which relative axis the scale runs along. The band is rectified through
    the placement before it is projected, so a chart photographed rotated or with
    mild perspective is measured along the ruler's own direction rather than along
    the pixel grid.

    ``polarity`` is ``"dark"``, ``"light"`` or ``"auto"``. Every reflective chart
    here prints dark ticks on a light ground, which is what this originally
    assumed; a **positive** transmissive chart prints the opposite, and looking for
    dark ticks in a band of bright ones finds the noise between them and locks a
    "pitch" onto it -- 7.7 px on a rule whose divisions are 124. Auto is the
    default because polarity is a property of the specimen and not only of the
    model: one microform chart ships as both a positive and a negative, and the
    same declared band has to read both.
    """
    u0, v0, u1, v1 = bounds_relative
    quad = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=np.float64)
    corners = placement.to_pixel(quad)

    along = np.linalg.norm(corners[1] - corners[0]) if axis == "u" else np.linalg.norm(
        corners[3] - corners[0]
    )
    across = np.linalg.norm(corners[3] - corners[0]) if axis == "u" else np.linalg.norm(
        corners[1] - corners[0]
    )
    length, width = int(round(along)), int(round(across))
    if length < 40 or width < 4:
        raise RulerError(f"the declared band is {length}x{width} px, too small to read")

    # One warped pixel per image pixel along the ruler, so the pitch comes out in
    # the image's own pixels and needs no scale factor afterwards.
    #
    # ``corners`` is (u0,v0), (u1,v0), (u1,v1), (u0,v1). For a ``u`` rule the first
    # edge runs along the scale, for a ``v`` rule the *last* one does -- so the two
    # destinations differ in which axis gets ``length``, not in winding order.
    # Getting that wrong transposes the strip, and the projection below then runs
    # across the rule instead of along it: the ticks average out, the profile goes
    # flat, and every vertical rule reads as "not on the rule". That is what the
    # previous form did, and no shipped target declared a ``v`` rule to catch it.
    destination = (
        np.array([[0, 0], [length, 0], [length, width], [0, width]], dtype=np.float32)
        if axis == "u"
        else np.array([[0, 0], [width, 0], [width, length], [0, length]], dtype=np.float32)
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    x0 = int(max(corners[:, 0].min() - 4, 0))
    x1 = int(min(corners[:, 0].max() + 4, image.width))
    y0 = int(max(corners[:, 1].min() - 4, 0))
    y1 = int(min(corners[:, 1].max() + 4, image.height))
    if x1 <= x0 or y1 <= y0:
        # The band lies off the edge of the capture. A refusal rather than a
        # crash, because it is one: there are no pixels there to read a rule in.
        # Latent until the band became something searched for -- a declared band
        # is inside the chart by construction, and a swept one is not.
        raise RulerError(
            "the band falls outside the capture, so there are no pixels to read"
        )
    window = image.window(y0, y1, x0, x1)
    if getattr(window, "size", 0) == 0:
        raise RulerError(
            "the band falls outside the capture, so there are no pixels to read"
        )
    shifted = matrix @ np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64)
    size = (length, width) if axis == "u" else (width, length)
    strip = cv2.warpPerspective(to_luma(window).astype(np.float32), shifted, size)
    # An extreme percentile down the band, not the mean. Rulers draw every fifth or
    # tenth tick longer, and a mean over the band's height dilutes a short tick in
    # proportion to how short it is -- which makes the major ticks the strongest
    # thing in the profile and invites the period estimate to lock onto them
    # instead. A tick's most extreme pixel is as extreme whether the tick is long or
    # short. Both ends are taken, because which one is the ink is the next question.
    projection_axis = 0 if axis == "u" else 1
    if pattern == "blocks":
        # The MEAN down the band. A block fills its whole height, so every row
        # says the same thing and averaging them is free signal-to-noise -- which
        # is exactly what the tick reader cannot do, a tick filling only a few
        # rows of the band it is found in.
        positions, period, warnings = _block_edges(
            strip.mean(axis=projection_axis).astype(np.float64), expected_pitch_px
        )
        return _fit_positions(
            positions, period, warnings, region_id=region_id, axis=axis, unit=unit,
            division_mm=division_mm, what="block boundaries",
            minimum=MIN_BLOCK_EDGES,
            max_residual_fraction=MAX_BLOCK_RESIDUAL_FRACTION,
        )
    if pattern != "ticks":
        raise RulerError(f"pattern must be 'ticks' or 'blocks'; got {pattern!r}")
    dark = np.percentile(strip, PROJECTION_PERCENTILE, axis=projection_axis)
    light = np.percentile(strip, 100.0 - PROJECTION_PERCENTILE, axis=projection_axis)
    ink, polarity_note = _tick_polarity(
        dark.astype(np.float64), light.astype(np.float64), polarity
    )
    # Light ticks are negated rather than handled separately, so everything after
    # this point sees one shape of signal: a baseline with troughs in it.
    profile = dark.astype(np.float64) if ink == "dark" else -light.astype(np.float64)

    positions, period, warnings = _tick_positions(profile)
    if polarity_note:
        warnings.append(polarity_note)
    return _fit_positions(
        positions, period, warnings, region_id=region_id, axis=axis, unit=unit,
        division_mm=division_mm, what="ticks", minimum=MIN_DIVISIONS,
    )


def _fit_positions(
    positions: np.ndarray,
    period: float,
    warnings: list[str],
    *,
    region_id: str,
    axis: str,
    unit: str,
    division_mm: float,
    what: str = "ticks",
    minimum: int = MIN_BLOCK_EDGES,
    max_residual_fraction: float = MAX_RESIDUAL_FRACTION,
) -> RulerReading:
    """One comb fitted through located divisions, whatever located them.

    Shared by the tick reader and the block reader on purpose. They differ in how
    a division is *found* -- a narrow dark mark against a percentile, or a
    boundary between two filled cells against a gradient -- and in nothing after
    that. The comb numbering, the regularity gate and everything the reading
    carries are one piece of code, so a scale read either way is judged the same.
    """
    if len(positions) < minimum:
        raise RulerError(
            f"only {len(positions)} {what} were located, fewer than the "
            f"{minimum} a pitch should be fitted through"
        )
    positions, index, _ = _fit_to_comb(positions, period)
    if len(positions) < minimum:
        raise RulerError(
            f"only {len(positions)} {what} lie on a regular comb, fewer than the "
            f"{minimum} a pitch should be fitted through"
        )
    pitch, intercept = np.polyfit(index, positions, 1)
    residual = float(np.std(positions - (pitch * index + intercept)))
    if residual > max_residual_fraction * abs(pitch):
        raise RulerError(
            f"located {what} scatter {residual:.2f} px about a uniform pitch of "
            f"{abs(pitch):.1f} px ({100 * residual / abs(pitch):.1f}%, against a "
            f"{100 * max_residual_fraction:.0f}% limit); the band is probably not on "
            "the rule, or has caught text or another feature"
        )
    return RulerReading(
        region_id=region_id,
        axis=axis,
        unit=unit,
        division_mm=division_mm,
        divisions_used=len(positions),
        span_px=float(positions[-1] - positions[0]),
        pitch_px=abs(float(pitch)),
        residual_px=residual,
        warnings=warnings,
    )


def _tick_polarity(
    dark: np.ndarray, light: np.ndarray, declared: str
) -> tuple[str, str | None]:
    """Which way the ticks go, and a note when the answer was not obvious.

    Decided on how far each end of the band departs from its own local baseline,
    after the same detrend the tick search uses. Ticks occupy a minority of a
    band's area whichever polarity they are, so the majority is the ground and the
    minority is the ink -- and the ink is whichever side departs further.
    """
    if declared in ("dark", "light"):
        return declared, None
    if declared != "auto":
        raise RulerError(f"polarity must be 'dark', 'light' or 'auto'; got {declared!r}")

    def excursion(profile: np.ndarray, sign: float) -> float:
        span = max(31, (len(profile) // 4) | 1)
        smooth = cv2.GaussianBlur(
            profile.reshape(-1, 1).astype(np.float32), (1, span), 0
        ).ravel()
        return float(np.percentile(sign * (profile - smooth), 98.0))

    dark_ink = excursion(dark, -1.0)
    light_ink = excursion(light, +1.0)
    if dark_ink >= light_ink:
        ratio = dark_ink / max(light_ink, 1e-9)
        chosen = "dark"
    else:
        ratio = light_ink / max(dark_ink, 1e-9)
        chosen = "light"
    if ratio < POLARITY_MARGIN:
        return chosen, (
            f"Tick polarity was ambiguous ({chosen} ink only {ratio:.2f}x the other "
            "way). Read as " + chosen + "; if this scale is misread, declare "
            "'polarity' on the region rather than moving the band."
        )
    return chosen, None



def _block_edges(
    profile: np.ndarray, expected_period: float | None
) -> tuple[np.ndarray, float, list[str]]:
    """Sub-pixel boundaries of a row of alternating blocks, and their period.

    The profile is the band's **mean** down its height, not an extreme percentile.
    A block fills the band, so every row says the same thing and averaging them is
    free signal-to-noise; a tick fills a few rows of it, which is why the tick
    reader cannot do the same and takes a percentile instead.

    A boundary is a peak in the gradient's magnitude, located to sub-pixel by
    fitting a parabola to the three samples around it. Direction is deliberately
    ignored: the cells alternate, so boundaries run light-to-dark and dark-to-light
    in turn and both are equally boundaries.

    **The two rows are complementary**, which is a hazard rather than a help: a
    band spanning both averages them into a flat line with no boundaries in it at
    all. That is a refusal, and the band search moves off it.
    """
    warnings: list[str] = []
    gradient = np.abs(np.gradient(profile.astype(np.float64)))
    if not np.isfinite(gradient).all() or gradient.max() <= 0:
        raise RulerError("the band has no contrast in it, so it holds no blocks")
    separation = max(4.0, (expected_period or 0.0) * 0.4)
    peaks, _ = find_peaks(
        gradient,
        height=BLOCK_EDGE_THRESHOLD * float(gradient.max()),
        distance=max(1, int(separation)),
    )
    located: list[float] = []
    for peak in peaks:
        if 0 < peak < len(gradient) - 1:
            a, b, c = gradient[peak - 1], gradient[peak], gradient[peak + 1]
            denominator = a - 2.0 * b + c
            offset = (a - c) / (2.0 * denominator) if abs(denominator) > 1e-12 else 0.0
            located.append(float(peak) + float(np.clip(offset, -0.5, 0.5)))
        else:
            # A boundary against the band's own edge, which is the band being cut
            # through a cell rather than a printed boundary.
            warnings.append(
                "A block boundary was found against the edge of the band and left "
                "out: the band is cut through a cell there rather than ending at a "
                "printed boundary."
            )
    positions = np.array(sorted(located), dtype=np.float64)
    if len(positions) < MIN_BLOCK_EDGES:
        raise RulerError(
            f"only {len(positions)} block boundar(ies) were located, fewer than the "
            f"{MIN_BLOCK_EDGES} a pitch should be fitted through"
        )
    gaps = np.diff(positions)
    period = float(np.median(gaps))
    if period < MIN_PERIOD_PX:
        raise RulerError(
            f"the block boundaries repeat every {period:.1f} px, which is too fine "
            "to be a printed pattern and is far more likely to be noise"
        )
    return positions, period, warnings


def _tick_positions(profile: np.ndarray) -> tuple[np.ndarray, float, list[str]]:
    """Sub-pixel tick positions, and the period they repeat at.

    The period is estimated first, by autocorrelation, and then used to space the
    peak search. Doing it the other way round -- find peaks, infer the period from
    the gaps -- needs a minimum separation before it knows what the separation is,
    and picks up the long every-tenth ticks twice on a ruler whose short ticks are
    faint.
    """
    warnings: list[str] = []
    # The detrend window has to be wider than the tick period or it removes the very
    # signal being looked for. The period is not known yet, but the band is required
    # to hold at least MIN_DIVISIONS of them, so a quarter of its length is safely
    # more than two periods.
    span = max(31, (len(profile) // 4) | 1)
    smooth = cv2.GaussianBlur(
        profile.reshape(-1, 1).astype(np.float32), (1, span), 0
    ).ravel()
    comb = profile - smooth                      # removes the rule's own baseline
    comb = comb.max() - comb                     # ticks are dark; make them peaks

    centerd = comb - comb.mean()
    correlation = np.correlate(centerd, centerd, mode="full")[len(centerd) - 1:]
    lo = 4
    hi = max(lo + 2, len(correlation) // 3)
    if hi <= lo:
        raise RulerError("the band is too short to estimate a tick period")
    # The *fundamental*, not the tallest peak. A ruler draws every fifth or tenth
    # tick longer, and those major ticks correlate more strongly than the divisions
    # themselves -- so taking the global maximum reads the scale five or ten times
    # too coarse, and fits a beautifully straight line through every fifth tick
    # while doing it.
    #
    # The fundamental has to *divide* the strongest lag, because a comb's harmonics
    # sit at integer multiples of it. Searching instead for "the smallest peak above
    # a threshold" looks equivalent and is not: on a short or noisy band it finds a
    # peak at some unrelated small lag and reports a pitch that is not a scale at
    # all -- on the DLT's 30 px/mm rule that produced a confident 202 ppi where the
    # answer is nearer 760.
    peaks, _ = find_peaks(correlation[lo:hi])
    if peaks.size == 0:
        raise RulerError("no repeating tick pattern was found in the band")
    peaks = peaks + lo
    strongest_lag = int(peaks[int(np.argmax(correlation[peaks]))])
    strongest = float(correlation[strongest_lag])
    period = float(strongest_lag)
    for k in range(2, MAX_TICKS_PER_MAJOR + 1):
        candidate = strongest_lag / k
        if candidate < MIN_PERIOD_PX:
            break
        window = int(round(candidate * 0.15)) + 1
        near = [int(q) for q in peaks if abs(q - candidate) <= window]
        if near and max(correlation[q] for q in near) > FUNDAMENTAL_THRESHOLD * strongest:
            period = float(near[int(np.argmax([correlation[q] for q in near]))])
    if period < MIN_PERIOD_PX:
        raise RulerError(f"estimated tick period is {period:.1f} px, too fine to resolve")

    peaks, _ = find_peaks(comb, distance=max(3, int(round(period * 0.6))))
    if len(peaks) < 3:
        raise RulerError("no regular tick pattern was found in the band")

    positions = []
    half = max(2, int(round(period * 0.35)))
    for p in peaks:
        # A tick whose centroid window runs off the end of the band is measured
        # through a clipped window and lands several pixels out. The band edge also
        # produces peaks of its own, from a neighboring feature cut in half.
        if p < half or p >= len(comb) - half:
            continue
        a, b = p - half, p + half + 1
        weight = comb[a:b] - comb[a:b].min()
        if weight.sum() > 0:
            positions.append(float((np.arange(a, b) * weight).sum() / weight.sum()))
    located = np.asarray(sorted(positions))

    # Refine the period against the ticks it just found. The autocorrelation lag is
    # an integer, and the caller numbers each tick by its distance from the first
    # over the period -- so half a pixel of error starts mis-numbering once it
    # accumulates to half a division, after period / (2 * error) ticks. On the UTT's
    # millimetre rule that is 0.34 in 15.66 and the numbering slips at the 46th tick
    # of 179: ticks collide on one index, are dropped as duplicates, and a rule
    # straight to 0.12 px is refused for scattering 1.68.
    #
    # Only gaps already near the period vote, which is what makes this a refinement
    # of the chosen comb rather than a second, worse attempt at choosing one. A
    # missed tick leaves a doubled gap and a spurious tick two half ones; both fall
    # outside the window and neither moves the estimate. If the autocorrelation
    # locked onto a harmonic, every gap falls outside, no gap votes, and the period
    # is left exactly as it was. The mean and not the median, because a rule this
    # fine quantises its gaps -- 15.66 px prints as a run of 15s and 16s whose
    # median is 16 and whose mean is the answer.
    if located.size >= 3:
        gaps = np.diff(located)
        near = gaps[np.abs(gaps - period) <= 0.25 * period]
        if near.size >= 2:
            period = float(near.mean())
    return located, period, warnings



def _fit_to_comb(
    positions: np.ndarray, period: float, *, tolerance: float = 0.2, passes: int = 3
) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep the ticks that lie on one regular comb, and number them.

    Numbering by a running count would be wrong the moment a tick is missed or an
    extra one found; numbering by rounded position over the period survives both.
    The refit-and-reject loop is there because one spurious tick is enough to bend
    the line, and a bent line then mis-numbers the ticks at the far end -- on the
    ISA 4x5's imperial scale a single half-tick at the band edge moved the fitted
    pitch by 1 px in 125 and the residual from 0.4 to 11.7.
    """
    kept = np.asarray(positions, dtype=np.float64)
    dropped = 0
    index = np.round((kept - kept[0]) / period)
    for _ in range(passes):
        index = np.round((kept - kept[0]) / period)
        unique, first = np.unique(index, return_index=True)
        if len(unique) != len(index):
            dropped += len(index) - len(unique)
            kept, index = kept[first], unique
        if len(kept) < 3:
            break
        slope, intercept = np.polyfit(index, kept, 1)
        residual = kept - (slope * index + intercept)
        keep = np.abs(residual) <= tolerance * abs(slope)
        if keep.all():
            period = abs(float(slope))
            break
        dropped += int((~keep).sum())
        kept, index = kept[keep], index[keep]
        period = abs(float(slope))
        if len(kept) < 3:
            break
    if kept.size == 0:
        # Rejection can take everything: a band holding a handful of unrelated
        # marks has no comb to keep. The caller turns an empty result into a
        # RulerError, which is the honest answer -- indexing kept[0] here instead
        # raised IndexError out of the middle of the library.
        return kept, np.empty(0, dtype=np.float64), dropped
    index = np.round((kept - kept[0]) / period)
    return kept, index, dropped


def unit_agreement(readings: Sequence[RulerReading]) -> dict[str, Any]:
    """How far the chart's *unit families* disagree, and how far they could by noise.

    Separating this out of the pooled spread matters because the two halves are
    different quantities with different causes. Rules of the same unit at different
    places disagree because the magnification varies across the field -- that is
    geometric distortion, and it is what a QC run is trying to measure. A chart's
    metric and imperial engravings disagreeing is a property of the *chart*: it
    shifts every rule of one family together, so it is not distortion, and pooling
    the two puts a printing error into the lens's figure.

    That is not hypothetical. On both ISA FADGI - ISO 19264 captures the millimetre
    comb reads about 0.75% coarser than the eighth-inch comb printed on the same
    rule, and it is the chart rather than the reader: measured in raw unwarped
    pixels, both combs are located with no missing ticks and gap scatter of a
    quarter of a pixel, and the millimetre comb's own decade markers agree with its
    fine ticks to 0.03%. Two scales, both read cleanly, disagreeing.

    The floor comes from the readings' own comb fits rather than from a fixed
    percentage, because a family mean averages down: the same absolute disagreement
    is noise on two rough rules and a finding on ten good ones.
    """
    families: dict[str, list[RulerReading]] = {}
    for reading in readings:
        families.setdefault(reading.unit, []).append(reading)

    by_unit = {
        unit: {
            "ppi": round(float(np.mean([r.ppi for r in group])), 3),
            "rules": len(group),
            "axes": sorted({getattr(r, "axis", "u") for r in group}),
        }
        for unit, group in families.items()
    }
    if len(families) < 2:
        return {"by_unit": by_unit, "disagreement": 0.0, "floor": 0.0, "real": False}

    means = np.array([entry["ppi"] for entry in by_unit.values()], dtype=np.float64)
    disagreement = float(means.max() - means.min()) / float(means.mean())

    # Standard error of each family's mean, as a fraction, then combined for the
    # difference between the two extremes.
    def family_error(group: list[RulerReading]) -> float:
        errors = np.array([
            r.standard_error_px / r.pitch_px for r in group if r.pitch_px
        ])
        return float(np.sqrt((errors**2).sum())) / max(len(errors), 1) if errors.size else 0.0

    errors = sorted(
        (family_error(group) for group in families.values()), reverse=True
    )[:2]
    floor = float(np.sqrt(sum(e * e for e in errors)))
    return {
        "by_unit": by_unit,
        "disagreement": round(disagreement, 6),
        "floor": round(floor, 6),
        "real": bool(disagreement > UNIT_AGREEMENT_FLOOR_MULTIPLE * floor),
    }


def sampling_rate(readings: list[RulerReading]) -> dict[str, Any]:
    """Reduce the rulers on a chart to one sampling rate, and cross-check them.

    A chart printing both a metric and an imperial scale measures the same capture
    twice through independently engraved rules. Agreement is not a formality: it is
    the only check available that the divisions were read as the chart intends,
    since a ruler misread by a factor of two still produces a beautiful straight
    line through its ticks.

    The pooled ``agreement`` is kept as it was, and :func:`unit_agreement` splits it
    into the part that belongs to the chart and the part that belongs to the field.
    """
    if not readings:
        return {}
    rates = np.array([r.ppi for r in readings], dtype=np.float64)
    mean = float(rates.mean())
    spread = float(rates.max() - rates.min()) / mean if mean else 0.0
    units = unit_agreement(readings)
    out: dict[str, Any] = {
        "ppi": mean,
        "rulers": {
            r.region_id: {
                "unit": r.unit,
                "division_mm": r.division_mm,
                "divisions_used": r.divisions_used,
                "pitch_px": round(r.pitch_px, 4),
                "px_per_mm": round(r.px_per_mm, 4),
                "ppi": round(r.ppi, 2),
                "fit_residual_px": round(r.residual_px, 4),
            }
            for r in readings
        },
        "agreement": round(spread, 5),
        "unit_agreement": units,
    }
    if len(readings) > 1 and spread > CROSS_CHECK_TOLERANCE:
        out["warning"] = (
            f"The chart's rulers disagree by {spread:.2%} "
            f"({', '.join(f'{r.region_id} {r.ppi:.1f} ppi' for r in readings)}). They "
            "measure one capture through different printed scales, so a real "
            "disagreement means one was misread -- most often a scale whose declared "
            "division does not match what the chart actually prints."
        )
    elif units["real"]:
        named = ", ".join(
            f"{unit} {entry['ppi']:.1f} ppi from {entry['rules']} rule(s)"
            for unit, entry in sorted(units["by_unit"].items())
        )
        out["unit_warning"] = (
            f"The chart's scales disagree by {units['disagreement']:.2%} between "
            f"units ({named}), which is more than {UNIT_AGREEMENT_FLOOR_MULTIPLE:.0f}x "
            f"the {units['floor']:.2%} their own comb fits could account for. Every "
            "rule of one unit shifts together, so this is the chart's printing (or a "
            "wrongly declared division) rather than anything the capture did -- and "
            "it means only one of these scales can be a dimensional reference. It is "
            "held out of geometric distortion for that reason; it is still inside the "
            "sampling rate above, which is the mean of both."
        )
    return out
