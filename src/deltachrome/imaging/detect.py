"""Finding a chart's placement in a capture, without anybody clicking on it.

Placement has been manual since the beginning, and the roadmap put automatic
detection in phase 4 on the assumption that it needed a detector accurate enough
to sample from. It does not, and that is the whole reason this module is small.

**Detection does not have to be accurate.** :func:`~deltachrome.imaging.refine.
refine_placement` already does the precision work by snapping a placement onto the
chart's own printed features, and its capture range is ``MAX_SHIFT_FRACTION`` --
five percent of the chart's shorter side. From errors it has actually been given,
22 px on the ISA transmissive charts and 40 to 50 on the ISA 120, it has recovered
corners to a fraction of a pixel and taken geometric distortion from INCOMPLETE to
a pass. So what a finder here has to deliver is "within five percent of the chart's
size", not "right", and every finder below is followed by a refinement.

**The finders already existed.** Each of the eight ``fiducials.method`` values the
shipped targets declare had a working locator in ``tools/``, written to *measure*
the definitions rather than to place a run. What they could not do was run inside
the engine: their tuning constants are in pixels of one capture, they exit the
process rather than returning a refusal, and several of them do work at full
resolution that a placement does not need. The algorithms here are those, with
three things changed and nothing else:

* **Every length is a fraction of a coarse first pass at the chart's extent**
  (:func:`chart_extent`), so the same constant works on a 3400 px capture and on
  an 11600 px one.
* **A refusal is a value, not an exit.** :class:`Detection` carries ``found``
  and a ``reason``, and the caller decides what to do about it.
* **The work happens on a decimated raster** sized to the smallest feature the
  finder must resolve, because a registration mark is hundreds of pixels across
  and locating it to a pixel is precision this stage does not need and the next
  stage supplies.

**The dispatch key is not the declared method alone.** ``patch_array_corners``
covers four targets and needs three different finders -- a 13x10 array of patches
with gutters between them, a 22x12 IT8 lattice whose patches are butted, and a
34:1 strip are not one problem. The distinguishing facts are in the definition
rather than in the target id: a strip declares one row, and butted patches declare
a size equal to their spacing. Keying on those means a *new* chart of a known shape
dispatches correctly without being named anywhere here, which keying on the id
would not.

**A wrong placement is worse than none**, because it returns plausible numbers --
which is the reason the ROI overlays exist at all. So detection refuses rather
than guessing, and every refusal is a sentence the operator reads before placing
the chart by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from deltachrome.imaging.loader import LoadedImage, to_luma
from deltachrome.text import plural

#: Long edge of the raster the coarse extent pass runs on. The chart is a large
#: fraction of the frame by construction -- it is what the capture is *of* -- so
#: locating its bounding box to a part in a thousand is ample, and doing it at full
#: resolution would mean connected components over a hundred megapixels to find
#: something several thousand pixels across.
EXTENT_LONG_EDGE = 1000

#: How much of the frame's edge is taken as a sample of the surround. The chart is
#: what the capture is of, so it is in the middle; whatever runs all the way to the
#: border is the table, the lightbox or the mask it was shot against.
BORDER_FRACTION = 0.02

#: How far from the surround's own level a pixel must sit to count as chart, as a
#: floor and as a multiple of the surround's median absolute deviation. The floor
#: matters on a clean surround, where the deviation is nearly zero and any multiple
#: of it would mark sensor noise as chart; the multiple matters on a textured one.
SURROUND_FLOOR = 0.05
SURROUND_SIGMAS = 5.0

#: Levels the card's own body is looked for between, once the extent is known. A
#: chart's mount is a mid grey by design on every target here -- the point of it is
#: to be neither of the extremes the patches reach -- and the level it reads at is
#: what every darkness threshold below is a fraction of.
CARD_LEVELS = (0.30, 0.80)

#: Fractions of the frame outside which the located region is not a chart. Below
#: the first it is a blob on an otherwise empty capture; above the second nothing
#: was separated at all. Both fall back to the whole frame, which is the honest
#: answer for a chart that fills its capture rather than a degraded one.
CARD_AREA_RANGE = (0.05, 0.995)


@dataclass
class Extent:
    """A coarse first pass at where the chart is, in full-resolution pixels.

    Deliberately loose. Measured against the seventeen committed placements it
    contains the fiducial quad every time and overshoots it by 4% to 47% of the
    quad's long side -- which is what it is for: every constant below is a
    fraction of this, so one number can serve a 3400 px capture and an 11600 px
    one, and none of them is tighter than a factor of two.
    """

    box: tuple[int, int, int, int]  # x0, y0, x1, y1
    #: The card's own modal level, 0..1. Every darkness threshold below is a
    #: fraction of this rather than an absolute, because a chart's mount reads
    #: differently under every illumination and the marks printed on it move with
    #: it.
    level: float
    #: Whether a chart was actually separated from a surround, or the whole frame
    #: was returned because nothing did.
    isolated: bool
    reason: str = ""

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)


def working_luma(image: LoadedImage, long_edge: int) -> tuple[np.ndarray, float]:
    """A decimated single-plane view of the whole capture, and its scale.

    Strided rather than resampled, and strided *before* the conversion to float:
    a 606 MB master widened to float64 across three records is 2.4 GB, which is
    the wrong way to begin looking for something a thousand pixels across. The
    stride costs aliasing that matters to a measurement and not to a search --
    nothing here reports a number read off this raster, only where to look.

    Returns the raster and the factor its coordinates are multiplied by to reach
    full-resolution pixels.
    """
    step = max(1, int(round(max(image.width, image.height) / float(long_edge))))
    luma = to_luma(image.pixels[::step, ::step, :]) / float(image.scale)
    return np.ascontiguousarray(luma, dtype=np.float32), float(step)


def chart_extent(image: LoadedImage) -> Extent:
    """Roughly where the chart is, and what level its own body sits at.

    **Found as the complement of the surround, not as the chart.** Looking for the
    chart directly was tried first, as the largest connected region of near-modal
    mid grey, which is what ``tools/locate_printed_fiducials.py`` does -- and it is
    right for a chart whose mount is most of its area and wrong for one that is
    mostly patches. On the ColorChecker SG it returned a box covering the right
    two thirds of the array, because the mount between 140 patches is not one
    connected region; on the DT Next Generation Target it returned a ninth of the
    chart. Both are the same failure: the thing being looked for is not
    contiguous.

    What *is* contiguous, on every capture here, is the surround -- and it can be
    sampled without being searched for, because it is whatever runs to the frame's
    own border. So the border ring is measured, everything unlike it is marked, and
    the largest such region is the chart. Against the seventeen committed
    placements that box contains the fiducial quad in every case.

    **It is allowed to fail, and says so.** A chart that fills its frame has no
    surround to be told apart from -- the early ISA FADGI-ISO 19264 capture is one
    -- and neither has a capture cropped to the chart. Those return the whole frame
    with ``isolated`` false, which is the correct extent for them.
    """
    luma, step = working_luma(image, EXTENT_LONG_EDGE)
    height, width = luma.shape

    def whole(reason: str) -> Extent:
        return Extent(
            box=(0, 0, image.width, image.height),
            level=_card_level(luma),
            isolated=False,
            reason=reason,
        )

    ring = max(2, int(BORDER_FRACTION * min(height, width)))
    border = np.concatenate([
        luma[:ring].ravel(), luma[-ring:].ravel(),
        luma[:, :ring].ravel(), luma[:, -ring:].ravel(),
    ])
    surround = float(np.median(border))
    spread = float(np.median(np.abs(border - surround)))
    tolerance = max(SURROUND_FLOOR, SURROUND_SIGMAS * spread)

    mask = (np.abs(luma - surround) > tolerance).astype(np.uint8)
    # Closed over a fiftieth of the frame, which is wider than any gutter between
    # patches and narrower than any chart, so an array of patches closes into one
    # region and two charts in one frame would not.
    span = max(3, (min(height, width) // 50) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((span, span), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return whole(
            "nothing in the frame is unlike its border, so the chart fills the "
            "capture; using the whole frame"
        )
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _area = stats[index]
    fraction = (w * h) / float(width * height)
    if not (CARD_AREA_RANGE[0] <= fraction <= CARD_AREA_RANGE[1]):
        return whole(
            f"the region unlike the border covers {fraction:.0%} of the frame, "
            "which is not a chart against a surround; using the whole frame"
        )

    # Grown by one working pixel each way before scaling, so the box does not cut
    # through the boundary that located it.
    x0 = max(int((x - 1) * step), 0)
    y0 = max(int((y - 1) * step), 0)
    x1 = min(int((x + w + 1) * step), image.width)
    y1 = min(int((y + h + 1) * step), image.height)
    inside = luma[max(y - 1, 0):y + h + 1, max(x - 1, 0):x + w + 1]
    return Extent(
        box=(x0, y0, x1, y1),
        level=_card_level(inside),
        isolated=True,
        reason=(
            f"a region {x1 - x0} x {y1 - y0} px unlike the frame's border "
            f"(level {surround:.3f})"
        ),
    )


#: How many candidate regions one frame is offered up as possibly holding a chart.
#:
#: Generous on purpose. The cost of a candidate that is not a chart is one finder
#: pass that refuses, and the cost of missing one is a chart nobody measures.
MOST_CHARTS = 12

#: Smallest region, as a fraction of the frame's area, offered as a candidate.
#:
#: Below this nothing could be placed anyway: the crosshair finder works in corner
#: windows a sixth of the extent across and every refiner needs printed features
#: several pixels wide. The four QA-62 cards in the largest capture here are 1.2% of
#: their frame, six times this; the lint on its backdrop is 0.15% to 0.27%, under it.
MULTI_AREA_FLOOR = 0.002


def chart_extents(image: LoadedImage, *, most: int = MOST_CHARTS) -> list[Extent]:
    """Every region of the frame that might hold a chart, largest first.

    **This does not decide what is a chart, and must not be read as though it did.**
    :func:`chart_extent` answers "where is the chart" and is what a single-chart
    placement uses; this answers "where might I look", which is a different and much
    easier question. What settles it is whether a finder can place a chart in a
    region, which is a far better test than any property of a blob -- so the list is
    deliberately over-generous and :func:`detect_placements` does the deciding.

    Trying to make the segmentation itself decide was the first attempt and it does
    not work. Four QA-62 cards on a board cover half of what the board's own
    bounding box holds, and so do the ISA 4x5's wedges and patch blocks within the
    chart they belong to: coverage reads 0.50 for both. Size agreement fares no
    better -- the DT Next Generation Target's five 59 px registration marks agree
    with each other far more closely than four hand-laid cards do. Both measurements
    are real and neither is the distinction, because the distinction is not
    geometric: it is whether the thing is a chart, and only a finder knows that.

    So what comes back is the union of three searches, in this order:

    * :func:`chart_extent`'s own answer, always and first, so that every capture this
      repository already places correctly keeps exactly the extent it had.
    * Every other region unlike the frame's border above the area floor -- which on a
      board that fills the frame is already the individual cards.
    * Every region lying *on* the largest of those, which is what a board
      photographed against a backdrop needs, since the first pass finds the board.
    * Every region **at the card's own level**, which is what a chart lying on
      something else needs -- see :func:`_at_card_level`.

    They overlap heavily and that is fine: two candidates for one chart cost a
    duplicate placement, which :func:`detect_placements` drops, where a missing
    candidate costs the chart.

    **The three take turns, and no one of them may crowd the others out.** Only
    ``most`` candidates are followed up, because each one is a finder pass, and
    the cap is reached on seven of the twenty captures this repository places --
    the DT Next Generation Target offers 25 and the Avian NGT 24. Appended in
    order, the last search is the one cut, which is what happened: on a
    Smithsonian specimen sheet dense with ruled dotted lines the card-level
    search found the chart correctly and it was the **sixteenth** of twenty-three
    candidates. Round robin needs no new number and states the rule directly.
    :func:`chart_extent`'s own answer still comes first, which is the
    compatibility guarantee and is not part of the rotation.
    """
    luma, step = working_luma(image, EXTENT_LONG_EDGE)
    height, width = luma.shape
    candidates = [chart_extent(image)]
    floor = MULTI_AREA_FLOOR * width * height

    def offer(box, surround: float) -> None:
        extent = _extent_of(image, luma, step, box, surround)
        if not any(_same_box(extent.box, seen.box) for seen in candidates):
            candidates.append(extent)

    searches: list[tuple[list[tuple[int, int, int, int]], float]] = []
    surround, mask = _unlike_border(luma)
    if mask is not None:
        found = _components(mask, floor)
        searches.append((found, surround))
        if found:
            searches.append((_regions_within(luma, mask, found[0], floor), surround))

    level = candidates[0].level
    at_level = _at_card_level(luma, level)
    if at_level is not None:
        searches.append((_components(at_level, floor), level))

    for rank in range(max((len(boxes) for boxes, _ in searches), default=0)):
        for boxes, surround_of in searches:
            if rank < len(boxes):
                offer(boxes[rank], surround_of)
    return candidates[:most]


def _at_card_level(luma: np.ndarray, level: float) -> np.ndarray | None:
    """Everything sitting at the chart's own mount level, as a mask.

    **The search above finds what is unlike the frame's border, and a chart lying on
    something else is not it.** On a Smithsonian Odonata capture -- a specimen sheet
    carrying a Golden Thread Object Level strip along the bottom -- the sheet, the
    insect, the typed labels and the strip are *all* unlike the black backdrop, and
    the close that joins an array of patches into one region joins them to each other
    too. So the largest region is 98% of the frame and looking *into* it returns the
    sheet's contents: seven regions whose aspects run 1.12 to 8.78, and the strip in
    none of them. There is nothing wrong with any of those answers; the strip is
    simply not a thing that differs from the border, because the sheet it lies on
    differs from the border in the same direction and by more.

    What it *does* differ from is the sheet, and what says so is the level the first
    search already computed. ``Extent.level`` is the modal level of the chart's own
    mount -- a mid grey by design on every target here, which is the whole point of
    ``CARD_LEVELS`` -- and every darkness threshold in every finder is already a
    fraction of it. Taking everything within ``SURROUND_FLOOR`` of that level and
    closing it at the usual span puts the strip's card back as the **largest**
    component, at 7.2:1 rather than the quad's 21:1, within a few pixels of a box
    drawn round it by hand. The right chart is then named on **eighteen** of those
    twenty frames, where it was named on none.

    Reusing ``SURROUND_FLOOR`` rather than choosing a width is the point: it already
    means "how far from a level a pixel must sit before it is something else", which
    is exactly the question here read the other way round. Measured on the Odonata
    frames the strip comes back at 0.05, 0.08 and 0.10 alike and is swallowed at
    0.15, so the constant is not near an edge.

    **This does not decide what is a chart** any more than the searches above do. A
    band of one level is a crude thing and will return the sheet's own margin, a
    shadow, or half a wing as readily as a card; what it costs is a finder pass that
    refuses. ``None`` where nothing in the frame sits at that level at all.
    """
    mask = (np.abs(luma - level) <= SURROUND_FLOOR).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(_span(luma), np.uint8))
    return mask if int(mask.max()) else None


def _same_box(first: tuple[int, ...], second: tuple[int, ...]) -> bool:
    """Whether two extents are the same region found twice.

    Judged against the smaller box's own short side rather than an absolute number
    of pixels, because these boxes are struck at the working raster's resolution and
    scaled up by as much as fourteen.
    """
    short = min(
        first[2] - first[0], first[3] - first[1],
        second[2] - second[0], second[3] - second[1],
    )
    return all(abs(a - b) <= 0.05 * short for a, b in zip(first, second))


def _unlike_border(luma: np.ndarray) -> tuple[float, np.ndarray | None]:
    """The surround's level, and the mask of everything unlike it.

    Lifted out of :func:`chart_extent` unchanged so the two searches cannot drift
    apart, and returns ``None`` for the mask where the frame holds nothing its own
    border does not.
    """
    height, width = luma.shape
    ring = max(2, int(BORDER_FRACTION * min(height, width)))
    border = np.concatenate([
        luma[:ring].ravel(), luma[-ring:].ravel(),
        luma[:, :ring].ravel(), luma[:, -ring:].ravel(),
    ])
    surround = float(np.median(border))
    spread = float(np.median(np.abs(border - surround)))
    tolerance = max(SURROUND_FLOOR, SURROUND_SIGMAS * spread)
    mask = (np.abs(luma - surround) > tolerance).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(_span(luma), np.uint8))
    return surround, (mask if int(mask.max()) else None)


def _span(luma: np.ndarray) -> tuple[int, int]:
    """The morphological span: a fiftieth of whatever is being segmented.

    A fraction rather than a constant is what lets one rule serve the whole frame and
    one region inside it -- the second search is the first applied to a smaller
    picture. It is wider than any gutter between patches and narrower than any chart,
    which is what closes an array of patches into one region and leaves two charts
    apart, and it is therefore also the constant deciding how close two charts may
    sit before they are found as one.
    """
    size = max(3, (min(luma.shape) // 50) | 1)
    return (size, size)


def _components(mask: np.ndarray, floor: float) -> list[tuple[int, int, int, int]]:
    """Connected regions of ``mask`` above an area floor, as boxes, largest first."""
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area >= floor:
            boxes.append((area, (x, y, x + w, y + h)))
    return [box for _area, box in sorted(boxes, key=lambda entry: -entry[0])]


def _crop(luma: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return luma[y0:y1, x0:x1]


def _regions_within(
    luma: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], floor: float
) -> list[tuple[int, int, int, int]]:
    """What lies *on* the largest region, in the frame's own coordinates.

    **The parent's mask is opened before it is looked into.** A first-pass region's
    bounding box can overshoot badly: on the QA-62 capture shot against velvet, lint
    on the backdrop is bridged into the board's own component by the close, and the
    box grows to take in a third of the frame that is not board. Opening at the
    close's own span removes bridges narrower than the constant that made them, which
    is the only self-consistent width to remove them at.
    """
    x0, y0, x1, y1 = box
    body = cv2.morphologyEx(
        _crop(mask, box), cv2.MORPH_OPEN, np.ones(_span(luma), np.uint8)
    )
    if not int(body.max()):
        return []
    window = _crop(luma, box)
    # The plain median of the region's body, not _card_level's mid-grey mode: this
    # region is being treated as a *surround*, exactly as the frame's border ring is
    # in the first search, and a surround is whatever it is. The board the QA-62
    # cards lie on reads 0.24, below the range a chart's mount is looked for in, so
    # the mid-grey rule fell back to a median of the extremes and returned 0.50 -- a
    # level nothing on the board sits at, and a tolerance wide enough to mark nothing.
    values = window[body > 0]
    level = float(np.median(values))
    spread = float(np.median(np.abs(values - level)))
    tolerance = max(SURROUND_FLOOR, SURROUND_SIGMAS * spread)
    inner = ((np.abs(window - level) > tolerance) & (body > 0)).astype(np.uint8)
    inner = cv2.morphologyEx(inner, cv2.MORPH_CLOSE, np.ones(_span(window), np.uint8))
    return [
        (a + x0, b + y0, c + x0, d + y0) for a, b, c, d in _components(inner, floor)
    ]


def _extent_of(
    image: LoadedImage,
    luma: np.ndarray,
    step: float,
    box: tuple[int, int, int, int],
    surround: float,
) -> Extent:
    """One found region as an :class:`Extent`, in full-resolution pixels.

    Grown by one working pixel each way before scaling, exactly as
    :func:`chart_extent` does, so the box does not cut through the boundary that
    located it.
    """
    x0, y0, x1, y1 = box
    inside = luma[max(y0 - 1, 0):y1 + 1, max(x0 - 1, 0):x1 + 1]
    full = (
        max(int((x0 - 1) * step), 0), max(int((y0 - 1) * step), 0),
        min(int((x1 + 1) * step), image.width), min(int((y1 + 1) * step), image.height),
    )
    return Extent(
        box=full,
        level=_card_level(inside),
        isolated=True,
        reason=(
            f"a region {full[2] - full[0]} x {full[3] - full[1]} px unlike its "
            f"surround (level {surround:.3f})"
        ),
    )


def _card_level(luma: np.ndarray) -> float:
    """The modal level of the chart's own mount, which thresholds key on.

    Restricted to the mid range because the patches are the extremes by design and
    the mount is not. Where nothing at all falls in that range -- a chart of two
    tones, or a badly exposed capture -- the median of everything is returned, on
    the grounds that a threshold derived from a wrong level fails visibly while a
    threshold derived from nothing raises.
    """
    mid = luma[(luma > CARD_LEVELS[0]) & (luma < CARD_LEVELS[1])]
    return float(np.median(mid if mid.size > 0.02 * luma.size else luma))


# -- the result, and what a refusal looks like --------------------------------


@dataclass
class Detection:
    """A proposed placement, or a refusal, and enough to judge either.

    A refusal is a first-class outcome rather than an exception. Automatic
    placement is only safe if it hands over cleanly: a wrong placement returns
    plausible numbers from the wrong pixels, which is precisely the failure the
    ROI overlays exist to catch, so every finder here would rather say nothing
    than say something it cannot support.
    """

    #: Four points in image pixels, in the order the target declares its corners.
    #: Empty on a refusal.
    corners: list[tuple[float, float]] = field(default_factory=list)
    found: bool = False
    #: The ``fiducials.method`` the target declares.
    method: str = ""
    #: Which finder ran. Not the same thing: ``patch_array_corners`` reaches three.
    finder: str = ""
    #: One sentence, in either outcome, saying what happened. On a refusal this is
    #: what the operator reads before placing the chart by hand.
    reason: str = ""
    notes: list[str] = field(default_factory=list)
    extent: Extent | None = None
    diagnostics: dict = field(default_factory=dict)
    #: The refinement applied afterwards, when one was. Detection lands within a
    #: few percent of the chart and refinement takes it to a fraction of a pixel,
    #: so the two are reported together -- the corners above are the refined ones.
    refinement: object | None = None


class DetectionRefused(Exception):
    """Raised inside a finder, caught by the dispatcher, reported as a refusal.

    The finders read as straight-line code because of this: each one says what it
    cannot do at the point it discovers it, rather than threading an optional
    return through six steps. Nothing outside this module sees it.
    """


# -- putting four found points into the target's own order --------------------


#: How nearly a set of four points must map onto itself a quarter turn round before
#: its shape stops saying which way up it is. As a fraction of the quad's own size,
#: so it reads the same on every capture.
#:
#: Asked of the **found** marks, to decide whether to warn that the chart's
#: orientation cannot be told from where they sit at all. It was once asked of the
#: *declared* corners too, as a second discriminator; see :func:`order_corners` for
#: why that was removed.
SQUARE_QUAD_RESIDUAL = 0.02

#: How far the quad's placed aspect may sit from the one its declared grid implies,
#: in natural log units, before that disagreement counts as evidence. 0.05 is 5%;
#: measured against the seventeen committed placements the agreement is better than
#: 1.1% on sixteen of them and better than 0.5% on thirteen.
CELL_ASPECT_TOLERANCE = 0.05


def implied_quad_aspect(target) -> float | None:
    """How much wider than tall the fiducial quad must be, from the grid alone.

    **A chart's cells are square**, which sounds like an assumption and is close to a
    measurement: across the seventeen committed placements the aspect this predicts
    agrees with the aspect actually placed to better than 1.1% on sixteen of them and
    better than 0.5% on thirteen. A cell is ``patch_spacing_relative`` across a quad
    whose relative coordinates run 0..1 in both axes, so if the cell is physically
    square the quad's own width-to-height ratio is fixed at ``v / u``.

    That matters because it is the only reference-free thing here that knows which way
    up a chart is. The four points a finder returns are the same four points whichever
    is called top-left, and a target's declared corners are a unit square on nearly
    every chart, so all four assignments fit the outline exactly as well and something
    outside the outline has to choose. This does, for the ninety-degree case: a chart
    placed a quarter turn out has a placed aspect of ``1/a`` against an implied ``a``,
    so the wrong answer is wrong by *twice* the log of the aspect -- a wide margin on
    any chart that is not square.

    **The one chart it is wrong about is the one that does not need it.** The Golden
    Thread Device Level Target's patches are not square -- its strips are 0.029 by
    0.063 of a quad 1.75 times wider than tall -- and this predicts 2.15 against a real
    1.75, a 21% error. Its declared corners are a *diamond*, so its own frame decides
    and this is never consulted. Even there the error is nowhere near enough to pick
    the wrong axis, because the alternative is out by 132%: the margin this test works
    on is so much larger than the error in its own premise that the premise being
    approximate does not reach the answer.

    **Where the chart states its own physical extent, that is used instead**, and it
    is not an inference at all: ``fiducial_extent_mm`` is the quad's width and height
    in millimetres, so their ratio is the answer the square-cell premise is reaching
    for. Five shipped targets declare it and on all five the two agree to 0.1%, so
    preferring it changes no placement here -- what it changes is the chart whose
    cells are *not* square, where the premise is not approximate but wrong: the 0.5X
    Object Level Target's quad is 22.9 times wider than tall and its cells imply 34.

    Returns ``None`` for a target declaring no patch grid.
    """
    extent = getattr(target, "fiducial_extent_mm", None)
    if extent and extent[0] and extent[1]:
        return float(extent[0] / extent[1])
    grids = [region.grid for region in target.grid_regions if region.grid is not None]
    if not grids:
        return None
    grid = max(grids, key=lambda g: g.rows * g.columns)
    across, down = grid.patch_spacing_relative
    width, height = grid.patch_size_relative
    u, v = across or width, down or height
    if u <= 0 or v <= 0:
        return None
    return float(v / u)


#: How far apart, in relative units, two placement points may sit on one axis and
#: still count as one side of a rectangle. The early ISA FADGI-ISO 19264's corner
#: squares and dot agree to 0.0025; the Device Level Target's diamond is 0.07 at
#: its closest.
RECTANGLE_TOLERANCE = 0.01


def shape_deviation(target, corners) -> float | None:
    """How far a placement's proportions sit from the ones this chart must have.

    The fractional difference between the aspect of the chart's own relative frame
    as placed -- by :func:`_placed_aspect`, through whatever the declared corners
    are -- and the aspect :func:`implied_quad_aspect` says that frame has.

    **It replaced a comparison against the card**, which was the wrong object.
    ``physical_size_mm`` is the whole card, and a quad is placed on the fiducials,
    which bound something else on most charts: on both Object Level Targets the
    quad is the patch strip, 34:1 on the 1X against a card of 9.25:1, so the
    warning fired on every correct placement of either. Measured across the
    committed placements, a correct one sits within 1% of what this predicts and a
    quarter turn is 40% or more out, so the threshold has room on both sides.

    ``None`` where nothing can be said: no patch grid, or declared corners that are
    not a rectangle on a chart not placed on its own patches. The Golden Thread Device Level Target's are a
    diamond and its cells are not square, so this predicts 2.15 against a real
    1.75 -- the case :func:`implied_quad_aspect` already says it is never consulted
    for, and the warning would fire on every correct placement of that chart too.
    """
    implied = implied_quad_aspect(target)
    declared = np.asarray(target.fiducials.corner_positions or [], dtype=np.float64)
    if implied is None or declared.shape != (4, 2):
        return None
    # The premise fails on one chart, the Device Level Target, and what marks it
    # out is that its placement points are a diamond. So the check runs wherever
    # they are a RECTANGLE -- the frame's own corners, a keyline outside it, three
    # squares and a dot -- or wherever the chart is placed on its own patches,
    # whose lattice IS its frame whichever four are declared (the DT Nexus charts
    # place on four circles one in from the corners, a corner cell being opaque).
    # Measured across the committed placements without this guard: every correct
    # placement reads 1.0% or less except that one chart's 18.8%.
    xs, ys = np.sort(declared[:, 0]), np.sort(declared[:, 1])
    rectangle = (xs[1] - xs[0] < RECTANGLE_TOLERANCE and xs[3] - xs[2] < RECTANGLE_TOLERANCE
                 and ys[1] - ys[0] < RECTANGLE_TOLERANCE and ys[3] - ys[2] < RECTANGLE_TOLERANCE)
    if not rectangle and target.fiducials.method not in PLACED_ON_ITS_OWN_PATCHES:
        return None
    try:
        placed = _placed_aspect(declared, np.asarray(corners, dtype=np.float64))
    except cv2.error:
        return None
    if not placed > 0:
        return None
    return float(abs(placed - implied) / implied)


def order_corners(
    found: list[tuple[float, float]],
    declared: list[tuple[float, float]],
    *,
    cell_aspect: float | None = None,
) -> tuple[list[tuple[float, float]], dict]:
    """Four unordered points, put into the order the target declares.

    **Not a sort into rows and columns.** That is what every locating tool did, and it
    is wrong for the Golden Thread Device Level Target, whose four discs sit as a
    *diamond* about the chart's centre with one point at each extreme: a
    top-row/bottom-row sort puts two of them in the same row by accident of a few
    pixels and can return a bow tie, which ``place_chart`` accepts as a valid
    projective transform and which makes every ROI read the frame's average.

    So the points are matched to the declaration instead. A convex quadrilateral has
    exactly one cyclic order, and the declared corner positions are in one too -- the
    schema's winding check enforces it -- so the question is only which of the four
    rotations lines them up.

    **The reflections are not considered**, and that is a correctness fix rather than
    an economy. Relative space and pixel space have the same handedness, so a found
    ring sorted by angle already winds the way the declaration does; admitting the
    mirror image let a *flipped* placement win the tie-break below, because a
    reflection composed with a small rotation reads as almost no rotation at all.

    **Which rotation is chosen is decided by the aspect the target's grid implies**,
    from :func:`implied_quad_aspect`, and by nothing else -- with being roughly
    upright breaking a tie it cannot break. It is decisive on every chart that is not
    square, and it is the only reference-free thing here that knows a chart has been
    shot a quarter turn round, which is not a rare mistake but the ordinary way to fit
    a wide chart onto a portrait sensor. The X-Rite ColorChecker SG capture in this
    repository is exactly that, and before this test existed its four crosshairs were
    found correctly and labelled a quarter turn out.

    **The obvious second test was tried and removed**, and it is worth saying why
    rather than leaving it to be reinvented. The *declared corners' own shape* looks
    like better evidence -- it is a direct statement about where a chart's fiducials
    sit, where the aspect is an inference from its cells being square -- and it is
    silent on fourteen of the seventeen shipped targets, because relative coordinates
    run 0..1 across the quad in both axes and so make the declared corners a unit
    square whatever the chart's proportions. On the one target it was supposed to
    decide, the Golden Thread Device Level Target's diamond, **it is wrong**: its four
    rotations score 0.248, 0.255, 0.264 and 0.285, the lowest of them is the placement
    a quarter turn out, and the correct one is the second highest. A test that is
    silent on most charts and mistaken on the chart it was for is not a test.

    **The aspect cannot tell a chart shot upside down from one shot upright.** Half a
    turn leaves it unchanged, so where two survivors are half a turn apart both are
    returned in ``assignment_tied`` for the caller to settle by reading the chart --
    see :func:`ramp_agreement`, which is what :func:`detect_placement` does with them.
    """
    points = np.asarray(found, dtype=np.float64).reshape(-1, 2)
    aim = np.asarray(declared, dtype=np.float64).reshape(-1, 2)
    if len(points) != 4 or len(aim) != 4:
        raise DetectionRefused(
            f"{len(points)} points to place against {len(aim)} declared corners."
        )

    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    ring = points[np.argsort(angles)]

    aspect_decides = (
        cell_aspect is not None
        and cell_aspect > 0
        and abs(np.log(cell_aspect)) >= CELL_ASPECT_TOLERANCE
    )

    scored = []
    for turn in range(4):
        candidate = np.roll(ring, turn, axis=0)
        _residual, rotation = _similarity_residual(aim, candidate)
        placed = _placed_aspect(aim, candidate)
        scored.append({
            "corners": candidate,
            "aspect": (
                abs(float(np.log(placed / cell_aspect)))
                if aspect_decides and placed > 0 else 0.0
            ),
            "upright": abs(rotation),
        })

    def cost(entry):
        # Quantised, so a test that does not separate falls through to the next
        # rather than deciding on floating-point noise.
        return (round(entry["aspect"] / CELL_ASPECT_TOLERANCE), entry["upright"])

    scored.sort(key=cost)
    chosen = scored[0]
    tied = [entry for entry in scored if cost(entry)[0] == cost(chosen)[0]]
    # What NARROWED the field, not what picked the last one out of it: the aspect
    # rules out the two rotations a quarter turn away and cannot rule out the one
    # half a turn away, so reporting only the final step would say the aspect
    # decided nothing on exactly the charts where it did the work.
    decided_by = (
        "the aspect this chart's patch grid implies"
        if len(tied) < len(scored)
        else "nothing but the chart being roughly upright in the frame"
    )

    notes = []
    if len(tied) == len(scored):
        turn_residual, _ = _similarity_residual(np.roll(ring, 1, axis=0), ring)
        # One line each, in the warning review's form: these reach every run's
        # warnings, a batch's included, where there are no handles to check. The
        # fact is what the operator can act on -- if the guess is wrong, the ramp
        # and the patches are read the wrong way round.
        notes.append(
            "Orientation assumed nearest upright: a square chart can't show a "
            "quarter turn."
            if turn_residual < SQUARE_QUAD_RESIDUAL else
            "Orientation assumed upright: nothing on the chart shows if it's "
            "upside down."
        )

    return (
        [(float(x), float(y)) for x, y in chosen["corners"]],
        {
            "assignment_decided_by": decided_by,
            "assignment_aspect_error": float(chosen["aspect"]),
            "assignment_rotation_deg": float(np.degrees(chosen["upright"])),
            "assignment_notes": notes,
            #: Every assignment the geometry could not separate from the chosen one,
            #: chosen first. Two of these means the chart could be either way up.
            "assignment_tied": [
                [(float(x), float(y)) for x, y in entry["corners"]] for entry in tied
            ],
        },
    )


#: How well the ramp must agree with its declaration, and by how much it must beat
#: the alternative, before it is allowed to decide which way up a chart is. Both are
#: on the scale below, where +1 is every step in the declared order and -1 is every
#: step against it.
#:
#: **The margin is the one that matters**, and the ColorChecker SG is why. Its ramp
#: is not a ladder read along a row: its twelve steps alternate between two rows
#: running in *opposite* directions, so a half turn maps the ramp onto itself with
#: only adjacent steps swapped rather than reversing it. Measured on the capture in
#: this repository, a rank correlation reads +1.00 the right way up and +0.96 the
#: wrong way -- both far above any floor, separated by four hundredths, and a
#: decision resting on that is a coin toss with a number beside it.
RAMP_AGREEMENT_FLOOR = 0.5
RAMP_AGREEMENT_MARGIN = 0.25

#: Fewest ramp *steps* that can say anything, and fewest comparisons between them.
#: Three points can agree by accident; a dozen cannot. A step is a value the ladder
#: aims at rather than a patch printing it -- see :func:`_ramp_steps`.
MIN_RAMP_STEPS = 6
MIN_RAMP_COMPARISONS = 5

#: How far apart two steps must be *expected* to read, as a fraction of the ladder's
#: own median step, before the direction between them counts as evidence.
#:
#: Against the median step rather than the whole range, because the range does not
#: know how many steps are in it: the LaserSoft declares ninety-six, so two percent
#: of its range is nearly two whole steps and every comparison was discarded.
#:
#: What it guards is a pair that is not really a step: two aims close enough that
#: the noise between two readings decides which way the comparison points. Patches
#: aiming at exactly the same value are a stronger form of the same thing and are
#: pooled before this is reached, which is what :func:`_ramp_steps` does and why the
#: three ISA transmissive charts, the two microform charts and the UTT no longer
#: arrive here interleaved.
MIN_RAMP_SEPARATION = 0.5

#: How far the copies of one step may disagree, as a multiple of the ladder's own
#: median step, on a target printing its ladder more than once. Not tuned on the
#: case it was written for: it is the point at which a step's copies stop agreeing
#: with each other better than with the step beside them. Measured on the six
#: targets that print one, the copies of a step span a median 0.02 to 0.10 of a
#: step on their own captures, worst 0.30; the microform's two read at a QA-62
#: placement span 11.1 to 11.7.
DUPLICATE_STEP_AGREEMENT = 1.0


def ramp_agreement(image: LoadedImage, target, corners) -> float | None:
    """Whether the chart's neutral ramp runs the way its definition says it does.

    The geometry cannot tell a chart shot upside down from one shot upright: half a
    turn leaves the quad's aspect unchanged, maps a rectangle onto itself, and -- on a
    grid centred in its own quad -- leaves every sampling region squarely on a patch.
    Every test that reads only *where* things are is blind to it, and the only thing
    that is not blind is what the patches contain.

    **The ramp is what to read, and every target here has one.** All sixteen shipped
    targets carrying patches declare an ``achromatic_ramp`` with a ``ramp_position``,
    from twelve steps to ninety-six. It is the right feature because a ramp is
    *ordered*: a chart's colour patches are not, and telling one arrangement of them
    from another would need reference values that two of these targets do not ship.

    **What is counted is adjacent steps, not the correlation of the whole ladder**,
    and the difference decides the case this exists for. Read upside down, the
    ColorChecker SG's ramp is not reversed -- its twelve steps alternate between two
    rows running in opposite directions, so a half turn swaps them in pairs and leaves
    the ladder almost monotone. A rank correlation reads +1.00 and +0.96 for the two
    orientations and separates nothing. Counting how many consecutive steps run the
    declared way reads +1.00 and **-0.09**, because every one of the six pairs is
    inverted; the same six inversions that a correlation over sixty-six pairs barely
    notices are all of the evidence there is, and they are adjacent.

    **What the ramp is compared against is the reference where there is one**, and the
    declared order otherwise. That distinction is load-bearing: the convention that
    step 1 is the light end holds for a positive reflective chart and inverts for a
    negative, and this repository ships a negative -- the 35mm microform negative,
    whose ladder as captured runs dark to light with position. Comparing against its
    own nominal values gets that right with no special case, because a definition's
    aim values describe the specimen they are for. The two targets shipping no nominal
    values, the ColorChecker SG and the LaserSoft, are both positive reflective
    charts, where the convention is safe.

    **A step is a value the ladder aims at, not a patch printing it.** Six targets
    here print several patches per step, and reading them as separate steps is what
    let a photograph of a dragonfly score a perfect +1.00 -- see :func:`_ramp_steps`,
    which pools them.

    Returns a number from +1 (every consecutive step in the declared order) to -1
    (every one against it), or ``None`` where there is nothing to read: no ramp, too
    few steps, or a placement whose regions did not sample.
    """
    sampled = _sample(image, target, corners)
    return None if sampled is None else _ramp_from(target, sampled)


def _sample(image: LoadedImage, target, corners):
    """This target's patches, read at this placement, or ``None`` if it cannot be.

    One place, because identifying a chart asks three questions of one sampling --
    how much of the array landed on patches, whether the ramp runs the declared way,
    and how far the colours sit from the aim values -- and asking each of them
    separately sampled the same capture three times over.

    Noise measurement is the expensive half of sampling and nothing here reads it:
    measured, 1324 ms against 287 for the same 140 patches.
    """
    from deltachrome.imaging.geometry import place_chart
    from deltachrome.imaging.sampling import sample_patches

    try:
        placement = place_chart(
            corners, image_size=(image.width, image.height),
            corner_positions_relative=target.fiducials.corner_positions or None,
        )
        return sample_patches(image, placement, target, measure_noise=False)
    except Exception:  # noqa: BLE001 -- a quad this bad simply has no reading
        return None


def _ramp_steps(target, sampled) -> list[tuple[float, float]] | None:
    """The ladder as *steps*, lightest expected first: one reading per aim value.

    **A step is a value the ladder aims at, not a patch printing it**, and the
    difference is the whole of why a dragonfly could be named an Image Engineering
    UTT. Six of the shipped targets print more than one patch per step -- the UTT
    four, as four parallel twenty-step greyscale strips around its border, the three
    ISA transmissive charts and both microform charts two, as a ladder printed twice
    in different places. Keyed by patch and sorted by aim, those arrive
    **interleaved**: strip one step 1, strip two step 1, strip three step 1, and the
    only adjacent pairs whose aim actually steps are the ones that cross from the
    last strip of one step to the first strip of the next.

    On a chart that is harmless, because every strip reads the same. On a
    *photograph* it is the failure: all eighteen comparisons the UTT counted were
    its right-hand strip against its top strip, so one fact about a specimen sheet's
    brightness gradient was counted eighteen times as if it were eighteen
    independent pieces of evidence, and ``MIN_RAMP_COMPARISONS`` -- which exists to
    insist on enough evidence -- was satisfied trivially by it. Monotonicity *within*
    a strip, which is what a ramp test is for, was never read at all: those pairs
    have a zero aim gap and ``MIN_RAMP_SEPARATION`` filters them out before the
    count. The UTT's top strip reads 0.73, 0.54, 0.64, 0.78, 0.83, 0.80, 0.73, 0.54,
    0.66, 0.46 down its length on one such frame -- wildly non-monotonic, and
    invisible.

    Pooling the patches that share an aim into one reading is the whole fix, and it
    **needs no new threshold**: for the sixteen targets declaring all-distinct aims
    it is a no-op by construction, every group holding one patch. Measured over
    twenty Smithsonian Odonata frames, photographs passing the ramp gate fall from
    20/20 to 1/20 while all seven real charts here still read exactly +1.00, against
    a ``RAMP_AGREEMENT_FLOOR`` that does not move.

    Aims are pooled on exact equality rather than within a tolerance, because that
    is a property of the *declaration* -- a target either names one aim for several
    patches or it does not -- and every one of the six is exact. Two aims that are
    merely close are a different question and ``MIN_RAMP_SEPARATION``'s.

    Returns ``None`` where too few steps sampled to say anything, or where a ladder
    printed more than once disagrees with itself -- see ``DUPLICATE_STEP_AGREEMENT``.
    """
    expectation = _ramp_expectation(target)
    if expectation is None:
        return None

    groups: dict[float, list[float]] = {}
    for patch_id, expected in expectation.items():
        sample = sampled.samples.get(patch_id)
        if sample is not None and sample.usable:
            groups.setdefault(expected, []).append(float(np.mean(sample.rgb_mean)))
    if len(groups) < MIN_RAMP_STEPS:
        return None
    steps = [
        (expected, float(np.mean(readings)))
        for expected, readings in sorted(groups.items(), key=lambda kv: -kv[0])
    ]
    # **A ladder printed twice has to read twice the same.** Pooling is what stops
    # the copies being counted as independent evidence; this is what stops pooling
    # hiding that they disagree. Read at a QA-62's placement, the 35mm microform's
    # ten-step L lands on a monotone stretch of the QA-62's own ring and scores a
    # perfect +1.00, so a board of QA-62 cards shortlisted it -- while its two
    # copies of each step sat on different ring cells, eleven steps apart.
    spreads = [max(r) - min(r) for r in groups.values() if len(r) > 1]
    if spreads:
        step = float(np.median(np.abs(np.diff([read for _, read in steps]))))
        if step > 0.0 and float(np.median(spreads)) > DUPLICATE_STEP_AGREEMENT * step:
            return None
    return steps


def _ramp_from(target, sampled) -> float | None:
    """:func:`ramp_agreement`, given a sampling already made."""
    steps = _ramp_steps(target, sampled)
    if steps is None:
        return None

    # Sorted lightest-expected first, so every consecutive pair should descend --
    # but only pairs that are a real step apart are asked, for the reason in
    # MIN_RAMP_SEPARATION.
    aim = np.array([step[0] for step in steps], dtype=np.float64)
    read = np.array([step[1] for step in steps], dtype=np.float64)
    gaps = -np.diff(aim)
    real = gaps[gaps > 0.0]
    if not len(real):
        return None
    differences = np.diff(read)
    counts = differences[
        (gaps >= MIN_RAMP_SEPARATION * float(np.median(real))) & (differences != 0.0)
    ]
    if len(counts) < MIN_RAMP_COMPARISONS:
        return None
    return float(2.0 * np.mean(counts < 0.0) - 1.0)


def _ramp_expectation(target) -> dict[str, float] | None:
    r"""What each ramp patch should read, relative to the others.

    The target's own nominal L\* where it ships any, and the negated step index
    otherwise -- both are monotone in lightness, which is all that is asked of them.

    Several patches may carry the same aim, and on six of the shipped targets
    several do; :func:`_ramp_steps` is what turns this into a ladder.
    """
    ramp = [patch for patch in target.patches if patch.ramp_position is not None]
    if not ramp:
        return None

    path = target.nominal_reference_path
    if path is not None and path.exists():
        try:
            from deltachrome.reference import read_reference

            reference = read_reference(path)
        except Exception:  # noqa: BLE001 -- fall back to the declared order
            reference = None
        if reference is not None:
            aims = {}
            for patch in ramp:
                measured = reference.get(patch.patch_id)
                if measured is not None and measured.lab is not None:
                    aims[patch.patch_id] = float(measured.lab[0])
            # Counted as distinct aims rather than as patches, because that is what
            # MIN_RAMP_STEPS means: a reference stating one value for four patches
            # of a strip describes a ladder that long, however many patches print it.
            if len(set(aims.values())) >= MIN_RAMP_STEPS:
                return aims
    return {patch.patch_id: -float(patch.ramp_position) for patch in ramp}


def _placed_aspect(declared: np.ndarray, candidate: np.ndarray) -> float:
    """How much wider than tall the chart's own relative frame comes out.

    Measured by mapping the unit square through the placement this assignment would
    give, rather than by measuring the found quad -- because the found marks need not
    be at the corners of the chart's frame at all. On the Device Level Target they are
    a diamond about its centre, and on four others they sit outside it.
    """
    matrix = cv2.getPerspectiveTransform(
        declared.astype(np.float32), candidate.astype(np.float32)
    )
    unit = cv2.perspectiveTransform(
        np.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], np.float32), matrix
    )[0].astype(np.float64)
    width = (np.linalg.norm(unit[1] - unit[0]) + np.linalg.norm(unit[2] - unit[3])) / 2.0
    height = (np.linalg.norm(unit[3] - unit[0]) + np.linalg.norm(unit[2] - unit[1])) / 2.0
    return float(width / height) if height > 0 else 0.0


def _similarity_residual(
    declared: np.ndarray, found: np.ndarray
) -> tuple[float, float]:
    """How well one pairing fits a shift, rotation and uniform scale.

    In closed form and in the complex plane, which is what a similarity is: with
    both sets centred, the best ``w = c*z`` has ``c = <z, w> / <z, z>``, its
    argument is the rotation and the residual falls out. Four points and four
    degrees of freedom, so this is a fit with two equations spare -- which is the
    whole point, since a wrong pairing has nowhere to hide.

    Reported relative to the found quad's own size, so the number means the same
    on a 3400 px capture and an 11600 px one.
    """
    z = declared[:, 0] + 1j * declared[:, 1]
    w = found[:, 0] + 1j * found[:, 1]
    z = z - z.mean()
    w = w - w.mean()
    if not np.any(z) or not np.any(w):
        return float("inf"), 0.0
    c = np.vdot(z, w) / np.vdot(z, z)
    residual = float(np.sqrt(np.mean(np.abs(w - c * z) ** 2)))
    size = float(np.sqrt(np.mean(np.abs(w) ** 2)))
    return residual / size if size else float("inf"), float(np.angle(c))


# -- reading only the pixels a scan actually looks at --------------------------

#: How many lines each side is scanned along. The boundary is fitted through them,
#: so this is a sample size rather than a resolution: a thousand points through two
#: parameters is already so over-determined that the fit's residual is a property
#: of the print rather than of the count.
SCAN_LINES = 1000

#: How far in from the extent a boundary is looked for, as a fraction of the
#: extent's shorter side. The methods that use this declare a mark "just inside the
#: chart's border", and what keeps the scan from wandering into the middle of the
#: chart and finding a patch edge is this together with the consensus in
#: :func:`_choose_sides`.
#:
#: It was 0.30, which is the loosest committed placement's 0.24 with room -- and
#: room measured against the *chart* is not room measured against the extent. A
#: 35mm slide is shot in its mount, so the extent is the mount and the chart is a
#: rectangle inside it: on the ISA slide capture the keyline sits 2475 px down and
#: the scan stopped at 2451, twenty-four pixels short, and the whole chart went
#: unfound. At 0.40 it is found with a 0.20 px fit, and at 0.50 the answer is
#: identical -- the deeper scan finds the same line rather than a different one,
#: which is what says the depth is not doing any of the discriminating.
SEARCH_DEPTH = 0.40


def _band(
    image: LoadedImage, box: tuple[int, int, int, int], *, along: int, lines: int
) -> tuple[np.ndarray, float]:
    """Full-resolution luma for the lines one scan reads, and nothing else.

    A side scan looks at a thousand lines out of eleven thousand, at full depth
    down each. Converting the whole band would be a quarter of a gigabyte to read
    six pixels of printed rule; striding along the scan axis and keeping full
    resolution across it costs a hundredth of that and loses nothing, because the
    precision that matters here is across the boundary and not along it.

    ``along`` is the axis the scan steps down: 1 for a horizontal side, whose
    lines are columns, 0 for a vertical one.
    """
    x0, y0, x1, y1 = box
    span = (x1 - x0) if along == 1 else (y1 - y0)
    step = max(1, span // max(lines, 1))
    if along == 1:
        block = image.pixels[y0:y1, x0:x1:step, :]
    else:
        block = image.pixels[y0:y1:step, x0:x1, :]
    return np.asarray(to_luma(block) / float(image.scale), dtype=np.float32), float(step)


#: How far a candidate may sit from a line hypothesis and still count as on it,
#: as a fraction of the extent's shorter side. A printed rule is straight to a
#: fraction of a pixel over its whole length; this is loose enough to absorb the
#: scan's own noise and tight enough that two parallel rules a few pixels apart --
#: a keyline and the card edge beside it -- are separate hypotheses.
INLIER_TOLERANCE = 0.0004

#: How many separate straight lines are pulled out of one side's candidates, and
#: how many random pairs each search draws. Three is enough: the competitors seen
#: so far are one apiece -- the LaserSoft's barcode below its frame rule, the ISA
#: 120's mask edge outside its keyline.
MAX_HYPOTHESES = 3
HYPOTHESIS_TRIALS = 400

#: How far a side's angle may sit from what the other three imply before it is
#: taken to have been found on something other than the rectangle, in degrees.
#: The four sides of a printed rectangle are parallel in pairs; the shipped
#: captures fit their four sides to within 0.02 degrees of each other.
SIDE_ANGLE_TOLERANCE = 0.10

#: Most candidate boundaries kept per scan line. More than one is the entire
#: point -- see :func:`_line_hypotheses` -- and six is past anything the shipped
#: captures need.
MAX_CANDIDATES = 6


def _fit_boundary(points: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """A straight line through a found boundary, with the strays thrown out.

    Three tightening passes rather than one: the first is dragged by whatever the
    scan mistook for the boundary, and the residual it reports only means anything
    once those are gone.
    """
    keep = points
    for _ in range(3):
        fit = np.polyfit(keep[:, 0], keep[:, 1], 1)
        residual = keep[:, 1] - np.polyval(fit, keep[:, 0])
        narrowed = keep[np.abs(residual) < max(0.8, 2.5 * residual.std())]
        if len(narrowed) >= 8:
            keep = narrowed
    return np.polyfit(keep[:, 0], keep[:, 1], 1), keep


def _line_hypotheses(
    points: np.ndarray, tolerance: float, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The straight lines a side's candidate boundaries actually lie on.

    **Why a side has more than one answer.** Taking the first dark thing each scan
    line meets was how this worked, and it is wrong in a way that produces a
    confident wrong placement rather than a failure. Two captures here show the
    two directions it fails in: the LaserSoft prints a *barcode* below its frame
    rule, so a scan up from the bottom meets the bars first; the ISA 120 is shot
    against a mask whose edge lies outside its keyline, so a scan in from the
    right meets the mask first. In both, the sides that were not deceived were
    fitted to a hundredth of a degree, which is what makes the deception findable.

    So every qualifying run on every line is kept, and the lines those candidates
    lie on are pulled out one at a time, strongest first, by random pairs and
    inlier counting. A rule printed across the whole chart is supported by every
    scan line; a barcode occupying a third of the width cannot be, whatever it
    does to the third it occupies. Which of the surviving hypotheses is the
    rectangle is then decided across the four sides together, in
    :func:`_choose_sides`, because one side alone cannot know.
    """
    rng = np.random.default_rng(seed)
    remaining = points
    hypotheses: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(MAX_HYPOTHESES):
        if len(remaining) < 8:
            break
        first = rng.integers(0, len(remaining), HYPOTHESIS_TRIALS)
        second = rng.integers(0, len(remaining), HYPOTHESIS_TRIALS)
        a, b = remaining[first], remaining[second]
        run = b[:, 0] - a[:, 0]
        usable = np.abs(run) > 1e-6
        if not np.any(usable):
            break
        slope = np.where(usable, (b[:, 1] - a[:, 1]) / np.where(usable, run, 1.0), 0.0)
        intercept = a[:, 1] - slope * a[:, 0]
        predicted = slope[:, None] * remaining[None, :, 0] + intercept[:, None]
        inliers = (np.abs(predicted - remaining[None, :, 1]) < tolerance) & usable[:, None]
        counts = inliers.sum(axis=1)
        best = int(np.argmax(counts))
        if counts[best] < 8:
            break
        on_line = remaining[inliers[best]]
        fit, kept = _fit_boundary(on_line, tolerance)
        hypotheses.append((fit, kept))
        # Removed by the FITTED line rather than by the sampled pair, so a rule
        # is taken off in one piece and its neighbours are not nibbled at.
        residual = np.abs(remaining[:, 1] - np.polyval(fit, remaining[:, 0]))
        remaining = remaining[residual >= tolerance]
    return hypotheses


def _choose_sides(
    candidates: dict[str, list[tuple[np.ndarray, np.ndarray]]], *,
    prefer: str = "strongest",
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """One line per side, chosen so the four of them make a rectangle.

    A rectangle's opposite sides are parallel, and in the parameterisation used
    here -- ``y = a*x + b`` along a horizontal side and ``x = a*y + b`` down a
    vertical one -- all four of its slopes have the same magnitude, positive on
    one pair and negative on the other. So the chart's own rotation is a quantity
    four independent measurements agree on, and a side found on something else
    disagrees with it.

    The rotation is taken as the median of each side's strongest hypothesis, which
    survives one side being wrong; any side then disagreeing is re-chosen among its
    own alternatives, and if none of them agrees either the whole detection is
    refused rather than a rectangle being forced out of four unrelated lines.
    """
    def angle(fit: np.ndarray, side: str) -> float:
        turn = np.degrees(np.arctan(fit[0]))
        return -turn if side in {"left", "right"} else turn

    strongest = {side: angle(options[0][0], side) for side, options in candidates.items()}
    rotation = float(np.median(list(strongest.values())))

    chosen: dict[str, np.ndarray] = {}
    report: dict[str, dict] = {}
    for side, options in candidates.items():
        agreeing = [
            (fit, kept) for fit, kept in options
            if abs(angle(fit, side) - rotation) <= SIDE_ANGLE_TOLERANCE
        ]
        if not agreeing:
            raise DetectionRefused(
                f"the {side} boundary was found at "
                f"{angle(options[0][0], side):+.2f} degrees where the other three "
                f"sides say {rotation:+.2f}, and none of the alternatives on that "
                "side agrees either -- so what was found there is not part of the "
                "chart's own rectangle"
            )
        if prefer == "outermost":
            # A block's own outer boundary is the shallowest of the parallel
            # boundaries a scan meets, and the ones behind it are the seams
            # between its first two rows. Support still has to be real, so a line
            # is only preferred for being outer if it is held up by half as many
            # scan lines as the best-supported one.
            floor = 0.5 * max(len(kept) for _fit, kept in agreeing)
            fit, kept = min(
                (option for option in agreeing if len(option[1]) >= floor),
                key=lambda option: float(np.median(option[1][:, 1])),
            )
        else:
            fit, kept = max(agreeing, key=lambda option: len(option[1]))
        residual = kept[:, 1] - np.polyval(fit, kept[:, 0])
        chosen[side] = fit
        report[side] = {
            "lines": len(options),
            "kept": int(len(kept)),
            "deg": angle(fit, side),
            "rms_px": float(residual.std()),
            "reselected": not np.allclose(fit, options[0][0]),
        }
    report["rotation_deg"] = rotation
    return chosen, report


def _intersect(horizontal: np.ndarray, vertical: np.ndarray) -> tuple[float, float]:
    """Where a fitted row-line and a fitted column-line meet.

    The two are parameterised differently on purpose -- ``y = a*x + b`` for a
    nearly horizontal rule and ``x = a*y + b`` for a nearly vertical one -- so each
    is fitted along the axis it actually extends in and neither goes singular.
    """
    denominator = 1.0 - vertical[0] * horizontal[0]
    if abs(denominator) < 1e-9:
        raise DetectionRefused(
            "two of the located boundaries are parallel, so they have no corner"
        )
    x = (vertical[0] * horizontal[1] + vertical[1]) / denominator
    return float(x), float(horizontal[0] * x + horizontal[1])


SIDES = ("top", "bottom", "left", "right")


def _scan_sides(
    image: LoadedImage, extent: Extent, locate, *,
    box: tuple[int, int, int, int] | None = None,
    depth: int | None = None,
    prefer: str = "strongest",
) -> tuple[list, dict]:
    """The four boundaries of a rectangle printed inside the extent, intersected.

    ``locate`` is handed one line's profile, running inward from the side, and
    returns every sub-pixel depth at which it found a boundary. That one function
    is the whole difference between finding a printed keyline and finding the edge
    of a patch strip, which is why the scanning, fitting and intersecting are here
    and not copied into each.
    """
    x0, y0, x1, y1 = box or extent.box
    depth = depth or max(8, int(SEARCH_DEPTH * extent.short_side))
    tolerance = max(1.5, INLIER_TOLERANCE * extent.short_side)
    candidates: dict[str, list] = {}

    for index, side in enumerate(SIDES):
        horizontal = side in {"top", "bottom"}
        if side == "top":
            box = (x0, y0, x1, min(y0 + depth, y1))
        elif side == "bottom":
            box = (x0, max(y1 - depth, y0), x1, y1)
        elif side == "left":
            box = (x0, y0, min(x0 + depth, x1), y1)
        else:
            box = (max(x1 - depth, x0), y0, x1, y1)

        block, step = _band(image, box, along=1 if horizontal else 0, lines=SCAN_LINES)
        # Every scan runs inward from its own side, so one ``locate`` serves all
        # four: the profile is reversed and the depth measured back from the far
        # edge for the two that start at the bottom and the right.
        profiles = block.T if horizontal else block
        inward = side in {"top", "left"}
        base = (box[1] if horizontal else box[0]) if inward else (
            (box[3] if horizontal else box[2]) - 1
        )
        origin = box[0] if horizontal else box[1]

        points = [
            (origin + line * step, base + (found if inward else -found))
            for line, profile in enumerate(profiles)
            for found in locate(profile if inward else profile[::-1])
        ]
        if len(points) < 8:
            raise DetectionRefused(
                f"the {side} boundary was found on only {len(points)} of "
                f"{len(profiles)} scan lines"
            )
        found_lines = _line_hypotheses(np.asarray(points, float), tolerance, index)
        if not found_lines:
            raise DetectionRefused(
                f"nothing along the {side} side lies on a straight line"
            )
        candidates[side] = found_lines

    fits, report = _choose_sides(candidates, prefer=prefer)
    corners = [
        _intersect(fits["top"], fits["left"]),
        _intersect(fits["top"], fits["right"]),
        _intersect(fits["bottom"], fits["right"]),
        _intersect(fits["bottom"], fits["left"]),
    ]
    return corners, {"sides": report}


# -- printed_frame_corners ----------------------------------------------------

#: How dark, as a fraction of the card's own level, counts as the printed rule.
#:
#: Taken from ``tools/locate_printed_fiducials.py``, where it was swept: the four
#: charts placed this way print keylines at 0.51, 0.60, 0.62 and 0.665 of their
#: card, three of them do not move by a tenth of a pixel anywhere from 0.65 to
#: 0.85, and the fourth fails below 0.70 and is stable from 0.75 up. This is the
#: middle of that flat region.
FRAME_DARK_FRACTION = 0.80

#: Thickest run of dark that can still be a rule, as a fraction of the extent's
#: shorter side. It exists because of the LaserSoft, which prints a *barcode*
#: below its frame: scanning up from the card for the first dark thing finds the
#: bars, and the bottom rule then fits through nothing. Its rule is 6 px and its
#: bars are nearer 300, so a bound anywhere between rejects the bars, walks past
#: them and finds the rule above.
FRAME_MAX_THICKNESS = 0.006


def find_printed_frame(image: LoadedImage, target, extent: Extent) -> tuple[list, dict]:
    """The four corners of a thin dark rule printed just inside the chart's border.

    Four of the shipped targets place this way -- the three ISA transmissive film
    charts on their keyline, and the LaserSoft ISO 12641-2 on its black frame --
    and the corner is taken as the intersection of two *fitted* sides rather than
    as the darkest place near a corner. A rule's corner is where two lines meet,
    and a thousand points along each line locate that far better than any
    neighbourhood of the corner itself can.
    """
    dark = extent.level * FRAME_DARK_FRACTION
    thickest = max(4, int(FRAME_MAX_THICKNESS * extent.short_side))

    def locate(profile: np.ndarray) -> list[float]:
        depths = []
        for start, end in _runs(profile < dark):
            if not (2 <= end - start + 1 <= thickest):
                continue
            lo, hi = max(start - 1, 0), min(end + 2, len(profile))
            weight = np.clip(extent.level - profile[lo:hi], 0.0, None)
            if weight.sum() <= 0:
                continue
            depths.append(float(np.arange(lo, hi) @ weight / weight.sum()))
            if len(depths) >= MAX_CANDIDATES:
                break
        return depths

    corners, report = _scan_sides(image, extent, locate)
    report["dark_below"] = dark
    report["thickest_rule_px"] = thickest
    return corners, report


def _runs(line: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous stretches of a boolean profile, as (first, last) index pairs."""
    where = np.flatnonzero(line)
    if not len(where):
        return []
    breaks = np.flatnonzero(np.diff(where) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(where) - 1]))
    return [(int(where[a]), int(where[b])) for a, b in zip(starts, ends)]


# -- printed_discs ------------------------------------------------------------

#: A circle fills pi/4 of its bounding box, and both bounds are wide enough to
#: survive the threshold moving thirty levels while being tight enough that
#: nothing else on the Device Level Target -- not a patch, not a letter, not a
#: resolution field -- gets through. Measured over the sweep below, its four discs
#: sit at 0.783 to 0.787.
DISC_FILL = np.pi / 4
DISC_FILL_TOLERANCE = 0.06
#: How square a disc's bounding box must be, and how far its area may sit from the
#: others'. A disc is round, so both are tight.
DISC_ASPECT_TOLERANCE = 0.12
DISC_AREA_SPREAD = 0.15
#: Smallest a disc may be, as a fraction of the extent's shorter side. A printed
#: registration disc is a feature of the chart rather than a speck on it.
DISC_MIN_FRACTION = 0.01
#: Long edge of the raster the discs are looked for on. They are hundreds of
#: pixels across on the capture this was measured on, so a coarse raster costs
#: nothing that refinement does not give back, and it turns fourteen passes of
#: connected components over a hundred megapixels into fourteen over one.
DISC_LONG_EDGE = 1500
#: Fractions of the card's own level the sweep runs over. Measured rather than
#: guessed: below 0.45 the discs have not fully separated from the card and their
#: centroids wander a third of a pixel; from 0.45 to 0.80 the four hold to about a
#: tenth while each disc's apparent area grows nine percent, which is the
#: signature of a real edge being crossed rather than a threshold finding noise.
DISC_LEVELS = np.arange(0.45, 0.80, 0.025)


def find_printed_discs(image: LoadedImage, target, extent: Extent) -> tuple[list, dict]:
    """The four solid dark circles the Device Level Target prints as a diamond.

    Chosen by **circularity**, which is the one property nothing else on that
    chart has: a disc fills pi/4 of its bounding box where a patch fills 1 and a
    letter something in between. Swept over a range of thresholds rather than
    fixed at one, and the answer taken as the mean where the sweep is flat --
    which is not caution for its own sake. The discs are printed on a mid-grey
    card whose level moves with the illumination, so any single threshold is a
    guess about the capture; over the swept range the four centroids move about a
    tenth of a pixel while each disc's measured area grows nine percent. **The
    mark staying put while its apparent size changes is what says the threshold
    does not matter, and therefore that the marks are real.** One threshold could
    not tell that from having found something else entirely.

    Thresholded against the card's own level and searched inside the extent, not
    across the frame. A chart shot on a dark table puts the surround far below
    anything printed on the chart, so a threshold taken from the whole image lands
    in the table and finds the chart itself as one enormous disc.
    """
    luma, step = working_luma(image, DISC_LONG_EDGE)
    x0, y0, x1, y1 = (int(v / step) for v in extent.box)
    window = luma[y0:y1, x0:x1]
    if window.size == 0:
        raise DetectionRefused("the located chart extent holds no pixels")
    smallest = max(6, int(DISC_MIN_FRACTION * extent.short_side / step))
    blurred = cv2.GaussianBlur(window, (9, 9), 0)

    per_level = []
    for fraction in DISC_LEVELS:
        mask = (blurred < extent.level * fraction).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        hits = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if width < smallest or height < smallest:
                continue
            if abs(width / height - 1.0) > DISC_ASPECT_TOLERANCE:
                continue
            if abs(area / (width * height) - DISC_FILL) > DISC_FILL_TOLERANCE:
                continue
            hits.append((
                (float(centroids[index][0]) + x0) * step,
                (float(centroids[index][1]) + y0) * step,
                float(area),
            ))
        # Only a set of four whose areas agree can be one chart's own marks.
        if len(hits) != 4:
            continue
        areas = np.array([hit[2] for hit in hits])
        if float(np.ptp(areas) / areas.mean()) > DISC_AREA_SPREAD:
            continue
        per_level.append(_ring(np.array([(hit[0], hit[1]) for hit in hits])))

    if not per_level:
        raise DetectionRefused(
            "No set of four agreeing circles was found. This chart is placed on "
            "solid discs filling pi/4 of their bounding box; a capture whose card "
            "reads darker than its marks, or one cropped so that a disc is off the "
            "frame, will find none."
        )
    stacked = np.array(per_level)
    mean = stacked.mean(axis=0)
    return (
        [tuple(map(float, point)) for point in mean],
        {
            "thresholds": len(per_level),
            "stability_px": float(np.max(np.linalg.norm(stacked - mean, axis=2))),
            "smallest_disc_px": smallest * step,
        },
    )


def _ring(points: np.ndarray) -> np.ndarray:
    """Four points in cyclic order about their centroid, starting at the topmost.

    Only so that thresholds can be averaged against each other -- the answer has
    to be in the *same* order at every level or the mean is nonsense. Which
    corner is which is decided later and elsewhere, by :func:`order_corners`.
    """
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


# -- printed_crosshairs -------------------------------------------------------

#: How much of a corner of the extent one crosshair is looked for in, as a
#: fraction of the extent's shorter side. The method's name is the justification:
#: these marks sit near the corners of the chart by definition, and searching only
#: there rejects everything printed in the middle of it for free.
#:
#: It was 0.28, which covers the three charts whose marks sit just outside their
#: patch array and misses the 35mm microform chart entirely: its crosses bound one
#: inch where the printed frame around them is 1.64, so they sit 0.47 of the way in
#: from an extent that is the whole piece of film. At 0.45 the four windows can
#: overlap on a chart whose marks are close to its corners, which costs nothing --
#: each window takes the mark furthest towards its own corner, so two windows
#: cannot choose the same one, and the check below says so rather than assuming it.
CROSS_WINDOW = 0.45

#: Bounds on a crosshair's bounding box: a lower one in pixels, because a
#: registration mark below a few pixels is not resolvable whatever the capture,
#: and an upper one as a fraction of its search window.
CROSS_MIN_PX = 8
CROSS_MAX_FRACTION = 0.25

#: How much of the bounding box may be ink. A hairline cross fills 0.13 of its box
#: and a stubby plus 0.40, so this is wide on purpose -- it is a sanity bound, and
#: the shape test is what decides.
CROSS_FILL = (0.05, 0.70)

#: How many levels each corner window is thresholded at, spanning that window's
#: own range, and both polarities at each.
#:
#: **Spanning the window rather than the card**, which was the first attempt and
#: is wrong in one direction only. Fractions of the card's level cover a dark mark
#: on a light chart and cannot reach a light mark on a dark one -- and the 35mm
#: microform chart prints exactly that: a bright plus on a dark ground, where
#: every threshold below the card level leaves the ground and the mark on the same
#: side of it and the finder sees one enormous component. Taking the levels from
#: the window itself needs no assumption about which way round the chart is
#: printed, which is the same reason ``rulers.polarity`` defaults to ``auto``.
CROSS_LEVELS = 24
CROSS_LEVEL_INSET = 0.05

#: How much ink a corner of the mark's bounding box may hold, and how far its arms
#: must reach across it. See :func:`_is_cross`; the reach is the test that says the
#: component is a plus rather than something else the size of one.
CROSS_CORNER_INK = 0.10
CROSS_ARM_REACH = 0.8

#: How far the four chosen marks' areas may disagree, and how many candidates a
#: corner window contributes to the search over combinations. Both are bounds on a
#: cheap exhaustive search: at six apiece there are 1296 combinations to score,
#: which is nothing beside the thresholding that produced them.
CROSS_AREA_SPREAD = 0.75
MAX_CROSS_CANDIDATES = 6

#: How far two thresholds' answers may sit apart, as a fraction of the extent's
#: shorter side, and still count as the same answer. Loose enough to absorb a
#: mark's centroid shifting as its threshold moves -- measured, about a pixel --
#: and far tighter than the distance to the next printed mark on any of these
#: charts.
CROSS_STEADY = 0.005

#: How many thresholds must agree before a group counts as real rather than as one
#: level's accident. Only used where outwardness rather than consensus chooses --
#: see :func:`_steadiest` -- since otherwise consensus is the ranking itself.
CROSS_MIN_AGREEING = 2


def find_printed_crosshairs(
    image: LoadedImage, target, extent: Extent, *, prefer: str = "steadiest"
) -> tuple[list, dict]:
    """The four printed plus marks near the corners of the chart.

    Five shipped targets place this way and the marks could hardly look less
    alike: the ColorChecker SG's is a hairline cross 94 px across with 6 px
    strokes, the DT Next Generation Target V2's a stubby grey plus 57 px across
    with 15 px strokes, the commercial ISA FADGI-ISO 19264's a 61 px plus just
    inside the chart's own border, the 35mm microform chart's a *bright* plus on a
    dark ground, and the Image Engineering UTT's the outermost of thirty-six.
    Nothing about size or fill separates any of them from the printed letters
    beside it -- an ``H`` at 60x70 px with fill 0.51 beats a crosshair on all three
    of those tests. **What separates a plus is its shape**, and that is scale-free,
    which is why one function serves all five.

    Two things then choose between the candidates the shape test admits, and
    neither is the obvious one.

    **The four are chosen together rather than one corner at a time**, because four
    registration marks on one chart are the same printed size and four unrelated
    blobs are not. Every combination of one candidate per corner is scored by how
    far their areas disagree and thrown out entirely if they disagree too much,
    which is what keeps the commercial ISA chart off a mark below and right of its
    real bottom-right plus that passes the shape test and would win any per-corner
    contest by sitting further out.

    **What is chosen among the survivors is the answer the most thresholds agree
    on.** Ranking by agreement in size puts the UTT on four identical bright marks
    inside its own quad; ranking by outwardness puts the commercial ISA chart 4451
    px out. Both are the same mistake, which is to believe one threshold. The
    marks that are really printed on the chart are found in the same places over a
    run of thresholds while their apparent areas grow -- **the mark staying put
    while its size changes is what says the threshold does not matter, and
    therefore that the mark is real** -- and nothing else in these captures does
    that. On the UTT the true four hold to a pixel across four consecutive
    thresholds where every competitor drifts; the answer is then the mean over
    them, which is the same argument and the same construction the disc finder
    uses.
    """
    window = int(CROSS_WINDOW * extent.short_side)
    x0, y0, x1, y1 = extent.box
    corners = {
        "TL": (x0, y0, min(x0 + window, x1), min(y0 + window, y1)),
        "TR": (max(x1 - window, x0), y0, x1, min(y0 + window, y1)),
        "BR": (max(x1 - window, x0), max(y1 - window, y0), x1, y1),
        "BL": (x0, max(y1 - window, y0), min(x0 + window, x1), y1),
    }
    blocks = {
        name: np.asarray(
            image.luma_window(box[1], box[3], box[0], box[2]), dtype=np.float32
        )
        for name, box in corners.items()
    }
    low = float(min(block.min() for block in blocks.values()))
    high = float(max(block.max() for block in blocks.values()))
    if high - low <= 0:
        raise DetectionRefused(
            "The corners of this capture hold no variation at all, so there is "
            "nothing there to find."
        )
    inset = CROSS_LEVEL_INSET * (high - low)

    solutions = []
    for level in np.linspace(low + inset, high - inset, CROSS_LEVELS):
        for polarity in (1, -1):
            candidates = {
                name: _cross_candidates(block, level, polarity, window, name)
                for name, block in blocks.items()
            }
            if any(not found for found in candidates.values()):
                continue
            choice = _agreeing_marks(candidates, prefer)
            if choice is None:
                continue
            spread, marks = choice
            solutions.append((
                np.array([
                    (marks[name][0] + corners[name][0], marks[name][1] + corners[name][1])
                    for name in ("TL", "TR", "BR", "BL")
                ]),
                spread,
                polarity,
            ))

    if not solutions:
        raise DetectionRefused(
            "No four plus-shaped marks of one size were found near the corners of "
            "the chart. Placement on printed crosshairs needs all four in the "
            "frame and clear of anything overlapping them."
        )
    agreed, spread, polarity = _steadiest(
        solutions, CROSS_STEADY * extent.short_side, prefer
    )
    # The windows may overlap, so it has to be said rather than assumed that four
    # windows chose four marks: two corners landing on one mark gives a degenerate
    # quad, and the honest report of that is here rather than a geometry error
    # three steps later.
    apart = np.linalg.norm(agreed[:, None, :] - agreed[None, :, :], axis=2)
    if float(np.min(apart[np.triu_indices(4, 1)])) < window * 0.1:
        raise DetectionRefused(
            "Two corners of the chart were placed on the same printed mark, so "
            "fewer than four were found."
        )
    return (
        [(float(x), float(y)) for x, y in agreed],
        {
            "thresholds_agreeing": int(spread[1]),
            "thresholds_tried": len(solutions),
            "marks": "light" if polarity > 0 else "dark",
            "area_spread": spread[0],
            "stability_px": spread[2],
        },
    )


def _steadiest(solutions: list, tolerance: float, prefer: str):
    """The quad the thresholds agree on, and how far they wander from it.

    Solutions are grouped by every corner landing within ``tolerance`` of the
    group's first member. ``steadiest`` then takes the largest group, which is the
    answer for a chart printing exactly the four marks being looked for; ties go
    to the group whose marks agree best in size.

    ``outermost`` takes the group sitting furthest into the corners, among those
    steady enough to be real. That is for the Image Engineering UTT, whose declared
    placement is *the outermost of the six crosshairs* around each corner module --
    so on that chart being outermost is not a tie-break, it is the definition, and
    consensus alone puts it on four steady bright marks well inside its own quad.
    The steadiness floor is what keeps the same rule off a single threshold's
    accident: a group of one has agreed with nothing.

    The answer is the chosen group's mean, so a threshold's own quantisation
    averages out rather than being inherited from whichever level happened to be
    tried.
    """
    groups: list[list] = []
    for solution in solutions:
        for group in groups:
            if np.max(np.linalg.norm(solution[0] - group[0][0], axis=1)) <= tolerance:
                group.append(solution)
                break
        else:
            groups.append([solution])

    if prefer == "outermost":
        steady = [group for group in groups if len(group) >= CROSS_MIN_AGREEING]
        best = max(steady or groups, key=_outwardness)
    else:
        best = max(groups, key=lambda group: (len(group), -min(s[1] for s in group)))
    stacked = np.array([solution[0] for solution in best])
    mean = stacked.mean(axis=0)
    return (
        mean,
        (
            float(min(solution[1] for solution in best)),
            len(best),
            float(np.max(np.linalg.norm(stacked - mean, axis=2))),
        ),
        best[0][2],
    )


def _agreeing_marks(candidates: dict[str, list], prefer: str):
    """One mark per corner, from the combinations whose four agree in size.

    The size spread is a **filter** either way: four registration marks on one
    chart agree in area to within the print and the threshold, four unrelated blobs
    do not, and a combination whose areas disagree past
    :data:`CROSS_AREA_SPREAD` is thrown out whatever else it has going for it.

    What is preferred among the survivors is ``prefer``, and it has to be the same
    preference :func:`_steadiest` applies across thresholds rather than a different
    one -- which was the subtler half of the UTT's failure. Ranking within a level
    by size agreement never even *offered* that chart's true quad to the
    consensus step: at every threshold where the real four could be found, a
    better-agreeing wrong four could be found too, so the right answer was
    discarded one level at a time and no amount of agreement across levels could
    recover it.
    """
    import itertools

    names = sorted(candidates)
    best = None
    for combination in itertools.product(*(candidates[name] for name in names)):
        areas = np.array([area for _centre, area, _reach in combination])
        spread = float(np.ptp(areas) / areas.mean()) if areas.mean() > 0 else np.inf
        if spread > CROSS_AREA_SPREAD:
            continue
        outward = float(sum(reach for _centre, _area, reach in combination))
        rank = (-outward, spread) if prefer == "outermost" else (spread, -outward)
        if best is None or rank < best[0]:
            best = (rank, dict(zip(
                names, [centre for centre, _area, _reach in combination]
            )))
    if best is None:
        return None
    spread = best[0][1] if prefer == "outermost" else best[0][0]
    return spread, best[1]


def _cross_candidates(
    block: np.ndarray, level: float, polarity: int, window: int, corner: str
) -> list:
    """Every plus-shaped mark in one corner window, furthest-out first."""
    mask = ((block > level) if polarity > 0 else (block < level)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    upper = CROSS_MAX_FRACTION * window
    towards = {
        "TL": (-1.0, -1.0), "TR": (1.0, -1.0), "BR": (1.0, 1.0), "BL": (-1.0, 1.0),
    }[corner]
    found = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if not (CROSS_MIN_PX < max(w, h) < upper):
            continue
        if not (0.75 < w / max(h, 1) < 1.33):
            continue
        if not (CROSS_FILL[0] < area / float(w * h) < CROSS_FILL[1]):
            continue
        if not _is_cross(labels[y:y + h, x:x + w] == index):
            continue
        centre = (float(centroids[index][0]), float(centroids[index][1]))
        found.append(
            (centre, float(w * h), centre[0] * towards[0] + centre[1] * towards[1])
        )
    found.sort(key=lambda mark: -mark[2])
    return found[:MAX_CROSS_CANDIDATES]


def _is_cross(mask: np.ndarray) -> bool:
    """Whether a component is a plus rather than a glyph, a disc or a patch.

    Two tests, and between them they are the whole finder.

    **The four corners of the bounding box must be clear of ink.** No letter with a
    bowl or a stem in a corner survives it, and neither does a T, an L or the
    corner of a printed rule -- which is what the shape test is really for, because
    nothing about size, squareness or fill separates a crosshair from the letters
    printed beside it: on the ColorChecker SG an ``H`` at 60x70 px with fill 0.51 is
    a better match for those three than the crosshair is.

    **One column must be nearly the full height and one row nearly the full
    width**, and they must meet in ink. That is what a plus *is*, said directly.
    It replaced five three-pixel probes at the bounding box's own centre and arm
    midpoints, which assumed the mark sits centred in its box -- and the Image
    Engineering UTT's does not: its horizontal bar is six rows tall and sits 55% of
    the way down a 56-row box, so the probes for the left and right arms sampled
    the row above the bar and the real crosshair was rejected at every threshold.
    Measuring where the ink actually is costs nothing and assumes nothing about how
    the mark is drawn.

    Both tests are scale-free, which is why one function serves marks that could
    hardly look less alike: the SG's hairline cross 94 px across with 6 px strokes,
    the DT Next Generation Target's stubby grey plus, the 35mm microform chart's
    bright plus on a dark ground.
    """
    height, width = mask.shape
    if height < 5 or width < 5:
        return False
    corner = max(2, min(width, height) // 6)
    if max(mask[:corner, :corner].mean(), mask[:corner, -corner:].mean(),
           mask[-corner:, :corner].mean(), mask[-corner:, -corner:].mean()) > CROSS_CORNER_INK:
        return False

    down, across = mask.sum(axis=0), mask.sum(axis=1)
    stem, bar = int(np.argmax(down)), int(np.argmax(across))
    if down[stem] < CROSS_ARM_REACH * height or across[bar] < CROSS_ARM_REACH * width:
        return False
    return bool(mask[bar, stem])


# -- patch_array_corners ------------------------------------------------------

#: Long edge of the raster a patch array is looked for on. A patch is hundreds of
#: pixels across and its boundary is located from a gradient ridge rather than
#: from a single pixel, so full resolution buys nothing here and costs a minute
#: per capture.
ARRAY_LONG_EDGE = 2400

#: How far a blob's area may sit from the array's own modal area. A patch array is
#: uniform by construction, so anything well off the mode is lettering, a mount
#: screw or two patches merged into one.
ARRAY_AREA_TOLERANCE = (0.55, 1.8)
#: How square a blob must be. Circular patches are square in bounding box too, so
#: this separates patches from labels and rules without caring which shape the
#: chart uses.
ARRAY_ASPECT_TOLERANCE = 0.22
#: How much of its own bounding box a patch fills. Bounded on both sides: the
#: upper bound is what tells a circular patch from a rectangle sitting at the same
#: pitch.
ARRAY_FILL = (0.60, 1.05)

#: Edge percentiles the blob search is swept over. What counts as an edge depends
#: on how much of the frame the chart fills and how much surface texture the print
#: has, so the setting that harvests the most patch-shaped regions without
#: exceeding the array's own size wins -- a self-tuning step with a stated
#: criterion rather than a constant tuned on one capture.
#:
#: **The floor is a statement about how much of the window is chart.** A
#: percentile of the whole window is an edge fraction of the whole window, so a
#: chart filling it wants a high one and a chart that is a third of it -- the rest
#: velvet and masking tape -- wants a low one. It started at 78, which every
#: committed capture satisfied because every one was shot to fill its frame, and the
#: first that was not -- a Wolf Faust IT8.7/2 beside a ColorChecker Passport --
#: found 28 patches there and 138 at the 44th. Sweeping further costs nothing in
#: judgement, since the criterion still chooses, and about 0.1 s in time.
ARRAY_PERCENTILES = np.arange(40.0, 96.0, 2.0)

#: Fewest patches that may be located before a lattice is fitted. A projective map
#: has eight degrees of freedom, so twenty is already more than twice
#: over-determined -- and the residual, which is the real check, means nothing
#: below about that many.
MIN_PATCHES = 20

#: How far the fitted lattice may sit from the patches it was fitted to, as a
#: fraction of the quad's shorter side, before the fit is reported as resting on
#: something that is not the array.
ARRAY_RESIDUAL_LIMIT = 0.01

#: How nearly a grid's patch size must equal its spacing before its patches count
#: as **butted** -- printed edge to edge with no mount between them. A fact about
#: the definition rather than about the capture, so a chart always takes the same
#: path, and what turns on it is only whether the located lattice is allowed to
#: assume it saw the whole array.
BUTTED_RATIO = 0.95

#: How wide a strip is taken across one lattice line, in pitches, when asking
#: whether there is a column of patches there. Comfortably inside the cell, so the
#: strip holds one column and none of its neighbours.
STRIP_WIDTH = 0.7

#: How much variation a strip must carry, against the median of the strips known
#: to be patches, before it counts as one too. A column of an IT8 is twelve
#: different patches down its length; the card beside the array is one level with
#: a printed row letter on it, and the letter is what this has to survive -- so the
#: measure is the spread of the strip's own row means, which a glyph barely moves
#: and twelve patches move a great deal.
STRIP_PATCH_FRACTION = 0.25


def find_patch_array_corners(
    image: LoadedImage, target, extent: Extent
) -> tuple[list, dict]:
    """Whichever of the three patch-array finders this chart's array calls for.

    ``patch_array_corners`` is declared by four shipped targets and they are not
    one problem: a 13 x 10 array of patches with a mount between them, a 22 x 12
    IT8 whose patches are butted, and a 34:1 strip. The key is in the definition
    rather than in the target id: a grid declaring one row is a strip, and cannot
    be fitted as a lattice at all because its centres are collinear and a
    homography through collinear points is degenerate; a grid whose patch size
    equals its spacing is butted, and cannot assume it saw its own ends. A chart of
    a known shape that nothing here has heard of therefore dispatches correctly,
    which keying on the target id could not do.
    """
    grid, _region = _largest_grid(target)
    if min(grid.rows, grid.columns) == 1:
        return find_patch_strip(image, target, extent)
    width, height = grid.patch_size_relative
    across, down = grid.patch_spacing_relative
    butted = (
        (across <= 0 or width / across >= BUTTED_RATIO)
        and (down <= 0 or height / down >= BUTTED_RATIO)
    )
    return (find_patch_block if butted else find_patch_array)(image, target, extent)


def find_patch_array(image: LoadedImage, target, extent: Extent) -> tuple[list, dict]:
    """The placement a chart's own array of patches implies.

    Four shipped targets place on their patch array rather than on a printed mark,
    because they print none: the Avian NGT's two small circles are mounting holes,
    and the IT8 charts carry nothing but the array itself.

    **The fit is made in relative coordinates, not in index space.** The obvious
    route -- fit index space to pixels, then read off the corner patches' outer
    corners -- needs the patch *size* measured as well as the lattice, and the size
    is the hard part: the regions a blob search returns are carved out of a dilated
    edge mask, so each is a few pixels smaller than the patch that made it and by a
    different amount, because a black patch on a black mount has a fainter boundary
    than a white one and loses more of it. The definition already states where every
    patch centre sits in the chart's own frame, so fitting *those* against the found
    centres gives the relative-to-pixel homography directly, and the corners fall
    out of it wherever the target declares them -- inside the array, outside it, or
    nowhere near it.

    That also means the whole array votes. Thirteen by ten patches through eight
    parameters leaves 122 of them as a check rather than an input, and the residual
    is what says the lattice was fitted to the array and not to something else
    shaped a little like it.

    **Patches are found as the regions between edges**, not as regions of some
    lightness, because on every one of these charts some patch is darker than its
    own mount -- and an array found by thresholding is an array missing its blacks,
    which is where a lattice fit is most constrained.
    """
    return _fit_patch_lattice(image, target, extent, grow=False)


def find_patch_block(image: LoadedImage, target, extent: Extent) -> tuple[list, dict]:
    """The placement a solid block of butted patches implies.

    The same fit as :func:`find_patch_array` and one correction, which is the whole
    difference. On an IT8 the patches are **butted**, and where two neighbours
    happen to be the same colour there is no boundary between them: a region finder
    returns 92 of 264, and the ones it merges are not randomly placed -- they
    cluster in the dark and the pale columns, so two or three columns at one end of
    the array go missing entirely. The lattice then fitted to what remains is
    perfectly regular and perfectly wrong, indexing every patch two columns over.
    Measured on the transmissive IT8 that put the placement 480 px out on a chart
    5143 px wide while the fit's own residual stayed at five pixels, which is
    exactly the failure this module exists to avoid: a confident answer read off the
    wrong pixels.

    Nothing in the located patches can detect that, because they are self-
    consistent. What can is the card: **a strip where a column ought to be either
    looks like a column or does not.** Twelve butted patches down a strip give its
    row means a large spread; the card beside the array gives almost none, and the
    printed row letter sitting on it does not change that, which is the property
    being relied on. So the lattice is grown outward one pitch at a time from the
    patches that were found, for as many pitches as the declared array has spare,
    and stops at the first strip that is not a column.
    """
    return _fit_patch_lattice(image, target, extent, grow=True)


def _fit_patch_lattice(
    image: LoadedImage, target, extent: Extent, *, grow: bool
) -> tuple[list, dict]:
    grid, region = _largest_grid(target)
    luma, step = working_luma(image, ARRAY_LONG_EDGE)
    x0, y0, x1, y1 = (int(v / step) for v in extent.box)
    window = luma[y0:y1, x0:x1]

    blobs, percentile = _patch_blobs(window, grid.rows * grid.columns)
    if len(blobs) < MIN_PATCHES:
        raise DetectionRefused(
            f"Only {len(blobs)} patch-shaped regions were found where "
            f"{target.display_name} prints {grid.rows * grid.columns}. Placement on a "
            "patch array needs most of the array visible and in focus."
        )
    # A chart shot a quarter turn round lies with its columns down the frame, and
    # ranking along x would then sort thirteen columns into ten. Which way up the
    # chart is gets decided afterwards, from its own cells and ramp; all the lattice
    # has to settle is which axis the columns run along. The turned fit is tried
    # only when the upright one fails, so a chart that placed before places exactly
    # as it did -- a lower residual is not evidence of the right lattice, which is
    # the lesson of the butted IT8 -- and a square lattice is never turned at all.
    turns = (False, True) if grid.rows != grid.columns else (False,)
    first_refusal = None
    for quarter_turn in turns:
        try:
            residual, corners, short, centres, lattice, _ = _fit_lattice_once(
                blobs[:, :2], grid, x0, y0, step, target,
                window=window if grow else None, quarter_turn=quarter_turn,
            )
        except DetectionRefused as refusal:
            first_refusal = first_refusal or refusal
            continue
        if short <= 0 or residual > ARRAY_RESIDUAL_LIMIT * short:
            first_refusal = first_refusal or DetectionRefused(
                f"The lattice fitted to {len(centres)} patches misses them by "
                f"{residual:.0f} px, more than {ARRAY_RESIDUAL_LIMIT:.0%} of the "
                "chart. What was found is not a regular array of this shape."
            )
            continue
        break
    else:
        raise first_refusal
    if quarter_turn:
        lattice = {**lattice, "columns_run_down_the_frame": True}
    return (
        [(float(x), float(y)) for x, y in corners],
        {
            "patches_located": int(len(centres)),
            "patches_declared": grid.rows * grid.columns,
            "lattice_residual_px": residual,
            "edge_percentile": percentile,
            "grid": region.region_id,
            **lattice,
        },
    )


def _fit_lattice_once(
    found, grid, x0, y0, step, target, *, window, quarter_turn
):
    """One lattice fit, with the columns taken along x or, turned, along y.

    Returns ``(residual, corners, short side, centres, report, quarter_turn)`` and
    leaves judging the residual to the caller, which has both fits to compare.
    """
    indices, lattice = _assign_indices(
        found, grid.rows, grid.columns, window=window, quarter_turn=quarter_turn
    )
    keep = _one_blob_per_cell(indices, found)
    indices, centres = indices[keep], found[keep]
    if len(centres) < MIN_PATCHES:
        raise DetectionRefused(
            f"Only {len(centres)} of the found regions fell in distinct cells of a "
            f"{grid.rows} x {grid.columns} lattice."
        )

    relative = np.array(
        [grid.center_relative(int(row), int(col)) for col, row in indices],
        dtype=np.float64,
    )
    pixels = (centres + (x0, y0)) * step
    homography, _ = cv2.findHomography(relative, pixels, cv2.RANSAC, 3.0 * step)
    if homography is None:
        raise DetectionRefused(
            "The patch-shaped regions found do not lie on one lattice, so they are "
            "not this chart's array."
        )
    predicted = cv2.perspectiveTransform(
        relative.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    residual = float(np.sqrt(np.mean(np.sum((predicted - pixels) ** 2, axis=1))))

    corners = cv2.perspectiveTransform(
        np.asarray(target.fiducials.corner_positions, np.float64).reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    short = min(
        float(np.linalg.norm(corners[1] - corners[0])),
        float(np.linalg.norm(corners[3] - corners[0])),
    )
    return residual, corners, short, centres, lattice, quarter_turn


def _largest_grid(target):
    """The patch grid a placement should be fitted to, and the region holding it.

    The largest, because a chart carrying more than one -- the IT8's twelve-row
    colour array beside its single-row grey ladder -- is placed on the big one, and
    because more patches is a better determined fit whatever the chart.
    """
    grids = [(region.grid, region) for region in target.grid_regions
             if region.grid is not None]
    if not grids:
        raise DetectionRefused(
            f"{target.display_name} declares no patch grid to place from."
        )
    return max(grids, key=lambda pair: pair[0].rows * pair[0].columns)


def _patch_blobs(luma: np.ndarray, expected: int) -> tuple[np.ndarray, float]:
    """Centroids and sizes of everything shaped like a patch."""
    blur = cv2.GaussianBlur(luma.astype(np.float32), (0, 0), 2.0)
    magnitude = np.hypot(
        cv2.Scharr(blur, cv2.CV_32F, 1, 0), cv2.Scharr(blur, cv2.CV_32F, 0, 1)
    )
    best: np.ndarray = np.empty((0, 5))
    best_percentile = 0.0
    for percentile in ARRAY_PERCENTILES:
        edge = (magnitude > np.percentile(magnitude, percentile)).astype(np.uint8)
        edge = cv2.dilate(edge, np.ones((3, 3), np.uint8))
        found = _regions_between(1 - edge)
        if len(best) < len(found) <= expected:
            best, best_percentile = found, float(percentile)
    return best, best_percentile


def _regions_between(mask: np.ndarray) -> np.ndarray:
    """Patch-shaped connected components of ``mask``, as (x, y, w, h, area)."""
    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 4)
    boxes, centres = stats[1:], centroids[1:]
    if len(boxes) == 0:
        return np.empty((0, 5))
    areas = boxes[:, cv2.CC_STAT_AREA].astype(float)
    widths = boxes[:, cv2.CC_STAT_WIDTH].astype(float)
    heights = boxes[:, cv2.CC_STAT_HEIGHT].astype(float)
    seed = _modal_area(areas)
    keep = (areas > seed * ARRAY_AREA_TOLERANCE[0]) & (areas < seed * ARRAY_AREA_TOLERANCE[1])
    keep &= np.abs(widths - heights) / np.maximum(widths, heights) < ARRAY_ASPECT_TOLERANCE
    fill = areas / np.maximum(widths * heights, 1.0)
    keep &= (fill > ARRAY_FILL[0]) & (fill < ARRAY_FILL[1])
    return np.column_stack([centres[keep], widths[keep], heights[keep], areas[keep]])


def _modal_area(areas: np.ndarray) -> float:
    """The area a patch has, taken as the commonest area in the frame.

    The array's defining property is that it repeats: a hundred and thirty
    components share one area to within the print tolerance, and nothing else in a
    capture does. So the mode is the patch, and it survives both a frame component
    larger than every patch put together and a scattering of dust smaller than any
    of them -- neither of which a median or a rank could be told apart from the
    array itself.
    """
    large = areas[areas >= 64.0]
    if len(large) == 0:
        return float(np.median(areas)) if len(areas) else 0.0
    log_area = np.log10(large)
    bins = np.arange(log_area.min(), log_area.max() + 0.08, 0.08)
    counts, _ = np.histogram(log_area, bins=bins)
    if len(counts) == 0:
        return float(np.median(large))
    # Widened by one bin either side: a patch's area varies a little with its own
    # boundary contrast, and a peak straddling a bin edge would read half.
    padded = np.convolve(counts, [1, 1, 1], mode="same")
    peak = int(np.argmax(padded))
    lo, hi = 10 ** bins[max(peak - 1, 0)], 10 ** bins[min(peak + 2, len(bins) - 1)]
    within = large[(large >= lo) & (large <= hi)]
    return float(np.median(within)) if len(within) else float(np.median(large))


def _assign_indices(
    centres: np.ndarray, rows: int, columns: int, *, window: np.ndarray | None = None,
    quarter_turn: bool = False,
) -> tuple[np.ndarray, dict]:
    """Each found centre's (col, row) place in the lattice.

    An array is close enough to axis-aligned on a copy stand or a flatbed that
    ranking by coordinate indexes it, and the projective fit that follows absorbs
    the residual rotation and keystone.

    ``window`` turns on the correction described in :func:`find_patch_block`: with
    it, the lattice is allowed to start before the first patch that was found.
    Without it every located patch is assumed to be in its rightful place, which is
    true of an array whose patches have a mount between them and false of one whose
    patches are butted.

    ``quarter_turn`` takes the columns down the frame and the rows across it, as a
    chart turned a quarter clockwise lies: its first column at the top and its
    first row at the right. That is a rotation rather than a transpose, so the fit
    stays orientation-preserving and the usual reading of which way up applies.
    """
    report: dict = {}
    assigned = []
    axes = (
        ((0, rows, "rows"), (1, columns, "columns")) if quarter_turn
        else ((0, columns, "columns"), (1, rows, "rows"))
    )
    for axis, groups, name in axes:
        lines = _lattice_lines(centres[:, axis], groups)
        if lines is None:
            raise DetectionRefused(
                f"{len(centres)} located regions could not be separated into "
                f"{groups} {name}."
            )
        if window is not None and groups > 1:
            offset = _lattice_offset(window, centres, lines, axis, groups)
            if offset:
                lines = lines - offset * float(np.median(np.diff(lines)))
                report[f"{name}_before_the_first_found"] = offset
        assigned.append(
            np.argmin(np.abs(centres[:, axis, None] - lines[None, :]), axis=1)
        )
    if quarter_turn:
        across, down = assigned
        return np.column_stack([down, rows - 1 - across]), report
    return np.column_stack(assigned), report


def _lattice_offset(
    window: np.ndarray, centres: np.ndarray, lines: np.ndarray, axis: int, groups: int
) -> int:
    """How many lattice lines sit outside the patches that were found.

    Asked of the capture rather than assumed, because the answer cannot be got from
    the located patches: they are self-consistent whether or not the array extends
    past them. A strip is cut where the next line out would be, and it either looks
    like a column of patches or it looks like card. See :func:`find_patch_block` for
    why that is the discriminating question and :func:`_strip_variation` for how it
    is measured.

    Only as many lines as the array has spare are ever tried, so a fully detected
    array cannot be grown at all and this costs nothing on the common case.
    """
    pitch = float(np.median(np.diff(lines)))
    if pitch <= 0:
        return 0
    seen = int(round((centres[:, axis].max() - centres[:, axis].min()) / pitch)) + 1
    spare = groups - seen
    if spare <= 0:
        return 0

    span = (float(centres[:, 1 - axis].min()), float(centres[:, 1 - axis].max()))
    known = np.array([
        _strip_variation(window, line, pitch, axis, span) for line in lines[:seen]
    ])
    floor = STRIP_PATCH_FRACTION * float(np.median(known))

    offset = 0
    for step_out in range(1, spare + 1):
        if _strip_variation(
            window, lines[0] - step_out * pitch, pitch, axis, span
        ) < floor:
            break
        offset = step_out
    return offset


def _strip_variation(
    window: np.ndarray, at: float, pitch: float, axis: int, span: tuple[float, float]
) -> float:
    """How much a strip across one lattice line varies along its own length.

    The spread of its block means, in eight blocks. A column of butted patches is
    several different levels down its length and scores high; the card beside the
    array is one level and scores near zero, and the printed row letter on it -- the
    one thing out there that is not card -- moves one block a little and the spread
    hardly at all.
    """
    half = STRIP_WIDTH * pitch / 2.0
    lo, hi = int(round(at - half)), int(round(at + half))
    a, b = int(round(span[0])), int(round(span[1]))
    if axis == 0:
        lo, hi = max(lo, 0), min(hi, window.shape[1])
        a, b = max(a, 0), min(b, window.shape[0])
        strip = window[a:b, lo:hi]
    else:
        lo, hi = max(lo, 0), min(hi, window.shape[0])
        a, b = max(a, 0), min(b, window.shape[1])
        strip = window[lo:hi, a:b]
    if strip.size == 0 or min(strip.shape) < 2:
        return 0.0
    # Averaged ACROSS the strip and kept along it, so what is measured is how the
    # strip varies down its own length rather than how wide it is.
    along = strip.mean(axis=1) if axis == 0 else strip.mean(axis=0)
    blocks = np.array_split(along, min(8, len(along)))
    return float(np.std([block.mean() for block in blocks]))


def _lattice_lines(values: np.ndarray, groups: int) -> np.ndarray | None:
    """Where each of ``groups`` lines of the lattice sits along one axis.

    **Not a clustering.** A lattice is regular, and that is worth more than any
    general method: the values are split at their own large gaps, the pitch is read
    from the run of resulting group centres, and the full set of line positions is
    then *predicted* from the pitch. A column with no located patch in it -- an
    all-dark column, or one a mount screw crosses -- therefore still gets its place,
    where a clustering forced to return k groups would split a neighbour in two to
    make up the number.

    The lines are anchored on the first group actually seen, so this returns where
    the lattice sits *given* that the first found patch is the first patch. Whether
    that is true is :func:`_lattice_offset`'s question.
    """
    if len(values) < 3 or groups < 1:
        return None
    ordered = np.sort(values)
    span = float(ordered[-1] - ordered[0])
    if span <= 0:
        return None
    if groups == 1:
        return np.array([float(np.mean(ordered))])
    gaps = np.diff(ordered)
    # A gap sits *between* lines if it is a large fraction of the smallest plausible
    # pitch: with ``groups`` lines across ``span`` the pitch is at most
    # span/(groups-1), and half of that separates lines from within-line scatter.
    threshold = 0.5 * span / (groups - 1)
    boundaries = np.flatnonzero(gaps > threshold)
    starts = np.concatenate([[0], boundaries + 1])
    ends = np.concatenate([boundaries + 1, [len(ordered)]])
    seen = np.array([float(np.mean(ordered[a:b])) for a, b in zip(starts, ends)])
    if len(seen) < 2:
        return None
    steps = np.diff(seen)
    unit = float(np.median(steps))
    if unit <= 0:
        return None
    pitch = float(np.median(steps / np.maximum(np.round(steps / unit), 1)))
    if pitch <= 0:
        return None
    origin = float(seen[0] - round((seen[0] - ordered[0]) / pitch) * pitch)
    return origin + pitch * np.arange(groups, dtype=float)


def _one_blob_per_cell(indices: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Which blobs to keep where two landed in one cell.

    The one nearest that cell's own median, since the other is a highlight, a speck
    or a patch split in two by something crossing it.
    """
    keep = np.ones(len(indices), bool)
    seen: dict[tuple[int, int], list[int]] = {}
    for position, cell in enumerate(map(tuple, indices)):
        seen.setdefault(cell, []).append(position)
    for members in seen.values():
        if len(members) == 1:
            continue
        block = centres[members]
        middle = np.median(block, axis=0)
        best = members[int(np.argmin(np.linalg.norm(block - middle, axis=1)))]
        for position in members:
            keep[position] = position == best
    return keep


# -- a patch strip ------------------------------------------------------------

#: Saturation levels the strip's patches are sought at. The answer is the union
#: across all of them rather than the best single one: a brown patch and a cyan one
#: do not appear at the same threshold, and a set missing the strip's last patch
#: shortens the fit at exactly the end it most needs to reach.
STRIP_SATURATIONS = (0.04, 0.06, 0.08, 0.10, 0.14, 0.20)

#: Smallest a patch may be, as a fraction of the capture's shorter side, and how
#: solid its blob must be. The current unit's yellow border is saturated too, and is
#: one component whose bounding box spans the whole card -- it fails on fill, which
#: is the only one of these it fails.
STRIP_MIN_FRACTION = 0.06
STRIP_FILL = 0.85

#: How far a blob's shape may sit from the patch shape the target declares, as a
#: multiplicative factor either way.
#:
#: It used to be the absolute band ``(0.82, 1.22)``, which is this tolerance about a
#: **square** patch -- true of both Golden Thread units measured here and false of the
#: 0.5X, whose patches are half as wide as the 1X's and three quarters as tall, so
#: they stand 1.48 times taller than wide. Against a square band that chart returns
#: no patches at all and the finder refuses having found nothing, which is the same
#: failure :func:`find_patch_strip` had when it read the whole frame instead of its
#: extent: a gate whose constant describes one specimen rather than the definition in
#: front of it.
#:
#: The shape is not a new declaration. ``fiducial_extent_mm`` says how wide and tall
#: the quad really is and ``patch_size_relative`` what fraction of it a patch takes,
#: so their product is the patch in millimetres and, pixels being square, its aspect
#: in the capture. A square patch still yields exactly ``(0.82, 1.22)``.
STRIP_ASPECT_TOLERANCE = 1.22


def declared_patch_aspect(target) -> float | None:
    """How much wider than tall one patch is printed, from the definition alone.

    ``None`` where the target does not declare a physical extent, and the caller then
    falls back to assuming the patch is square -- which is what every gate here did
    unconditionally before, so a definition that says nothing is treated exactly as it
    was.
    """
    extent = getattr(target, "fiducial_extent_mm", None)
    if not extent or not extent[0] or not extent[1]:
        return None
    grids = [region.grid for region in target.grid_regions if region.grid is not None]
    if not grids:
        return None
    grid = max(grids, key=lambda g: g.rows * g.columns)
    width, height = grid.patch_size_relative
    if width <= 0 or height <= 0:
        return None
    return float((width * extent[0]) / (height * extent[1]))

#: Fewest slots that must be occupied before the fit is trusted. The 1X prints
#: thirty patches in the thirty-four slots its quad spans, in two blocks of fifteen
#: with the module between them.
STRIP_MIN_SLOTS = 12

#: How far from the card a cell must read, per channel, to be part of the strip.
STRIP_DEPARTURE = 0.06


def find_patch_strip(image: LoadedImage, target, extent: Extent) -> tuple[list, dict]:
    """The placement a single row of butted patches implies.

    The Golden Thread Object Level Target is a ruled strip whose fiducial quad is
    thirty-four patch pitches long and **one patch tall**, and that shape is what
    makes it a separate finder rather than a case of the others. A lattice fit needs
    points off a line and this array has none; a boundary scan gets the ends of a
    3400 px strip to a fraction of a pixel and its *height* to about two, because
    the older of the two units prints a white keyline around the block that the
    current one does not. On a quad thirty-four times longer than it is tall, two
    pixels of height is the wrong error to carry: every feature off the patch row is
    declared as a fraction of the short side, so the printed rules' bands are twelve
    pixels tall and two pixels of quad height moves them a sixth of their own width.

    So the strip is fitted from the patches themselves, in both axes. Thirty patch
    centres against their slot indices fix the pitch to a tenth of a pixel and the
    row's direction with them; the quad is then the declared number of pitches long,
    one pitch tall, lying on that fitted centre line. It is the same construction
    the definition's own geometry was measured with.

    **Neighbouring patches must not be allowed to merge.** They butt together with
    no gutter, so two adjacent saturated patches threshold into one component whose
    centroid sits on the boundary between them, half a slot from either -- which is
    what the solidity and squareness tests are for.
    """
    grid, region = _largest_grid(target)
    slots = max(grid.rows, grid.columns)
    if image.channels < 3:
        raise DetectionRefused(
            f"{target.display_name} is placed by finding its colored patches, and "
            "this capture carries one color record, so there is no chroma to tell "
            "them from the card."
        )
    # Cut from the extent, sized by the extent, gated by the extent. This alone used
    # to read the whole frame and scale every threshold to it, which is right by
    # accident when the strip fills its capture and wrong by a factor when it does
    # not: on a frame where the strip is a third of the width, the smallest
    # admissible patch came out larger than any patch the chart prints, and the
    # finder refused having found nothing. Every other finder here already works
    # inside its extent; this was the one that did not.
    left, top, right, bottom = extent.box
    step = max(1, int(round(max(right - left, bottom - top) / ARRAY_LONG_EDGE)))
    rgb = np.asarray(
        image.pixels[top:bottom:step, left:right:step, :3], np.float32
    ) / float(image.scale)
    saturation = rgb.max(2) - rgb.min(2)
    smallest = STRIP_MIN_FRACTION * min(rgb.shape[0], rgb.shape[1])
    # Square unless the definition says otherwise. See STRIP_ASPECT_TOLERANCE.
    shape = declared_patch_aspect(target) or 1.0
    low, high = shape / STRIP_ASPECT_TOLERANCE, shape * STRIP_ASPECT_TOLERANCE

    found = []
    for level in STRIP_SATURATIONS:
        mask = cv2.morphologyEx(
            (saturation > level).astype(np.uint8), cv2.MORPH_OPEN,
            np.ones((5, 5), np.uint8),
        )
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        found += [
            (centroids[i][0], centroids[i][1], float(stats[i][3]))
            for i in range(1, count)
            if min(stats[i][2], stats[i][3]) > smallest
            and low < stats[i][2] / max(stats[i][3], 1) < high
            and stats[i][4] > STRIP_FILL * stats[i][2] * stats[i][3]
        ]
    if len(found) < STRIP_MIN_SLOTS:
        raise DetectionRefused(
            f"Only {plural(len(found), 'colored patch', 'colored patches')} "
            f"found where {target.display_name} prints a row of them."
        )
    blobs = np.asarray(found, float)
    tall = float(np.median(blobs[:, 2]))
    middle = float(np.median(blobs[:, 1]))
    blobs = blobs[(np.abs(blobs[:, 2] - tall) < 0.25 * tall)
                  & (np.abs(blobs[:, 1] - middle) < 0.5 * tall)]
    if len(blobs) < STRIP_MIN_SLOTS:
        # Reachable only since this finder began working inside its extent, and
        # only on a region that is not this chart: enough saturated blobs to pass
        # the count above, none of which agree on a height or a row. It has to be
        # a refusal rather than an empty array reaching the fit below -- a finder
        # returns a reason or a placement, and never raises.
        raise DetectionRefused(
            f"{len(blobs)} of the {len(found)} colored regions found sit at one "
            "height in one row, so they are not this chart's strip."
        )

    along_axis = 0 if np.ptp(blobs[:, 0]) >= np.ptp(blobs[:, 1]) else 1
    along = blobs[:, along_axis]
    across = blobs[:, 1 - along_axis]

    centres, outline = _declared_strip_centres(target)
    if len(centres) < STRIP_MIN_SLOTS:
        raise DetectionRefused(
            f"{target.display_name} declares only {len(centres)} patches in a row, "
            "which is too few to fit a strip to."
        )
    # Where the row starts and ends is asked of the capture, not of the patches
    # that were found: they are equally consistent with being the whole row and
    # with being most of it, because a saturation search walks past every neutral
    # step. The walk below steps outward from the outermost found patch one pitch
    # at a time for as long as what is printed there is not the card.
    smallest = _declared_cell(centres)
    cells = max(int(round((outline[1] - outline[0]) / smallest)), 2)
    pitch = _strip_pitch(along, cells)
    if pitch is None:
        raise DetectionRefused(
            "The colored patches found do not sit at one regular pitch, so they "
            "are not this chart's strip."
        )
    seen = int(round(float(np.ptp(along)) / pitch)) + 1
    # In CELLS, which is what ``seen`` counts, and not in patches: the 1X's row is
    # thirty-four cells of which thirty carry a patch and four are the resolution
    # module. Bounding the walk by the patch count let it take no steps at all on a
    # chart where one cell was unaccounted for, and left the row's end half a patch
    # short -- which moved both Golden Thread placements 113 px, a whole patch.
    spare = max(cells - seen, 0)
    # Only the near end is walked for. How long the row is after that is the
    # chart's own declaration, not something to probe for a second time: asking the
    # capture at both ends over-determines the scale and the two answers disagree
    # -- one cell of slack at each end stretched the 1X by 2.4%, which is half a
    # patch by the far end and enough to shift the whole correspondence by one.
    walked = _strip_cells_beyond(rgb, float(along.min()), -pitch, blobs, along_axis, spare)
    # An UPPER BOUND on how many patches precede the first one found, not a count of
    # them. The walk asks the capture whether each cell holds something other than
    # the card, and both Golden Thread units print a solid disc just beyond the row
    # -- which departs from the card exactly as a patch does and sits one cell out.
    # Where the bound leaves room for it the walk takes it and the row starts a
    # patch early. Every position up to the bound is offered to the fit instead, and
    # the one the located patches actually agree with wins.
    starts = [float(along.min()) - (walked + 0.5) * pitch]
    fitted = _fit_declared_strip(
        along, across, centres, outline, starts, pitch / smallest
    )
    if fitted is None:
        raise DetectionRefused(
            "The colored patches found do not sit where "
            f"{target.display_name} prints its row of them."
        )
    line_along, line_across, matched, residual = fitted

    direction = np.array([line_along[0], line_across[0]], float)
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise DetectionRefused("the located patches give the strip no direction")
    direction /= norm
    # Half the quad's height across the strip. The quad's proportions are the
    # chart's own -- ``fiducial_extent_mm`` where it states one, and the square-cell
    # inference otherwise -- so this is half the fitted length divided by them. It
    # used to be half a pitch, which is the same number only where a patch is
    # square: the 0.5X Object Level Target's stand 1.49 times taller than wide, and
    # half a pitch there builds a quad 270 px tall across a patch row 388 px tall.
    aspect = implied_quad_aspect(target) or 1.0
    half = np.array([-direction[1], direction[0]]) * (norm / aspect / 2.0)
    ends = [
        np.array([np.polyval(line_along, t), np.polyval(line_across, t)])
        for t in (outline[0], outline[1])
    ]
    block = np.array([ends[0] - half, ends[1] - half, ends[1] + half, ends[0] + half])
    if along_axis == 1:
        block = block[:, ::-1]
    quad = _quad_from_block(
        block * step + np.array([left, top], float), target,
        origin=(outline[0], outline[2]),
        span=(outline[1] - outline[0], outline[3] - outline[2]),
    )
    pitch = abs(float(line_along[0])) * float(np.median(np.diff(np.sort(centres))))
    return (
        [(float(x), float(y)) for x, y in quad],
        {
            "patches_located": int(matched),
            "patches_declared": len(centres),
            "blocks": len({r.region_id for r in target.grid_regions
                           if r.grid is not None and r.grid.rows == 1}),
            "pitch_px": pitch * step,
            "lattice_residual_px": residual * step,
            "grid": region.region_id,
        },
    )


#: How near a found patch must land to a declared one, as a fraction of the
#: smallest gap between two declared centres, before the two are called the same
#: patch. Generous, because the fit that predicts the position is the thing being
#: tested and a third of a patch is still unambiguous.
STRIP_MATCH_TOLERANCE = 0.35

#: Fewest DIFFERENT declared patches an assignment must land on. Two fix the line
#: and are what the candidate was built from, so everything above that is the
#: check: six leaves four patches voting on a fit that two determined.
STRIP_MIN_DISTINCT = 6

#: How much worse than the closest-fitting correspondence another may fit and still
#: be preferred for explaining more patches. The floor beside it -- two percent of a
#: patch -- keeps the comparison from being decided by noise when the best fit is
#: already essentially exact.
#: How near a whole number of pitches a gap between two found patches must fall
#: before it counts as agreeing with that pitch.
STRIP_PITCH_TOLERANCE = 0.12


def _declared_strip_centres(target) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Where the chart says its patches sit along the strip, and the row's outline.

    Read from the *patches* rather than from a grid's column count, and across
    every one-row grid the target declares rather than the largest. Both of those
    are the same correction: a slot a chart leaves empty is not a patch and nothing
    will ever be found in it -- the 1X prints thirty patches in the thirty-four
    slots its quad spans, with the resolution module in four of them -- and a strip
    need not be one block at one pitch.

    **The 0.5X is two blocks**, 15 patches each, at pitches 1.4% apart, separated by
    a module spanning 4.37 pitches. No integer slot count describes that, which is
    why the fit below is against these positions and not against an index.
    """
    centres: list[float] = []
    u0, u1 = float("inf"), float("-inf")
    v0, v1 = float("inf"), float("-inf")
    for patch in target.patches:
        region = target.region_for(patch)
        grid = region.grid
        if grid is None or grid.rows != 1:
            continue
        cu, cv = grid.center_relative(patch.row, patch.col)
        centres.append(cu)
        ou, ov = grid.grid_origin_relative
        su, sv = grid.extent_relative
        u0, u1 = min(u0, ou), max(u1, ou + su)
        v0, v1 = min(v0, ov), max(v1, ov + sv)
    if not centres:
        return np.zeros(0), (0.0, 1.0, 0.0, 1.0)
    return np.array(sorted(centres), float), (u0, u1, v0, v1)


def _strip_pitch(along: np.ndarray, cells: int) -> float | None:
    """The row's pitch, from the gaps between the patches that were found.

    Not from a lattice fit. A lattice asked for thirty-four groups when nine
    patches were found locks onto a multiple of the real pitch -- 66 px against a
    true 53 on two of the twenty Odonata captures -- and everything downstream is
    then measured in the wrong unit: the walk steps to the wrong cells and the
    scale search is bounded around a wrong guess.

    The patches butt together, so *some* pair of the ones found is one pitch apart
    and every gap is a whole number of pitches. That makes the pitch the smallest
    spacing all the gaps agree on, which is what is searched for here -- and a gap
    of five pitches across the resolution module is then evidence for the same
    answer rather than a hole the fit has to bridge.
    """
    x = np.sort(np.asarray(along, float))
    if len(x) < 2:
        return None
    # The blob search keeps the union over six saturation levels, so one patch
    # arrives several times over and those near-zero spacings are not gaps between
    # patches. They have to be merged before anything is measured from the gaps,
    # and merging needs a length: the found patches span at most the whole row, so
    # their extent over the row's cell count is a floor under the true pitch and a
    # quarter of that is well inside it. Filtering by a fraction of the MEDIAN gap
    # instead does not work -- where the duplicates outnumber the patches, as they
    # do on half of these captures, the median IS a duplicate and the pitch comes
    # out as a fifth of a pixel.
    apart = 0.25 * float(np.ptp(x)) / max(cells, 1)
    kept = [x[0]]
    for value in x[1:]:
        if value - kept[-1] > apart:
            kept.append(value)
    gaps = np.diff(np.asarray(kept, float))
    if len(gaps) == 0:
        return None
    best = None
    for candidate in np.unique(gaps):
        if candidate <= 0:
            continue
        multiples = gaps / candidate
        whole = np.round(multiples)
        agree = (whole >= 1) & (np.abs(multiples - whole) < STRIP_PITCH_TOLERANCE)
        if not agree.any():
            continue
        # The LARGEST spacing the gaps agree on, not the smallest: half a pitch
        # divides everything a whole pitch divides, and so does a quarter, so
        # preferring small takes the search straight down to noise. What is wanted
        # is the gaps' common divisor, and among equally agreeing candidates that
        # is the biggest one -- five pitches across the module fails because the
        # single-pitch gaps are not multiples of it.
        score = (int(agree.sum()), float(candidate))
        if best is None or score > best[0]:
            best = (score, candidate)
    if best is None:
        return None
    candidate = best[1]
    whole = np.round(gaps / candidate)
    keep = (whole >= 1) & (np.abs(gaps / candidate - whole) < STRIP_PITCH_TOLERANCE)
    return float(gaps[keep].sum() / whole[keep].sum())


def _declared_cell(centres: np.ndarray) -> float:
    """How far apart two neighbouring patches sit, in the chart's own frame.

    The MEDIAN gap, not the smallest. Most gaps between declared centres are one
    cell and the exceptions are exceptional in both directions: a strip's module
    leaves one gap of several cells, and a patch the block cut short leaves one
    *smaller* than a cell -- the 0.5X's patch 15 is two thirds the width of its
    neighbours, so taking the minimum made a cell two thirds of what a cell is,
    and with it the match tolerance and the cell count the walk is bounded by.
    """
    if len(centres) < 2:
        return 1.0
    return float(np.median(np.diff(centres)))


def _fit_declared_strip(
    along: np.ndarray, across: np.ndarray, centres: np.ndarray,
    outline: tuple[float, float, float, float], starts: list[float], guess: float,
):
    """Map the chart's declared patch positions onto the ones that were found.

    Two things are needed and they come from different places. **Where the row
    begins** is asked of the capture -- ``starts``, from a walk outward from the
    outermost patch found -- because the located patches cannot answer it: they are
    equally consistent with being the whole row and with being most of it, and on a
    row at one pitch throughout, shifting every one of them a slot along fits
    exactly as well. That walk is the older Golden Thread unit's placement, which
    lands 113 px out without it. It bounds the answer rather than giving it: both
    units print a disc a cell beyond the row, and it departs from the card exactly
    as a patch does.

    **How long the row is** is then searched: each declared patch in turn is tried
    as the one the outermost found patch is, and the assignment putting the most of
    the rest on a declared position, most closely, wins. ``guess`` is the scale the
    observed pitch implies and only bounds that search.

    The search exists because a single pitch cannot describe every strip. The 0.5X
    is two blocks of fifteen at pitches 1.4% apart, separated by a module spanning
    4.37 of them; taking the scale from the observed pitch alone leaves it 1.8%
    long, which is half a patch by the far end and misassigns everything there.

    Returns the two fitted lines, how many declared patches were matched, and the
    residual.
    """
    if guess == 0 or len(centres) < 2:
        return None
    smallest = _declared_cell(centres)
    order = np.argsort(along)
    x, y = along[order], across[order]
    candidates = []
    for start in starts:
        far = x[-1] if abs(x[-1] - start) >= abs(x[0] - start) else x[0]
        for last in range(len(centres)):
            reach = centres[last] - outline[0]
            if reach == 0:
                continue
            scale = (far - start) / reach
            if not 0.75 <= abs(scale / guess) <= 1.25:
                continue
            found = _strip_assignment(x, y, centres, start, scale, outline, smallest)
            if found is not None:
                candidates.append(found)
    if not candidates:
        return None
    # How WELL the patches fit chooses, and how MANY of them only breaks the tie.
    # The other way round loses: the match tolerance is a third of a patch, so a
    # scale a percent long still collects most of the row -- it collects it badly,
    # at four times the residual, and counting alone cannot see that.
    return min(candidates, key=lambda c: (round(c[3], 3), -c[2]))


def _strip_assignment(x, y, centres, start, scale, outline, smallest):
    """One candidate scale, matched and refitted. ``None`` where too little sticks."""
    # Matched ONCE, at the anchored start, and then fitted. Re-matching against the
    # fitted line and going round again looks like a refinement and is not: on a row
    # whose patches sit at one pitch, an assignment shifted a slot along fits just
    # as tightly, so the loop is free to walk off the start the capture established
    # and does -- 14 delta-E instead of 2.6, at a residual that still reads clean.
    # What the anchor decides, the fit is not allowed to reopen.
    offset = start - scale * outline[0]
    predicted = scale * centres + offset
    gaps = np.abs(x[:, None] - predicted[None, :])
    nearest = gaps.argmin(axis=1)
    near = gaps[np.arange(len(x)), nearest]
    hit = near < STRIP_MATCH_TOLERANCE * abs(scale) * smallest
    chosen = nearest[hit]
    keys = np.array(sorted(set(chosen.tolist())), int)
    # The blob search runs at six saturation levels and keeps the union, so one
    # patch arrives several times over. Those are the same patch and are averaged
    # rather than being allowed to vote several times -- what counts is how many
    # DIFFERENT declared patches an assignment lands on.
    if hit.sum() < STRIP_MIN_SLOTS or len(keys) < STRIP_MIN_DISTINCT:
        return None
    key = centres[keys]
    mid_along = np.array([x[hit][chosen == k].mean() for k in keys])
    mid_across = np.array([y[hit][chosen == k].mean() for k in keys])
    line_along = np.polyfit(key, mid_along, 1)
    line_across = np.polyfit(key, mid_across, 1)
    residual = float(np.std(np.hypot(
        np.polyval(line_along, key) - mid_along,
        np.polyval(line_across, key) - mid_across,
    )))
    return line_along, line_across, len(keys), residual


def _strip_cells_beyond(
    rgb: np.ndarray, first: float, stride: float, blobs: np.ndarray,
    along_axis: int, spare: int,
) -> int:
    """How many cells of the strip lie past the first patch that was found.

    Asked of the capture, because the located patches cannot answer it: they are
    equally consistent with being the whole strip and with being most of it. A cell
    is on the strip if what is printed there is not the **card**, which is sampled
    from the band just outside the row itself -- so a neutral patch the saturation
    search missed still reads as a patch here, which is the entire point.

    One missing cell is stepped over. A patch that happens to match the card exactly
    would otherwise end the strip early, and the cost of allowing it is only that a
    mark one cell beyond the true end could extend it -- which the declared cell
    count bounds, since the strip is never grown past the number of cells its own
    definition says are unaccounted for.
    """
    if spare <= 0:
        return 0
    middle = float(np.median(blobs[:, 1]))
    tall = float(np.median(blobs[:, 2]))
    span = (float(blobs[:, along_axis].min()), float(blobs[:, along_axis].max()))
    card = _strip_card(rgb, middle, tall, span, along_axis)
    if card is None:
        return 0

    reached = 0
    for step_out in range(1, spare + 1):
        at = first + step_out * stride
        centre = (at, middle) if along_axis == 0 else (middle, at)
        # The cell itself, AND the boundary back to the cell before it. The row's
        # patches butt together, so the line between two of them is patch on both
        # sides and departs from the card as they do; anything printed beyond the
        # row's end has card between it and the row, and that is what the second
        # test finds. Both Golden Thread units print a solid disc exactly one cell
        # past the row, so position alone cannot exclude it -- it looks like the
        # next patch in every way except that the row does not reach it.
        joint = at - 0.5 * stride
        seam = (joint, middle) if along_axis == 0 else (middle, joint)
        if (_departs(rgb, centre, abs(stride), tall, card)
                and _departs(rgb, seam, abs(stride), tall, card)):
            reached = step_out
        elif step_out - reached > 1:
            break
    return reached


def _strip_card(rgb, middle: float, tall: float, span, along_axis: int):
    """The colour of the card the strip is printed on, from beside the strip."""
    limit = rgb.shape[0] if along_axis == 0 else rgb.shape[1]
    lo, hi = int(round(middle + 0.8 * tall)), int(round(middle + 1.5 * tall))
    if hi > limit:
        lo, hi = int(round(middle - 1.5 * tall)), int(round(middle - 0.8 * tall))
    lo, hi = max(lo, 0), min(hi, limit)
    if hi - lo < 2:
        return None
    a, b = int(round(span[0])), int(round(span[1]))
    band = rgb[lo:hi, a:b] if along_axis == 0 else rgb[a:b, lo:hi]
    if band.size == 0:
        return None
    return np.median(band.reshape(-1, band.shape[-1]), axis=0)


def _departs(rgb, centre, pitch: float, tall: float, card) -> bool:
    """Whether what is printed at one cell is something other than the card."""
    half_x, half_y = 0.3 * pitch, 0.3 * tall
    x0, x1 = int(round(centre[0] - half_x)), int(round(centre[0] + half_x))
    y0, y1 = int(round(centre[1] - half_y)), int(round(centre[1] + half_y))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, rgb.shape[1]), min(y1, rgb.shape[0])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    # The MEDIAN, so this asks whether most of the cell is something other than the
    # card rather than whether the cell's average has been pulled off it. Both
    # Golden Thread units print a solid disc a cell beyond the row: it overlaps
    # under a fifth of the sampled box, being smaller than a patch and sitting high
    # in the row, and that fifth is dark enough to move the mean past the threshold
    # and start the row a patch early. It does not move the median.
    cell = np.median(rgb[y0:y1, x0:x1].reshape(-1, rgb.shape[-1]), axis=0)
    return bool(np.max(np.abs(cell - card)) > STRIP_DEPARTURE)


def _quad_from_block(
    block: np.ndarray, target, *, origin: tuple[float, float], span: tuple[float, float]
) -> np.ndarray:
    """The fiducial quad implied by the located outline of a patch block.

    The definition states where that outline sits in the chart's own frame -- its
    top-left corner and its size -- so four located corners give a homography from
    relative to pixel, and the fiducial corners come out of it wherever the target
    declares them.

    The outline is passed in rather than read off one grid, because a strip may be
    declared as several blocks and the outline is then their union.
    """
    origin_u, origin_v = origin
    span_u, span_v = span
    if span_u <= 0 or span_v <= 0:
        raise DetectionRefused("the declared patch grid has no extent")
    outline = np.array([
        [origin_u, origin_v],
        [origin_u + span_u, origin_v],
        [origin_u + span_u, origin_v + span_v],
        [origin_u, origin_v + span_v],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(outline, block.astype(np.float32))
    return cv2.perspectiveTransform(
        np.asarray(target.fiducials.corner_positions, np.float64).reshape(-1, 1, 2),
        matrix,
    ).reshape(-1, 2)


# -- dispatch -----------------------------------------------------------------


def find_whole_frame(image: LoadedImage, target, extent: Extent) -> tuple[list, dict]:
    """The corners of the frame itself.

    For the flat-field 'target', which is not a chart: a white sheet meant to fill
    the frame, whose fiducials are the corners of the *area to be measured*. There
    is nothing printed to find, and the whole frame is both the right default and
    the one the operator adjusts inward when the sheet does not quite reach it.
    """
    width, height = float(image.width - 1), float(image.height - 1)
    return (
        [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)],
        {"note": "the whole frame; there is nothing printed here to find"},
    )


def _outwardness(group: list) -> float:
    """How far a group's four marks sit into the corners of the frame."""
    quad = group[0][0]
    centre = quad.mean(axis=0)
    return float(np.sum(np.abs(quad - centre)))


def find_module_crosshairs(
    image: LoadedImage, target, extent: Extent
) -> tuple[list, dict]:
    """The outermost registration crosshair on each of the four corner modules.

    :func:`find_printed_crosshairs` with one preference changed, because this chart
    prints thirty-six of the marks the others print four of, and its definition
    names the outermost. See :func:`_steadiest`.
    """
    return find_printed_crosshairs(image, target, extent, prefer="outermost")


#: Finder per declared ``fiducials.method``. Several methods share one: three
#: charts place on printed crosshairs, one on the crosses of a microform chart and
#: one on the outermost of thirty-six crosshairs, and all five are the same plus
#: found the same way. ``patch_array_corners`` reaches a dispatcher of its own,
#: because the four targets declaring it need three different finders and what
#: separates them is the shape of the array rather than the name of the method.
FINDERS = {
    "printed_frame_corners": find_printed_frame,
    "printed_discs": find_printed_discs,
    "printed_crosshairs": find_printed_crosshairs,
    "corner_crosses": find_printed_crosshairs,
    "module_corner_crosshairs": find_module_crosshairs,
    "patch_array_corners": find_patch_array_corners,
    "chart_corners": find_whole_frame,
}


def _detect_placement_once(image: LoadedImage, target, *, refine: bool = True) -> Detection:
    """Find where the chart sits in the capture, and refine what is found.

    The two halves are deliberately unequal. Detection has to land within
    ``MAX_SHIFT_FRACTION`` -- five percent of the chart's shorter side -- and
    refinement takes it from there to a fraction of a pixel on every chart printing
    features it can use. A finder that is merely close is therefore doing its whole
    job, and one that is confidently wrong is the only real failure.
    """
    method = target.fiducials.method
    finder = FINDERS.get(method)
    # Before the extent pass, not after it. Whether a finder exists is a property
    # of the definition, so this refusal is knowable without touching a pixel --
    # which lets a caller ask for a placement it knows cannot be made and get the
    # answer immediately, rather than paying for a scan of the frame first.
    if finder is None:
        return Detection(
            method=method,
            reason=(
                f"{target.display_name} places on its {method.replace('_', ' ')}, "
                "which nothing here can find yet. Place the chart by hand."
            ),
        )

    return _place_in(image, target, chart_extent(image), refine=refine)


#: How far outside a placement made on ANOTHER frame a chart is allowed to have
#: moved, as a fraction of that placement's longer side.
#:
#: Measured rather than chosen: across the twenty Smithsonian Odonata captures the
#: Object Level Target's position varies by 186 px in x and 46 px in y on an
#: 11608 x 9108 frame, against a placement 8906 px long -- 2.1% of its own length.
#: Five percent is twice that, and the cost of being generous is small because the
#: region only has to be tight enough for the finders' size gates: at this margin
#: the smallest patch the strip finder will admit is 77 px against the 267 px the
#: chart prints.
HINT_MARGIN = 0.05


def extent_from_corners(
    image: LoadedImage, corners, *, margin: float = HINT_MARGIN
) -> Extent:
    """Where to look, from a placement made on a different frame.

    A batch of production captures is the case this exists for: the chart is in
    the same place in every frame, having been laid on the copy stand once, and
    the operator has checked exactly one placement. That placement is worth far
    more than its four corners -- it says where to look in the other ninety-nine.

    **It narrows the search and never replaces it.** Every frame is still detected
    and refined inside the region, so a chart that shifted is measured where it
    actually is rather than where the first one was, and a frame missing its chart
    refuses rather than returning the seed. What the region buys is that the search
    is possible at all: :func:`chart_extent` returns the whole frame on a capture
    whose chart is a sixth of it against a busy subject, and every finder's size
    gate is a fraction of the extent, so the whole-frame answer puts the smallest
    admissible patch above the largest the chart prints.
    """
    quad = np.asarray(corners, float).reshape(-1, 2)
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    pad = margin * max(x1 - x0, y1 - y0)
    box = (
        max(int(x0 - pad), 0), max(int(y0 - pad), 0),
        min(int(x1 + pad), image.width), min(int(y1 + pad), image.height),
    )
    luma, step = working_luma(image, EXTENT_LONG_EDGE)
    inside = luma[int(box[1] / step):int(box[3] / step),
                  int(box[0] / step):int(box[2] / step)]
    return Extent(
        box=box,
        level=_card_level(inside) if inside.size else 0.5,
        isolated=True,
        reason=(
            f"a region {box[2] - box[0]} x {box[3] - box[1]} px around a placement "
            "checked on another capture"
        ),
    )


def _detect_placement_near_once(
    image: LoadedImage, target, corners, *, margin: float = HINT_MARGIN,
    refine: bool = True,
) -> Detection:
    """:func:`detect_placement`, told roughly where the chart sits.

    Same finders, same refinement, same refusals; only the extent differs. See
    :func:`extent_from_corners` for why a batch needs one.
    """
    method = target.fiducials.method
    if FINDERS.get(method) is None:
        return Detection(
            method=method,
            reason=(
                f"{target.display_name} places on its {method.replace('_', ' ')}, "
                "which nothing here can find yet. Place the chart by hand."
            ),
        )
    extent = extent_from_corners(image, corners, margin=margin)
    x0, y0, x1, y1 = extent.box
    if x1 <= x0 or y1 <= y0:
        # The seed was checked on a larger frame and the region it names lies
        # wholly past this one's edge. Clipped, it is a box of negative size, and
        # handed on it reached a finder as an empty image and raised from inside
        # OpenCV -- the one refusal in a batch nobody could read.
        return Detection(
            method=method,
            reason=(
                f"The seed placement lies outside this {image.width} x "
                f"{image.height} px frame, so this capture is not framed like "
                "the one the seed was checked on."
            ),
        )
    return _place_in(image, target, extent, refine=refine)


def _place_in(image: LoadedImage, target, extent: Extent, *, refine: bool) -> Detection:
    """:func:`detect_placement`, given the extent to look in.

    Split out because reading a frame that holds several charts is this function
    called once per candidate region, and nothing else about a placement changes
    when there is more than one chart in the picture.
    """
    method = target.fiducials.method
    finder = FINDERS[method]
    try:
        found, diagnostics = finder(image, target, extent)
        corners, ordering = order_corners(
            found, target.fiducials.corner_positions,
            cell_aspect=implied_quad_aspect(target),
        )
    except DetectionRefused as refusal:
        return Detection(
            method=method, finder=finder.__name__, extent=extent, reason=str(refusal),
        )
    diagnostics.update(ordering)
    notes = list(diagnostics.pop("assignment_notes", []))
    # The geometry has said all it can. Where more than one assignment survives it,
    # the two are half a turn apart and only the chart's own contents can separate
    # them -- so the ramp is read, and where that says nothing the note the geometry
    # left standing is the honest answer.
    tied = diagnostics.pop("assignment_tied", [corners])
    if len(tied) > 1:
        corners, settled, settled_notes = _settle_orientation(image, target, tied)
        diagnostics.update(settled)
        if settled_notes is not None:
            notes = settled_notes
    detection = Detection(
        corners=corners, found=True, method=method, finder=finder.__name__,
        extent=extent, diagnostics=diagnostics, notes=notes,
        reason=f"Placed automatically from the chart's {method.replace('_', ' ')}.",
    )
    if refine:
        _refine(image, target, detection)
    return detection


def _detect_placements_once(
    image: LoadedImage, target, *, refine: bool = True, most: int = MOST_CHARTS
) -> list[Detection]:
    """Every placement of ``target`` this frame supports, in reading order.

    A capture holding four QA-62 cards is four placements and, downstream, four runs
    sharing one decode. It is deliberately not one run with four charts in it: a run
    is one capture, one target, one placement, one verdict, and widening it would
    make every record in the library a different shape. ``session_from_records``
    already exists to combine several runs into a system-level verdict, which is the
    question a board of charts is asking.

    **Which candidate regions are offered is a separate and generous question** --
    see :func:`chart_extents`. Two things then decide, and the second is the one that
    matters.

    **A finder has to place a chart in the region**, which most candidates fail by
    refusing outright, and two placements of one chart from two overlapping regions
    count once.

    **Each surviving placement has to fit the capture**, by
    :func:`fits_the_capture` -- the same three gates that decide whether a target may
    be *named*, for the same reason and against the same failure. The failure here is
    specific and was watched happening: with four cards on a board, the board is
    itself a candidate region, and asked for four crosshairs near *its* corners the
    finder returns one mark from each of four different cards. Four marks of one
    printed size, agreeing across a run of thresholds, and a perfectly valid quad
    spanning the whole board -- nothing about it looks wrong until the numbers come
    out. What says it is wrong is the chart's own contents: read at that placement
    the QA-62's twenty-step ramp scores -0.37 where a real card scores +1.

    Geometry was tried first and does not work. A quad made of one corner from each
    of four charts does not enclose any of them -- it is inset from all four -- so
    "a chart does not contain another chart" never fires; and where it does fire it
    points the wrong way, since the ISA 4x5's own correct placement encloses a
    partial one found inside it.

    **The gate applies only when there is more than one placement to choose
    between.** A lone placement is returned exactly as :func:`detect_placement`
    would return it, ungated: the gate needs a readable neutral ramp, every shipped
    target carrying patches has one but a custom definition need not, and refusing a
    single correct placement for want of a ramp would be a worse failure than the
    one this prevents. Where several are found and none fits, the single-chart
    answer is returned rather than a shortlist, because "several charts, none of
    which is this one" is not a result anything downstream can use.
    """
    method = target.fiducials.method
    if FINDERS.get(method) is None:
        return [_detect_placement_once(image, target, refine=refine)]

    placed: list[Detection] = []
    for extent in chart_extents(image, most=most):
        detection = _place_in(image, target, extent, refine=refine)
        if not detection.found or detection.corners is None:
            continue
        quad = np.asarray(detection.corners, dtype=np.float64)
        if any(_same_quad(quad, np.asarray(kept.corners, float)) for kept in placed):
            continue
        placed.append(detection)

    if len(placed) < 2:
        return placed or [_detect_placement_once(image, target, refine=refine)]

    kept = []
    for detection in placed:
        candidate = _score_candidate(image, target, [detection.corners])
        detection.diagnostics["fit"] = candidate.explain()
        if fits_the_capture(candidate):
            kept.append(detection)
    if not kept:
        return [_detect_placement_once(image, target, refine=refine)]
    return sorted(kept, key=lambda d: _reading_order(np.asarray(d.corners, float)))


def _quad_scale(quad: np.ndarray) -> float:
    """A quad's own size, as the mean distance of its corners from its centre."""
    return float(np.mean(np.linalg.norm(quad - quad.mean(axis=0), axis=1)))


def _same_quad(first: np.ndarray, second: np.ndarray) -> bool:
    """Whether two placements are one chart found from two candidate regions."""
    scale = min(_quad_scale(first), _quad_scale(second))
    return bool(np.all(np.linalg.norm(first - second, axis=1) < 0.05 * scale))


def _reading_order(quad: np.ndarray) -> tuple[float, float]:
    """Top to bottom, then left to right, banded so a row is not split by a wobble.

    Charts laid out on a board are rarely level to the pixel, so sorting on raw y
    interleaves the top and bottom rows. Banding by the chart's own height puts every
    chart within one of its own heights of another on the same row.
    """
    centre = quad.mean(axis=0)
    band = max(_quad_scale(quad), 1.0)
    return (round(float(centre[1]) / band), float(centre[0]))


def _settle_orientation(image: LoadedImage, target, tied: list):
    """Which of the assignments the geometry left standing the chart's own ramp supports.

    Returns the chosen corners, what to record, and -- when the reading was decisive
    -- the notes to replace the geometry's own caution with. Where the ramp cannot be
    read, or reads no better one way than the other, nothing is replaced: the
    placement stays as the geometry left it and so does its warning, because a
    coin toss dressed as a measurement is worse than an admitted coin toss.

    **The ramp is read two ways, and the second only when the first is silent.**
    Counting adjacent steps (:func:`_ramp_from`) is what separates the ColorChecker
    SG's half turn, where a rank correlation reads +1.00 and +0.96. The QA-62 is the
    opposite case: its ring is one monotone spiral, so a quarter turn moves every
    step five places along and breaks a single adjacent pair -- +1.00 against +0.89,
    inside the margin -- while the ladder as a whole is scrambled, and its rank
    correlation reads +1.00 against -0.13 and -0.50 on all four cards of the board.
    Each reading is held to the same floor and margin, and the second is consulted
    only where the first decided nothing, so no chart the first test settles can be
    settled differently by the second.
    """
    sampled = [(_sample(image, target, corners), corners) for corners in tied]
    readings = []
    for name, read in (("adjacent steps", _ramp_from), ("rank order", _ramp_rank)):
        scores = [
            (score, corners) for score, corners in (
                (None if sample is None else read(target, sample), corners)
                for sample, corners in sampled
            ) if score is not None
        ]
        if not scores:
            continue
        scores.sort(key=lambda entry: entry[0], reverse=True)
        (best_score, best_corners), rest = scores[0], scores[1:]
        runner_up = rest[0][0] if rest else None
        readings.append((name, best_score, runner_up))
        if best_score >= RAMP_AGREEMENT_FLOOR and (
            runner_up is None or best_score - runner_up >= RAMP_AGREEMENT_MARGIN
        ):
            return (
                best_corners,
                {
                    "orientation_from_ramp": (
                        f"the chart's neutral ramp runs the way it is declared to "
                        f"by {name} ({best_score:+.2f}) in this orientation and "
                        f"{runner_up:+.2f} in the next best"
                        if runner_up is not None else
                        f"the chart's neutral ramp agrees with its declaration by "
                        f"{name} ({best_score:+.2f})"
                    ),
                    "ramp_agreement": best_score,
                },
                # No note. Settled means nothing is wrong, and a sentence saying so
                # on every run is the method disclosure the warning review cut; the
                # reading is kept above, in the diagnostics. An empty list still
                # replaces the "assumed" note the geometry left, which is the point.
                [],
            )

    if not readings:
        return tied[0], {"orientation_from_ramp": "the ramp could not be read"}, None
    _, best_score, runner_up = readings[0]
    return (
        tied[0],
        {
            "orientation_from_ramp": "; ".join(
                f"by {name} the ramp reads {best:+.2f} one way up and {second:+.2f} "
                "the other, which does not separate them"
                if second is not None else
                f"by {name} the ramp reads only {best:+.2f}, which decides nothing"
                for name, best, second in readings
            ),
            "ramp_agreement": best_score,
        },
        None,
    )


def _ramp_rank(target, sampled) -> float | None:
    """The ladder's rank correlation with its declaration, from +1 to -1.

    The second of the two readings :func:`_settle_orientation` takes, and blind
    where the first sees: it is what cannot separate the ColorChecker SG's half
    turn, and what does separate the QA-62's quarter turn. Spearman's coefficient
    over the pooled steps of :func:`_ramp_steps`, written out rather than imported
    because the steps carry no ties to correct for -- one reading per aim value.
    """
    steps = _ramp_steps(target, sampled)
    if steps is None:
        return None
    # Lightest expected first, so a ladder reading as declared descends; the
    # correlation is negated to put agreement at +1, as _ramp_from does.
    aim = np.argsort(np.argsort([-step[0] for step in steps])).astype(np.float64)
    read = np.argsort(np.argsort([-step[1] for step in steps])).astype(np.float64)
    if np.std(aim) == 0.0 or np.std(read) == 0.0:
        return None
    return float(np.corrcoef(aim, read)[0, 1])


def _refine(image: LoadedImage, target, detection: Detection) -> None:
    """Snap a detected placement onto the chart's printed features, in place."""
    from deltachrome.errors import GeometryError
    from deltachrome.imaging.geometry import corners_outside, place_chart
    from deltachrome.imaging.refine import refinable, refine_placement

    # A chart cut by the frame's edge is found by its marks and then cannot be
    # placed whole. Said as that, rather than as place_chart's own refusal, which
    # is written for somebody who typed four corners and tells them to check the
    # order -- advice that is wrong for a placement nobody typed.
    if corners_outside(detection.corners, (image.width, image.height)):
        detection.corners = []
        detection.found = False
        detection.reason = "The chart runs off the edge of the frame."
        return
    try:
        placement = place_chart(
            detection.corners, image_size=(image.width, image.height),
            corner_positions_relative=target.fiducials.corner_positions or None,
        )
    except GeometryError as exc:
        detection.corners = []
        detection.found = False
        detection.reason = (
            f"The marks found do not form a usable quadrilateral: {exc}"
        )
        return
    result = refine_placement(image, placement, target)
    detection.refinement = result
    if result.applied:
        detection.corners = list(result.corners)
    elif result.features_used and refinable(target)[0]:
        # Only where refinement could ever succeed. The QA-62 declares one
        # locatable feature against the three a fit needs, so this fired on every
        # run of it -- a chart's design reported as news, which the warning review
        # cut everywhere else. Its placement is made on four printed marks and
        # needs no refining; the refusal is still in ``detection.refinement``.
        detection.notes.append("Unable to refine ROI placement.")


def color_agreement(image: LoadedImage, target, corners) -> float | None:
    """How far this chart's patches read from the aim values the target ships.

    The median Delta E2000 over every patch that sampled and has an aim value, which
    is the same comparison the analysis itself makes and is here for a different
    purpose: **colour is what identifies a colour chart.** Two charts can share a
    layout exactly and share almost none of their colours -- the two ISA FADGI-ISO
    19264 versions do -- so where geometry has run out this is the only thing left,
    and where geometry has *not* run out it is a strong confirmation that costs one
    conversion per patch.

    The median rather than the mean, because a wrong target puts its regions on
    arbitrary patches and a handful of those will agree by luck; and against the
    target's own **nominal** values rather than a per-unit file, because
    identification happens before anyone has said which chart this is, so there is no
    per-unit file to reach for. Fifteen of the seventeen shipped targets carry
    nominal values. The two that do not, the ColorChecker SG and the LaserSoft, are
    identified on their geometry and their ramp alone.

    Returns ``None`` where the target ships no aim values or too few patches sampled.
    """
    sampled = _sample(image, target, corners)
    return None if sampled is None else _color_from(image, target, sampled)


def _color_from(image: LoadedImage, target, sampled) -> float | None:
    """:func:`color_agreement`, given a sampling already made."""
    from deltachrome.imaging.color import delta_e, rgb_to_lab

    path = target.nominal_reference_path
    if path is None or not path.exists():
        return None
    try:
        from deltachrome.reference import read_reference

        reference = read_reference(path)
    except Exception:  # noqa: BLE001 -- no reading is a fine answer here
        return None

    errors = []
    for patch in target.patches:
        measurement = reference.get(patch.patch_id)
        sample = sampled.samples.get(patch.patch_id)
        if measurement is None or measurement.lab is None or sample is None:
            continue
        if not sample.usable:
            continue
        try:
            read = rgb_to_lab(sample.rgb_mean, image.profile)
        except Exception:  # noqa: BLE001
            return None
        errors.append(float(delta_e(read, np.asarray(measurement.lab, float))))
    if len(errors) < MIN_RAMP_STEPS:
        return None
    return float(np.median(errors))


# -- which chart is in the frame ----------------------------------------------

#: How much better the best candidate must be than the next before it is named
#: rather than shortlisted. On the scale below, where every term is zero for a
#: perfect fit and one for a total mismatch.
IDENTIFY_MARGIN = 0.25

#: How much of a target's declared array must sample before it counts as an
#: explanation of the capture at all. A target of nearly the right shape can put its
#: regions in the gaps.
IDENTIFY_MIN_SAMPLED = 0.9

#: How many of the regions that did sample may carry a sampling flag before the
#: candidate stops being an explanation of the capture.
#:
#: **Deliberately far looser than any right answer needs.** Across every capture
#: here the correct target flags 0% to 15% of its regions, and the wrong ones that
#: got as far as being scored flag 73% to 100% -- a 35mm slide read as a microform
#: chart flags 90% where the right answer flags 0. A gate anywhere in between would
#: work on this evidence, and it is set high on purpose: a *clipped* capture is a
#: real thing an operator has and will flag heavily through no fault of the chart,
#: and refusing to name a chart because the exposure was blown would be answering a
#: question nobody asked. Regions landing on junk and regions landing on a blown
#: patch are different faults, and only the first should stop an identification.
IDENTIFY_MAX_FLAGGED = 0.6

#: An aspect disagreement past which a candidate is not worth sampling for. Two
#: thirds of an octave: a chart placed against the wrong definition is usually out
#: by far more, and pruning here is what keeps identification to a few seconds
#: rather than a few tens of them.
IDENTIFY_ASPECT_LIMIT = 0.30

#: The median Delta E2000 at which a chart's colours count as agreeing with a
#: target's aim values not at all, so that the colour term saturates rather than
#: letting one wild candidate dominate the sum.
COLOR_MISMATCH = 20.0

#: Finders that read nothing from the target they are given, so their answer is a
#: property of the capture alone and can be computed once and shared by every
#: candidate declaring that method. Without this, identifying a capture costs
#: seventeen finder passes instead of ten.
#: Methods that are not evidence of *which* chart this is. A target declaring one of
#: these prints no registration marks, so its placement is derived from the thing
#: being measured rather than from anything identifying.
PLACED_ON_ITS_OWN_PATCHES = frozenset({"patch_array_corners", "chart_corners"})

TARGET_BLIND_FINDERS = frozenset({
    "find_printed_frame", "find_printed_discs", "find_printed_crosshairs",
    "find_module_crosshairs", "find_whole_frame",
})


@dataclass
class Candidate:
    """One target considered as an explanation of the capture."""

    target_id: str
    display_name: str
    corners: list[tuple[float, float]] = field(default_factory=list)
    #: How far the placed quad's proportions sit from what this target's grid
    #: implies, in natural log units. Zero is exact.
    aspect_error: float = 0.0
    #: How well this target's declared neutral ramp reads, from +1 to -1, or None
    #: where the target has none or it could not be read.
    ramp: float | None = None
    #: Fraction of this target's declared patches whose region sampled at all.
    sampled: float = 0.0
    #: Fraction of the sampled regions carrying a sampling flag.
    flagged: float = 0.0
    #: Median Delta E2000 against this target's own nominal aim values, or None
    #: where it ships none.
    color: float | None = None
    #: Whether this target was found by registration marks it prints, rather than by
    #: the outline of its own patch array. See :func:`identify_target`.
    marks: bool = True
    #: Sum of the three costs below. Lower is a better explanation.
    cost: float = float("inf")
    reason: str = ""

    def explain(self) -> str:
        ramp = "unreadable" if self.ramp is None else f"{self.ramp:+.2f}"
        color = "no aim values" if self.color is None else f"{self.color:.1f} dE"
        return (
            f"aspect off by {self.aspect_error * 100:.0f}%, ramp {ramp}, "
            f"color {color}, {self.sampled:.0%} of its patches sampled, "
            f"{self.flagged:.0%} flagged"
        )


@dataclass
class Identification:
    """Which chart the capture holds, or why that cannot be said."""

    target_id: str | None = None
    #: The placement found for the winning target, so identifying a chart and
    #: placing it are one pass rather than two.
    detection: Detection | None = None
    #: Every candidate that got as far as being scored, best first.
    candidates: list[Candidate] = field(default_factory=list)
    found: bool = False
    reason: str = ""
    notes: list[str] = field(default_factory=list)


def _identifiable(target) -> bool:
    """Whether identification could ever name this target: patches, and a finder."""
    return bool(target.patches) and target.fiducials.method in FINDERS


def identifiable_targets() -> list[str]:
    """Every installed target identification can name, in id order.

    What Settings offers to turn off. A target identification never considers --
    the frame-filling white field, which has nothing on it to identify -- is not
    offered, since turning it off would change nothing and say that it did.
    """
    from deltachrome.targets.registry import available_targets, load_target

    return [t for t in available_targets() if _identifiable(load_target(t))]


def _identify_target_once(
    image: LoadedImage,
    target_ids: list[str] | None = None,
    *,
    extent: Extent | None = None,
) -> Identification:
    """Which of the installed targets the capture is a capture of.

    ``extent`` names the region to look in, and defaults to :func:`chart_extent` --
    the whole search, unchanged, for a frame holding one chart. Passing one is what
    lets :func:`identify_targets` ask the same question of each region of a frame
    holding several, and nothing else about identification changes when it does.

    ``target_ids`` narrows the candidates, and ``None`` means every installed target.
    Narrowing is the cheapest defence against a wrong name there is: every target
    left out is one fewer chance for the wrong chart to fit, and the two pairs that
    share a layout -- the IT8s, the two 19264 versions -- are named outright for
    anybody who owns only one of each. It is also what Settings' list of owned
    targets passes.

    Placement answers *where* a chart is; this answers *which* chart, and it is the
    harder half. It is built out of the same two pieces, and adds nothing:

    **The search is per method, not per target.** Seventeen targets declare eight
    ``fiducials.method`` values, and five of the finders read nothing at all from the
    target handed to them -- what they look for is a property of the capture. So a
    capture of an unknown chart costs about ten finder passes rather than seventeen,
    and most candidates eliminate themselves by their finder simply refusing.

    **What ranks the survivors is what already ranks an orientation**: the aspect the
    target's grid implies against the aspect actually placed, and whether that
    target's declared neutral ramp reads the way it is declared to. Both are
    properties of the *pixels* rather than of the placement, which is the quality
    wanted here -- a wrong target placed on a real chart gives a quad of the wrong
    proportions and a ramp that lands on arbitrary patches. A third term, how much of
    the declared array sampled cleanly, catches the case where a target of nearly the
    right shape puts its regions in the gaps.

    **Geometry cannot separate two charts of one layout**, and this ships one such
    pair: the two ISA FADGI-ISO 19264 versions share a layout exactly and share
    almost none of their colours, so nothing measured here can tell them apart and
    only their reference data could. That is what the shortlist is for. Where the
    best candidate does not beat the next by :data:`IDENTIFY_MARGIN`, no target is
    named and the candidates are returned in order for the operator to choose
    between -- which is a better answer than a coin toss, and the same rule the rest
    of this module follows.

    **A target with no patches is not a candidate.** The flat field is a
    frame-filling white sheet rather than a chart; there is nothing on it to identify
    and nothing that would distinguish it from an unfocused photograph of anything.
    """
    from deltachrome.targets.registry import available_targets, load_target

    extent = chart_extent(image) if extent is None else extent
    candidates: list[Candidate] = []
    placements: dict[str, object] = {}

    # ``None`` is every installed target; an empty list is none of them. They used
    # to be read alike, so an operator who had turned every target off in Settings
    # got the whole search back -- the one reading of "none" that cannot be right.
    for target_id in sorted(available_targets() if target_ids is None else target_ids):
        target = load_target(target_id)
        if not _identifiable(target):
            continue
        finder = FINDERS[target.fiducials.method]
        key = finder.__name__
        if key not in TARGET_BLIND_FINDERS or key not in placements:
            try:
                placements[key if key in TARGET_BLIND_FINDERS else target_id] = (
                    finder(image, target, extent)
                )
            except DetectionRefused as refusal:
                placements[key if key in TARGET_BLIND_FINDERS else target_id] = refusal
        found = placements[key if key in TARGET_BLIND_FINDERS else target_id]
        if isinstance(found, DetectionRefused):
            continue

        try:
            corners, ordering = order_corners(
                found[0], target.fiducials.corner_positions,
                cell_aspect=implied_quad_aspect(target),
            )
        except DetectionRefused:
            continue
        scored = _score_candidate(
            image, target, ordering.get("assignment_tied") or [corners]
        )
        scored.marks = target.fiducials.method not in PLACED_ON_ITS_OWN_PATCHES
        candidates.append(scored)

    candidates.sort(key=lambda candidate: candidate.cost)
    if not candidates:
        return Identification(
            reason=(
                "No installed target could be found in this capture. Choose the "
                "chart yourself, and place it by hand if it cannot be found."
            ),
            candidates=candidates,
        )

    # A target is named only if it FITS -- its ramp reads the way the definition says
    # and its regions land on the chart -- and not merely by being the least bad of a
    # field of candidates. Ranking alone named the ColorChecker SG as the chart in a
    # capture of the early ISA FADGI-ISO 19264, whose own method has no finder, purely
    # because nothing better was available: its ramp read -0.09 and 88% of its regions
    # were flagged, and it still won. Each gate is a separate claim that can be read
    # and disagreed with; a ranking is not.
    fits = [candidate for candidate in candidates if fits_the_capture(candidate)]
    if not fits:
        return Identification(
            candidates=candidates,
            reason=(
                "No installed target explains this capture: its patches did not read "
                "the way any of their definitions say they should. Choose the chart "
                "yourself."
            ),
        )

    best = fits[0]
    close = [c for c in fits if c.cost - best.cost < IDENTIFY_MARGIN]
    notes: list[str] = []
    if len(close) > 1 and best.marks and not any(c.marks for c in close[1:]):
        # **A chart found by marks it prints is better identified than one found by
        # the outline of its own patch array.** ``patch_array_corners`` is declared by
        # charts that print no registration marks, so finding "an array of this shape"
        # is satisfied by any array of that shape, where finding four printed pluses
        # exactly where a definition says four printed pluses are is evidence about
        # *which* chart this is. The two Next Generation Targets are the case: one
        # maker, one 13x10 array, aim values agreeing to a Delta E, and one of them
        # prints crosshairs.
        #
        # It may only ever confirm the leader, never replace it, and that is not
        # caution. Allowed to promote any tied candidate, it named the ISA 4x5 film
        # target as the chart in a capture of an IT8 -- a worse fit that happened to
        # be placed on a printed frame. Being found by marks breaks a tie between
        # explanations that are equally good; it does not make a worse one better.
        close = [best]
    if len(close) > 1:
        runner_up = close[1]
        return Identification(
            candidates=fits,
            # WHERE the chart is, even though WHICH chart it is was not settled. The
            # shortlisted targets fitted the same four corners -- that is what made
            # them a shortlist -- so the leader's placement is the chart's placement
            # whichever of them it turns out to be. A caller that hands the choice to
            # somebody can then show them the chart rather than only the question.
            detection=_place_in(
                image, load_target(best.target_id), extent, refine=True
            ),
            reason=(
                f"This capture could be {best.display_name} or "
                f"{runner_up.display_name}, and nothing measured here separates them "
                f"({best.explain()}, against {runner_up.explain()}). Choose which."
            ),
        )
    return Identification(
        target_id=best.target_id,
        detection=_place_in(
            image, load_target(best.target_id), extent, refine=True
        ),
        candidates=fits,
        found=True,
        reason=f"{best.display_name}, identified from the capture: {best.explain()}.",
        notes=notes,
    )


#: How much of a decoded code's own area a region must hold before the region is
#: called a code block. Half, and it is deliberately a fraction of the *code*
#: rather than of the region: the two things being told apart are a chart that
#: happens to print a code somewhere inside it and a region that is nothing but
#: the code, and the second is the one where the code fills what was found. A
#: ratio of areas would need a threshold tuned on one capture -- the ColorChecker
#: SG's QR is 3.5% of its own chart and 34% of the spurious region beside it,
#: which is not a gap worth resting on.
CODE_REGION_SHARE = 0.5


def _identify_targets_once(
    image: LoadedImage,
    target_ids: list[str] | None = None,
    *,
    most: int = MOST_CHARTS,
    code_boxes: Sequence[tuple[int, int, int, int]] = (),
) -> list[Identification]:
    """What each region of the frame holds, one answer per chart, in reading order.

    :func:`identify_target` asks "which chart is this capture of" and is the right
    question for a frame holding one. This asks it of every candidate region, which
    is the same question repeated and not a harder one -- the search, the scoring and
    the two gates are all unchanged, and each region simply gets its own answer.

    **Nothing a region said is thrown away**, with two exceptions. An outcome comes
    back for every region that produced a scored candidate at all, whether it named
    a chart, shortlisted two it could not separate, or declined because nothing
    fitted. A region where no finder placed anything is silent, because a region
    with no candidate is not evidence of a chart; and an unnamed outcome sitting on
    a chart this frame *did* name is silent, because those pixels were measured
    (:func:`_not_a_named_chart_again`). Anything else is reported, so a caller can
    run what was named and *say* what was not, rather than a chart quietly going
    unmeasured.

    **What this cannot do is notice a chart it never separated**, and two attempts to
    make it were measured and thrown out rather than shipped. Two charts closer
    together than the extent search can tell apart come back as one region; a finder
    places whichever of them it can, correctly, with no sign that the other is there.

    *Flagging a chart that covers little of its region* looked like the tell -- a
    chart normally fills its own -- and does not survive contact with the corpus: the
    Golden Thread Object Level strip covers **21%** of its region on a capture holding
    nothing else, because its fiducial quad is the patch row and the card carries
    printed rules well outside it, while a Device Level Target sharing a frame with a
    ColorChecker covers 14%. Those ranges overlap, so the test would call a correct
    single-chart placement suspect.

    *Blanking what was placed and searching the remainder* is threshold-free and does
    not work either, for a reason worth keeping: blanking does not change the
    segmentation. The region is still the whole frame, so the second chart's finder
    still looks in the frame's corners rather than the chart's, and finds nothing.

    So a frame whose charts the search cannot separate reads as holding fewer charts
    than it does, and that is a stated limitation rather than a hidden one. The
    distance it takes is small: the two charts it was measured on sit **12 pixels
    apart on an 11600 pixel frame**, where the morphological close that merges them
    spans 180. Nothing between those two numbers can both separate charts that close
    and still close a ColorChecker's own 140 patches into one region.

    **A region that is a printed code is not a chart**, and ``code_boxes`` is how a
    caller that has already decoded one says so. Nothing about the pixels could:
    a QR block is exactly what this search looks for -- a compact region of
    high-contrast structure unlike its surround -- so the ColorChecker SG's own QR
    is offered as a second target in its frame, shortlisted between the two
    microform charts because a five-by-six lattice of black and white cells is
    what a microform target's density ladder also is. Knowing that a code was
    decoded there settles it, and only a decoder can know.

    The claim is made in two halves so that it cannot swallow a real chart. The
    region must hold most of a decoded code's own area, and **no target may have
    been named for it** -- a chart that prints its code inside its own outline is
    named on its patches and is untouched, which is the ColorChecker itself.
    **What is skipped is not reported.** It was, as evidence about the capture, and
    on every run of the SG that made the run say a region "looked like a chart and
    couldn't be identified" about a code the application had just read.
    """
    seen: list[np.ndarray] = []
    outcomes: list[tuple[tuple[float, float], Identification, tuple]] = []
    for extent in chart_extents(image, most=most):
        result = _identify_target_once(image, target_ids, extent=extent)
        if not result.candidates:
            continue
        if not result.found and _code_in(extent.box, code_boxes) is not None:
            continue
        detection = result.detection
        if result.found and detection is not None and detection.corners:
            quad = np.asarray(detection.corners, dtype=np.float64)
            if any(_same_quad(quad, kept) for kept in seen):
                continue
            seen.append(quad)
            outcomes.append((_reading_order(quad), result, quad))
        else:
            middle = (0.5 * (extent.box[0] + extent.box[2]),
                      0.5 * (extent.box[1] + extent.box[3]))
            outcomes.append((
                (round(middle[1] / max(extent.short_side, 1)), middle[0]),
                result,
                _place_of(result, extent),
            ))
    ordered = sorted(outcomes, key=lambda entry: entry[0])
    kept = _one_chart_per_place([result for _key, result, _place in ordered])
    return _not_a_named_chart_again(kept, [place for _key, _result, place in ordered])


def unidentified_regions_note(count: int) -> str | None:
    """The run warning for regions that looked like charts and were not named.

    One line, and not the regions' own refusals joined together: each says why no
    target fitted, which is the same sentence for every such region and answers
    nothing an operator can act on. What they can act on is that a chart may have
    gone unmeasured, and the count is that.
    """
    if count <= 0:
        return None
    if count == 1:
        return "1 other region looked like a chart and couldn't be identified."
    return f"{count} other regions looked like charts and couldn't be identified."


def _place_of(result: Identification, extent: Extent):
    """Where an unnamed region's outcome sits: its best guess's quad, else its box.

    A guess that fitted four corners says more exactly where it was looking than
    the region it was looking in, which is deliberately loose; a guess that fitted
    nothing has only the region, as a box ``(x0, y0, x1, y1)``.
    """
    detection = result.detection
    if detection is not None and len(detection.corners) == 4:
        return np.asarray(detection.corners, dtype=np.float64)
    for candidate in result.candidates:
        if len(candidate.corners) == 4:
            return np.asarray(candidate.corners, dtype=np.float64)
    return tuple(extent.box)


def _quads_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    """Whether two placed outlines share any area at all.

    No threshold, on the argument :func:`shares_a_place` rests on: two charts
    cannot occupy the same pixels, so outlines placed correctly on two different
    charts share none.
    """
    try:
        area, _points = cv2.intersectConvexConvex(
            first.astype(np.float32), second.astype(np.float32)
        )
    except cv2.error:
        # A quad that is not convex is not a placement of anything; fall back to
        # the rule that needs no polygon.
        return shares_a_place(quad_bounds(first.tolist()), quad_bounds(second.tolist()))
    return area > 0.0


def _not_a_named_chart_again(
    found: list[Identification], places: list[tuple]
) -> list[Identification]:
    r"""Drop an unnamed outcome that sits on a chart this frame has already named.

    **The region search is over-generous on purpose** and returns pieces of a chart
    as well as charts, and every target's finder is tried on every piece. On the
    Golden Thread Device Level Target's own sample, a capture holding that chart and
    nothing else, six regions come back: the chart, three pieces of it on which the
    LaserSoft's and the ISA 120's finders fitted something before their patches
    refused them, and two strips of backdrop that fitted nothing. The three pieces
    were reported as *further regions that held something a chart could be placed
    in and were not measured* -- which on a one-chart frame is untrue, and printed
    the same refusal three times over on every run of it.

    **It is the rule two charts cannot be in the same place**, asked of a refusal
    rather than of a second name, and with no threshold. A guess that fitted four
    corners is dropped if its outline shares *any* area with a named chart's --
    the centre test :func:`shares_a_place` asks was tried first and missed a DT
    Nexus guess covering the right half of a ColorChecker SG, its centre landing
    just past the card. A guess that fitted nothing has only its region's box,
    which is deliberately loose, so that one asks the centre test instead. A guess
    sitting on a named chart is that chart read by the wrong finder, and its pixels
    *were* measured -- so dropping it does not make a chart read as absent, which is
    what reporting a skipped region exists to prevent. A region elsewhere in the
    frame is untouched and still reported. That includes what
    :func:`_one_chart_per_place` demotes, a second name for a named chart's pixels:
    it is the same pixels, so it goes too.

    What this widens is a limitation already stated: a genuine second chart that
    could not be named, close enough to the first that the wrong finder's outline
    for it spills onto the first, is no longer mentioned.
    """
    named = [place for result, place in zip(found, places)
             if result.found and isinstance(place, np.ndarray)]

    def on_a_named_chart(place) -> bool:
        if isinstance(place, np.ndarray):
            return any(_quads_overlap(place, other) for other in named)
        return any(shares_a_place(place, quad_bounds(other.tolist())) for other in named)

    return [
        result for result, place in zip(found, places)
        if result.found or place is None or not on_a_named_chart(place)
    ]


def extra_charts(image: LoadedImage, target, measured) -> list[str]:
    r"""Further charts of the measured one's model in this frame, not measured.

    A batch measures one chart per frame and, until this existed, said nothing
    about the rest of the frame: a folder of boards was measured one card per frame
    and the other three went unreported, so a chart nobody measured read as a chart
    that was not there. That is the same failure the rest of this module refuses --
    *a region that held a chart and could not be named is reported as skipped* --
    applied to a batch.

    **It only ever warns.** Nothing here changes what is measured or judged. A
    batch calls it once, on its first capture (``batch._first_frame_notice``).

    **Only the declared model is looked for**, which is :func:`detect_placements`:
    one finder pass per candidate region, and the board case -- four QA-62 cards,
    one measured. On the real QA-62 board it finds the other three at 7.9 s against
    a 10 s decode, which is why it runs once per batch and not once per frame.
    Asking every frame, and looking for other models too, was a batch option and
    is deferred past the beta. A different chart in the frame goes
    unreported.

    It is a *separate search* from the run: a batch places from a seed, this does
    not, so on material where the seed is what makes placement work the two can
    disagree. That is why the comparison is :func:`shares_a_place` against the
    placement actually measured rather than a count of what was found -- a check
    that found nothing is silent rather than claiming the measured chart is extra.
    """
    here = quad_bounds(measured)
    if here is None:
        return []

    found: list[tuple[str, tuple[float, float, float, float]]] = []
    for detection in detect_placements(image, target):
        if detection.found and detection.corners:
            box = quad_bounds(detection.corners)
            if box is not None:
                found.append((target.display_name, box))

    return [
        f"{name} was found in this frame at "
        f"({box[0]:.0f}, {box[1]:.0f})-({box[2]:.0f}, {box[3]:.0f}) and was not "
        "measured: a batch measures one chart per capture, the one its target and "
        "placement name. Run this capture on its own to measure it."
        for name, box in found if not shares_a_place(here, box)
    ]


def quad_bounds(corners) -> tuple[float, float, float, float] | None:
    """A placement's bounding box, or ``None`` if it has no corners."""
    if not corners:
        return None
    quad = np.asarray(corners, dtype=np.float64)
    return (quad[:, 0].min(), quad[:, 1].min(), quad[:, 0].max(), quad[:, 1].max())


def shares_a_place(first, second) -> bool:
    """Whether two placements are two readings of one chart rather than two charts.

    One placement's centre lying inside the other's bounding box. **There is no
    threshold in it**, which is the point: two placements of one chart agree within
    a few pixels and two different charts share none at all. The frontend has held
    this since a repeated *Search a Region* started adding a target each time, and
    it is the same rule :func:`_one_chart_per_place` and :func:`extra_charts` ask.
    """
    def holds(box, other) -> bool:
        x = 0.5 * (other[0] + other[2])
        y = 0.5 * (other[1] + other[3])
        return box[0] <= x <= box[2] and box[1] <= y <= box[3]
    return holds(first, second) or holds(second, first)


def _names(result: Identification) -> str:
    """What to call this outcome's chart in a sentence a person reads."""
    if result.candidates:
        return result.candidates[0].display_name
    return result.target_id or "an unnamed chart"


def _explains(result: Identification) -> str:
    """This outcome's own account of how well it fitted."""
    return result.candidates[0].explain() if result.candidates else "no reading"


def _one_chart_per_place(found: list[Identification]) -> list[Identification]:
    r"""Where two named charts occupy the same pixels, keep the one that fits.

    **Two charts cannot be in the same place**, so two names for one set of pixels
    are two explanations of one thing rather than two charts. The frontend has held
    this about *placements* since a repeated *Search a Region* started putting one
    more target in the frame each time; this is the same rule one level up, about
    which chart a region was said to hold.

    The case it is for was measured on twenty Smithsonian Odonata frames. Four of
    them name a second chart -- a microform target, an Image Engineering UTT, an ISA
    transmissive 4x5 -- placed across very nearly the whole frame and **enclosing**
    the Object Level strip that was correctly found inside it. Five of those six
    read 1.000 on the test below, at ramps of 0.56 to 0.78 and 12 to 31 dE against
    the strip's +1.00 and 2.1.

    **Which one is kept is not a new judgement.** ``Candidate.cost`` already orders
    a shortlist, and :func:`detect_placements` already chooses this way when it has
    more than one placement to choose between. Nothing here is close: the scores
    differ by a factor rather than by a margin somebody would have to defend.

    **And the loser is reported here, never dropped** -- though
    :func:`identify_targets` then drops it, its pixels being the winner's. A chart that quietly disappears
    reads as "there was no chart there", which is a different and more misleading
    claim than "there was something there and it could not be accounted for" -- the
    same reason a region that held a chart and could not be named is reported as
    skipped rather than passed over. So it comes back as an outcome with no target
    on it, in its own place in reading order, which is what every caller already
    treats as a region to mention and not to measure.

    **The test is centre containment, not an area overlap**, and it is the
    frontend's own: one placement's centre lying inside the other's bounding box.
    There is no threshold in it, which is the point -- two placements of one chart
    agree within a few pixels and two different charts share none at all. Measured
    on the one capture here holding two charts of different models both correctly
    named, the Golden Thread Device Level and ColorChecker SG composite, it reads
    **0.000**: the case this could break is not in the corpus.
    """
    def bounds(result: Identification):
        detection = result.detection
        if not (result.found and detection is not None and detection.corners):
            return None
        quad = np.asarray(detection.corners, dtype=np.float64)
        return (quad[:, 0].min(), quad[:, 1].min(), quad[:, 0].max(), quad[:, 1].max())

    def cost_of(result: Identification) -> float:
        return result.candidates[0].cost if result.candidates else float("inf")

    boxes = [bounds(result) for result in found]
    out = list(found)
    for i, first in enumerate(boxes):
        if first is None or out[i] is not found[i]:
            continue
        for j, second in enumerate(boxes):
            if j <= i or second is None or out[j] is not found[j]:
                continue
            if not shares_a_place(first, second):
                continue
            loser = j if cost_of(found[i]) <= cost_of(found[j]) else i
            winner = i if loser == j else j
            out[loser] = Identification(
                candidates=found[loser].candidates,
                reason=(
                    f"This region was read as {_names(found[loser])} and was not "
                    f"measured: it occupies the same pixels as {_names(found[winner])}, "
                    f"which fits the capture better ({_explains(found[winner])} against "
                    f"{_explains(found[loser])}). Two charts cannot be in the same "
                    "place, so these are two readings of one."
                ),
            )
            # A chart that has just been displaced cannot go on displacing others:
            # everything after this would be demoted on the strength of a reading
            # this loop has already decided against.
            if loser == i:
                break
    return out


def _code_in(box, code_boxes) -> tuple[int, int, int, int] | None:
    """The first decoded code this region holds most of, if any."""
    x0, y0, x1, y1 = box
    for code in code_boxes:
        cx0, cy0, cx1, cy1 = code
        area = max(1.0, float(cx1 - cx0) * float(cy1 - cy0))
        overlap = (max(0.0, min(x1, cx1) - max(x0, cx0))
                   * max(0.0, min(y1, cy1) - max(y0, cy0)))
        if overlap / area >= CODE_REGION_SHARE:
            return (int(cx0), int(cy0), int(cx1), int(cy1))
    return None


def fits_the_capture(candidate: Candidate) -> bool:
    """Whether a candidate explains the capture, rather than merely leading a field.

    Three claims, each separately readable and separately arguable, which is the
    point of their being gates rather than a score: this target's neutral ramp reads
    the way its definition says, its regions landed on its array, and those regions
    came back looking like patches. A ranking cannot be disagreed with in that way,
    and ranking alone named the ColorChecker SG as the chart in a capture of the
    early ISA FADGI-ISO 19264 -- whose own method no finder covers, so the right
    answer was not a candidate at all -- on a ramp reading -0.09 with 88% of its
    regions flagged, purely because nothing better was available.
    """
    return (
        candidate.ramp is not None
        and candidate.ramp >= RAMP_AGREEMENT_FLOOR
        and candidate.sampled >= IDENTIFY_MIN_SAMPLED
        and candidate.flagged <= IDENTIFY_MAX_FLAGGED
    )


def _score_candidate(image: LoadedImage, target, orientations: list) -> Candidate:
    """How well one target explains the capture, in costs that read zero for a
    perfect fit and one for a total mismatch.

    Every orientation the geometry could not rule out is scored and the best kept,
    so identifying a chart settles which way up it is at the same time and for the
    same reason -- both are questions the outline cannot answer and the pixels can.

    Summed with equal weight, which is a stated default rather than a calibration:
    what names a target is the gates in :func:`identify_target`, each of which is a
    separate and individually meaningful claim, and the sum only orders a shortlist.
    The components travel with the answer so a reader can see which did the work.
    """
    candidate = Candidate(target_id=target.target_id, display_name=target.display_name)
    implied = implied_quad_aspect(target)
    declared = np.asarray(target.fiducials.corner_positions, float)

    best = None
    for corners in orientations:
        aspect_error = 0.0
        if implied:
            placed = _placed_aspect(declared, np.asarray(corners, float))
            if placed > 0:
                aspect_error = abs(float(np.log(placed / implied)))
        if aspect_error > IDENTIFY_ASPECT_LIMIT:
            continue
        reading = _sample(image, target, corners)
        if reading is None:
            continue
        usable = [s for s in reading.samples.values() if s.usable]
        sampled = len(usable) / max(len(target.patches), 1)
        flagged = (sum(1 for s in usable if s.flags) / len(usable)) if usable else 1.0
        ramp = _ramp_from(target, reading)
        color = _color_from(image, target, reading)
        # The MEAN of the evidence there is, not the sum, and the aspect is not in
        # it. Both were failures worth keeping. Summing gave a target shipping no aim
        # values a free pass on the colour term, so absence of evidence read as
        # agreement and the LaserSoft outranked the right answer on an IT8 capture.
        # And the aspect is a *filter* here rather than evidence: it is an inference
        # from cells being square, which is false on the one chart whose patches are
        # not, and counting its 21% error against the Device Level Target nearly cost
        # that chart its own identification.
        terms = [(1.0 - ramp) / 2.0 if ramp is not None else None,
                 1.0 - sampled,
                 flagged,
                 min(color / COLOR_MISMATCH, 1.0) if color is not None else None]
        known = [term for term in terms if term is not None]
        # A candidate whose ramp could not be read has not been tested against the
        # one thing that identifies a chart, so it is not ranked as though it had:
        # without this, a target with no readable ramp and no aim values scored a
        # perfect zero on the single term it did have.
        cost = float(np.mean(known)) if (known and ramp is not None) else 1.0
        if best is None or cost < best[0]:
            best = (cost, corners, aspect_error, ramp, sampled, flagged, color)

    if best is None:
        candidate.cost = float("inf")
        candidate.reason = "the wrong proportions for this chart"
        return candidate
    (candidate.cost, candidate.corners, candidate.aspect_error, candidate.ramp,
     candidate.sampled, candidate.flagged, candidate.color) = best
    return candidate


# -- an untagged capture in linear light -----------------------------------------


def through_srgb_curve(image: LoadedImage) -> LoadedImage | None:
    """The capture's values put through the sRGB curve, for searching only.

    ``None`` for a capture carrying a profile: its values are whatever its profile
    says, and what detection was tuned on. An untagged capture may be linear --
    straight off a raw converter, about to be profiled -- and every darkness
    threshold a finder applies is a fraction of the mount's level, set on encoded
    captures. In linear light the same fractions fall elsewhere: the LaserSoft
    4x5 refuses linear and places on its printed frame through this curve.
    Nothing is measured from what this returns; it is where to look.
    """
    if not isinstance(image, LoadedImage) or image.profile is not None:
        return None
    pixels = image.pixels
    if pixels.dtype in (np.uint8, np.uint16):
        levels = np.arange(int(np.iinfo(pixels.dtype).max) + 1, dtype=np.float64)
        lut = (_encode_srgb(levels / float(image.scale)) * float(image.scale))
        lut = np.clip(np.round(lut), 0, np.iinfo(pixels.dtype).max).astype(pixels.dtype)
        encoded = lut[pixels]
    else:
        values = np.clip(pixels.astype(np.float32) / float(image.scale), 0.0, 1.0)
        encoded = (_encode_srgb(values) * float(image.scale)).astype(pixels.dtype)
    return replace(image, pixels=encoded)


def _encode_srgb(linear):
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * np.power(np.maximum(linear, 0.0031308), 1 / 2.4) - 0.055)


def _found(result) -> bool:
    if isinstance(result, list):
        return any(_found(item) for item in result)
    return bool(result.found)


def _mark_encoded(result) -> None:
    """Record, on every placement, that it was found through the sRGB curve."""
    if isinstance(result, list):
        for item in result:
            _mark_encoded(item)
        return
    detection = result if isinstance(result, Detection) else result.detection
    if detection is not None:
        detection.diagnostics["searched_through_srgb_curve"] = True


def _retrying_encoded(image: LoadedImage, attempt):
    """``attempt(image)``, and once more through the sRGB curve if that found
    nothing on an untagged capture. Only on failure: an untagged capture that is
    already encoded would be pushed twice through the curve, and is found as it is.
    """
    first = attempt(image)
    if _found(first):
        return first
    encoded = through_srgb_curve(image)
    if encoded is None:
        return first
    second = attempt(encoded)
    if not _found(second):
        return first
    _mark_encoded(second)
    return second


def detect_placement(image: LoadedImage, target, *, refine: bool = True) -> Detection:
    """Find where the chart sits in the capture; see :func:`_detect_placement_once`."""
    return _retrying_encoded(
        image, lambda im: _detect_placement_once(im, target, refine=refine))


def detect_placement_near(
    image: LoadedImage, target, corners, *, margin: float = HINT_MARGIN,
    refine: bool = True,
) -> Detection:
    """:func:`detect_placement`, told roughly where the chart sits."""
    return _retrying_encoded(image, lambda im: _detect_placement_near_once(
        im, target, corners, margin=margin, refine=refine))


def detect_placements(
    image: LoadedImage, target, *, refine: bool = True, most: int = MOST_CHARTS
) -> list[Detection]:
    """Every placement of ``target`` this frame supports; see
    :func:`_detect_placements_once`."""
    return _retrying_encoded(image, lambda im: _detect_placements_once(
        im, target, refine=refine, most=most))


def identify_target(
    image: LoadedImage,
    target_ids: list[str] | None = None,
    *,
    extent: Extent | None = None,
) -> Identification:
    """Which installed target the capture holds; see :func:`_identify_target_once`."""
    return _retrying_encoded(
        image, lambda im: _identify_target_once(im, target_ids, extent=extent))


def identify_targets(
    image: LoadedImage,
    target_ids: list[str] | None = None,
    *,
    most: int = MOST_CHARTS,
    code_boxes: Sequence[tuple[int, int, int, int]] = (),
) -> list[Identification]:
    """What each region of the frame holds; see :func:`_identify_targets_once`."""
    return _retrying_encoded(image, lambda im: _identify_targets_once(
        im, target_ids, most=most, code_boxes=code_boxes))
