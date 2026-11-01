"""Several charts in one frame, and a chart that compares none of its patches.

Both arrived with the Applied Image QA-62, and they are separate changes that
happen to have one cause. The QA-62 is a small card that reads one field position,
so it is shot four or more to a board -- the first target here not meant to fill its
own frame -- and it ships no aim values for the twenty-step ring it prints, so it is
the first target whose patches take part in nothing.

Every capture below is drawn rather than photographed, as everywhere else in this
suite: the real masters are not in the repository. What is drawn is the QA-62's own
declared geometry read back out of its definition, which makes these tests say
something a hand-drawn approximation could not -- that the numbers in the definition
and the code that finds them agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.errors import TargetError
from deltachrome.imaging.detect import chart_extents, detect_placement, detect_placements
from deltachrome.imaging.loader import load_image
from deltachrome.targets.registry import load_target

from tests.fixtures import adobe_rgb_icc, write_tiff_with_profile

QA62 = "applied_image_qa62_sfr_v1"

#: How far a re-derived corner may sit from where it was drawn, as a fraction of the
#: card's own side. Detection only has to land inside refinement's capture range,
#: which is five percent; this is far tighter because the marks are drawn as clean
#: pluses and there is nothing here for a threshold to be wrong about.
CORNER_TOLERANCE = 0.01


def _draw_qa62(canvas: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """One QA-62 card drawn into ``canvas``, from its own definition. Returns its quad.

    ``box`` is where the *card* goes; the fiducial quad is inset within it, because
    that is how the real card is printed -- the four crosshairs bound the measurement
    field and a third of the card lies below the bottom pair.
    """
    target = load_target(QA62)
    grid = target.regions[0].grid
    square = target.regions[1].geometry

    x0, y0, x1, y1 = box
    canvas[y0:y1, x0:x1] = 0.93  # the white card
    side = min(x1 - x0, y1 - y0)
    quad_side = int(0.74 * side)
    left, top = x0 + (x1 - x0 - quad_side) // 2, y0 + int(0.06 * side)

    def place(u: float, v: float) -> tuple[int, int]:
        return left + int(round(u * quad_side)), top + int(round(v * quad_side))

    # The twenty-step ring, lightest at the top-left corner and running clockwise,
    # over a uniform mid-grey field that fills the rest of the lattice.
    origin_u, origin_v = grid.grid_origin_relative
    pitch_u, pitch_v = grid.patch_spacing_relative
    field_a = place(origin_u, origin_v)
    field_b = place(origin_u + 6 * pitch_u, origin_v + 6 * pitch_v)
    canvas[field_a[1]:field_b[1], field_a[0]:field_b[0]] = 0.57
    for patch in target.patches:
        a = place(origin_u + patch.col * pitch_u, origin_v + patch.row * pitch_v)
        b = place(origin_u + (patch.col + 1) * pitch_u, origin_v + (patch.row + 1) * pitch_v)
        canvas[a[1]:b[1], a[0]:b[0]] = 0.90 - 0.036 * (patch.ramp_position - 1)

    # The slanted square, drawn by filling the rotated quad its declaration implies.
    import cv2

    angle = np.radians(float(square["tilt_deg"]))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    unit = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    corners = np.array(square["center"]) + (unit @ rotation.T) * np.array(square["half_size"])
    cv2.fillConvexPoly(
        canvas, np.array([place(u, v) for u, v in corners], np.int32), (0.31, 0.31, 0.31)
    )

    # The four printed pluses last, so nothing is drawn over them.
    stroke = max(3, quad_side // 100)
    span = max(9, quad_side // 15)
    for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        x, y = place(u, v)
        canvas[y - stroke:y + stroke + 1, x - span:x + span + 1] = 0.05
        canvas[y - span:y + span + 1, x - stroke:x + stroke + 1] = 0.05
    return np.array([place(u, v) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))], float)


def _board(tmp_path, boxes, *, surround=0.06, board=None, name="board.tif"):
    """A capture holding one QA-62 per box, optionally on a mid-grey board."""
    canvas = np.full((1400, 1800, 3), surround, np.float64)
    if board is not None:
        bx0, by0, bx1, by1 = board
        canvas[by0:by1, bx0:bx1] = 0.24
    quads = [_draw_qa62(canvas, box) for box in boxes]
    encoded = np.round(np.clip(canvas, 0.0, 1.0) * 65535.0).astype(np.uint16)
    image = load_image(write_tiff_with_profile(tmp_path / name, encoded, adobe_rgb_icc()))
    return image, quads


def _matches(found, drawn, side) -> bool:
    return bool(np.all(np.linalg.norm(np.asarray(found) - drawn, axis=1) < CORNER_TOLERANCE * side))


FOUR_ON_A_BOARD = [(180, 160, 700, 660), (1080, 160, 1600, 660),
                   (180, 760, 700, 1260), (1080, 760, 1600, 1260)]


# -- several charts in one frame ----------------------------------------------


def test_four_charts_on_a_board_that_fills_the_frame_are_all_placed(tmp_path):
    """The arrangement the QA-62 is sold for, and the one the board itself defeats.

    Four cards, no surround: the board *is* the frame's border, so the first search
    finds the four cards directly.
    """
    image, drawn = _board(tmp_path, FOUR_ON_A_BOARD, surround=0.24)
    placed = detect_placements(image, load_target(QA62))

    assert len(placed) == 4
    side = np.linalg.norm(drawn[0][1] - drawn[0][0])
    for detection, quad in zip(placed, drawn):
        assert detection.found
        assert _matches(detection.corners, quad, side)


def test_four_charts_on_a_board_against_a_backdrop_are_all_placed(tmp_path):
    """The same four cards, on a board photographed against something darker.

    A different problem and the reason :func:`chart_extents` looks *inside* the
    largest region it finds: here the first search finds the board and the cards are
    on it, so a search that stopped at the first pass would offer one candidate
    covering all four.
    """
    image, drawn = _board(tmp_path, FOUR_ON_A_BOARD, board=(100, 80, 1700, 1340))
    placed = detect_placements(image, load_target(QA62))

    assert len(placed) == 4
    side = np.linalg.norm(drawn[0][1] - drawn[0][0])
    for detection, quad in zip(placed, drawn):
        assert _matches(detection.corners, quad, side)


def test_a_quad_spanning_four_charts_does_not_fit_although_it_places_cleanly(tmp_path):
    """The failure the fit gate exists for, stated directly rather than via a board.

    Asked for four crosshairs near the corners of the *board*, the finder returns one
    mark from each of four different cards: four marks of one printed size, agreeing
    across a run of thresholds, and a perfectly valid quad. Nothing about the geometry
    says it is wrong, which is why the gate cannot be geometric -- a quad made of one
    corner from each of four charts is inset from all four, so it encloses none of
    them and "a chart does not contain another chart" never fires.

    What says it is wrong is the chart's own contents read at that placement. Built
    here from the drawn quads rather than waited for, so the assertion is about the
    gate and not about whichever region the segmentation happened to offer.
    """
    from deltachrome.imaging.detect import _score_candidate, fits_the_capture

    image, drawn = _board(tmp_path, FOUR_ON_A_BOARD, surround=0.24)
    target = load_target(QA62)
    spanning = [
        tuple(drawn[0][1]), tuple(drawn[1][0]), tuple(drawn[3][2]), tuple(drawn[2][3])
    ]

    assert fits_the_capture(_score_candidate(image, target, [[tuple(p) for p in drawn[0]]]))
    assert not fits_the_capture(_score_candidate(image, target, [spanning]))


def test_one_chart_in_a_frame_gives_exactly_the_placement_it_always_gave(tmp_path):
    """The compatibility claim, and the one that matters most.

    Every capture this repository already places correctly must keep the placement it
    had, so :func:`chart_extents` puts :func:`chart_extent`'s own answer first and the
    gate never runs on a lone placement.
    """
    image, drawn = _board(tmp_path, [(180, 160, 700, 660)])
    placed = detect_placements(image, load_target(QA62))

    assert len(placed) == 1
    assert placed[0].corners == detect_placement(image, load_target(QA62)).corners


def test_charts_come_back_top_to_bottom_then_left_to_right(tmp_path):
    """Reading order, banded by the chart's own size so a wobble does not split a row."""
    image, _drawn = _board(tmp_path, FOUR_ON_A_BOARD, surround=0.24)
    centres = [
        np.asarray(d.corners, float).mean(axis=0)
        for d in detect_placements(image, load_target(QA62))
    ]
    assert centres[0][0] < centres[1][0] and centres[0][1] < centres[2][1]
    assert centres[2][0] < centres[3][0]


def test_the_candidate_list_is_generous_and_the_placements_are_not(tmp_path):
    """The division of labour: the segmentation offers, the finders decide.

    Stated as a test because it is easy to read :func:`chart_extents` as answering
    "where are the charts" and tune it as though it did. It does not; it answers
    "where might I look", and on a board of four it offers more than four.
    """
    image, _drawn = _board(tmp_path, FOUR_ON_A_BOARD, board=(100, 80, 1700, 1340))
    assert len(chart_extents(image)) > 4
    assert len(detect_placements(image, load_target(QA62))) == 4


def test_a_chart_lying_on_something_else_is_offered_at_its_own_level():
    r"""The Smithsonian Odonata failure, reduced to the raster that causes it.

    A specimen sheet in a plastic sleeve, on a black backdrop, with a Golden Thread
    Object Level strip along the bottom. Every one of those -- sleeve, sheet, typed
    labels, insect, strip -- is unlike the backdrop, and the close that joins an
    array of patches into one region joins them to each other, so the search that
    asks "what is unlike the frame's border" returns **one** region covering nearly
    the whole frame, and looking into that region returns the same region again.
    There is nothing wrong with either answer: the strip does not differ from the
    border, because the sheet it lies on differs from the border in the same
    direction and by more.

    What it does differ from is the sheet, and the level the first search already
    computed is what says so. Drawn as a raster rather than as a capture because
    what is being pinned is the segmentation and not a finder: on the real frames
    this is the difference between naming an Image Engineering UTT across a
    photograph of a dragonfly on thirteen of twenty captures and naming the right
    chart, placed within 2.5 px, on all twenty.
    """
    from deltachrome.imaging.detect import (
        MULTI_AREA_FLOOR, _at_card_level, _card_level, _components,
        _regions_within, _unlike_border,
    )

    luma = np.full((760, 1000), 0.02, np.float32)
    luma[20:740, 15:985] = 0.10                        # the sleeve
    luma[40:620, 40:960] = 0.86                        # the sheet
    for i in range(12):
        luma[70 + i * 44:88 + i * 44, 90:700] = 0.04   # typed labels on it
    luma[650:730, 60:940] = 0.50                       # the chart, below the sheet
    floor = MULTI_AREA_FLOOR * luma.size

    _surround, mask = _unlike_border(luma)
    unlike = _components(mask, floor)
    assert unlike == [(15, 20, 985, 740)], "one region, and it is everything"
    assert _regions_within(luma, mask, unlike[0], floor) == unlike, (
        "looking into it returns the same region: the chart is not separated"
    )

    assert _card_level(luma) == pytest.approx(0.50)
    assert _components(_at_card_level(luma, _card_level(luma)), floor) == [
        (60, 650, 940, 730)
    ]


# -- a chart that compares none of its patches --------------------------------


def test_a_chart_declaring_no_reference_source_runs_with_no_reference_file(tmp_path):
    """A target may carry patches and state that none of them is compared.

    Two refusals stand between such a chart and a run, and both were written for a
    different fault: "no reference file was supplied and this target ships no nominal
    values", and "no patch could be both sampled and matched". The second guards
    against a reference file for the wrong model of chart, which it can only do
    because a patch without a reference is normally dropped -- so it must not fire
    where no reference was ever in play.
    """
    from deltachrome.pipeline import analyze

    image, drawn = _board(tmp_path, [(180, 160, 700, 660)])
    result = analyze(
        image,
        target=load_target(QA62),
        profile="fadgi_3star_documents_unbound_general_2023",
        corners=[tuple(point) for point in drawn[0]],
    )

    assert result.overall in {"PASS", "FAIL", "INCOMPLETE", "MEASURED"}
    assert len(result.run.patches) == 20, "the patches are still sampled and recorded"
    assert not any("reference value" in w for w in result.warnings), (
        "a definition's own statement must not be reported back as a shortfall"
    )


def test_a_chart_declaring_no_reference_source_must_declare_every_role(tmp_path):
    """Nothing can resolve a role from measured colour when there is no measurement.

    So the claim carries an obligation, and the definition making it has to meet the
    obligation itself rather than leave a patch unassigned that no selection matches.
    """
    from deltachrome.pipeline import EMPTY_REFERENCE
    from deltachrome.targets.roles import resolve_roles

    target = load_target(QA62)
    resolve_roles(target, EMPTY_REFERENCE)  # every role declared: nothing to do

    target.patches[3] = target.patches[3].with_role("unassigned")
    with pytest.raises(TargetError, match="unassigned"):
        resolve_roles(target, EMPTY_REFERENCE)


# -- the definition itself ----------------------------------------------------


def test_the_qa62_ring_is_the_perimeter_of_its_lattice_and_runs_clockwise():
    """Twenty of thirty-six cells occupied, and the order the ramp check depends on.

    The QA-62's fiducial quad is square and so are its cells, so nothing geometric can
    tell one rotation of it from another and the ring's declared order is the only
    thing that can. A ring numbered any other way would place the chart correctly by
    luck.
    """
    target = load_target(QA62)
    grid = target.regions[0].grid
    assert (grid.rows, grid.columns) == (6, 6)
    assert len(target.patches) == 20

    cells = {(p.row, p.col) for p in target.patches}
    assert cells == {
        (r, c) for r in range(6) for c in range(6)
        if r in (0, 5) or c in (0, 5)
    }
    assert sorted(p.ramp_position for p in target.patches) == list(range(1, 21))

    clockwise = (
        [(0, c) for c in range(6)] + [(r, 5) for r in range(1, 5)]
        + [(5, c) for c in range(5, -1, -1)] + [(r, 0) for r in range(4, 0, -1)]
    )
    ordered = sorted(target.patches, key=lambda p: p.ramp_position)
    assert [(p.row, p.col) for p in ordered] == clockwise


def test_the_qa62_lattice_is_square_and_its_patches_are_butted():
    """Both are facts the placement depends on, and both were measured on four units.

    Square cells are why ``implied_quad_aspect`` reads 1.0 and cannot choose a
    rotation; butted patches are why the ROI inset is the only thing holding a sample
    off a boundary.
    """
    grid = load_target(QA62).regions[0].grid
    assert grid.patch_size_relative == grid.patch_spacing_relative
    horizontal, vertical = grid.patch_spacing_relative
    assert abs(horizontal / vertical - 1.0) < 0.001


# -- the placement screen -----------------------------------------------------


def test_the_placement_screen_offers_the_charts_it_found():
    """Several charts in one frame are offered one at a time, not run together.

    A run is one chart: each is analyzed, saved and re-scorable on its own, and the
    library already combines several saved runs into a verdict for the system --
    which is the workflow the session screen was built around and states in its own
    words. So the screen gains a selector and nothing else.

    Hidden for one chart, which is every other capture here, because a control
    offering a choice of one asks the operator to consider an arrangement they have
    not got.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from PySide6.QtWidgets import QApplication
    from deltachrome_ui.placement_view import ChartOffer, PlacementView

    QApplication.instance() or QApplication([])
    view = PlacementView()

    view.offer_charts([ChartOffer()])
    assert not view.chart_combo.isVisibleTo(view)

    seen: list[int] = []
    view.chartSelected.connect(seen.append)
    view.offer_charts([ChartOffer() for _ in range(4)])
    assert view.chart_combo.isVisibleTo(view)
    assert view.chart_combo.count() == 4
    assert view.chart_combo.itemText(0) == "Target 1 of 4"
    assert not seen, "offering the charts must not look like choosing one"


# -- the declared scale -------------------------------------------------------


def test_the_qa62_declares_its_crosshair_spacing_and_not_a_scale_capability():
    """Applied Image state 2.625 in (66.675 mm) centre to centre; the card has no rule.

    Both halves matter. The declared extent is what makes sampling frequency
    measurable, and it is ungated and one-sided -- 294 ppi against 601 measured -- so a
    card printed a percent small cannot move it. ``reproduction_scale_accuracy`` is a
    two percent agreement check against the file's own header, and a rate derived from
    a *declared* dimension carries the chart's print tolerance into the quantity being
    judged: a small card and a wrong header read identically. So the capability stays
    disclaimed and the metric reports N/A rather than a number nothing could check.
    """
    target = load_target(QA62)
    assert target.fiducial_extent_mm == (66.675, 66.675)
    assert target.fiducial_extent_mm[0] == pytest.approx(2.625 * 25.4)
    assert "reproduction_scale" in target.not_capable_of
    assert target.dimensional_reference["available"] is True
    assert not [r for r in target.regions if r.kind == "scale"], (
        "this chart prints no rule; if one is ever declared, the metric should read "
        "it in preference and the capability should follow"
    )


def test_the_declared_extent_agrees_with_the_committed_placements():
    """The check that lets a design dimension be declared per model rather than measured.

    Four cards in one capture, placed independently, must imply one sampling rate. They
    give 601.1 to 602.1 ppi against a nominal 600 -- which a wrong figure would not do
    on four specimens at once. Asserted against the quads
    ``tools/render_target_overlays.py`` commits, so a change to either has to face this.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    from render_target_overlays import QA62_BOARD

    extent = load_target(QA62).fiducial_extent_mm[0]
    rates = []
    for entry in QA62_BOARD:
        quad = np.asarray(entry.corners, float)
        side = np.mean([
            np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]),
            np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]),
        ])
        rates.append(side / extent * 25.4)

    assert len(rates) == 4
    assert min(rates) > 600.0 and max(rates) < 603.0, rates
    assert (max(rates) - min(rates)) / np.mean(rates) < 0.005, (
        f"four specimens should agree on the rate to well under a percent: {rates}"
    )


# -- several charts of different models ---------------------------------------


def _two_targets_drawn(tmp_path):
    """A frame holding a QA-62 and a plain patch array, well apart.

    Two models rather than two of one, which is the case per-chart identification
    exists for. The second is drawn as the ColorChecker SG's lattice because it is
    the chart whose geometry the suite already knows how to draw badly and well.
    """
    canvas = np.full((1500, 3000, 3), 0.06, np.float64)
    quad = _draw_qa62(canvas, (200, 400, 800, 1000))
    # A second, unmistakably different chart: a coarse grid of neutral steps on a
    # light card, nothing like the QA-62's ring.
    canvas[400:1100, 1900:2800] = 0.62
    for row in range(4):
        for col in range(6):
            level = 0.88 - 0.11 * (row * 6 + col) / 23.0
            y, x = 460 + row * 150, 1960 + col * 140
            canvas[y:y + 110, x:x + 110] = level
    encoded = np.round(np.clip(canvas, 0.0, 1.0) * 65535.0).astype(np.uint16)
    return load_image(
        write_tiff_with_profile(tmp_path / "two.tif", encoded, adobe_rgb_icc())
    ), quad


def test_each_region_gets_its_own_answer(tmp_path):
    """identify_targets asks the same question of every region, not a harder one.

    A region that named a chart, one that shortlisted two it could not separate and
    one that declined are all outcomes, and all three come back. Only a region where
    no finder placed anything at all is silent, because that is not evidence of a
    chart.
    """
    from deltachrome.imaging.detect import identify_targets

    image, quad = _two_targets_drawn(tmp_path)
    results = identify_targets(image)

    assert results, "a frame with two charts in it must yield outcomes"
    named = [r for r in results if r.found]
    assert any(r.target_id == QA62 for r in named), (
        f"the QA-62 should be among {[r.target_id for r in named]}"
    )
    for result in results:
        assert result.reason, "every outcome says what it decided and why"
        assert result.candidates, "a silent region must not be reported at all"


def test_a_named_chart_carries_the_placement_of_its_own_region(tmp_path):
    """The placement returned is the one made in that region, not a second search.

    It used to re-run detect_placement over the whole frame, which is the same answer
    on a single-chart capture and the wrong region's answer on a frame holding two.
    """
    from deltachrome.imaging.detect import identify_targets

    image, quad = _two_targets_drawn(tmp_path)
    qa62 = next(
        (r for r in identify_targets(image) if r.found and r.target_id == QA62), None
    )
    assert qa62 is not None and qa62.detection is not None and qa62.detection.found
    side = np.linalg.norm(quad[1] - quad[0])
    assert _matches(qa62.detection.corners, quad, side)


def test_one_chart_in_a_frame_is_identified_exactly_as_it_always_was(tmp_path):
    """The compatibility claim for identification, as for placement."""
    from deltachrome.imaging.detect import identify_target, identify_targets

    image, _quad = _board(tmp_path, [(180, 160, 700, 660)])
    several = identify_targets(image)
    alone = identify_target(image)

    assert len(several) == 1
    assert several[0].found == alone.found
    assert several[0].target_id == alone.target_id


# -- the strip finder, which used to read the whole frame ---------------------


def test_the_strip_finder_works_inside_the_extent_it_is_handed(tmp_path):
    """It took an extent and used it only in the signature.

    Every size gate was a fraction of the whole frame, which is right by accident
    when the strip fills its capture and wrong by a factor when it does not: on a
    frame where the strip is a third of the width the smallest admissible patch came
    out larger than any patch the chart prints, and it refused having found nothing.
    Asserted by handing one finder two extents of very different size over the same
    pixels -- a finder that ignores its extent cannot tell them apart.
    """
    import inspect
    from deltachrome.imaging import detect

    source = inspect.getsource(detect.find_patch_strip)
    assert "extent.box" in source, (
        "find_patch_strip must read the extent it is given; every other finder does"
    )
    assert "image.pixels[::step, ::step" not in source, (
        "the saturation raster must be cut from the extent, not from the whole frame"
    )


def test_a_finder_refuses_rather_than_raising_on_regions_that_are_not_its_chart():
    """A finder returns a reason or a placement, and never raises.

    The strip finder had a path that reached a zero-length array and threw a numpy
    ValueError: enough saturated blobs to pass its count, none of them agreeing on a
    height or a row. Unreachable while it read whole frames, ordinary once it reads
    small regions -- which is the shape of bug that arrives with a change of scope
    rather than with a change of logic.
    """
    import inspect
    from deltachrome.imaging import detect

    source = inspect.getsource(detect.find_patch_strip)
    guard = source.index("blobs = blobs[")
    fit = source.index("along_axis = 0 if")
    assert "raise DetectionRefused" in source[guard:fit], (
        "the height/row filter can empty the array, and what follows reduces over it"
    )


# -- the frontend, per chart --------------------------------------------------


def test_the_placement_screen_switches_target_with_the_chart():
    """Every detected chart is treated the same and carries its own target.

    Same-model is the case where they coincide, not a separate mode: a board of
    QA-62 cards is four placements of one target and a device-level target beside a
    colour chart is two placements of two, and the screen does not know the
    difference. Selecting a chart therefore switches the target, so the sampled
    regions drawn over it, the reference requirement and the run that follows all
    belong to the chart on screen.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from PySide6.QtWidgets import QApplication
    from deltachrome_ui.app import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    sg, qa62 = load_target("colorchecker_sg_v1"), load_target(QA62)

    from deltachrome.imaging.detect import Detection

    def placed(corners):
        return Detection(corners=corners, found=True, method="printed_crosshairs",
                         reason="placed")

    window._offer_charts(
        [placed([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]),
         placed([(20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0)])],
        [sg, qa62],
    )
    window._select_chart(0)
    assert window.placement.choices.target_id == "colorchecker_sg_v1"
    window._select_chart(1)
    assert window.placement.choices.target_id == QA62


def test_a_region_that_could_not_be_named_reaches_the_record():
    """A chart nobody measured has to be visible in the archive.

    Identification declines rather than coin-tossing where two targets fit equally,
    and a run that quietly measured three of four charts would be worse than one that
    measured three and said so -- the fourth would read as absent rather than
    undecided. So the count travels into the run's own warnings, not only onto a
    screen somebody has since closed.

    **The count and not the reasons.** Each reason is why no target fitted, the
    same sentence for every such region, and three of them joined by semicolons was
    a paragraph nobody could act on.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from PySide6.QtWidgets import QApplication
    from deltachrome_ui.app import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._unnamed = ["This capture could be A or B, and nothing separates them."]

    assert window._placement_warnings() == [
        "1 other region looked like a chart and couldn't be identified."
    ]
    window._unnamed = ["first", "second", "third"]
    assert window._placement_warnings() == [
        "3 other regions looked like charts and couldn't be identified."
    ]
    window._unnamed = []
    assert window._placement_warnings() == []


def _outcome(box, *, named):
    """An identification of ``box``, named or not, as identify_targets builds one."""
    from deltachrome.imaging.detect import Candidate, Detection, Identification

    x0, y0, x1, y1 = box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return Identification(
        target_id="named" if named else None,
        detection=Detection(found=True, corners=corners) if named else None,
        candidates=[Candidate(target_id="t", display_name="T", corners=corners)],
        found=named,
        reason="why",
    )


def test_a_refusal_sitting_on_a_named_chart_is_not_another_chart():
    """The Device Level Target's sample is one chart and reported three more.

    The region search returns pieces of a chart as well as charts, and other
    targets' finders fitted something on three pieces of it before their patches
    refused them. Those pixels were measured -- they are the named chart -- so a
    refusal there is not a chart that went unmeasured. One elsewhere still is.
    """
    from deltachrome.imaging.detect import _not_a_named_chart_again

    def quad(x0, y0, x1, y1):
        return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=np.float64)

    chart = _outcome((3372, 2667, 8238, 5713), named=True)
    on_it = _outcome((4378, 2341, 8561, 6030), named=False)
    # Centre past the chart's edge and half of it over the chart: the SG's case.
    straddling = _outcome((6000, 2667, 10000, 5713), named=False)
    # Only a region box, its centre inside the chart: the Device Level's region 3.
    region_only = _outcome((2628, 2064, 4296, 6360), named=False)
    beside = _outcome((9500, 2667, 11000, 5713), named=False)
    found = [on_it, chart, straddling, region_only, beside]
    places = [quad(4378, 2341, 8561, 6030), quad(3372, 2667, 8238, 5713),
              quad(6000, 2667, 10000, 5713), (2628, 2064, 4296, 6360),
              quad(9500, 2667, 11000, 5713)]

    assert _not_a_named_chart_again(found, places) == [chart, beside]
    # Nothing named, nothing to sit on: every refusal is reported.
    assert _not_a_named_chart_again([on_it, beside], [places[0], places[4]]) == [
        on_it, beside
    ]


def test_a_shortlisted_region_is_offered_on_screen_and_skipped_on_the_command_line():
    """The asymmetry this application already rests on, applied to identification.

    Naming the wrong chart is worse than naming none *when nobody is looking*. The
    desktop operator sees the chart, its target and its sampled regions and confirms
    them before anything is measured; a scripted run has no such moment. So a region
    two targets fit equally is offered there with its leading candidate selected and
    the ambiguity stated, and skipped here -- and dropping it on screen would make a
    chart that is certainly present read as absent rather than as undecided.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from PySide6.QtWidgets import QApplication
    from deltachrome.imaging.detect import Candidate, Detection, Identification
    from deltachrome_ui.app import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()

    corners = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    shortlist = Identification(
        found=False,
        reason="This capture could be A or B, and nothing separates them.",
        candidates=[
            Candidate(target_id=QA62, display_name="QA-62", corners=corners),
            Candidate(target_id="microform_35mm_negative_v1", display_name="Microform"),
        ],
        detection=Detection(corners=corners, found=True,
                            method="printed_crosshairs", reason="placed"),
    )
    window._on_charts_identified([shortlist])

    assert len(window._detections) == 1, "a region a chart was placed in is offered"
    assert not window._unnamed, "it is offered rather than reported as skipped"
    assert window._chart_targets[0].target_id == QA62, "the leader is selected"

    window._select_chart(0)
    assert window.placement.choices.target_pill.text() == "Confirm the Target", (
        "the chip must say the choice is not settled, not present it as a decision"
    )
    assert any("could not be named outright" in note
               for note in window._placement_notes), (
        "and the record must carry that somebody confirmed it"
    )


def test_a_shortlist_still_says_where_the_chart_is():
    """The shortlisted targets fitted the same four corners -- that is what made them
    a shortlist -- so the leader's placement is the chart's placement whichever of
    them it turns out to be. Carrying it is what lets a caller show the operator the
    chart rather than only the question."""
    from deltachrome.imaging.detect import identify_target

    import inspect
    from deltachrome.imaging import detect

    source = inspect.getsource(detect._identify_target_once)
    branch = source.index("if len(close) > 1:")
    rest = source[branch:source.index("target_id=best.target_id")]
    assert "detection=_place_in(" in rest, (
        "a shortlist must carry the leading candidate's placement"
    )


# -- running a board -----------------------------------------------------------


def test_the_primary_action_says_how_many_charts_it_will_run():
    """A board is laid out to measure a system rather than a card.

    Stepping through the chart selector is for *checking* each placement before
    committing; the button commits all of them. Going back to the placement screen
    between charts was the alternative and is too much to ask -- five charts would be
    five round trips through a screen whose work was already done before the first
    run.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from PySide6.QtWidgets import QApplication
    from deltachrome_ui.placement_view import ChartOffer, PlacementView

    QApplication.instance() or QApplication([])
    view = PlacementView()

    view.offer_charts([ChartOffer()])
    assert view.continue_button.text() == "Run Analysis"
    view.offer_charts([ChartOffer() for _ in range(5)])
    assert view.continue_button.text() == "Run 5 Targets"
    view.offer_charts([ChartOffer()])
    assert view.continue_button.text() == "Run Analysis", "and back again"


def test_a_freshly_measured_board_can_be_walked_before_anything_is_saved():
    """The session's captures are looked up in memory first, then in the library.

    A session assembled from saved records can send the operator to each run's own
    report because the library holds them. A board just measured has not been saved,
    so the library cannot answer for it -- and being unable to look at chart three
    until you have committed all five is exactly backwards.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from PySide6.QtWidgets import QApplication
    from deltachrome_ui.app import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()

    class FakeRun:
        run_id = "fresh-1"

    class FakeResult:
        run = FakeRun()
        evaluation = None

    window._fresh_results = {"fresh-1": FakeResult()}
    seen = {}
    window.result.set_result = lambda result, saved, **rest: seen.update(
        result=result, saved=saved, **rest
    )
    window.result.set_proxy = lambda proxy: None

    window._open_run_by_id("fresh-1")
    assert seen["result"] is window._fresh_results["fresh-1"]
    assert seen["saved"] is False, "a run that has not been saved must not claim to be"


def test_leaving_a_report_returns_to_the_session_it_was_opened_from():
    """The combined verdict is the answer a board was shot for; the individual
    reports are the detail underneath it. Sending the operator to the library
    instead would lose the thing they were reading."""
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from PySide6.QtWidgets import QApplication
    from deltachrome_ui.app import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    window.session.session = object()

    window._came_from_session = True
    window._leave_report()
    assert window.stack.currentWidget() is window.session

    window._leave_report()
    assert window.stack.currentWidget() is window.library, (
        "and only once -- a second back goes out, not round in a circle"
    )


# -- the report, chart by chart ------------------------------------------------


def test_the_report_offers_the_capture_s_other_charts():
    """Switching charts on the report is a read, not a run.

    Every chart's readings were measured when the capture was and are in the record
    beside the one on screen, so a selector costs a parse and a redraw. That is what
    makes it the right control -- and what made returning to the placement screen
    between charts the wrong one, since the placements were already found and checked
    before the first chart ran.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from PySide6.QtWidgets import QApplication
    from deltachrome_ui.result_view import ResultView

    QApplication.instance() or QApplication([])
    view = ResultView()

    class Result:
        def __init__(self, index, count):
            self.chart_index, self.chart_count = index, count

    view.chart_index, view.chart_count = 0, 1
    assert view.chart_count == 1, "a lone chart offers no choice"

    asked: list[int] = []
    view.chartRequested.connect(asked.append)
    view.chart_index, view.chart_count = 2, 5
    view.chartRequested.emit(3)
    assert asked == [3]


def test_saving_a_board_writes_one_record_holding_every_chart():
    """A board is one capture of one system at one moment; five records in a library
    is five things to find again and keep together by memory."""
    from deltachrome.results.models import CaptureRecord
    from deltachrome.session import capture_record

    class Fake:
        def __init__(self, run, target_name):
            self.run = run
            self.evaluation = type("E", (), {
                "results": [],
                "profile": "fadgi_3star_documents_unbound_general_2023",
                "overall": "PASS", "conformance_claim": True,
            })()
            self.target = type("T", (), {
                "target_id": target_name, "display_name": target_name,
            })()
            self.overall = "PASS"
            self.conformance_claim = True
            self.warnings: list[str] = []

    from deltachrome.results.models import AnalysisRun

    def run_of(run_id):
        return AnalysisRun(
            run_id=run_id, created_utc="2026-01-01T00:00:00+00:00",
            image={"filename": "board.tif"}, target={"display_name": "QA-62"},
            reference={}, profile={"profile_id": "p"}, geometry={}, sampling={},
            color_management={}, patches=[], edges=[], edge_field_summary={},
            registration=[], registration_summary={}, rulers={}, metrics=[],
            overall="PASS", overall_explanation="",
        )

    runs = [Fake(run_of("a"), "QA-62"), Fake(run_of("b"), "QA-62")]
    record = capture_record(runs, labels=["chart 1", "chart 2"])

    assert isinstance(record, CaptureRecord)
    assert record.capture_id == "a", "named for the capture, not for a chart"
    assert len(record.charts) == 2
    assert record.image["filename"] == "board.tif"
    assert [c.to_dict() for c in record.charts] == [r.run.to_dict() for r in runs]


def test_the_chart_selector_is_there_on_a_fresh_run_as_well_as_a_saved_one():
    """The two kinds of result reach the report by different routes, and the screen
    has to be *told* which chart it is showing rather than sniff for it.

    It sniffed, with getattr defaults of 0 and 1. A saved capture carries those
    attributes and a freshly measured chart does not, so a board that had just been
    measured reported one chart and hid the selector -- available only after saving,
    which is exactly backwards: the moment you most want to compare five charts is
    before you have committed to any of them.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from PySide6.QtWidgets import QApplication
    from deltachrome_ui.result_view import ResultView

    QApplication.instance() or QApplication([])
    view = ResultView()

    import inspect

    signature = inspect.signature(ResultView.set_result)
    assert "chart" in signature.parameters, (
        "the position is passed in, not read off the result"
    )
    assert signature.parameters["chart"].default == (0, 1), (
        "and one chart is the default, so a single run behaves as it always did"
    )
    source = inspect.getsource(ResultView.set_result)
    assert 'getattr(result, "chart' not in source, (
        "sniffing for an attribute is what hid the selector on a fresh run"
    )


def test_a_fresh_report_is_not_a_view_onto_a_saved_record():
    """The selector asks a record for another chart when the report came from one,
    and the in-memory list otherwise. A row left over from a previously opened
    capture would send it to the wrong file -- so showing a fresh result clears it."""
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from PySide6.QtWidgets import QApplication
    from deltachrome_ui.app import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    window._open_record = object()

    class Result:
        run = type("R", (), {"run_id": "x", "image": {}, "target": {},
                             "metrics": [], "patches": [], "edges": []})()

    window.result.set_proxy = lambda proxy: None
    window.result.set_result = lambda *a, **k: None
    window._on_analysed(Result())
    assert window._open_record is None


# -- editing the list of charts ------------------------------------------------


def _shell_with_charts(tmp_path, count=2):
    """A shell holding a real capture, because the canvas is part of what is tested.

    The corners live on the canvas between switches, and a canvas with no image
    accepts none -- so a stub proxy would test the bookkeeping and miss the thing
    that was actually broken.
    """
    from PySide6.QtWidgets import QApplication
    from deltachrome.imaging.detect import Detection
    from deltachrome_ui.app import MainWindow
    from deltachrome_ui.imageproxy import ImageProxy

    QApplication.instance() or QApplication([])
    window = MainWindow()
    canvas = np.full((200, 260, 3), 0.5, np.float64)
    image = load_image(write_tiff_with_profile(
        tmp_path / "frame.tif",
        np.round(canvas * 65535.0).astype(np.uint16), adobe_rgb_icc(),
    ))
    window._proxy = ImageProxy(image)
    window._image_path = tmp_path / "frame.tif"
    window.placement.set_proxy(window._proxy, window._image_path)
    window.placement.set_target(load_target(QA62))
    window._offer_charts(
        [Detection(corners=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
                   found=True, method="printed_crosshairs", reason="placed")
         for _ in range(count)],
        [load_target(QA62) for _ in range(count)],
    )
    return window


def _charts_handed_to_the_runner(window):
    """The per-chart ``(target, corners, notes, placed)`` ``_run_every_chart``
    would hand the runner.

    Reached by capturing what it builds rather than by rerunning its arithmetic
    here, so the test reads the same list the record is written from.
    """
    captured = []

    class Stop(Exception):
        pass

    def spy(**kwargs):
        captured.extend(kwargs["charts"])
        raise Stop

    import deltachrome_ui.app as app_module

    original = app_module.ChartsRunner
    app_module.ChartsRunner = spy
    try:
        window._run_every_chart()
    except Stop:
        pass
    finally:
        app_module.ChartsRunner = original
    return captured


def _notes_of_every_chart(window):
    return [note for _t, _c, notes, _p in _charts_handed_to_the_runner(window)
            for note in notes]


def test_how_each_chart_was_placed_is_handed_on_for_the_record(tmp_path):
    """Found, found and then moved, or added by hand -- the record's geometry says
    which, where it used to be a warning on every run."""
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

    window = _shell_with_charts(tmp_path, 3)
    window._chart_dragged[1] = True
    window._chart_added[2] = True
    placed = [p for _t, _c, _n, p in _charts_handed_to_the_runner(window)]
    assert placed == ["automatic", "automatic_then_moved", "by_hand"]
    assert not any("placed automatically" in note or "placed by hand" in note
                   for note in _notes_of_every_chart(window))


def test_a_chart_can_be_added_by_hand_to_a_frame_that_found_one(tmp_path):
    """The frame this exists for is the one where the search found a single chart
    and the operator can see two: charts closer together than the extent search can
    separate come back as one region, and no amount of looking at that region finds
    the second. Rather than leave it unmeasurable, it is added and dragged onto."""
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

    window = _shell_with_charts(tmp_path, 1)
    window._add_chart()

    assert len(window._detections) == 2
    assert window._chart_added == [False, True]
    assert window.placement.chart_combo.count() == 2
    assert window.placement.drop_chart_button.text() == "Remove This Target", (
        "an added chart is removed outright: there is no detection under it to keep"
    )


def test_removing_a_chart_takes_it_out_of_the_frame_entirely(tmp_path):
    """Keeping a *found* chart on screen, greyed, was the earlier answer, and it
    answered the archive's question with the screen.

    The commonest reason to press this is that the search returned a region that
    is not a chart at all -- a QR block on a ColorChecker SG is the case that
    prompted the change -- and a non-chart left in the selector is still something
    the operator steps through, with its own place in `Target 2 of 2` and its own
    ROIs drawn over the capture. So it goes, and its placement with it.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

    window = _shell_with_charts(tmp_path, 3)
    window._drop_chart(1)

    assert len(window._detections) == 2, "removed, not greyed"
    assert window.placement.continue_button.text() == "Run 2 Targets"
    assert window._dropped_charts == 1


def test_what_was_removed_is_still_in_the_record(tmp_path):
    """The evidence argument was right and the screen was the wrong place for it.
    A capture that held more than was measured has to say so in the archive, which
    is the only thing a reader a year later will have."""
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

    window = _shell_with_charts(tmp_path, 3)
    window._drop_chart(1)
    assert any("left out of the run by the operator" in note
               for note in _notes_of_every_chart(window))


def test_removing_every_chart_leaves_nothing_to_run(tmp_path):
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

    window = _shell_with_charts(tmp_path, 2)
    window._drop_chart(0)
    window._drop_chart(0)

    assert not window._detections
    assert not window.placement.continue_button.isEnabled()
    assert "No targets left" in window.placement.continue_button.toolTip()


def test_a_drag_on_one_chart_survives_switching_to_another(tmp_path):
    """Nothing wrote corrections back into the stored placements, so on a
    multi-chart frame every drag was discarded at run time -- which made the
    placement screen's whole purpose a lie: the operator checked and corrected five
    placements and the run read the five the search had stored.

    Read back off the canvas rather than pushed by a signal, at the two moments it
    matters, so a drag, a refinement and a Reset are all caught without each new way
    of moving the handles having to remember to announce itself.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

    window = _shell_with_charts(tmp_path, 2)
    window._select_chart(0)
    moved = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
    window.placement.set_corners(moved)

    window._select_chart(1)
    window._select_chart(0)

    assert [tuple(c) for c in window._detections[0].corners] == moved
    assert window._chart_dragged[0] is True
    assert window._chart_dragged[1] is False, "only the chart that moved"


def _window_holding_two_charts():
    """A shell with two placements in the frame, as a multi-chart capture leaves it."""
    from PySide6.QtWidgets import QApplication
    from deltachrome.imaging.detect import Detection
    from deltachrome_ui.app import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    dlt = load_target("isa_golden_thread_dlt_v1")

    def placed(x):
        return Detection(
            corners=[(x, 0.0), (x + 10, 0.0), (x + 10, 10.0), (x, 10.0)],
            found=True, method=dlt.fiducials.method, reason="placed",
        )

    window.placement.set_target(dlt)
    window._offer_charts([placed(0.0)], [dlt])
    window._select_chart(0)
    # The same call *Add a Target* makes: another chart in the frame, and nothing
    # yet saying which chart it is.
    window._append_chart(None, placed(20.0), doubt=None, added=True)
    return window, dlt


def test_naming_one_chart_does_not_discard_the_others():
    """Reported from the running application, and the multi-chart flow's whole
    point: on a capture whose two charts sit too close for the extent search to
    separate, the operator adds the second by hand and then says which model it is.

    That said "start over" -- the frame was thrown away and searched again for the
    new model -- so the added chart *and* the found one both vanished and the frame
    came back holding one. The combo names the chart in front of you, and on this
    screen that is the chart selected.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    window, dlt = _window_holding_two_charts()
    assert len(window._detections) == 2

    window._on_target_chosen("colorchecker_sg_v1")

    assert len(window._detections) == 2, "the frame must keep both charts"
    assert [t.target_id for t in window._chart_targets] == [
        "isa_golden_thread_dlt_v1", "colorchecker_sg_v1",
    ]
    assert window.placement.target.target_id == "colorchecker_sg_v1"


def test_naming_a_hand_placed_chart_keeps_the_placement_somebody_made():
    """There is no detection under an added chart to redo, and the operator put it
    where it is. Searching again would discard a placement made deliberately."""
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    window, _dlt = _window_holding_two_charts()
    window._select_chart(1)
    corners = list(window._detections[1].corners)

    window._on_target_chosen("colorchecker_sg_v1")

    assert window._detections[1].corners == corners


def test_one_found_chart_still_starts_a_fresh_search():
    """The case the old behaviour was right for, kept exactly. With one chart that
    the search found, the combo really does mean "you named the capture wrongly,
    find this instead" -- so it searches the frame rather than the chart's region.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from PySide6.QtWidgets import QApplication
    from deltachrome.imaging.detect import Detection
    from deltachrome_ui.app import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    dlt = load_target("isa_golden_thread_dlt_v1")
    window.placement.set_target(dlt)
    window._offer_charts([Detection(
        corners=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        found=True, method=dlt.fiducials.method, reason="placed")], [dlt])
    window._select_chart(0)

    assert not window._retargets_one_chart()


def test_an_added_chart_shows_no_regions_until_it_is_placed():
    """Reported from the running application. *Add a Target* puts a chart on the
    canvas's default rectangle in the middle of the frame, and the overlay drew its
    ROIs there immediately -- over pixels nobody had claimed were a chart.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    window, _dlt = _window_holding_two_charts()

    # The added chart is shown, and has nowhere to be yet.
    assert window._shown_chart == 1
    assert window._chart_added[1] and not window._chart_dragged[1]
    assert not window.placement.placement_settled
    assert not window.placement.drawing_regions

    # The found chart beside it is a placement, and keeps its regions.
    window._select_chart(0)
    assert window.placement.placement_settled


def test_a_chart_that_was_moved_keeps_its_regions_when_you_come_back():
    """The first drag settles it, and stepping away and back must not un-settle it
    -- the operator placed it, and that fact outlives the selection."""
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    window, _dlt = _window_holding_two_charts()
    window._chart_dragged[1] = True

    window._select_chart(0)
    window._select_chart(1)
    assert window.placement.placement_settled


def test_a_new_capture_holds_none_of_the_last_one_s_charts():
    """Reported from the running application: add a target, go back to the
    library, open the same capture, and the selector reads "Target 3 of 3".

    Nothing cleared the list at the way in. ``_apply_target`` used to do it on the
    way through and now short-circuits when the frame already holds several -- so
    the previous capture's charts survived and every visit added to the count.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    window, _dlt = _window_holding_two_charts()
    assert len(window._detections) == 2

    window._forget_charts()

    assert window._detections == []
    assert window._chart_targets == []
    assert window._chart_added == []
    assert window._chart_dragged == []
    assert window._shown_chart is None
    # And with the frame empty, a target change is a fresh search again rather
    # than a retarget of a chart that is no longer there.
    assert not window._retargets_one_chart()


def test_a_capture_nothing_was_named_in_holds_no_charts():
    """Reported as a frame offering three targets where two were asked for.

    The setup panel outlives the capture, so the previous one's chart stayed
    selected when identification declined -- and the screen then ran a search for
    *that* chart, placed whatever it found, and offered it as a chart of a capture
    nothing had been identified in. Add a Target then made it two, and a target
    with no definition at all made it three.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from deltachrome.imaging.detect import Identification

    window, _dlt = _window_holding_two_charts()
    assert len(window._detections) == 2

    window._on_charts_identified([Identification(
        reason="No installed target could be found in this capture.")])

    assert window._detections == []
    assert not window.placement.choices.has_target
    assert not window.placement.target_settled


def test_a_target_cannot_be_added_before_one_is_chosen():
    """A chart with no definition has no patches, no edges and nothing to measure.
    Built anyway, it crashed the selection handler on the first press -- which is
    what made a frame read as holding a target that was not there."""
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from PySide6.QtWidgets import QApplication
    from deltachrome_ui.app import MainWindow

    QApplication.instance() or QApplication([])
    window = MainWindow()
    window.placement.choices.clear_target()
    window.placement.set_target(None)
    window.placement.update_readiness()

    assert not window.placement.add_chart_button.isEnabled()
    assert "Search a Region" in window.placement.add_chart_button.toolTip()

    # And the guard holds where a caller is not a button.
    window._add_chart()
    assert window._detections == []


def test_a_region_search_fills_an_abandoned_placeholder():
    """A sequence reported from the running application: the DLT is found,
    *Add a Target* is pressed, and then *Search a Region* is used instead of
    dragging it. That gave three charts where two were wanted -- the one found, the
    empty one abandoned, and the one the region named.

    An added chart nobody has moved is a placeholder rather than a chart: it sits
    on a default rectangle, it is not a placement of anything, and its regions are
    already withheld for that reason. Pressing *Add a Target* states an intent to
    place a chart; doing it another way carries out that intent.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from deltachrome.imaging.detect import Detection

    window, _dlt = _window_holding_two_charts()
    assert len(window._detections) == 2
    assert window._unplaced_chart() == 1

    sg = load_target("colorchecker_sg_v1")
    window._append_chart(sg, Detection(
        corners=[(40.0, 40.0), (50.0, 40.0), (50.0, 50.0), (40.0, 50.0)],
        found=True, method=sg.fiducials.method, reason="found in a drawn region"),
        doubt=None, added=False)

    assert len(window._detections) == 2, "the placeholder is filled, not followed"
    assert [t.target_id for t in window._chart_targets] == [
        "isa_golden_thread_dlt_v1", "colorchecker_sg_v1",
    ]
    assert window._chart_added == [False, False]
    assert window._unplaced_chart() is None


def test_a_placed_chart_is_never_overwritten_by_one():
    """The rule is about a placeholder, not about the last chart. A hand-placed
    chart somebody actually moved is a placement and a region search adds beside
    it."""
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    from deltachrome.imaging.detect import Detection

    window, _dlt = _window_holding_two_charts()
    window._chart_dragged[1] = True
    assert window._unplaced_chart() is None

    sg = load_target("colorchecker_sg_v1")
    window._append_chart(sg, Detection(
        corners=[(40.0, 40.0), (50.0, 40.0), (50.0, 50.0), (40.0, 50.0)],
        found=True, method=sg.fiducials.method, reason="found"),
        doubt=None, added=False)

    assert len(window._detections) == 3


def test_add_a_target_twice_leaves_one_placeholder():
    """Two identical default rectangles are two things to step through and nothing
    to look at."""
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    window, _dlt = _window_holding_two_charts()
    before = len(window._detections)

    window._add_chart()

    assert len(window._detections) == before
    assert window._shown_chart == window._unplaced_chart()


def test_adding_a_target_names_no_chart_and_draws_nothing():
    """Carrying the shown chart's target over was the obvious thing and it is the
    wrong offer: a chart of the same model at a default rectangle, with reticles up
    and its ROIs drawn, invites dragging — which is the way of placing a chart this
    screen wants reached for last. Blank says what is true, that the frame holds
    another chart and nothing yet says which or where.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    window, _dlt = _window_holding_two_charts()

    assert window._chart_targets[1] is None
    assert not window.placement.choices.has_target
    assert window.placement.target is None
    # Unsettled is what hides the handles and the regions both. That it does so on
    # a real canvas is pinned in test_ui_canvas; this window has no capture, so
    # the canvas has no handles to hide.
    assert not window.placement.target_settled


def test_naming_the_added_chart_brings_its_reticles_back():
    """The other half: choosing a target is what says there is something to place,
    so that is when the handles appear."""
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    window, _dlt = _window_holding_two_charts()

    window._on_target_chosen("colorchecker_sg_v1")

    assert window._chart_targets[1].target_id == "colorchecker_sg_v1"
    assert window.placement.target_settled
    # The chart that was already there is untouched.
    assert window._chart_targets[0].target_id == "isa_golden_thread_dlt_v1"
    assert len(window._detections) == 2


def _found(x, y, size=10.0, target_id="colorchecker_sg_v1"):
    from deltachrome.imaging.detect import Detection
    return Detection(
        corners=[(x, y), (x + size, y), (x + size, y + size), (x, y + size)],
        found=True, method=load_target(target_id).fiducials.method,
        reason="found in a drawn region",
    )


def test_searching_the_same_chart_again_does_not_add_one():
    """Reported from the running application: every press of *Search a Region*
    put one more target in the frame.

    Drawing a region, disliking what comes back and drawing again is "try this
    again", not "and another chart here". Two charts cannot occupy the same
    pixels, so any real overlap between two placements means they are two
    placements of one chart -- which is a comparison with no threshold to tune.
    """
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    window, _dlt = _window_holding_two_charts()
    sg = load_target("colorchecker_sg_v1")

    window._append_chart(sg, _found(500.0, 500.0), doubt=None, added=False)
    assert len(window._detections) == 2  # filled the placeholder

    # Drawn again over the same chart, landing a few pixels off.
    window._append_chart(sg, _found(503.0, 497.0), doubt=None, added=False)
    assert len(window._detections) == 2

    # And again, with a different target named in the same place.
    window._append_chart(load_target("avian_ngt_v1"), _found(498.0, 502.0),
                         doubt=None, added=False)
    assert len(window._detections) == 2
    assert window._chart_targets[1].target_id == "avian_ngt_v1"


def test_searching_somewhere_else_still_adds_a_chart():
    """The rule is about overlap, not about repetition. A region drawn over a
    different part of the frame is a different chart, which is the whole point of
    being able to draw one."""
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    window, _dlt = _window_holding_two_charts()
    sg = load_target("colorchecker_sg_v1")

    window._append_chart(sg, _found(500.0, 500.0), doubt=None, added=False)
    assert len(window._detections) == 2

    window._append_chart(sg, _found(2000.0, 2000.0), doubt=None, added=False)
    assert len(window._detections) == 3


def test_a_placeholder_never_claims_a_placement_as_its_own():
    """An added chart nobody has moved sits on the canvas's default rectangle -- a
    large box in the middle of the frame that would swallow the centre of almost
    any real chart. It is not a placement of anything, so it is not compared
    against one; it is *filled* instead, which is a different rule."""
    pytest.importorskip("PySide6", reason="frontend tests need the gui extra")
    window, _dlt = _window_holding_two_charts()
    # The placeholder's own box, which encloses the middle of the frame.
    window._detections[1].corners = [(0.0, 0.0), (9000.0, 0.0),
                                     (9000.0, 9000.0), (0.0, 9000.0)]

    assert window._chart_already_here(_found(4000.0, 4000.0)) is None
    assert window._unplaced_chart() == 1


# -- two names for one set of pixels -------------------------------------------


def _answer(target_id, box, *, cost, ramp=1.0, color=2.0):
    """One identification, placed on ``box``, as :func:`identify_targets` returns it."""
    from deltachrome.imaging.detect import Candidate, Detection, Identification

    x0, y0, x1, y1 = box
    candidate = Candidate(
        target_id=target_id, display_name=target_id, ramp=ramp, sampled=1.0,
        flagged=0.0, color=color, cost=cost,
    )
    return Identification(
        target_id=target_id,
        detection=Detection(corners=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)], found=True),
        candidates=[candidate], found=True,
        reason=f"{target_id}, identified from the capture.",
    )


def test_two_charts_cannot_be_in_the_same_place_and_the_better_fit_is_kept():
    r"""The Odonata residual, as the rule rather than as the capture.

    Four of twenty frames named a second chart placed across nearly the whole frame
    and *enclosing* the Object Level strip correctly found inside it -- five of those
    six at ramps of 0.56 to 0.78 and 12 to 31 dE against the strip's +1.00 and 2.1.
    Two charts cannot be in the same place, so those are two readings of one.
    """
    from deltachrome.imaging.detect import _one_chart_per_place

    strip = _answer("strip", (1400, 7400, 10300, 7900), cost=0.05, ramp=1.0, color=2.1)
    frame = _answer("whole_frame", (900, 50, 11300, 8700), cost=0.40, ramp=0.7, color=31.0)

    kept = _one_chart_per_place([frame, strip])
    named = [r.target_id for r in kept if r.found]
    assert named == ["strip"]


def test_the_chart_that_loses_is_reported_and_not_dropped():
    """A chart that quietly disappears reads as "there was no chart there", which is
    a different and more misleading claim than "there was something there and it
    could not be accounted for". So the loser keeps its place in the list, with no
    target on it and a reason naming what displaced it."""
    from deltachrome.imaging.detect import _one_chart_per_place

    strip = _answer("strip", (1400, 7400, 10300, 7900), cost=0.05)
    frame = _answer("whole_frame", (900, 50, 11300, 8700), cost=0.40, ramp=0.7, color=31.0)

    kept = _one_chart_per_place([frame, strip])
    assert len(kept) == 2, "nothing is removed from the list"
    loser = kept[0]
    assert loser.target_id is None and not loser.found
    assert "whole_frame" in loser.reason and "strip" in loser.reason
    assert "same pixels" in loser.reason
    assert loser.candidates, "what was read there is kept, so the record has it"


def test_two_charts_that_share_no_pixels_are_both_kept():
    """The case this must not break, and the one capture here that exercises it: the
    Golden Thread Device Level and ColorChecker SG composite, whose two correct
    answers share no pixels at all."""
    from deltachrome.imaging.detect import _one_chart_per_place

    left = _answer("device_level", (974, 1790, 5840, 4836), cost=0.05)
    right = _answer("colorchecker", (9476, 1191, 15432, 5436), cost=0.06)

    kept = _one_chart_per_place([left, right])
    assert [r.target_id for r in kept] == ["device_level", "colorchecker"]


def test_a_region_that_named_nothing_is_left_exactly_as_it_was():
    """The rule is about two *names* for one place. A region that named nothing has
    no placement to compare and is passed through untouched."""
    from deltachrome.imaging.detect import Identification, _one_chart_per_place

    skipped = Identification(reason="This region is the chart's printed code.")
    strip = _answer("strip", (1400, 7400, 10300, 7900), cost=0.05)

    kept = _one_chart_per_place([skipped, strip])
    assert kept[0] is skipped
    assert kept[1].target_id == "strip"


def test_a_displaced_chart_does_not_go_on_displacing_others():
    """Three readings: a whole-frame one, and two real charts sitting apart inside
    it. The whole-frame reading loses to the first, and must then stop counting --
    otherwise the second is displaced by a reading already decided against."""
    from deltachrome.imaging.detect import _one_chart_per_place

    frame = _answer("whole_frame", (0, 0, 11600, 9100), cost=0.40, ramp=0.7, color=31.0)
    left = _answer("left_chart", (900, 1000, 5000, 4000), cost=0.05)
    right = _answer("right_chart", (6500, 5000, 10800, 8000), cost=0.06)

    kept = _one_chart_per_place([frame, left, right])
    assert [r.target_id for r in kept if r.found] == ["left_chart", "right_chart"]
    assert kept[0].target_id is None


def test_a_board_says_which_cards_a_batch_did_not_measure(tmp_path):
    """The case the whole feature exists for, on a drawn board of four.

    A batch measures one chart per capture. Before this, the other three went
    unreported -- and a chart nobody measured reads as a chart that was not there,
    which is the failure the rest of the module refuses.
    """
    from deltachrome.imaging.detect import extra_charts

    image, drawn = _board(tmp_path, FOUR_ON_A_BOARD, surround=0.24)
    target = load_target(QA62)

    said = extra_charts(image, target, [tuple(p) for p in drawn[0]])
    assert len(said) == 3, said
    assert all("was not measured" in line for line in said)
    assert all(target.display_name in line for line in said)


def test_a_frame_holding_one_chart_says_nothing(tmp_path):
    """The property that decides whether this is usable: it must be silent on the
    ordinary case. Measured on twenty Smithsonian Odonata frames that hold one
    chart each, the declared-target search warned on none of them."""
    from deltachrome.imaging.detect import extra_charts

    image, drawn = _board(tmp_path, [(180, 160, 700, 660)])
    target = load_target(QA62)

    assert extra_charts(image, target, [tuple(p) for p in drawn[0]]) == []


def test_a_placement_that_found_nothing_claims_nothing(tmp_path):
    """The check is a *separate* search from the run -- a batch places from a seed
    and this does not -- so the two can disagree on material where the seed is what
    makes placement work. Comparing against the measured placement rather than
    counting what was found is what keeps that quiet."""
    from deltachrome.imaging.detect import extra_charts

    image, _drawn = _board(tmp_path, [(180, 160, 700, 660)])
    assert extra_charts(image, load_target(QA62), []) == []
