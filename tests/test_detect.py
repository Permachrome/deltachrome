"""Finding a chart in a capture, without anybody clicking on it.

Every capture here is drawn rather than photographed, for the reason the rest of
this suite is: the real sample masters are the Library of Congress's and the
operator's to distribute and are not in the repository, so a test that needed one
would pass on one machine. What is drawn is deliberately crude -- a grey card with
four black pluses on it is not a ColorChecker SG -- because what these tests are
for is the *logic*: that a plus is told from a letter by its shape, that four
points come back in the order the target declares, that a chart that cannot be
found says so instead of guessing.

How well the finders do on real captures is a separate question and is measured
separately, against the seventeen committed placements in
``tools/render_target_overlays.py``: every one of them is re-derived to within
0.63% of the chart's shorter side and thirteen to under a pixel. That number
cannot live here.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deltachrome.imaging.detect import (
    FINDERS,
    implied_quad_aspect,
    ramp_agreement,
    shape_deviation,
    Detection,
    DetectionRefused,
    chart_extent,
    detect_placement,
    find_patch_array,
    find_printed_crosshairs,
    find_printed_discs,
    find_printed_frame,
    order_corners,
)
from deltachrome.imaging.loader import load_image
from deltachrome.targets.registry import available_targets, load_target

from tests.fixtures import adobe_rgb_icc, write_tiff_with_profile

#: Methods no finder covers yet, and why. Named rather than merely absent so that
#: adding a target declaring a new method fails this test rather than silently
#: joining a list of things that quietly cannot be placed.
NOT_YET_FINDABLE = {
    "corner_squares_and_dot": (
        "three black corner squares and a dot that is one of 75 on a regular grid; no "
        "finder has been written for them, as none could be for the dpi fields the "
        "chart used to place on"
    ),
}


def _card(width=900, height=700, level=0.55, channels=3):
    """A blank chart card on a dark surround, as float in 0..1."""
    canvas = np.full((height, width, channels), 0.06, np.float64)
    canvas[40:height - 40, 40:width - 40] = level
    return canvas


def _write(tmp_path, canvas, name="capture.tif"):
    encoded = np.round(np.clip(canvas, 0.0, 1.0) * 65535.0).astype(np.uint16)
    return load_image(
        write_tiff_with_profile(tmp_path / name, encoded, adobe_rgb_icc())
    )


def _plus(canvas, centre, *, span=40, stroke=6, level=0.05):
    x, y = int(centre[0]), int(centre[1])
    half, arm = span // 2, stroke // 2
    canvas[y - arm:y + arm + 1, x - half:x + half + 1] = level
    canvas[y - half:y + half + 1, x - arm:x + arm + 1] = level


# -- dispatch -----------------------------------------------------------------


def test_every_shipped_target_says_how_it_is_placed_and_is_answered():
    """No target may declare a placement method nothing has heard of.

    The failure this prevents is silent: a new target whose method is not in
    ``FINDERS`` and not on the list above would simply never be found, and the
    operator would see a refusal with no way to tell a chart that cannot be
    placed from one nobody has taught it to place.
    """
    unanswered = {}
    for target_id in available_targets():
        method = load_target(target_id).fiducials.method
        if method not in FINDERS and method not in NOT_YET_FINDABLE:
            unanswered[target_id] = method
    assert not unanswered, (
        f"targets declaring a placement method with no finder and no reason: "
        f"{unanswered}"
    )


def test_no_finder_is_registered_for_a_method_no_target_declares():
    """A dead dispatch entry is a claim about a chart that does not exist."""
    declared = {load_target(t).fiducials.method for t in available_targets()}
    assert not set(FINDERS) - declared


def test_a_method_with_no_finder_refuses_and_names_itself(tmp_path):
    """The two-target case: a clear refusal, not a crash and not a guess."""
    image = _write(tmp_path, _card())
    for target_id in available_targets():
        target = load_target(target_id)
        if target.fiducials.method not in NOT_YET_FINDABLE:
            continue
        detection = detect_placement(image, target, refine=False)
        assert not detection.found
        assert not detection.corners
        assert target.fiducials.method.replace("_", " ") in detection.reason
        assert "by hand" in detection.reason


def test_a_blank_card_is_refused_by_every_finder(tmp_path):
    """Nothing printed, nothing found -- and nothing raised, either.

    A finder that throws takes the whole run down where a refusal only sends the
    operator to the placement screen they would have used anyway, so every path
    out of :func:`detect_placement` has to be a value.
    """
    image = _write(tmp_path, _card())
    for target_id in available_targets():
        target = load_target(target_id)
        # The flat field is a white sheet whose placement IS the whole frame, so
        # it is found on a blank card and correctly so.
        if target.fiducials.method == "chart_corners":
            continue
        detection = detect_placement(image, target, refine=False)
        assert isinstance(detection, Detection)
        assert not detection.found, f"{target_id} claimed to find a blank card"
        assert detection.reason


def test_the_flat_field_is_placed_on_the_whole_frame(tmp_path):
    image = _write(tmp_path, _card())
    detection = detect_placement(image, load_target("flat_field_v1"), refine=False)
    assert detection.found
    assert detection.corners[0] == (0.0, 0.0)
    assert detection.corners[2] == (image.width - 1.0, image.height - 1.0)


# -- ordering -----------------------------------------------------------------


def test_corners_come_back_in_the_targets_own_order():
    """A rectangle found in any order is returned TL, TR, BR, BL."""
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    scrambled = [(100, 900), (900, 100), (100, 100), (900, 900)]
    ordered, _report = order_corners(scrambled, square)
    assert ordered == [(100.0, 100.0), (900.0, 100.0), (900.0, 900.0), (100.0, 900.0)]


def test_a_diamond_is_not_sorted_into_rows():
    """The Device Level Target's discs, which a row-and-column sort bow-ties.

    Its four marks sit one at each extreme about the chart's centre rather than
    two to a row, so a sort into a top row and a bottom row puts two of them in
    the same row by accident of a few pixels -- and the resulting bow tie is a
    perfectly valid projective transform that makes every region read the frame's
    average.
    """
    target = load_target("isa_golden_thread_dlt_v1")
    truth = [(6100.8, 2666.7), (8238.5, 3870.1), (5816.1, 5712.2), (3372.1, 4206.6)]
    ordered, _report = order_corners(
        list(reversed(truth)), target.fiducials.corner_positions,
        cell_aspect=implied_quad_aspect(target),
    )
    assert np.asarray(ordered) == pytest.approx(np.asarray(truth), abs=0.1)


def test_a_mirrored_assignment_is_never_chosen():
    """Relative space and pixel space share their handedness, so a flip is wrong.

    It has to be excluded rather than merely disfavoured: a reflection composed
    with a small rotation reads as almost *no* rotation, so on any tie-break that
    prefers the least-turned answer the mirror image wins. On the Golden Thread
    Object Level Target it returned the right quad upside down.
    """
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    strip = [(200.0, 300.0), (3600.0, 305.0), (3600.0, 405.0), (200.0, 400.0)]
    ordered, _report = order_corners(strip, square)
    assert ordered[0][1] < ordered[3][1], "top-left came back below bottom-left"


def test_a_square_set_of_marks_says_its_orientation_cannot_be_told():
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    _ordered, report = order_corners(
        [(100, 100), (900, 100), (900, 900), (100, 900)], square
    )
    assert any("quarter turn" in note for note in report["assignment_notes"])


def test_the_note_says_which_kind_of_ambiguity_it_is():
    """Square marks cannot say which quarter turn; oblong ones cannot say which half.

    Two different states of knowledge and two different things for an operator to
    check, so they are not one message.
    """
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    _ordered, oblong = order_corners(
        [(100, 100), (900, 100), (900, 400), (100, 400)], square
    )
    assert "upside down" in oblong["assignment_notes"][0]
    _ordered, boxy = order_corners(
        [(100, 100), (900, 100), (900, 900), (100, 900)], square
    )
    assert "quarter turn" in boxy["assignment_notes"][0]


def test_the_aspect_a_grid_implies_places_a_chart_shot_a_quarter_turn_round():
    """The bug this test exists for was found in the application, not here.

    The X-Rite ColorChecker SG capture is shot in portrait, because that is how a
    wide chart fits a portrait sensor. Its four crosshairs were found correctly and
    labelled a quarter turn out, and nothing in the outline could say so: the four
    points are the same four points whichever is called top-left, and the SG's
    declared corners -- like almost every target's -- are a unit square.
    """
    target = load_target("colorchecker_sg_v1")
    implied = implied_quad_aspect(target)
    assert implied == pytest.approx(1.404, abs=0.01)

    # A quad taller than it is wide: the chart on its side.
    portrait = [(1000.0, 1000.0), (3800.0, 1000.0), (3800.0, 4900.0), (1000.0, 4900.0)]
    ordered, report = order_corners(
        portrait, target.fiducials.corner_positions, cell_aspect=implied,
    )
    assert report["assignment_decided_by"].startswith("the aspect")
    # Top-left goes to the chart's own top-left, which is the frame's bottom-left.
    assert ordered[0] == (1000.0, 4900.0)
    # The aspect rules out the two rotations a quarter turn away and cannot rule out
    # the one half a turn away, which is what the ramp is for.
    assert len(report["assignment_tied"]) == 2


def _placed_on_its_points(target, scale=2000.0):
    """The four pixel positions an exact placement of ``target`` would have.

    The frame is mapped to pixels at the proportions the chart's own grid implies,
    so the result is a placement that is right by construction, and the declared
    points are read off it -- wherever on the chart they are."""
    aspect = implied_quad_aspect(target)
    frame = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    pixels = (frame * np.array([scale * aspect, scale], np.float32)) + 500
    matrix = cv2.getPerspectiveTransform(frame, pixels)
    declared = np.asarray(target.fiducials.corner_positions, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(declared, matrix).reshape(-1, 2)


@pytest.mark.parametrize("target_id", [
    "dt_nexus_transmissive_ntpt_rev_a_v1", "dt_nexus_transmissive_ntvt_rev_b_v1",
    "isa_fadgi_19264_v1", "isa_fadgi_19264_commercial_v1", "isa_transmissive_4x5_v1",
    "colorchecker_sg_v1",
])
def test_the_shape_check_reads_charts_placed_off_their_frame_corners(target_id):
    """The quarter-turn warning measures the chart's FRAME through the placement,
    so it does not need the placement points to be the frame's corners. It used to
    require them to be, and so was silent on every chart placed anywhere else --
    and went silent on two more when the DT Nexus charts and the early ISA FADGI-ISO
    19264 moved onto marks a person can see. A right placement reads near zero and
    the same four points a quarter turn round read far over the threshold."""
    target = load_target(target_id)
    corners = _placed_on_its_points(target)
    assert shape_deviation(target, corners) == pytest.approx(0.0, abs=1e-3)
    assert shape_deviation(target, np.roll(corners, 1, axis=0)) > 0.3


def test_the_shape_check_stays_off_the_device_level_target():
    """The one chart its square-cell premise is wrong about: the Device Level
    Target's strips are not square and the grid implies 2.15 against a real 1.75,
    which would warn on every correct placement. Its placement points are a diamond,
    and a diamond is the one layout the check declines."""
    target = load_target("isa_golden_thread_dlt_v1")
    assert shape_deviation(target, _placed_on_its_points(target)) is None


def test_the_declared_frame_would_choose_the_wrong_rotation():
    """Why the obvious second test is not one, pinned so it is not reinvented.

    A target's declared corners look like better evidence than an inference from its
    cells being square -- they are a direct statement about where its fiducials sit.
    They are silent on fourteen of the seventeen shipped targets, and on the one they
    were meant to decide they are **wrong**: the Golden Thread Device Level Target's
    diamond fits its four rotations to 0.248, 0.255, 0.264 and 0.285, and the best of
    those is the placement a quarter turn out.
    """
    from deltachrome.imaging.detect import _similarity_residual

    target = load_target("isa_golden_thread_dlt_v1")
    truth = np.array(
        [(6100.8, 2666.7), (8238.5, 3870.1), (5816.1, 5712.2), (3372.1, 4206.6)]
    )
    aim = np.asarray(target.fiducials.corner_positions, float)
    residuals = [
        _similarity_residual(aim, np.roll(truth, turn, axis=0))[0] for turn in range(4)
    ]
    assert int(np.argmin(residuals)) != 0, (
        "the declared frame now prefers the correct rotation; if that is a real "
        "improvement it can be used again, but check every target before doing so"
    )


def test_four_points_are_needed():
    with pytest.raises(DetectionRefused):
        order_corners([(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1), (0, 1)])


# -- the finders, on drawn charts ---------------------------------------------


def test_printed_crosshairs_are_found_where_they_were_drawn(tmp_path):
    canvas = _card()
    marks = [(150, 130), (750, 130), (750, 570), (150, 570)]
    for mark in marks:
        _plus(canvas, mark)
    image = _write(tmp_path, canvas)
    found, report = find_printed_crosshairs(
        image, load_target("colorchecker_sg_v1"), chart_extent(image)
    )
    assert np.ravel(sorted(found)) == pytest.approx(np.ravel(sorted(marks)), abs=1.0)
    assert report["marks"] == "dark"


def test_a_letter_is_not_mistaken_for_a_crosshair(tmp_path):
    """The trap that decided the shape test.

    Nothing about size, squareness or fill separates a printed capital from a
    registration mark -- on the SG an ``H`` beats the real crosshair on all three
    -- so what has to reject it is that its bounding box has ink in the corners.
    """
    canvas = _card()
    for mark in [(150, 130), (750, 130), (750, 570)]:
        _plus(canvas, mark)
    # An H where the fourth crosshair should be.
    canvas[550:591, 130:137] = 0.05
    canvas[550:591, 164:171] = 0.05
    canvas[567:574, 130:171] = 0.05
    image = _write(tmp_path, canvas)
    with pytest.raises(DetectionRefused):
        find_printed_crosshairs(
            image, load_target("colorchecker_sg_v1"), chart_extent(image)
        )


def test_a_bright_cross_on_a_dark_ground_is_found(tmp_path):
    """Polarity belongs to the specimen, not to the model.

    The 35mm microform chart ships as a positive and a negative through one set of
    declared marks, and the positive prints white crosses on a dark ground. It is
    the case that took the threshold sweep off fractions of the card's level,
    which can only ever reach downward from it.
    """
    canvas = _card(level=0.20)
    for mark in [(150, 130), (750, 130), (750, 570), (150, 570)]:
        _plus(canvas, mark, level=0.95)
    image = _write(tmp_path, canvas)
    found, report = find_printed_crosshairs(
        image, load_target("microform_35mm_positive_v1"), chart_extent(image)
    )
    assert report["marks"] == "light"
    assert len(found) == 4


def test_a_printed_frame_is_found_at_its_line_intersections(tmp_path):
    canvas = _card()
    box = (120, 100, 780, 600)
    # A thin rule, drawn thin on purpose: a printed keyline is a few pixels
    # on a real master of several thousand, and the finder rejects anything
    # thicker as being something other than a rule.
    cv2.rectangle(canvas, box[:2], box[2:], (0.05, 0.05, 0.05), thickness=2)
    image = _write(tmp_path, canvas)
    found, report = find_printed_frame(
        image, load_target("isa_transmissive_4x5_v1"), chart_extent(image)
    )
    ordered, _ = order_corners(found, [(0, 0), (1, 0), (1, 1), (0, 1)])
    assert ordered[0] == pytest.approx((box[0], box[1]), abs=1.5)
    assert ordered[2] == pytest.approx((box[2], box[3]), abs=1.5)
    assert all(entry["rms_px"] < 1.0 for entry in report["sides"].values()
               if isinstance(entry, dict))


def test_a_frame_is_told_from_a_thick_bar_beside_it(tmp_path):
    """The LaserSoft prints a barcode below its frame rule.

    A scan up from the card for the first dark thing finds the bars, and the
    bottom side then fits through nothing -- so a boundary is only a rule if its
    dark run is thin, and the sides that were not deceived are what say so.
    """
    canvas = _card()
    box = (120, 100, 780, 600)
    cv2.rectangle(canvas, box[:2], box[2:], (0.05, 0.05, 0.05), thickness=2)
    canvas[620:650, 200:700] = 0.04
    image = _write(tmp_path, canvas)
    found, _report = find_printed_frame(
        image, load_target("isa_transmissive_4x5_v1"), chart_extent(image)
    )
    ordered, _ = order_corners(found, [(0, 0), (1, 0), (1, 1), (0, 1)])
    assert ordered[2] == pytest.approx((box[2], box[3]), abs=1.5)


def test_printed_discs_are_found_as_a_diamond(tmp_path):
    canvas = _card()
    marks = [(450, 120), (790, 350), (450, 580), (110, 350)]
    for mark in marks:
        cv2.circle(canvas, mark, 34, (0.06, 0.06, 0.06), thickness=-1)
    image = _write(tmp_path, canvas)
    found, report = find_printed_discs(
        image, load_target("isa_golden_thread_dlt_v1"), chart_extent(image)
    )
    assert np.ravel(sorted(found)) == pytest.approx(np.ravel(sorted(marks)), abs=2.0)
    # The mark holding still while its apparent size changes is what says the
    # threshold does not matter, and therefore that the mark is real.
    assert report["thresholds"] > 1
    assert report["stability_px"] < 2.0


def test_a_square_is_not_mistaken_for_a_disc(tmp_path):
    """Discs are chosen by circularity: a circle fills pi/4 of its box, a square 1."""
    canvas = _card()
    for x, y in [(450, 120), (790, 350), (450, 580), (110, 350)]:
        canvas[y - 34:y + 34, x - 34:x + 34] = 0.06
    image = _write(tmp_path, canvas)
    with pytest.raises(DetectionRefused):
        find_printed_discs(
            image, load_target("isa_golden_thread_dlt_v1"), chart_extent(image)
        )


# -- a chart placed on its own array of patches --------------------------------


def test_a_patch_lattice_places_the_chart(tmp_path):
    """The whole array votes on where the chart is.

    Fitted in the chart's own relative coordinates rather than in index space, so
    the corners come out wherever the target declares them and the patch *size*
    -- which a blob search measures badly, because a dark patch on a dark mount
    loses more of its own boundary than a light one -- never enters the answer.
    """
    from tests.fixtures import SyntheticChart

    chart = SyntheticChart()
    path, corners = chart.render_capture(tmp_path / "sg.tif", skew=18)
    image = load_image(path)
    target = load_target("colorchecker_sg_v1")
    found, report = find_patch_array(image, target, chart_extent(image))
    ordered, _ = order_corners(found, target.fiducials.corner_positions)
    assert np.asarray(ordered) == pytest.approx(np.asarray(corners, float), abs=2.0)
    assert report["patches_located"] == 140
    assert report["lattice_residual_px"] < 1.0


def test_a_patch_lattice_shot_a_quarter_turn_round_is_still_found(tmp_path):
    """Its columns run down the frame, so ranking along x sorts fourteen columns
    into ten rows and the fit misses by a fifth of the chart. DT's own pair capture
    of the Nexus transmissive targets is shot this way and refused for exactly
    that reason. Which corner is which is decided downstream; what the lattice has
    to get right is the four points."""
    from tests.fixtures import SyntheticChart

    chart = SyntheticChart()
    path, corners = chart.render_capture(tmp_path / "sg.tif", skew=18)
    upright = load_image(path)
    pixels = upright.window(0, upright.height, 0, upright.width)
    image = _write(tmp_path, np.rot90(pixels, k=-1), name="turned.tif")
    # A clockwise quarter turn takes (x, y) to (H - 1 - y, x).
    expected = [(upright.height - 1 - y, x) for x, y in corners]

    target = load_target("colorchecker_sg_v1")
    found, report = find_patch_array(image, target, chart_extent(image))
    for point in expected:
        nearest = min(np.linalg.norm(np.subtract(point, f)) for f in found)
        assert nearest < 2.0
    assert report["columns_run_down_the_frame"] is True
    assert report["lattice_residual_px"] < 1.0


def test_an_upright_patch_lattice_is_never_tried_turned(tmp_path):
    """The turned fit is a fallback, not a rival: a lower residual is not evidence
    of the right lattice, so a chart that places upright places exactly as it did."""
    from tests.fixtures import SyntheticChart

    path, _ = SyntheticChart().render_capture(tmp_path / "sg.tif", skew=18)
    image = load_image(path)
    _, report = find_patch_array(
        image, load_target("colorchecker_sg_v1"), chart_extent(image)
    )
    assert "columns_run_down_the_frame" not in report


def test_too_few_patches_is_refused_rather_than_extrapolated(tmp_path):
    """Six blobs cannot determine a 14 x 10 lattice, and saying so is the answer.

    A projective map has eight degrees of freedom. Below about twenty patches its
    residual stops being a check on anything -- the fit simply passes through what
    it was given -- so the refusal is on the count and not on the residual.
    """
    canvas = _card(width=1200, height=900)
    for row in range(2):
        for column in range(3):
            x, y = 300 + column * 200, 300 + row * 200
            canvas[y:y + 90, x:x + 90] = 0.15 + 0.1 * ((row + column) % 3)
    image = _write(tmp_path, canvas)
    with pytest.raises(DetectionRefused, match="patch-shaped regions"):
        find_patch_array(image, load_target("colorchecker_sg_v1"), chart_extent(image))


def test_a_butted_array_is_not_indexed_from_the_first_patch_found():
    """The correction that keeps an IT8 off a placement two columns over.

    Where two butted neighbours share a colour there is no boundary between them,
    so whole columns at one end of the array go undetected -- and the lattice
    fitted to what remains is perfectly regular, self-consistent, and wrong. What
    detects it is not the patches, which agree with each other either way, but the
    card: a strip where a column ought to be either looks like a column of patches
    or it looks like card.
    """
    from deltachrome.imaging.detect import _lattice_lines, _lattice_offset

    # Ten columns of eight patches each, with the leftmost two columns' patches
    # left undetected -- as a blob search would leave them.
    window = np.full((400, 1000), 0.55, np.float32)
    for column in range(10):
        for row in range(8):
            x, y = 50 + column * 90, 40 + row * 44
            window[y:y + 40, x:x + 80] = 0.1 + 0.09 * ((row * 3 + column) % 8)
    centres = np.array(
        [(50 + column * 90 + 40, 40 + row * 44 + 20)
         for column in range(2, 10) for row in range(8)],
        dtype=np.float64,
    )
    lines = _lattice_lines(centres[:, 0], 10)
    assert _lattice_offset(window, centres, lines, 0, 10) == 2


def test_a_fully_detected_array_is_not_grown():
    """The correction must cost nothing on the common case: with no columns
    unaccounted for there is nowhere to grow to, and nothing is even looked at."""
    from deltachrome.imaging.detect import _lattice_lines, _lattice_offset

    window = np.full((400, 1000), 0.55, np.float32)
    centres = np.array(
        [(50 + column * 90, 40 + row * 44) for column in range(10) for row in range(8)],
        dtype=np.float64,
    )
    lines = _lattice_lines(centres[:, 0], 10)
    assert _lattice_offset(window, centres, lines, 0, 10) == 0


# -- a chart drawn from a definition, for the tests that read one --------------


def _render_target(tmp_path, target, *, name="drawn.tif", marks=True, size=1150.0):
    """Draw a target's declared geometry, and return the capture and its quad.

    The suite's ``SyntheticChart`` renders a ColorChecker SG and is not usable here:
    it assigns its synthetic aim values down the patch ids, where the SG's definition
    declares a ramp whose twelve steps alternate between two rows running in opposite
    directions. That disagreement is a simplification in the fixture rather than an
    error in either -- the real capture reads the way the definition says -- but a
    test of whether a ramp runs the declared way cannot be built on a chart that does
    not.

    So each patch is filled at its own declared position, ramp steps at the level
    their ``ramp_position`` calls for and everything else at a fixed spread, and a
    plus is drawn at each declared fiducial corner.

    **The quad's proportions come from the target**, not from the test. The aspect a
    chart's cells imply is what decides which way up it is placed, so drawing a chart
    at proportions its own definition disagrees with asks the detector to prefer the
    drawing over the definition -- which it correctly refuses to do. And the card is
    sized to the chart rather than the other way round, because the corner windows a
    mark is looked for in are a fraction of the card: a chart adrift in the middle of
    a much larger card puts its own marks outside them.
    """
    import cv2
    import tifffile

    from deltachrome.imaging.detect import _ramp_expectation, implied_quad_aspect
    from tests.fixtures import adobe_rgb_icc

    aspect = implied_quad_aspect(target) or 1.0
    span = (size, size / aspect) if aspect >= 1 else (size * aspect, size)
    pad = 0.12 * min(span)
    surround = 60
    origin = surround + pad
    corners = [
        (origin, origin), (origin + span[0], origin),
        (origin + span[0], origin + span[1]), (origin, origin + span[1]),
    ]
    width = int(round(span[0] + 2 * pad + 2 * surround))
    height = int(round(span[1] + 2 * pad + 2 * surround))

    canvas = np.full((height, width, 3), 0.06)
    canvas[surround:height - surround, surround:width - surround] = 0.5

    declared = np.asarray(target.fiducials.corner_positions, np.float32)
    matrix = cv2.getPerspectiveTransform(declared, np.asarray(corners, np.float32))

    def to_pixels(points):
        return cv2.perspectiveTransform(
            np.asarray(points, np.float32).reshape(1, -1, 2), matrix
        )[0]

    expectation = _ramp_expectation(target) or {}
    low, high = (min(expectation.values()), max(expectation.values())) if expectation else (0.0, 0.0)
    for index, patch in enumerate(target.patches):
        if patch.patch_id in expectation and high > low:
            share = (expectation[patch.patch_id] - low) / (high - low)
            level = 0.12 + 0.83 * share
        else:
            level = 0.20 + 0.65 * ((index * 37) % 11) / 10.0
        quad = to_pixels(target.patch_corners_relative(patch, inset=0.85))
        cv2.fillPoly(canvas, [np.round(quad).astype(np.int32)], (level,) * 3)

    if marks:
        for x, y in np.round(np.asarray(corners)).astype(int):
            canvas[y - 3:y + 4, x - 30:x + 31] = 0.02
            canvas[y - 30:y + 31, x - 3:x + 4] = 0.02

    encoded = np.round(np.clip(canvas, 0.0, 1.0) * 65535.0).astype(np.uint16)
    tifffile.imwrite(
        tmp_path / name, encoded, photometric="rgb",
        extratags=[(34675, "B", len(adobe_rgb_icc()), adobe_rgb_icc(), True)],
    )
    return load_image(tmp_path / name), corners



# -- reading the chart, where the outline has run out --------------------------


def test_a_ramp_read_upside_down_reads_backwards(tmp_path):
    """The half turn no geometry can see, and the one thing that can.

    Half a turn leaves the quad's aspect unchanged and, on a grid centred in its own
    quad, leaves every region on a patch -- so every test that reads only *where*
    things are is blind to it.
    """
    target = load_target("isa_fadgi_19264_commercial_v1")
    image, corners = _render_target(tmp_path, target)
    upright = ramp_agreement(image, target, corners)
    flipped = ramp_agreement(image, target, corners[2:] + corners[:2])
    assert upright == pytest.approx(1.0)
    assert flipped is not None and upright - flipped > 0.25


def test_a_ramp_printed_twice_still_reads(tmp_path):
    r"""A ladder printed as two matching wedges is the case that broke first.

    Three ISA transmissive charts print theirs twice, above and below the chart's
    centre, agreeing with each other to half an L\*. Sorted by what they should read,
    the two copies interleave, so half of every adjacent pair is two readings of one
    aim separated by noise -- and counting those inverted the answer on all three,
    scoring the correct placement 0.04 and the one half a turn out 0.83.
    """
    target = load_target("isa_transmissive_4x5_v1")
    image, corners = _render_target(tmp_path, target, marks=False)
    assert ramp_agreement(image, target, corners) == pytest.approx(1.0)


def _render_gradient(tmp_path, *, name="gradient.tif", size=(900, 1200)):
    """A capture with no chart in it at all: one smooth left-to-right gradient.

    It stands for the thing that actually broke this -- the brightness gradient
    across a specimen sheet -- and is drawn rather than photographed for the reason
    everything here is. What matters is that it carries no ladder anywhere, so any
    ramp read off it is read off nothing.
    """
    height, width = size
    field = np.tile(np.linspace(0.0, 1.0, width), (height, 1))
    canvas = np.repeat((0.15 + 0.7 * field)[:, :, None], 3, axis=2)
    path = write_tiff_with_profile(
        tmp_path / name,
        np.round(canvas * 65535.0).astype(np.uint16),
        adobe_rgb_icc(),
    )
    quad = [(100.0, 80.0), (width - 100.0, 80.0),
            (width - 100.0, height - 80.0), (100.0, height - 80.0)]
    return load_image(path), quad


def test_patches_sharing_an_aim_are_one_step_of_the_ladder():
    """A step is a value the ladder aims at, not a patch printing it.

    Six shipped targets print more than one patch per step. Read as separate steps
    they interleave when sorted by aim, and the only adjacent pairs whose aim
    actually moves are the ones crossing from one copy of the ladder to another --
    which is a comparison between two *places on the chart*, not between two tones.
    """
    from deltachrome.imaging.detect import _ramp_expectation, _ramp_steps

    class _Sample:
        usable = True
        flags = ()

        def __init__(self, level):
            self.rgb_mean = np.array([level, level, level], float)

    class _Sampled:
        def __init__(self, samples):
            self.samples = samples

    utt = load_target("image_engineering_utt_a3_v1")
    expectation = _ramp_expectation(utt)
    assert len(expectation) == 80 and len(set(expectation.values())) == 20

    # Every patch reads its own aim, so pooling can only average equals: the step
    # count is the number of aims and each step is that aim's own level.
    sampled = _Sampled({pid: _Sample(aim) for pid, aim in expectation.items()})
    steps = _ramp_steps(utt, sampled)
    assert len(steps) == 20
    assert [aim for aim, _read in steps] == sorted(set(expectation.values()), reverse=True)

    # And on a target whose aims are all distinct it is a no-op by construction.
    sg = load_target("colorchecker_sg_v1")
    sg_expectation = _ramp_expectation(sg)
    sg_sampled = _Sampled({pid: _Sample(aim) for pid, aim in sg_expectation.items()})
    assert len(_ramp_steps(sg, sg_sampled)) == len(sg_expectation) == 12


def test_a_gradient_is_not_a_ramp(tmp_path):
    """The dragonfly, reduced to the one property of it that mattered.

    Read as eighty separate steps, the UTT's four parallel greyscale strips score a
    **perfect +1.00** on a plain gradient carrying no chart whatever: sorted by aim
    the strips interleave, so all eighteen counted comparisons are its right-hand
    strip against its top strip -- one fact about the gradient, counted eighteen
    times, and ``MIN_RAMP_COMPARISONS`` satisfied trivially by it. That is how a
    photograph of an insect came to be named an Image Engineering UTT on thirteen of
    twenty Smithsonian Odonata frames.

    Pooled, the same gradient reads about zero, and the ColorChecker SG -- whose
    twelve aims are all distinct, so pooling cannot touch it -- reads what it always
    read. Both halves are asserted, because a fix that also moved the charts it was
    not about would not be this fix.

    The UTT may also read nothing at all: its four strips are four copies of one
    ladder, and on a gradient they disagree with each other by more than a step,
    which ``DUPLICATE_STEP_AGREEMENT`` refuses before the ladder is read. That is
    the stronger answer, so it is accepted -- but only where the gradient sampled,
    so a placement that read nothing cannot pass for one.
    """
    from deltachrome.imaging.detect import RAMP_AGREEMENT_FLOOR, _sample

    image, quad = _render_gradient(tmp_path)

    utt_target = load_target("image_engineering_utt_a3_v1")
    assert _sample(image, utt_target, quad) is not None
    utt = ramp_agreement(image, utt_target, quad)
    assert utt is None or utt < RAMP_AGREEMENT_FLOOR

    sg = ramp_agreement(image, load_target("colorchecker_sg_v1"), quad)
    assert sg is not None and sg < RAMP_AGREEMENT_FLOOR


def test_a_chart_shot_a_quarter_turn_round_is_placed_the_right_way_up(tmp_path):
    """End to end, on a chart drawn and then turned on its side.

    This is the failure as it was reported: the four printed marks found correctly
    and labelled a quarter turn out, which samples every region from a real patch and
    returns a full set of plausible numbers read from the wrong ones.
    """
    import cv2
    import tifffile

    from tests.fixtures import adobe_rgb_icc

    target = load_target("isa_fadgi_19264_commercial_v1")
    upright, corners = _render_target(tmp_path, target)
    with tifffile.TiffFile(tmp_path / "drawn.tif") as handle:
        pixels = handle.pages[0].asarray()
    turned = np.ascontiguousarray(cv2.rotate(pixels, cv2.ROTATE_90_CLOCKWISE))
    image = load_image(
        write_tiff_with_profile(tmp_path / "turned.tif", turned, adobe_rgb_icc())
    )
    # A clockwise turn sends (x, y) to (height - 1 - y, x).
    height = upright.height
    expected = [(height - 1 - y, x) for x, y in corners]

    detection = detect_placement(image, target, refine=False)
    assert detection.found
    assert np.asarray(detection.corners) == pytest.approx(
        np.asarray(expected, float), abs=4.0
    )


def test_identifying_a_capture_of_nothing_names_no_target(tmp_path):
    """Never a name for want of a better candidate.

    Ranking alone named the ColorChecker SG as the chart in a capture of a different
    target entirely, purely because nothing better was available -- its ramp read
    -0.09 and 88% of its regions were flagged, and it still won. What names a target
    is that it fits, not that it came first.
    """
    from deltachrome.imaging.detect import identify_target

    image = _write(tmp_path, _card())
    result = identify_target(image)
    assert not result.found
    assert result.target_id is None
    assert "Choose the chart" in result.reason


def test_identifying_a_drawn_chart_names_it(tmp_path):
    from deltachrome.imaging.detect import identify_target

    target = load_target("isa_fadgi_19264_commercial_v1")
    image, _corners = _render_target(tmp_path, target)
    result = identify_target(
        image,
        ["isa_fadgi_19264_commercial_v1", "colorchecker_sg_v1", "avian_ngt_v1"],
    )
    assert result.found
    assert result.target_id == "isa_fadgi_19264_commercial_v1"
    assert result.detection is not None and result.detection.found


# -- a chart that does not fill its own capture --------------------------------


def test_a_chart_inside_a_mount_is_still_found(tmp_path):
    """A 35mm slide is shot in its mount, so the extent is the mount.

    The chart is then a rectangle in the middle of something much larger, and "just
    inside the chart's border" is a long way inside the *extent*. On the ISA slide
    capture the keyline sat 2475 px down and the scan stopped at 2451 -- twenty-four
    pixels short -- so the chart was not found at all, and the capture was scored
    against every target whose marks could be mistaken for something in the mount.
    """
    mount = np.full((1800, 1800, 3), 0.05)
    mount[80:1720, 80:1720] = 0.15  # the slide mount: dark, but not the surround
    card = (520, 620, 1280, 1180)
    mount[card[1]:card[3], card[0]:card[2]] = 0.55
    box = (card[0] + 40, card[1] + 40, card[2] - 40, card[3] - 40)
    cv2.rectangle(mount, box[:2], box[2:], (0.05, 0.05, 0.05), thickness=2)
    image = _write(tmp_path, mount, name="mounted.tif")

    extent = chart_extent(image)
    assert extent.width > 1500, "the mount, not the chart -- which is the point"
    found, _report = find_printed_frame(
        image, load_target("isa_transmissive_4x5_v1"), extent
    )
    ordered, _ = order_corners(found, [(0, 0), (1, 0), (1, 1), (0, 1)])
    assert ordered[0] == pytest.approx((box[0], box[1]), abs=2.0)
    assert ordered[2] == pytest.approx((box[2], box[3]), abs=2.0)


def test_a_candidate_whose_regions_landed_on_junk_does_not_fit():
    """The gate the ISA slide needed, and the reason it is not tight.

    Read as a microform chart, that slide put 90% of its regions on flags where the
    right answer put none -- and it passed every other test, because its ramp
    happened to read monotone across whatever it landed on. A *clipped* capture also
    flags heavily and is a real thing an operator has, so the gate sits far above
    anything a right answer needs rather than just above it.
    """
    from deltachrome.imaging.detect import Candidate, fits_the_capture

    def candidate(**kwargs):
        return Candidate(
            target_id="x", display_name="X",
            **{"ramp": 1.0, "sampled": 1.0, "flagged": 0.0, **kwargs},
        )

    assert fits_the_capture(candidate())
    assert fits_the_capture(candidate(flagged=0.15)), "a right answer's worst case"
    assert fits_the_capture(candidate(flagged=0.55)), "a clipped capture still names"
    assert not fits_the_capture(candidate(flagged=0.90)), "regions on junk"
    assert not fits_the_capture(candidate(ramp=-0.09))
    assert not fits_the_capture(candidate(ramp=None))
    assert not fits_the_capture(candidate(sampled=0.5))


def test_an_untagged_capture_that_refuses_is_searched_again_through_the_srgb_curve(
        tmp_path, monkeypatch):
    """Only an untagged capture, and only after the search as given found nothing:
    one already encoded is found as it is and must not be pushed through twice."""
    from dataclasses import replace

    from deltachrome.imaging import detect

    tagged = _write(tmp_path, _card())
    image = replace(tagged, profile=None)
    seen = []

    def attempt(im):
        seen.append(im)
        found = len(seen) == 2
        return detect.Detection(found=found, corners=[(0, 0)] * 4 if found else [])

    result = detect._retrying_encoded(image, attempt)
    assert result.found and result.diagnostics["searched_through_srgb_curve"]
    assert seen[0] is image and seen[1] is not image
    # 18% grey: linear on the way in, encoded on the second look.
    grey = np.full((2, 2, 3), round(0.18 * 65535), np.uint16)
    encoded = detect.through_srgb_curve(replace(image, pixels=grey, scale=65535.0))
    assert abs(encoded.pixels[0, 0, 0] / 65535 - 0.4614) < 0.001

    assert detect.through_srgb_curve(tagged) is None
