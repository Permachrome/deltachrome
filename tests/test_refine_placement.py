"""Snapping a hand placement onto the chart's own printed features.

The cost of a slightly wrong placement is not what it looks like. On the Golden
Thread capture a 20 pixel error on an 11600 x 8700 master -- 0.2% of the frame --
moves the number of readable printed rules from 6 to 0, and geometric distortion
needs at least four of them. A placement can therefore look right, pass the
aspect-ratio check, produce good color and SFR figures across eighteen metrics,
and silently disable a nineteenth. That is the failure this project cares most
about: not a wrong number, but a missing measurement nobody noticed.

These tests are synthetic because the sample captures are gitignored, and because
a synthetic chart is the only way to know the right answer exactly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deltachrome.imaging.geometry import place_chart
from deltachrome.imaging.loader import load_image
from deltachrome.imaging.refine import (
    CLUSTERED_FEATURE_FRACTION,
    MAX_SHIFT_FRACTION,
    MIN_FEATURES,
    FEATURE_KINDS,
    _declared_features,
    _estimator,
    feature_separation,
    locatable_features,
    refinable,
    refine_placement,
)
from deltachrome.targets.registry import load_target
from deltachrome.targets.schema import target_from_dict
from fixtures import adobe_rgb_icc, write_tiff_with_profile

SIZE = 1200
#: The chart occupies an inset quad, not the whole frame, so a displaced
#: placement in these tests still has all four corners inside the image.
MARGIN = 90
#: Module centers in target-relative coordinates, and the half-size each declares.
MODULES = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.5, 0.5))
HALF = 0.09
TILT = 5.0


def module_target(count: int = 5, declared_half: float = HALF):
    """A chart declaring ``count`` square modules at known relative positions."""
    return target_from_dict({
        "target_id": "refine_probe",
        "display_name": "Refine probe",
        "capabilities": ["sfr"],
        "fiducials": {"method": "chart_corners", "corner_markers": [
            {"id": "TL", "relative_position": [0.0, 0.0]},
            {"id": "TR", "relative_position": [1.0, 0.0]},
            {"id": "BR", "relative_position": [1.0, 1.0]},
            {"id": "BL", "relative_position": [0.0, 1.0]}]},
        "regions": [
            {"region_id": f"module_{i}", "kind": "slanted_edge",
             "geometry": {"center": list(center),
                          "half_size": [declared_half, declared_half],
                          "tilt_deg": TILT}}
            for i, center in enumerate(MODULES[:count])
        ],
        "patches": [],
    })


#: The ISA transmissive line's fiducials: five apex-to-apex pairs, each disc
#: offset from its pair's centre. Ten features rather than five, which is what
#: lets a homography be over-determined -- see :data:`deltachrome.imaging.refine.MODELS`.
#:
#: Taken from the ISA 120's measured geometry rather than invented, because the
#: *size* of the feature relative to the chart is load-bearing here: a minimum-area
#: rectangle around a rasterised arc loses a fraction of a pixel off the curved side
#: and none off the two straight ones, so the measured centre creeps toward the
#: apex. On a 900 px disc that is 0.05% and invisible; on an 90 px one it is 3% and
#: dominates the test.
#:
#: Scaled to 0.8 of the ISA's own figures, and the modules moved in from the corners,
#: so that discs belonging to *adjacent* pairs do not overlap. On the real chart the
#: pairs sit half a chart apart; packing five of them into a fixture at 0.2/0.5/0.8
#: put neighbouring squares on top of each other, and the finder then snapped four
#: of the ten onto the wrong disc -- correctly, since the declared square did contain
#: two.
DISC_MODULES = ((0.12, 0.12), (0.88, 0.12), (0.88, 0.88), (0.12, 0.88), (0.5, 0.5))
DISC_OFFSET = (0.0797, 0.0936)
DISC_HALF = 0.0799


def disc_target():
    """A chart declaring five quarter-disc pairs, as the ISA film targets do."""
    return target_from_dict({
        "target_id": "disc_probe",
        "display_name": "Disc probe",
        "capabilities": ["sfr"],
        "fiducials": {"method": "quarter_disc_pair_midpoints", "corner_markers": [
            {"id": "TL", "relative_position": [0.0, 0.0]},
            {"id": "TR", "relative_position": [1.0, 0.0]},
            {"id": "BR", "relative_position": [1.0, 1.0]},
            {"id": "BL", "relative_position": [0.0, 1.0]}]},
        "regions": [
            {"region_id": f"module_{i}", "kind": "slanted_edge",
             "geometry": {
                 "feature": "quarter_disc_pair",
                 "discs": [
                     {"center": [cx - DISC_OFFSET[0], cy - DISC_OFFSET[1]],
                      "half_size": [DISC_HALF, DISC_HALF], "tilt_deg": TILT},
                     {"center": [cx + DISC_OFFSET[0], cy + DISC_OFFSET[1]],
                      "half_size": [DISC_HALF, DISC_HALF], "tilt_deg": TILT},
                 ]}}
            for i, (cx, cy) in enumerate(DISC_MODULES)
        ],
        "patches": [],
    })


def render_discs(tmp_path, name="discs.tif"):
    """Ten quarter discs, apex to apex in five pairs, on a light field.

    A quarter disc, not a square: the finder checks that the mask fills pi/4 of its
    bounding rectangle, which is what stops it accepting a solid block -- so a
    fixture drawing squares would exercise nothing.
    """
    field = np.full((SIZE, SIZE), 0.80)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    t = np.radians(TILT)
    span = SIZE - 2 * MARGIN
    half = DISC_HALF * span
    for cx_rel, cy_rel in DISC_MODULES:
        for sign in (-1.0, +1.0):
            cx = MARGIN + (cx_rel + sign * DISC_OFFSET[0]) * span
            cy = MARGIN + (cy_rel + sign * DISC_OFFSET[1]) * span
            u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
            v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
            # A quarter disc inscribed in the square the definition declares: apex
            # at the corner facing the pair's centre, radius equal to the full side,
            # so the shape fills exactly pi/4 of the square. That ratio is the
            # identity check the finder applies, so a fixture drawing anything else
            # -- a solid block, or a quarter disc whose bounding square is offset
            # from the declared one -- is correctly rejected and tests nothing.
            au, av = -sign * half, -sign * half
            inside = (u - au) ** 2 + (v - av) ** 2 < (2 * half) ** 2
            field[(np.abs(u) < half) & (np.abs(v) < half) & inside] = 0.20
    field = cv2.GaussianBlur(field, (0, 0), 0.8)
    data = (np.repeat(field[:, :, None], 3, axis=2) * 65535).astype(np.uint16)
    return write_tiff_with_profile(tmp_path / name, data, adobe_rgb_icc())


@pytest.fixture
def disc_chart(tmp_path):
    return load_image(render_discs(tmp_path))


def render(tmp_path, count: int = 5, printed_half: float = HALF, name="chart.tif"):
    """The chart as printed: dark rotated squares on a light field.

    ``printed_half`` is separate from what the definition declares, so a
    definition whose sizes are wrong -- which is the real case on the Golden
    Thread -- can be reproduced deliberately.
    """
    field = np.full((SIZE, SIZE), 0.78)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    t = np.radians(TILT)
    span = SIZE - 2 * MARGIN
    for cx_rel, cy_rel in MODULES[:count]:
        cx, cy = MARGIN + cx_rel * span, MARGIN + cy_rel * span
        u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
        v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
        field[(np.abs(u) < printed_half * span) & (np.abs(v) < printed_half * span)] = 0.22
    field = cv2.GaussianBlur(field, (0, 0), 0.8)
    data = (np.repeat(field[:, :, None], 3, axis=2) * 65535).astype(np.uint16)
    return write_tiff_with_profile(tmp_path / name, data, adobe_rgb_icc())


_FAR = float(SIZE - MARGIN)
TRUE_CORNERS = [
    (float(MARGIN), float(MARGIN)), (_FAR, float(MARGIN)), (_FAR, _FAR), (float(MARGIN), _FAR)
]


@pytest.fixture
def chart(tmp_path):
    return load_image(render(tmp_path))


def placed(image, corners, target):
    return place_chart(
        corners, image_size=(image.width, image.height),
        corner_positions_relative=target.fiducials.corner_positions or None,
    )


def worst_error(corners, truth=TRUE_CORNERS) -> float:
    return max(
        float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
        for a, b in zip(corners, truth)
    )


def flat(points):
    """Corner lists flattened for comparison.

    ``pytest.approx`` does not apply its tolerance to a list of tuples -- it
    compares them exactly and, when they differ, reports "0 mismatched elements",
    which is a confusing way to fail. Flattening makes the tolerance real.
    """
    return [value for point in points for value in point]


def shift(corners, dx, dy):
    return [(x + dx, y + dy) for x, y in corners]


# -- recovering a known error -------------------------------------------------


@pytest.mark.parametrize("offset", [(20, 0), (0, 20), (-25, -15), (30, 25)])
def test_a_displaced_placement_is_pulled_back_onto_the_features(chart, offset):
    target = module_target()
    start = shift(TRUE_CORNERS, *offset)
    assert worst_error(start) > 15

    result = refine_placement(chart, placed(chart, start, target), target)
    assert result.applied, result.reason
    assert worst_error(result.corners) < 3.0


def test_a_correct_placement_is_left_alone(chart):
    """And is *confirmed*, not merely left alone by failing to find anything.
    A correct placement makes every snap land where it was declared, which an
    earlier version mistook for five refusals."""
    target = module_target()
    result = refine_placement(chart, placed(chart, TRUE_CORNERS, target), target)
    assert result.applied, result.reason
    assert result.features_used == len(MODULES)
    assert result.max_shift_px < 3.0


def test_a_declining_refinement_is_told_from_a_confirming_one_by_identity(chart):
    """The contract refinement rests on: every refusal path in ``_refine_square``
    returns the array it was handed, and a success returns a new one. Value
    equality cannot separate them, because a correct placement produces a
    successful snap whose values are unchanged."""
    from deltachrome.imaging.edges import _refine_square

    blank = np.full((400, 400, 3), 0.5)
    nothing_there = np.array([[100.0, 100.0], [300.0, 100.0],
                              [300.0, 300.0], [100.0, 300.0]])

    class Flat:
        width = height = 400

        def window(self, y0, y1, x0, x1):
            return blank[y0:y1, x0:x1]

    assert _refine_square(Flat(), nothing_there) is nothing_there


def test_a_rectangular_feature_is_refined_anisotropically():
    """A circular annulus at a fraction of the corner-to-centre distance
    is the right shape for a square and the wrong one for a rectangle -- on a 3:2
    feature's short axis it sits mostly outside the feature, sampling its own
    surround as though it were the inside, which is what the ISA FADGI-ISO
    19264 charts' resolution-field panels hit. Built by hand at a declared
    aspect no committed target's own tests exercise, so a regression back to an
    isotropic annulus would not be caught by anything else here."""
    from deltachrome.imaging.edges import _refine_square

    height, width = 400, 400
    field = np.full((height, width, 3), 0.80)
    # A 3:2 rectangle -- 120 x 80 -- centred in the frame.
    half_w, half_h = 60.0, 40.0
    cx, cy = 200.0, 200.0
    yy, xx = np.mgrid[0:height, 0:width]
    field[(np.abs(xx - cx) < half_w) & (np.abs(yy - cy) < half_h)] = 0.20

    class Flat:
        width = 400
        height = 400

        def window(self, y0, y1, x0, x1):
            return field[y0:y1, x0:x1]

    # Declared a few pixels off centre, as a hand placement would be.
    declared = np.array([
        [cx - half_w + 4, cy - half_h + 3],
        [cx + half_w + 4, cy - half_h + 3],
        [cx + half_w + 4, cy + half_h + 3],
        [cx - half_w + 4, cy + half_h + 3],
    ])
    found = _refine_square(Flat(), declared)
    assert not np.array_equal(found, declared), "must actually refine, not decline"

    center = found.mean(axis=0)
    assert center[0] == pytest.approx(cx, abs=1.0)
    assert center[1] == pytest.approx(cy, abs=1.0)
    found_half_w = float(np.linalg.norm(found[1] - found[0])) / 2.0
    found_half_h = float(np.linalg.norm(found[3] - found[0])) / 2.0
    assert found_half_w == pytest.approx(half_w, abs=2.0)
    assert found_half_h == pytest.approx(half_h, abs=2.0)


def test_the_square_case_is_unchanged_by_the_anisotropic_formula():
    """The anisotropic construction has to reduce to the old circular one when
    the two axes are equal -- checked directly by comparing against the
    original formula it replaced, not merely by the existing square fixtures
    still passing, since those could pass under either formula by coincidence."""
    from deltachrome.imaging.edges import (
        SQUARE_INSIDE_T,
        SQUARE_OUTSIDE_T,
        _refine_square,
    )

    height, width = 400, 400
    field = np.full((height, width, 3), 0.80)
    half = 60.0
    cx, cy = 200.0, 200.0
    yy, xx = np.mgrid[0:height, 0:width]
    field[(np.abs(xx - cx) < half) & (np.abs(yy - cy) < half)] = 0.20

    class Flat:
        width = 400
        height = 400

        def window(self, y0, y1, x0, x1):
            return field[y0:y1, x0:x1]

    declared = np.array([
        [cx - half, cy - half], [cx + half, cy - half],
        [cx + half, cy + half], [cx - half, cy + half],
    ])

    # The formula this replaced: a single scalar "radius" (mean corner distance)
    # and a circular annulus at SQUARE_INSIDE_T/SQUARE_OUTSIDE_T of it.
    local = declared
    center = local.mean(axis=0)
    radius = np.linalg.norm(local - center, axis=1).mean()
    distance = np.hypot(xx - center[0], yy - center[1])
    old_inside = (distance > SQUARE_INSIDE_T[0] * radius) & (distance < SQUARE_INSIDE_T[1] * radius)
    old_outside = distance > SQUARE_OUTSIDE_T * radius
    old_level_in = float(np.median(field[..., 0][old_inside]))
    old_level_out = float(np.median(field[..., 0][old_outside]))

    found = _refine_square(Flat(), declared)
    assert not np.array_equal(found, declared)
    new_center = found.mean(axis=0)
    assert new_center[0] == pytest.approx(cx, abs=0.5)
    assert new_center[1] == pytest.approx(cy, abs=0.5)
    # The two formulas must have picked out the same two levels on a square --
    # this is what "reduces to the old formula" actually claims.
    assert old_level_in == pytest.approx(0.20, abs=0.01)
    assert old_level_out == pytest.approx(0.80, abs=0.01)


def test_the_inside_band_moved_and_the_outside_band_did_not():
    """The shipped values, so a later edit does not silently drift back
    towards the old tolerance or past the point that regressed the commercial
    ISA FADGI-ISO 19264 chart's top-left corner square. This only pins
    the numbers that real-capture measurements settled on."""
    from deltachrome.imaging.edges import SQUARE_INSIDE_T, SQUARE_OUTSIDE_T

    assert SQUARE_INSIDE_T == (0.20, 0.45)
    # Left exactly where it always was: widening this the same way regressed a
    # real feature (a corner square with chart border on two sides) on the
    # commercial chart, so the fix is the inside band alone.
    assert SQUARE_OUTSIDE_T == 1.15


def test_refining_twice_changes_nothing_the_second_time(chart):
    """A fixed point. Without one, "refine" would be a button whose result
    depends on how many times it was pressed."""
    target = module_target()
    once = refine_placement(chart, placed(chart, shift(TRUE_CORNERS, 25, 18), target), target)
    twice = refine_placement(chart, placed(chart, once.corners, target), target)
    assert twice.max_shift_px < 1.0


# -- the reason it fits centers, and only a similarity ------------------------


def test_declared_sizes_being_wrong_does_not_move_the_placement(chart, tmp_path):
    """The real case: on the Golden Thread the declared half-sizes are 4-6%
    larger than the squares actually printed, and inconsistent between modules.
    Fitting corner-to-corner asks the transform to reconcile sizes that do not
    match, and a homography has enough freedom to do it by shearing the chart."""
    printed = load_image(render(tmp_path, printed_half=HALF * 0.94, name="small.tif"))
    target = module_target(declared_half=HALF)  # declares the larger, wrong size

    result = refine_placement(printed, placed(printed, TRUE_CORNERS, target), target)
    assert result.applied, result.reason
    # The features are where they were declared to be; only their size is wrong,
    # so a correct method moves nothing.
    assert result.max_shift_px < 5.0


# -- refusing rather than guessing --------------------------------------------


def test_too_few_features_refuses_and_says_how_many_it_found(chart, tmp_path):
    sparse = load_image(render(tmp_path, count=2, name="sparse.tif"))
    target = module_target(count=2)
    result = refine_placement(sparse, placed(sparse, TRUE_CORNERS, target), target)
    assert not result.applied
    assert str(MIN_FEATURES) in result.reason


def test_a_target_printing_nothing_usable_says_so(chart):
    """A ColorChecker SG has no slanted edges. Refinement is not available for it,
    and saying which chart cannot do it beats a bare failure."""
    sg = load_target("colorchecker_sg_v1")
    result = refine_placement(chart, placed(chart, TRUE_CORNERS, sg), sg)
    assert not result.applied
    assert "prints none" in result.reason


def test_an_implausibly_large_correction_is_refused(chart):
    """A correction is out by pixels or tens of pixels. A fraction of the chart
    means a feature was mislocated, and moving there is worse than staying."""
    target = module_target()
    displaced = placed(chart, shift(TRUE_CORNERS, 30, 22), target)
    # The correction is real and about 37 px; the limit here is about 10.
    result = refine_placement(chart, displaced, target, max_shift_fraction=0.01)
    assert not result.applied
    assert "misdetection" in result.reason
    # And with the normal limit the same correction is accepted, so the guard is
    # rejecting the size of the move rather than the move itself.
    assert refine_placement(chart, displaced, target).applied


def test_a_refusal_returns_the_placement_it_was_given(chart):
    """So a caller can use the result unconditionally without checking a flag and
    silently getting corners it never asked for."""
    sg = load_target("colorchecker_sg_v1")
    start = shift(TRUE_CORNERS, 11, 7)
    result = refine_placement(chart, placed(chart, start, sg), sg)
    assert not result.applied
    assert flat(result.corners) == pytest.approx(flat(start))
    assert result.moved_px == [0.0] * 4


# -- what it reports ----------------------------------------------------------


def test_the_result_says_how_far_it_moved_and_on_what_evidence(chart):
    """Never applied silently: the caller has to be able to show the user what
    changed and decide whether to accept it."""
    target = module_target()
    result = refine_placement(chart, placed(chart, shift(TRUE_CORNERS, 22, 16), target), target)
    assert result.features_used == len(MODULES)
    assert len(result.moved_px) == 4
    assert result.residual_px == pytest.approx(result.residual_px)  # not NaN
    assert any("module_0" in note for note in result.notes)
    assert "px" in result.reason


def test_the_residual_is_reported_in_pixels_not_relative_units(chart):
    """A relative residual is unreadable; 'agrees to 2 px' is a sentence someone
    can act on."""
    target = module_target()
    result = refine_placement(chart, placed(chart, TRUE_CORNERS, target), target)
    assert 0.0 <= result.residual_px < 20.0


# -- quarter-disc targets ----------------------------------------------------
#
# The three ISA transmissive charts locate their edges from apex-to-apex disc
# pairs rather than from squares, and refinement skipped them entirely on the
# grounds that a quarter disc needs its apex identified before its geometry means
# anything. That is true of *extracting its edges* and irrelevant to finding where
# it sits: the bounding square's centre needs no apex, and the disc finder was
# already computing it -- with a stronger identity check than a square gets, since
# it also requires the mask to fill pi/4 of the rectangle.


def test_a_quarter_disc_chart_is_refined_rather_than_skipped(disc_chart):
    target = disc_target()
    result = refine_placement(
        disc_chart, placed(disc_chart, shift(TRUE_CORNERS, 22, -14), target), target
    )
    assert result.applied
    assert worst_error(result.corners) < 1.0


def test_each_disc_of_a_pair_votes_on_its_own(disc_chart):
    """Five pairs give ten correspondences, not five. That is the difference between
    a fit that over-determines a homography and one that cannot."""
    target = disc_target()
    result = refine_placement(disc_chart, placed(disc_chart, TRUE_CORNERS, target), target)
    assert result.features_used == 2 * len(DISC_MODULES)


def test_a_correct_quarter_disc_placement_is_left_where_it_is(disc_chart):
    target = disc_target()
    result = refine_placement(disc_chart, placed(disc_chart, TRUE_CORNERS, target), target)
    assert flat(result.corners) == pytest.approx(flat(TRUE_CORNERS), abs=1.0)


# -- which transform is fitted, and why --------------------------------------


def test_five_features_get_a_similarity_and_ten_get_a_homography(chart, disc_chart):
    """The whole rule: a model may be used when the features over-determine it. It
    reproduces two findings that look contradictory -- the Golden Thread's five
    modules measured *worse* under a homography, and the ISA transmissive line's ten
    discs measure far better -- without being tuned to either."""
    squares = module_target()
    discs = disc_target()
    from_squares = refine_placement(chart, placed(chart, TRUE_CORNERS, squares), squares)
    from_discs = refine_placement(
        disc_chart, placed(disc_chart, TRUE_CORNERS, discs), discs
    )
    assert (from_squares.features_used, from_squares.model) == (5, "similarity")
    assert (from_discs.features_used, from_discs.model) == (10, "homography")


def _shape(corners):
    """Side and diagonal lengths as ratios to the first side.

    Invariant under shift, rotation and uniform scale, and only under those -- which
    is exactly the class a similarity can express.
    """
    p = np.asarray(corners, dtype=np.float64)
    lengths = [float(np.linalg.norm(p[(i + 1) % 4] - p[i])) for i in range(4)]
    lengths += [float(np.linalg.norm(p[2] - p[0])), float(np.linalg.norm(p[3] - p[1]))]
    return [value / lengths[0] for value in lengths]


def test_a_similarity_leaves_the_placement_the_shape_it_was_given(chart):
    """The honest limit of the conservative model, stated as the property rather than
    as an outcome. Four corners each wrong in their own direction is a *perspective*
    error, and shift-rotate-scale cannot express one -- so whatever perspective error
    the operator introduced survives, however small it happens to be on one chart."""
    target = module_target()
    skewed = [(x + dx, y + dy) for (x, y), (dx, dy) in
              zip(TRUE_CORNERS, [(-14, -9), (11, -16), (17, 8), (-7, 13)])]
    result = refine_placement(chart, placed(chart, skewed, target), target)
    assert result.model == "similarity"
    # Preserved to a few parts in a hundred thousand -- the precision of the fit
    # chain -- while the shape error the operator introduced is two parts in a
    # hundred, which is where it stays.
    assert _shape(result.corners) == pytest.approx(_shape(skewed), abs=1e-4)
    assert max(abs(a - b) for a, b in
               zip(_shape(result.corners), _shape(TRUE_CORNERS))) > 1e-3


def test_a_homography_is_free_to_change_the_shape(disc_chart):
    """The other half: with enough features to pay for it, the correction may change
    the quad's shape -- which is what makes it able to undo a perspective error."""
    target = disc_target()
    skewed = [(x + dx, y + dy) for (x, y), (dx, dy) in
              zip(TRUE_CORNERS, [(-14, -9), (11, -16), (17, 8), (-7, 13)])]
    result = refine_placement(disc_chart, placed(disc_chart, skewed, target), target)
    assert result.model == "homography"
    assert _shape(result.corners) == pytest.approx(_shape(TRUE_CORNERS), abs=1e-3)


def test_a_homography_can(disc_chart):
    """The same error, on a chart with enough features to pay for the freedom."""
    target = disc_target()
    skewed = [(x + dx, y + dy) for (x, y), (dx, dy) in
              zip(TRUE_CORNERS, [(-14, -9), (11, -16), (17, 8), (-7, 13)])]
    result = refine_placement(disc_chart, placed(disc_chart, skewed, target), target)
    assert result.model == "homography"
    assert worst_error(result.corners) < 1.0


def test_the_model_is_named_in_the_result(disc_chart):
    """A similarity and a homography are different claims about what was wrong with
    the placement, so which was fitted is part of the answer."""
    target = disc_target()
    result = refine_placement(disc_chart, placed(disc_chart, TRUE_CORNERS, target), target)
    assert result.model in {"similarity", "homography"}
    assert result.model in result.reason
    assert any(result.model in note for note in result.notes)


def test_an_affine_fit_is_never_used(chart, disc_chart):
    """Six degrees of freedom is enough to distort the frame and not enough to
    describe a projection, and it measured worse than *both* its neighbours on one of
    the three real captures -- 6.1 px on an exact placement where the similarity gave
    0.35 and the homography 0.44."""
    from deltachrome.imaging.refine import MODELS

    assert [name for name, _ in MODELS] == ["homography", "similarity"]


def test_a_noisy_feature_set_is_reported_and_not_refused(disc_chart, tmp_path):
    """LMEDS survives a minority of mislocated features, so a residual threshold
    would refuse correct answers. The disagreement is worth saying out loud and is
    not a reason to discard the fit."""
    from deltachrome.imaging.refine import NOISY_FEATURE_FRACTION

    assert 0 < NOISY_FEATURE_FRACTION < MAX_SHIFT_FRACTION
    target = disc_target()
    result = refine_placement(disc_chart, placed(disc_chart, TRUE_CORNERS, target), target)
    # A clean chart says nothing about noise.
    assert not any("not where the definition says" in note for note in result.notes)


# -- markers count as features ------------------------------------------------
#
# Refinement needs a feature to be locatable, not measurable, and a ``marker`` is
# by definition a feature the chart carries that no metric reads. Excluding them
# cost the Golden Thread half its evidence: each of its five modules prints a white
# "dots per inch" field inside a dark mount, and the mount alone gives five
# correspondences where both give ten.


def marker_target(count: int = 5):
    """Squares declared as markers rather than as slanted edges."""
    data = {
        "target_id": "marker_probe",
        "display_name": "Marker probe",
        "capabilities": ["sfr"],
        "fiducials": {"method": "chart_corners", "corner_markers": [
            {"id": "TL", "relative_position": [0.0, 0.0]},
            {"id": "TR", "relative_position": [1.0, 0.0]},
            {"id": "BR", "relative_position": [1.0, 1.0]},
            {"id": "BL", "relative_position": [0.0, 1.0]}]},
        "regions": [
            {"region_id": f"mark_{i}", "kind": "marker",
             "geometry": {"center": list(center), "half_size": [HALF, HALF],
                          "tilt_deg": TILT}}
            for i, center in enumerate(MODULES[:count])
        ],
        "patches": [],
    }
    return target_from_dict(data)


def test_a_marker_region_supplies_a_correspondence(chart):
    target = marker_target()
    result = refine_placement(
        chart, placed(chart, shift(TRUE_CORNERS, 18, -11), target), target
    )
    assert result.features_used == len(MODULES)
    assert result.applied
    assert worst_error(result.corners) < 1.5


def test_markers_and_slanted_edges_are_counted_together(chart):
    """Which is what promotes a five-module chart to a homography: the Golden Thread
    prints a mount and a field per module, and neither alone is enough."""
    from deltachrome.imaging.refine import FEATURE_KINDS

    assert FEATURE_KINDS == {"slanted_edge", "marker"}
    both = target_from_dict({
        "target_id": "both_probe", "display_name": "Both probe",
        "capabilities": ["sfr"], "patches": [],
        "fiducials": {"method": "chart_corners", "corner_markers": [
            {"id": "TL", "relative_position": [0.0, 0.0]},
            {"id": "TR", "relative_position": [1.0, 0.0]},
            {"id": "BR", "relative_position": [1.0, 1.0]},
            {"id": "BL", "relative_position": [0.0, 1.0]}]},
        # The same five squares declared twice, under both kinds. Each declaration
        # finds the square that is there, so the count -- and therefore the model --
        # is what changes.
        "regions": [
            {"region_id": f"{prefix}_{i}", "kind": kind,
             "geometry": {"center": list(c), "half_size": [HALF, HALF],
                          "tilt_deg": TILT}}
            for prefix, kind in (("module", "slanted_edge"), ("mark", "marker"))
            for i, c in enumerate(MODULES)
        ],
    })
    result = refine_placement(chart, placed(chart, TRUE_CORNERS, both), both)
    assert result.features_used == 10
    assert result.model == "homography"


def test_a_marker_without_geometry_is_skipped_rather_than_a_failure(chart):
    """Most markers declare a band or a list of positions, not a centre and a size --
    the microform chart's bar-pattern arrays, the ISA 120's line wedges. Those carry
    no correspondence and must not be counted as one."""
    target = target_from_dict({
        "target_id": "banded_probe", "display_name": "Banded probe",
        "capabilities": ["sfr"], "patches": [],
        "fiducials": {"method": "chart_corners", "corner_markers": [
            {"id": "TL", "relative_position": [0.0, 0.0]},
            {"id": "TR", "relative_position": [1.0, 0.0]},
            {"id": "BR", "relative_position": [1.0, 1.0]},
            {"id": "BL", "relative_position": [0.0, 1.0]}]},
        "regions": [
            {"region_id": f"module_{i}", "kind": "slanted_edge",
             "geometry": {"center": list(c), "half_size": [HALF, HALF],
                          "tilt_deg": TILT}}
            for i, c in enumerate(MODULES)
        ] + [
            {"region_id": "wedge", "kind": "marker",
             "geometry": {"feature": "bar_pattern_array",
                          "bounds_relative": [0.3, 0.3, 0.4, 0.4]}},
        ],
    })
    result = refine_placement(chart, placed(chart, TRUE_CORNERS, target), target)
    assert result.features_used == len(MODULES)


# -- the Golden Thread's own geometry -----------------------------------------
#
# Its declared module sizes were 5.2% too large and its centres 1 to 6 px out --
# harmless for edge extraction, which snaps to the real square anyway, and the
# limiting error in refinement. Measured off the capture, the residual fell from
# 2.19 px to 0.23 and a 23 px misplacement recovered to 0.10 px rather than 6.07.
#
# The capture is gitignored, so what can be tested here is the *definition's* own
# consistency -- which is where a transcription error would land.

DLT = "isa_golden_thread_dlt_v1"


def dlt_regions(kind: str) -> dict:
    return {r.region_id: r.geometry for r in load_target(DLT).regions if r.kind == kind}


def test_the_golden_thread_declares_a_mount_and_a_field_for_every_module():
    mounts = dlt_regions("slanted_edge")
    fields = {k: v for k, v in dlt_regions("marker").items() if k.startswith("dpi_field")}
    assert len(mounts) == 5
    assert {k.replace("module_", "") for k in mounts} == {
        k.replace("dpi_field_", "") for k in fields
    }


def test_both_features_are_square_on_the_chart():
    """Both are printed square, so their relative half-sizes must carry the *quad's*
    aspect -- and the same one. A digit dropped in either would break that, and would
    otherwise be invisible: an ROI slightly the wrong shape still lands on the
    feature."""
    ratios = [
        geometry["half_size"][0] / geometry["half_size"][1]
        for kind in ("slanted_edge", "marker")
        for name, geometry in dlt_regions(kind).items()
        if "half_size" in geometry and (kind == "slanted_edge" or "dpi_field" in name)
    ]
    assert len(ratios) == 10
    assert max(ratios) - min(ratios) < 0.001


def test_the_field_sits_inside_its_mount():
    """420 px inside 760. A field declared larger than its mount would be a
    transcription error that still located *something*."""
    mounts = dlt_regions("slanted_edge")
    fields = dlt_regions("marker")
    for name, mount in mounts.items():
        field = fields[name.replace("module_", "dpi_field_")]
        for axis in (0, 1):
            assert field["half_size"][axis] < mount["half_size"][axis]
        assert field["half_size"][0] / mount["half_size"][0] == pytest.approx(
            420.30 / 760.70, abs=0.01
        )


def test_the_corner_fields_are_still_the_relative_frame_they_always_were():
    """Placement moved to the printed discs; the coordinate frame did not.

    That separation is the whole reason the change cost nothing: the four corner
    modules' resolution fields are still relative (0,0), (1,0), (1,1) and (0,1), so
    every patch, module, rule and marker coordinate in the definition still means
    what it did. The fiducials are four points expressed IN that frame rather than
    its corners, which is what `corner_positions_relative` is for."""
    fields = dlt_regions("marker")
    for position, corner in zip(
        ("tl", "tr", "br", "bl"), ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    ):
        declared = fields[f"dpi_field_{position}"]["center"]
        assert tuple(declared) == pytest.approx(corner, abs=1e-6)


def test_placement_is_on_the_four_printed_discs():
    """A diamond about the chart's centre, not a rectangle at its corners -- so the
    declared positions are nowhere near 0 and 1, and two of them fall outside the
    frame entirely. The winding check still has to pass, because a diamond listed
    in the wrong cycle maps the chart onto a bow tie exactly as a rectangle would."""
    target = load_target(DLT)
    assert target.fiducials.method == "printed_discs"
    assert target.fiducials.corner_ids == ["TOP", "RIGHT", "BOTTOM", "LEFT"]

    points = target.fiducials.corner_positions
    assert len(points) == 4
    # One point beyond each side of the frame, which is what a diamond about the
    # centre looks like in the frame's own coordinates.
    assert min(v for _, v in points) < 0.0 and max(v for _, v in points) > 1.0
    assert min(u for u, _ in points) < 0.0 and max(u for u, _ in points) > 1.0

    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        area += x0 * y1 - x1 * y0
    assert abs(area) > 0.5, "the four points must go round the quad, not across it"


def test_the_discs_are_chosen_for_being_findable_and_not_for_fitting_better():
    """The argument is recorded because the obvious one is wrong, and a later reader
    who assumes a bigger quad must be a better-conditioned one would undo this."""
    notes = load_target(DLT).fiducials.notes
    assert "not because a diamond is better conditioned" in notes
    assert "25% worse" in notes
    assert "0.16 px" in notes


def test_every_module_declares_the_same_size_and_tilt():
    """Five printings of one design. Differing figures would mean one was measured
    and the others guessed."""
    for kind, prefix in (("slanted_edge", "module_"), ("marker", "dpi_field_")):
        entries = [g for name, g in dlt_regions(kind).items() if name.startswith(prefix)]
        assert len(entries) == 5
        assert len({tuple(g["half_size"]) for g in entries}) == 1
        assert len({g["tilt_deg"] for g in entries}) == 1


def test_the_mount_is_slanted_and_the_field_inside_it_is_not():
    """Not a discrepancy but the design, and worth pinning because it looks like one.
    The mount is tilted because a slanted-edge ROI needs each row to catch the
    transition at a different sub-pixel phase; the field is square to the chart
    because the bar patterns printed in it are read against the chart's own axes."""
    mount = next(iter(dlt_regions("slanted_edge").values()))
    field = dlt_regions("marker")["dpi_field_tl"]
    assert 4.0 < mount["tilt_deg"] < 6.0
    assert abs(field["tilt_deg"]) < 1.0


# -- the overlay ---------------------------------------------------------------
#
# The picture is the only check on a class of error the numbers cannot show: a
# lattice shifted by a third of a patch still samples every patch and returns
# plausible values. It found one on the ISA 120, whose bottom resolution wedge was
# declared a third of a chart from the strip it names.


def test_the_overlay_draws_every_kind_of_region(chart, tmp_path):
    """A region kind the renderer silently skips is a region nobody ever looks at,
    which is how the 120's misplaced marker survived being written down."""
    import json

    from deltachrome.imaging.sampling import sample_patches
    from deltachrome.reporting.overlay import COLORS, render_geometry

    target = target_from_dict(json.loads(json.dumps({
        "target_id": "overlay_probe", "display_name": "Overlay probe",
        "capabilities": ["sfr"],
        "fiducials": {"method": "chart_corners", "corner_markers": [
            {"id": "TL", "relative_position": [0.0, 0.0]},
            {"id": "TR", "relative_position": [1.0, 0.0]},
            {"id": "BR", "relative_position": [1.0, 1.0]},
            {"id": "BL", "relative_position": [0.0, 1.0]}]},
        "regions": [
            {"region_id": "module_0", "kind": "slanted_edge",
             "geometry": {"center": [0.2, 0.2], "half_size": [HALF, HALF],
                          "tilt_deg": TILT}},
            {"region_id": "discs_0", "kind": "slanted_edge",
             "geometry": {"discs": [
                 {"center": [0.8, 0.2], "half_size": [HALF, HALF], "tilt_deg": TILT}]}},
            {"region_id": "straight_0", "kind": "slanted_edge",
             "geometry": {"straight_edges": [
                 {"id": "a", "start": [0.3, 0.7], "end": [0.7, 0.72]}]}},
            {"region_id": "ruler_0", "kind": "scale",
             "geometry": {"axis": "u", "division_mm": 1.0,
                          "bounds_relative": [0.1, 0.55, 0.9, 0.6]}},
            {"region_id": "wedge", "kind": "marker",
             "geometry": {"bounds_relative": [0.1, 0.85, 0.4, 0.92]}},
            {"region_id": "rings", "kind": "marker",
             "geometry": {"positions": [[0.5, 0.9]]}},
        ],
        "patches": [],
    })))
    placement = placed(chart, TRUE_CORNERS, target)
    picture = render_geometry(chart, placement, target, max_width=900)
    assert picture.shape[2] == 3
    assert picture.shape[1] <= 900

    present = {tuple(int(v) for v in c) for c in picture.reshape(-1, 3)}
    for name in ("quad", "declared", "marker"):
        assert tuple(COLORS[name]) in present, name
    # No readings were supplied, so the band draws as one that could not be read --
    # which is the state an operator most needs to see, and the reason the two are
    # different colours rather than the band being omitted.
    assert tuple(COLORS["rule_failed"]) in present
    assert tuple(COLORS["rule"]) not in present

    class Reading:
        region_id = "ruler_0"

    read = render_geometry(chart, placement, target, rulers=[Reading()], max_width=900)
    present = {tuple(int(v) for v in c) for c in read.reshape(-1, 3)}
    assert tuple(COLORS["rule"]) in present


def test_the_overlay_marks_a_flagged_patch_differently(chart, tmp_path):
    """Whether a sample was flagged is the one thing about it visible at a glance,
    and a picture that coloured them alike would hide the only pattern that matters
    -- flags clustered along one edge of the array is a geometry error, and flags
    scattered across it is dust."""
    from deltachrome.imaging.sampling import PatchSample, SamplingResult
    from deltachrome.reporting.overlay import COLORS, render_geometry

    target = module_target(count=1)
    placement = placed(chart, TRUE_CORNERS, target)
    quad = placement.to_pixel(
        np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]], dtype=np.float64)
    )
    sample = PatchSample(
        patch_id="X1", row=0, col=0, rgb_mean=np.zeros(3), rgb_median=np.zeros(3),
        rgb_std=np.zeros(3), lab_std=None, pixel_count=100,
        clipped_high_fraction=0.0, clipped_low_fraction=0.0, polygon_px=quad,
        flags=["clipped_high(9%)"],
    )
    sampling = SamplingResult(samples={"X1": sample}, inset=0.55)
    picture = render_geometry(chart, placement, target, sampling=sampling,
                              max_width=900)
    present = {tuple(int(v) for v in c) for c in picture.reshape(-1, 3)}
    assert tuple(COLORS["flagged"]) in present
    assert tuple(COLORS["patch"]) not in present


# -- which estimator, and why -------------------------------------------------


def declared_feature_points(target):
    """Every declared feature centre, in target-relative coordinates.

    Read from the engine's own declaration rather than re-derived here. This used
    to repeat the disc/square branching, which made it a third statement of a rule
    that already had two -- and the other copy of it, in the frontend, is exactly
    the one that drifted.
    """
    points = [
        center
        for region in target.regions
        if region.kind in FEATURE_KINDS
        for _label, center, _half, _tilt, _shape in _declared_features(region)
    ]
    return np.asarray(points, np.float32)


def test_feature_separation_is_the_closest_pair_over_the_widest():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    assert feature_separation(square) == pytest.approx(1.0 / np.sqrt(2.0))
    doubled = np.array([[0, 0], [0.001, 0.0], [1, 0], [1, 1], [0, 1]], np.float32)
    assert feature_separation(doubled) == pytest.approx(0.001 / np.sqrt(2.0), rel=1e-6)


def test_spread_features_are_fitted_by_lmeds():
    """The default, and it earns its keep: median-based, so a minority of
    mislocated features cannot drag it."""
    method, name = _estimator(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32))
    assert method == cv2.LMEDS
    assert name == "LMEDS"


def test_clustered_features_are_fitted_by_least_squares():
    """LMEDS draws minimal four-point subsets and keeps the best median. When two
    features nearly coincide a subset holding both is degenerate for a homography
    -- and it still scores well, because LMEDS is entitled to ignore half the data.
    It then returns a wild transform most of the features disagree with."""
    clustered = np.array(
        [[0, 0], [0.0005, 0.0005], [1, 0], [1, 1], [0, 1]], np.float32
    )
    method, name = _estimator(clustered)
    assert method == 0
    assert "least squares" in name and "cluster" in name


def test_only_the_device_level_target_counts_as_clustered():
    """The threshold is not tuned: across every shipped target with four or more
    features the statistic reads 0.14 or above, except the Golden Thread Device
    Level Target at 0.0014 -- two decades of empty space, so any value inside it
    separates them and the exact one cannot matter.

    The Device Level Target is clustered because each of its five modules declares
    TWO features, the dark mount and the white field inside it, whose centres are
    0.0009 apart in relative units -- about 5 px on its own master. Ten features,
    five places."""
    from deltachrome.targets.registry import available_targets

    ratios = {}
    for target_id in available_targets():
        points = declared_feature_points(load_target(target_id))
        if len(points) >= 4:
            ratios[target_id] = feature_separation(points)

    clustered = {t for t, r in ratios.items() if r < CLUSTERED_FEATURE_FRACTION}
    assert clustered == {"isa_golden_thread_dlt_v1"}
    # And the gap either side of the threshold is wide, which is the real claim:
    # everything sits at least a factor of two clear of it, in fact far more.
    # Stated against the threshold rather than as fixed bounds, because adding
    # features to a chart moves its statistic -- declaring eight registration dots
    # on the ISA FADGI-ISO 19264 charts took them from having none to 0.067.
    assert max(ratios[t] for t in clustered) < CLUSTERED_FEATURE_FRACTION / 2
    assert min(r for t, r in ratios.items() if t not in clustered) > (
        CLUSTERED_FEATURE_FRACTION * 2
    )


def test_a_clustered_chart_is_decided_by_its_definition_not_by_a_capture():
    """So a chart always takes the same path and no run can silently switch
    estimators on the same target."""
    points = declared_feature_points(load_target(DLT))
    assert _estimator(points)[0] == 0
    # A subset of the same features -- what a capture with one feature missing
    # would give -- is still clustered, because the pairs are what makes it so.
    assert _estimator(points[:8])[0] == 0


def test_the_residual_is_documented_as_measured_rather_than_argued():
    """The decision not to gate is the kind that quietly gets reversed by someone
    who sees a bad fit applied and reaches for a threshold. The evidence that it
    would not help is therefore kept where they will meet it, with its caveat."""
    import deltachrome.imaging.refine as refine

    doc = refine.__doc__ + open(refine.__file__, encoding="utf-8").read()
    # The measurement, not the anecdote.
    assert "511 seeded refinements" in doc
    assert "93%" in doc and "median of 15 px" in doc
    # Why refusing cannot help: it returns the placement you already had.
    assert "25 were STILL an improvement" in doc
    assert "101.8 px out" in doc
    # And what the measurement does not cover.
    assert "CAVEAT ON THE MEASUREMENT" in doc


# -- what a chart can be offered, before a pixel is read ----------------------
#
# The gate these pin was computed in the frontend and drifted from the engine in
# both directions at once: it counted rotated squares only, so the three ISA
# transmissive charts -- ten disc features apiece, and the charts this module
# recovers every corner of to 0.44 px -- were shown a dead button saying they
# print no marks to snap to, while both Object Level Targets were shown a live
# one that refuses on every press. Their single edge is declared as
# ``straight_edges``, which yields no centre at all, so they declare not too few
# features but none.


def test_every_shipped_target_is_gated_on_the_count_it_declares():
    """One rule, applied to every chart, knowable without touching a capture."""
    from deltachrome.targets.registry import available_targets

    for target_id in available_targets():
        target = load_target(target_id)
        offered, _why = refinable(target)
        assert offered == (locatable_features(target) >= MIN_FEATURES), target_id


def test_the_transmissive_charts_are_offered_refinement():
    """The regression. Each declares five quarter-disc pairs, and each disc votes
    on its own -- which is what takes them from barely determining a homography to
    over-determining one."""
    for target_id in (
        "isa_transmissive_4x5_v1",
        "isa_transmissive_35mm_v1",
        "isa_transmissive_120_v1",
    ):
        target = load_target(target_id)
        assert locatable_features(target) >= 10, target_id
        assert refinable(target) == (True, "")


def test_markers_are_counted_alongside_edges():
    """The Golden Thread's five modules and the five white fields inside them are
    ten features, not five -- which is the difference between a similarity and a
    homography on that chart."""
    target = load_target(DLT)
    assert locatable_features(target) == 10


def test_a_chart_declaring_too_few_says_how_many_and_how_many_are_needed():
    """Distinct from declaring none, because the two are different facts about the
    chart and only one of them reads as 'this will never work'."""
    target = load_target("applied_image_qa62_sfr_v1")
    assert locatable_features(target) == 1
    offered, why = refinable(target)
    assert not offered
    assert "1 mark " in why and str(MIN_FEATURES) in why


def test_a_chart_declaring_none_says_that_instead():
    target = load_target("isa_object_level_1x_v1")
    assert locatable_features(target) == 0
    offered, why = refinable(target)
    assert not offered
    assert "No marks" in why


def test_the_gate_is_an_upper_bound_on_what_a_run_actually_locates(chart, disc_chart):
    """The count answers "could this ever work" and the run answers "did it". So
    the run may find fewer than are declared and must never find more -- which is
    what ties the two branchings together: a feature located under a shape the
    count does not know about would break this rather than quietly widening the
    evidence.
    """
    for image, target in ((chart, module_target()), (disc_chart, disc_target())):
        result = refine_placement(
            image, placed(image, TRUE_CORNERS, target), target
        )
        assert result.features_used <= locatable_features(target)
        # On a placement that is already right, every declared feature is found.
        assert result.features_used == locatable_features(target)
