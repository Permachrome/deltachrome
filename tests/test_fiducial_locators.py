"""Finding the printed features a placement is made on.

Three of the shipped targets place on marks that no other tool could find, and each
of the three failed first in a way worth keeping a test for.

**A plus is not a blob.** The ColorChecker SG's crosshair is a hairline cross 94 px
across drawn with a 6 px stroke; the DT Next Generation Target's is a stubby plus 57
px across with a 15 px stroke. Neither size, squareness nor fill ratio separates
either from the printed column letters beside it -- an ``H`` scores better on all
three than the SG's crosshair does. What separates them is shape, tested with probes
that are three pixels across whatever the mark's size, because a probe scaled to the
bounding box is wider than a hairline stroke and rejects the real mark.

**The LaserSoft prints a barcode below its frame.** Scanning in from the card for the
first dark thing finds a bar, not the rule.

**The Object Level Target's quad is 34 times longer than it is tall.** Its height is
one patch, and the older unit prints a white keyline around the patch block that the
current one does not -- so measuring the block's boundary directly gets the height
wrong by the keyline's width on one unit and right on the other. Every feature off
the patch row is declared as a fraction of that height, so the fit is done on the
patch lattice instead, where thirty patches across 3400 px fix the pitch.

The captures these were written against are gitignored, so everything here is drawn.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from tools.locate_printed_fiducials import (  # noqa: E402
    STRIP_SLOTS,
    find_crosshairs,
    find_frame,
    find_strip,
    to_luma,
)


def _plus(canvas, centre, extent, stroke, value) -> None:
    x, y = centre
    half, arm = extent // 2, stroke // 2
    canvas[y - arm:y + arm + 1, x - half:x + half + 1] = value
    canvas[y - half:y + half + 1, x - arm:x + arm + 1] = value


#: Where the four pluses are drawn, and therefore what the finder must return.
CROSSHAIRS = ((200, 180), (3000, 180), (3000, 2220), (200, 2220))


def crosshair_chart(extent=94, stroke=6, *, letters=True) -> np.ndarray:
    """A dark card with four printed pluses, and the letters that fool a blob test."""
    gray = np.full((2400, 3200), 0.14, np.float32)
    gray[300:2100, 400:2800] = 0.30                      # the patch array's mount
    for x, y in CROSSHAIRS:
        _plus(gray, (x, y), extent, stroke, 0.85)
    if letters:
        # A column label: square, of the same size, and solidly filled at its corners.
        gray[140:140 + extent, 1000:1000 + extent] = 0.85
        # And one with a hole, which is what actually survives a fill-ratio test.
        gray[155:125 + extent, 1515:1485 + extent] = 0.85
        gray[170:110 + extent, 1530:1470 + extent] = 0.30
    return cv2.GaussianBlur(gray, (5, 5), 1.0)


# -- crosshairs ---------------------------------------------------------------


@pytest.mark.parametrize("extent, stroke", [(94, 6), (57, 15), (40, 4)])
def test_a_plus_is_found_whatever_its_stroke(extent, stroke):
    """The two shipped charts' marks differ by a factor of two and a half in stroke,
    and the finder must not be tuned to either."""
    points, report = find_crosshairs(crosshair_chart(extent, stroke))
    assert len(points) == 4
    found = sorted((round(x), round(y)) for x, y in points)
    assert found == sorted(CROSSHAIRS)
    assert report["area_spread"] < 0.05


def test_the_printed_letters_beside_the_marks_are_not_mistaken_for_them():
    """A solid square and a ring both beat a hairline cross on size, squareness and
    fill. Only the shape test rejects them, and this is the case that broke first."""
    points, _ = find_crosshairs(crosshair_chart())
    assert all(abs(x - 1000) > 200 and abs(x - 1500) > 200 for x, _ in points)


def test_dark_marks_on_a_light_card_are_found_too():
    """Polarity is swept rather than declared -- a transmissive chart inverts."""
    points, report = find_crosshairs(1.0 - crosshair_chart(letters=False))
    assert report["polarity"] == -1
    assert len(points) == 4


def test_a_chart_without_four_marks_is_refused_rather_than_guessed_at():
    gray = crosshair_chart(letters=False)
    gray[:280, :] = 0.14                       # rub out the top two
    with pytest.raises(SystemExit, match="no four-crosshair solution"):
        find_crosshairs(gray)


# -- frame --------------------------------------------------------------------


def frame_chart(*, barcode=True) -> np.ndarray:
    """A grey card carrying a thin dark rectangle -- and, below it, a barcode."""
    gray = np.full((1200, 1600), 0.98, np.float32)
    gray[60:1140, 80:1520] = 0.53                                    # the card
    for y in (200, 900):
        gray[y:y + 6, 200:1400] = 0.06
    for x in (200, 1394):
        gray[200:906, x:x + 6] = 0.06
    gray[300:880, 300:1300] = 0.35                                   # the patch array
    if barcode:
        for x in range(300, 1200, 40):
            gray[960:1100, x:x + 14] = 0.02
    return cv2.GaussianBlur(gray, (3, 3), 0.8)


def test_the_frames_four_rules_meet_where_they_are_drawn():
    corners, report = find_frame(frame_chart())
    expected = [(202.5, 202.5), (1396.5, 202.5), (1396.5, 902.5), (202.5, 902.5)]
    for found, want in zip(corners, expected):
        assert found == pytest.approx(want, abs=1.5)
    assert all(side["rms_px"] < 1.0 for side in report["sides"].values())


def test_the_barcode_below_the_frame_is_walked_past():
    """Its bars are darker than the rule and fifty times thicker. Scanning up from
    the card without a thickness bound fits the bottom line through them instead."""
    with_bars, without = find_frame(frame_chart())[0], find_frame(frame_chart(barcode=False))[0]
    for a, b in zip(with_bars, without):
        assert a == pytest.approx(b, abs=0.5)


# -- strip --------------------------------------------------------------------

#: Slots the Object Level Target leaves empty for its module, in the middle of 34.
GAP = range(15, 19)


def strip_chart(*, keyline=False, pitch=100, rotate_deg=0.0) -> np.ndarray:
    """Thirty butted colour patches in a 34-slot row, on a grey card."""
    rgb = np.full((340, 3800, 3), 0.45, np.float32)
    top, left = 120, 200
    if keyline:
        rgb[top - 3:top + pitch + 3, left - 3:left + STRIP_SLOTS * pitch + 3] = 0.96
    for slot in range(STRIP_SLOTS):
        if slot in GAP:
            continue
        angle = slot * 0.7
        colour = np.array([0.5 + 0.42 * np.sin(angle),
                           0.5 + 0.42 * np.sin(angle + 2.1),
                           0.5 + 0.42 * np.sin(angle + 4.2)], np.float32)
        x = left + slot * pitch
        rgb[top:top + pitch, x:x + pitch] = colour
        # The printed hairline between butted patches. Without it every patch above
        # the saturation threshold is one component 34 pitches long, and the tool
        # sees no patches at all -- which is what the real charts would look like if
        # they did not print it.
        rgb[top:top + pitch, x:x + 2] = 0.18
    if rotate_deg:
        centre = (rgb.shape[1] / 2.0, rgb.shape[0] / 2.0)
        matrix = cv2.getRotationMatrix2D(centre, rotate_deg, 1.0)
        rgb = cv2.warpAffine(rgb, matrix, (rgb.shape[1], rgb.shape[0]),
                             flags=cv2.INTER_LINEAR, borderValue=(0.45, 0.45, 0.45))
    return cv2.GaussianBlur(rgb, (3, 3), 0.8)


def test_the_strip_quad_is_thirty_four_pitches_by_one():
    corners, report = find_strip(strip_chart())
    quad = np.array(corners)
    assert report["pitch_px"] == pytest.approx(100.0, abs=0.3)
    assert np.linalg.norm(quad[1] - quad[0]) == pytest.approx(3400.0, abs=6.0)
    assert np.linalg.norm(quad[3] - quad[0]) == pytest.approx(100.0, abs=0.3)
    assert report["lattice_residual_px"] < 1.0
    assert corners[0] == pytest.approx((200.0, 120.0), abs=2.0)


def test_a_keyline_moves_the_boundary_fit_and_not_the_lattice_fit():
    """The whole reason the lattice fit exists. The older unit prints a white line
    around its patch block, so the block's own boundary reads several pixels large --
    and the quad's height is ONE patch, so that is a percent of it, applied to every
    band declared as a fraction of the short side."""
    plain, plain_report = find_strip(strip_chart())
    lined, lined_report = find_strip(strip_chart(keyline=True))

    boundary = np.array(lined_report["boundary_quad"]) - np.array(plain_report["boundary_quad"])
    assert np.abs(boundary).max() > 2.0, "the keyline should move the boundary fit"
    assert lined_report["pitch_px"] == pytest.approx(plain_report["pitch_px"], abs=0.2)
    for found, want in zip(lined, plain):
        assert found == pytest.approx(want, abs=1.0)


def test_a_rotated_strip_is_measured_along_its_own_axis():
    """A half-degree rotation drifts a quarter of a patch over 34 of them, so a quad
    built as an axis-aligned rectangle runs every off-row feature out of its band."""
    corners, report = find_strip(strip_chart(rotate_deg=0.44))
    assert report["deg"] == pytest.approx(-0.44, abs=0.05)
    quad = np.array(corners)
    assert np.linalg.norm(quad[1] - quad[0]) == pytest.approx(3400.0, abs=6.0)
    assert np.linalg.norm(quad[3] - quad[0]) == pytest.approx(100.0, abs=0.5)


def test_two_neighbours_of_one_colour_do_not_become_one_patch():
    """Patches butt with no gutter, so two alike neighbours threshold into a single
    component whose centroid sits half a slot from either. Admitting those took the
    fit's residual along the strip from 2 px to 14."""
    chart = strip_chart()
    chart[120:220, 200 + 5 * 100:200 + 7 * 100] = chart[150, 200 + 5 * 100]
    _, report = find_strip(chart)
    assert report["lattice_residual_px"] < 1.5


def test_the_luma_used_to_find_a_feature_is_never_used_to_measure_one():
    """A guard on the docstring's claim, and on anyone reaching for this helper: it
    weights three records into one and nothing that reports a number may do that."""
    rgb = np.zeros((4, 4, 3), np.float32)
    rgb[..., 1] = 1.0
    assert to_luma(rgb)[0, 0] == pytest.approx(0.7152)
