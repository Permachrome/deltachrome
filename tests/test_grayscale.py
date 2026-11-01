"""Single-record captures.

Grayscale is not a colour capture with the chroma set to zero. It carries one
colour record where an RGB file carries three, and the difference is not academic:
three of the metrics here compare the records against each other or read chroma out
of them, and on one record each of those returns a *perfect* score for a comparison
that was never made. White balance is Delta E(a*b*) against a neutral aim and would
report 0.00 on every patch of a chart the system rendered catastrophically wrong.

So the tests here are mostly about what must **not** happen. The engine reads a
grayscale master -- the format FADGI list for microfilm, and what a microfilm
scanner actually produces -- and refuses to let a colour metric pass on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.errors import ImageError, UnsupportedICCProfileError
from deltachrome.imaging.color import xyz_to_lab
from deltachrome.imaging.icc import CIE_D50, ICC_PCS_D50, parse_icc
from deltachrome.imaging.loader import load_image, to_display_records, to_luma
from deltachrome.metrics.base import CAPTURE_COLOR_CAPABILITIES
from deltachrome.pipeline import analyze
from tests.fixtures import (
    GRAY_GAMMA,
    RAMP_IDS,
    SyntheticChart,
    adobe_rgb_icc,
    build_gray_icc,
    write_tiff_with_profile,
)


# -- the profile ------------------------------------------------------------


def test_a_monochrome_profile_is_read_rather_than_refused():
    profile = parse_icc(build_gray_icc(2.2))
    assert profile.color_space == "GRAY"
    assert profile.channels == 1
    assert profile.is_grayscale
    assert profile.transform_kind == "matrix_trc"


def test_grey_maps_onto_the_pcs_illuminant():
    """ICC clause 8.3: a monochrome transform is ``Y = kTRC(g)`` scaled onto the PCS
    illuminant. So white is L* 100 exactly and every value is neutral by
    construction -- there is no colorant matrix that could tint it."""
    profile = parse_icc(build_gray_icc(2.2))
    xyz = profile.to_xyz(np.array([[0.0], [0.5], [1.0]]))
    assert xyz[2] == pytest.approx(ICC_PCS_D50, abs=1e-9)
    # A ``curv`` gamma is u8Fixed8, so 2.2 is stored as 563/256. The profile is the
    # authority on what it says, not the number handed to the writer.
    assert xyz[1][1] == pytest.approx(0.5 ** (563.0 / 256.0), abs=1e-9)

    lab = xyz_to_lab(xyz, white_point=CIE_D50)
    assert lab[2][0] == pytest.approx(100.0, abs=1e-6)
    assert np.allclose(lab[:, 1:], 0.0, atol=0.03)


def test_a_grayscale_profile_refuses_three_channel_input():
    """The shape check is the profile's own channel count, not a constant 3. Passing
    RGB through a monochrome transform has no defined meaning, and silently taking
    the first channel would measure the red record and call it luminance."""
    profile = parse_icc(build_gray_icc(2.2))
    with pytest.raises(ValueError, match="trailing axis of 1"):
        profile.to_xyz(np.array([0.3, 0.6, 0.9]))


def test_cmyk_is_still_refused():
    """Widening the reader to grayscale must not widen it to everything. CMYK needs
    a LUT with four inputs and a pixel pipeline that does not exist here."""
    header = bytearray(build_gray_icc())
    header[16:20] = b"CMYK"
    with pytest.raises(UnsupportedICCProfileError, match="CMYK"):
        parse_icc(bytes(header))


# -- the loader -------------------------------------------------------------


def test_a_single_channel_master_loads_and_says_so(tmp_path):
    pixels = np.linspace(0, 65535, 64 * 64, dtype=np.uint16).reshape(64, 64)
    path = write_tiff_with_profile(tmp_path / "grey.tif", pixels, build_gray_icc())
    image = load_image(path)
    assert image.channels == 1
    assert image.is_grayscale
    assert image.pixels.shape == (64, 64, 1)
    assert "grayscale" in image.summary()


def test_a_grayscale_master_is_not_widened_to_three(tmp_path):
    """Replication is one line and it is the wrong line: colour registration would
    then read exactly zero displacement between three identical records, and report
    it as a pass."""
    pixels = np.zeros((32, 32), dtype=np.uint16)
    path = write_tiff_with_profile(tmp_path / "grey.tif", pixels, build_gray_icc())
    assert load_image(path).pixels.shape[2] == 1


def test_pixels_and_profile_must_agree_about_the_space(tmp_path):
    """The two arrive from different places in the file and nothing makes them
    agree. Guessing either way is a measurement through the wrong transform."""
    grey = np.zeros((32, 32), dtype=np.uint16)
    path = write_tiff_with_profile(tmp_path / "mismatch.tif", grey, adobe_rgb_icc())
    with pytest.raises(ImageError, match="1 channel"):
        load_image(path)

    rgb = np.zeros((32, 32, 3), dtype=np.uint16)
    path = write_tiff_with_profile(tmp_path / "mismatch2.tif", rgb, build_gray_icc())
    with pytest.raises(ImageError, match="3 channel"):
        load_image(path)


def test_luma_and_display_helpers_handle_both_widths():
    one = np.full((4, 5, 1), 0.25)
    three = np.full((4, 5, 3), 0.25)
    assert to_luma(one).shape == (4, 5)
    assert to_luma(three).shape == (4, 5)
    assert to_luma(one) == pytest.approx(to_luma(three))
    assert to_display_records(one).shape == (4, 5, 3)
    assert to_display_records(three).shape == (4, 5, 3)


# -- the pipeline -----------------------------------------------------------


@pytest.fixture(scope="module")
def grayscale_run(tmp_path_factory):
    """One grayscale capture of the synthetic SG, analyzed against a colour profile.

    Deliberately the awkward pairing: a chart that carries colour, measured through
    a file that threw the colour away. Nothing in the file admits to that, which is
    why the engine has to.
    """
    tmp = tmp_path_factory.mktemp("grayscale")
    chart = SyntheticChart()
    reference = chart.write_reference(tmp / "ref.txt")
    image, corners = chart.render_capture(tmp / "grey.tif", grayscale=True)
    return analyze(
        image,
        target="colorchecker_sg_v1",
        reference_path=reference,
        profile="fadgi_3star_prints_photographs_2023",
        corners=corners,
    )


def _outcome(result, metric_id):
    return next(r for r in result.evaluation.results if r.metric_id == metric_id)


def test_the_run_completes_on_one_colour_record(grayscale_run):
    assert grayscale_run.image.channels == 1
    assert len(grayscale_run.sampling) == 140


@pytest.mark.parametrize("metric_id", ["average_color_accuracy", "white_balance"])
def test_colour_metrics_report_incomplete_and_never_pass(grayscale_run, metric_id):
    result = _outcome(grayscale_run, metric_id)
    assert result.outcome == "incomplete"
    assert result.passed is None
    assert "color records" in (result.reason or "")


def test_the_overall_verdict_cannot_be_a_pass(grayscale_run):
    """The point of the whole exercise."""
    assert grayscale_run.evaluation.overall != "PASS"
    incomplete = {r.metric_id for r in grayscale_run.evaluation.incomplete}
    assert {"average_color_accuracy", "color_accuracy_90th_percentile",
            "white_balance"} <= incomplete
    assert all(r.passed is not True for r in grayscale_run.evaluation.results
               if r.metric_id.startswith(("color_", "white_")))


def test_what_the_gate_is_actually_preventing(grayscale_run):
    """The danger made concrete rather than argued. Every patch on this capture --
    including the 128 chromatic ones -- comes back with a* and b* at zero, because a
    monochrome transform cannot produce anything else. Had white balance been allowed
    to run, it would have returned 0.00 on a capture that discarded the entire
    quantity it measures, and passed."""
    measured = np.array([d.measured_lab for d in grayscale_run.run.patches
                         if d.measured_lab is not None])
    assert len(measured) == 140
    assert np.allclose(measured[:, 1:], 0.0, atol=0.05)

    reference = np.array([d.reference_lab for d in grayscale_run.run.patches
                          if d.reference_lab is not None])
    assert np.abs(reference[:, 1:]).max() > 30.0


def test_the_incompleteness_is_blamed_on_the_capture_not_the_chart(grayscale_run):
    """N/A means the target cannot carry the measurement and nothing is wrong. Here
    the chart carries it perfectly well and the file does not, and a coverage line
    reading 'not measurable by this target' would send the operator to buy a
    different chart."""
    skipped = [r.metric_id for r in grayscale_run.evaluation.not_applicable]
    assert "white_balance" not in skipped
    assert "average_color_accuracy" not in skipped


def test_tone_response_still_round_trips_at_the_quantisation_floor(grayscale_run):
    """Everything that reads luminance is unaffected, and this is the proof: the
    neutral ramp was rendered from its own reference L* through a gamma 2.2 grey
    curve, and comes back through the monochrome transform to the same numbers."""
    result = _outcome(grayscale_run, "tone_response_oecf")
    assert result.outcome == "evaluated"
    assert result.value == pytest.approx(0.0, abs=0.05)
    assert {v.patch_id for v in result.per_patch} == set(RAMP_IDS)


def test_the_record_says_how_many_colour_records_the_capture_had(grayscale_run):
    """A reopened run has no pixels. Without this it could show colour metrics as
    incomplete and be unable to say why."""
    assert grayscale_run.run.image["color_records"] == 1
    assert grayscale_run.run.to_dict()["image"]["color_records"] == 1


def test_the_gate_lists_exactly_the_capabilities_that_need_three_records():
    """A guard on a hard-coded list of metric ids would go stale the moment a
    profile named a new one. It gates on the capability instead, and the set is
    small enough to assert outright."""
    assert CAPTURE_COLOR_CAPABILITIES == {
        "color_accuracy",
        "white_balance",
        "color_registration",
    }


def test_an_rgb_capture_is_untouched_by_any_of_this(tmp_path):
    """The whole change has to be invisible to the common case."""
    chart = SyntheticChart()
    reference = chart.write_reference(tmp_path / "ref.txt")
    image, corners = chart.render_capture(tmp_path / "colour.tif")
    result = analyze(
        image,
        target="colorchecker_sg_v1",
        reference_path=reference,
        profile="fadgi_3star_prints_photographs_2023",
        corners=corners,
    )
    assert result.image.channels == 3
    assert _outcome(result, "white_balance").outcome == "evaluated"
    assert result.run.image["color_records"] == 3


def test_grey_gamma_matches_what_the_fixture_encoded():
    """Guards the fixture itself: if GRAY_GAMMA and the profile it writes ever drift
    apart, the round-trip test above would be asserting a coincidence."""
    profile = parse_icc(build_gray_icc(GRAY_GAMMA))
    assert profile.trc is not None
    stored = round(GRAY_GAMMA * 256.0) / 256.0
    assert profile.trc[0](np.array([0.5]))[0] == pytest.approx(0.5**stored, abs=1e-6)
    assert abs(stored - GRAY_GAMMA) < 0.002
