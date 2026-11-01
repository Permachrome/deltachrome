"""MTF50 expressed in physical units.

Metamorfoze states its resolution requirement in line pairs per millimetre, where
FADGI states MTF50 as a percentage of the half-sampling frequency. They are the same
measurement: a frequency in cycles per pixel, times a property of the capture.

Which makes the sampling rate load-bearing. A reading in lp/mm is only as good as
the scale behind it, so the tests below care as much about where the scale came from
and what happens when there isn't one as about the arithmetic.
"""

from __future__ import annotations

import pytest

from deltachrome.errors import ProfileError
from deltachrome.imaging.loader import load_image
from deltachrome.pipeline import analyze
from deltachrome.profiles.schema import profile_from_dict
from test_sfr import edge_target, rotated_square_capture

MM_PER_INCH = 25.4
EXTENT = (100.0, 100.0)


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    path = rotated_square_capture(tmp_path_factory.mktemp("mtf50"))
    image = load_image(path)
    corners = [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)]
    return path, corners


def measure(capture, *, units=None, condition="value > 0", extent=EXTENT, ids=("mtf50",)):
    path, corners = capture
    profile = profile_from_dict({
        "profile_id": "units", "display_name": "units",
        "metrics": [
            {"metric_id": metric_id, "requires_capability": "sfr",
             "pass_condition": condition,
             "algorithm": {"frequency_units": units} if units else {}}
            for metric_id in ids
        ],
    })
    result = analyze(path, target=edge_target(), profile=profile, corners=corners,
                     fiducial_extent_mm=extent)
    return {r.metric_id: r for r in result.evaluation.results}


# -- the conversions ---------------------------------------------------------


def test_every_unit_is_the_same_measurement(capture):
    """One frequency, six labels. If these ever stop agreeing, one of the scale
    factors is wrong and the reading is quietly wrong with it."""
    values = {
        units: measure(capture, units=units)["mtf50"].value
        for units in ("cycles/px", "lp/mm", "cycles/mm", "lp/inch",
                      "% of half-sampling freq")
    }
    native = values["cycles/px"]
    ppi = measure(capture, units="lp/mm")["mtf50"].metadata["sampling_ppi"]

    assert values["% of half-sampling freq"] == pytest.approx(native * 200.0)
    assert values["lp/mm"] == pytest.approx(native * ppi / MM_PER_INCH)
    assert values["lp/inch"] == pytest.approx(native * ppi)
    # One line pair is one cycle: the same number under two names.
    assert values["cycles/mm"] == pytest.approx(values["lp/mm"])


def test_the_nyquist_percentage_agrees_exactly_with_sfr50(capture):
    """The two arrive by different routes -- sfr50 divides by Nyquist inside the
    analyzer, mtf50 scales afterwards -- so agreement is a real check."""
    results = measure(capture, ids=("mtf50", "sfr50"))
    as_percent = measure(capture, units="% of half-sampling freq")["mtf50"]
    assert as_percent.value == pytest.approx(results["sfr50"].value, abs=1e-9)


def test_lp_per_mm_is_the_default(capture):
    """Because it is the unit the requirement that motivated this is written in."""
    assert measure(capture)["mtf50"].units == "lp/mm"


def test_an_unknown_unit_is_refused(capture):
    with pytest.raises(ProfileError, match="Unknown frequency_units"):
        measure(capture, units="lines/furlong")


# -- the scale it depends on -------------------------------------------------


def test_a_physical_unit_without_a_scale_is_incomplete_not_guessed(capture):
    """The failure that matters: inventing a sampling rate would produce a
    confident lp/mm figure with no relationship to the capture."""
    result = measure(capture, units="lp/mm", extent=None)["mtf50"]
    assert result.outcome == "incomplete"
    assert result.value is None
    assert "needs the capture's sampling rate" in result.reason
    assert "cycles/px" in result.reason  # and says what would work instead


def test_a_dimensionless_unit_needs_no_scale(capture):
    """Which is what makes it the fallback worth naming."""
    for units in ("cycles/px", "% of half-sampling freq"):
        result = measure(capture, units=units, extent=None)["mtf50"]
        assert result.outcome == "evaluated"


def test_where_the_scale_came_from_travels_with_the_reading(capture):
    result = measure(capture, units="lp/mm")["mtf50"]
    assert result.metadata["sampling_source"] == "operator-supplied extent"
    assert result.metadata["sampling_ppi"] > 0
    assert any("does not transfer to another unit" in note for note in result.notes)


# -- what a lp/mm requirement can and cannot ask for -------------------------


def test_a_requirement_above_the_sampling_ceiling_fails_with_the_ceiling_recorded(capture):
    """A line pair needs two samples, so above that ceiling no lens can help. The
    ceiling travels in the record; it is no longer a note, being reachable only on
    a hand-written guideline stating MTF50 in lp/mm (2026-10-02 review)."""
    result = measure(capture, units="lp/mm", condition="value > 6")["mtf50"]
    assert result.passed is False
    assert 6 > result.metadata["nyquist_limit"]
    assert not any("ceiling" in note for note in result.notes)


def test_the_ceiling_is_recorded_whether_or_not_it_binds(capture):
    result = measure(capture, units="lp/mm", condition="value > 0.5")["mtf50"]
    ppi = result.metadata["sampling_ppi"]
    assert result.metadata["nyquist_limit"] == pytest.approx(ppi / MM_PER_INCH / 2, abs=0.01)


def test_the_unit_can_be_overridden_per_run(capture):
    """Which unit a frequency is printed in is presentation, not a change to the
    measurement -- so unlike a threshold, choosing one does not fork the profile
    into a local specification."""
    path, corners = capture
    profile = profile_from_dict({
        "profile_id": "units", "display_name": "units",
        "metrics": [{"metric_id": "mtf50", "requires_capability": "sfr",
                     "pass_condition": "value > 0", "units": "lp/mm"}],
    })
    result = analyze(path, target=edge_target(), profile=profile, corners=corners,
                     fiducial_extent_mm=EXTENT, frequency_units="cycles/px")
    metric = result.evaluation.results[0]
    assert metric.units == "cycles/px"
    assert metric.value < 0.5           # cycles per pixel cannot exceed Nyquist
    assert result.evaluation.conformance_claim is True


# -- the unit must not decide the verdict ------------------------------------
#
# It used to. The override rescaled the reading and left the profile's threshold
# in the unit the profile declared, so the same capture passed or failed
# depending on a control described in the interface as presentation. These pin
# the fix: a unit is how a number is written down, not what was measured.


def _metamorfoze_shaped(condition="value >= 38.25"):
    """Metamorfoze Full's MTF50 criterion, in the unit it states it in."""
    return profile_from_dict({
        "profile_id": "mz-shaped", "display_name": "mz-shaped",
        "metrics": [{
            "metric_id": "mtf50", "requires_capability": "sfr",
            "units": "% of half-sampling freq",
            "algorithm": {"frequency_units": "% of half-sampling freq"},
            "pass_condition": condition,
        }],
    })


def _run(capture, profile, units=None):
    path, corners = capture
    result = analyze(path, target=edge_target(), profile=profile, corners=corners,
                     fiducial_extent_mm=EXTENT, frequency_units=units)
    return result.evaluation.results[0]


@pytest.mark.parametrize("units", [None, "lp/mm", "cycles/mm", "lp/inch",
                                   "cycles/inch", "cycles/px",
                                   "% of half-sampling freq"])
def test_the_verdict_does_not_depend_on_the_unit_it_is_read_in(capture, units):
    """The regression that motivated all of this: on a passing capture the old
    code returned PASS in the profile's own unit and FAIL in every other."""
    metric = _run(capture, _metamorfoze_shaped(), units)
    assert metric.passed is True


@pytest.mark.parametrize("units", [None, "lp/mm", "cycles/px",
                                   "% of half-sampling freq"])
def test_a_failing_capture_fails_in_every_unit_too(capture, units):
    """The converse, so the fix is not just 'always passes now'."""
    metric = _run(capture, _metamorfoze_shaped("value >= 99"), units)
    assert metric.passed is False


def test_the_threshold_is_converted_with_the_reading(capture):
    """A report whose value and criterion are in different units cannot be read,
    and one where only the value moved cannot be trusted."""
    native = _run(capture, _metamorfoze_shaped())
    assert native.condition == "value >= 38.25"

    as_cycles = _run(capture, _metamorfoze_shaped(), "cycles/px")
    # 38.25% of the half-sampling frequency is 0.3825 x 0.5 cycles per pixel.
    assert as_cycles.condition == "value >= 0.19125"
    assert as_cycles.metadata["threshold"] == pytest.approx(0.19125)

    per_mm = _run(capture, _metamorfoze_shaped(), "lp/mm")
    ppi = per_mm.metadata["sampling_ppi"]
    assert per_mm.metadata["threshold"] == pytest.approx(0.19125 * ppi / MM_PER_INCH)


def test_the_value_to_threshold_ratio_is_the_invariant(capture):
    """Both sides scale together, so how much margin the capture has is the same
    number whichever unit it is read in.

    To within the criterion's own rounding: it is written to six significant
    figures because it is a string a person reads as well as the thing evaluated,
    so a converted threshold can move by 5e-7 of itself. Five orders below the
    precision of the reading it judges, and the alternative -- an exact threshold
    behind a rounded display -- is the split this whole fix exists to close.
    """
    margins = [
        (lambda m: m.value / m.metadata["threshold"])(
            _run(capture, _metamorfoze_shaped(), units)
        )
        for units in (None, "lp/mm", "lp/inch", "cycles/px",
                      "% of half-sampling freq")
    ]
    for margin in margins:
        assert margin == pytest.approx(margins[0], rel=1e-5)


def test_a_converted_threshold_is_exact_where_the_factor_is(capture):
    """The rounding is a property of writing the number down, not of the
    conversion: a factor that lands inside six figures lands exactly."""
    metric = _run(capture, _metamorfoze_shaped(), "cycles/px")
    assert metric.condition == "value >= 0.19125"
    assert metric.metadata["threshold"] == 0.19125


def test_a_two_sided_band_converts_both_bounds(capture):
    band = profile_from_dict({
        "profile_id": "band", "display_name": "band",
        "metrics": [{
            "metric_id": "mtf50", "requires_capability": "sfr",
            "units": "% of half-sampling freq",
            "algorithm": {"frequency_units": "% of half-sampling freq"},
            "pass_condition": "20 <= value <= 80",
        }],
    })
    assert _run(capture, band).passed is True
    converted = _run(capture, band, "cycles/px")
    assert converted.condition == "0.1 <= value <= 0.4"
    assert converted.passed is True


def test_the_profile_records_the_unit_it_declared(capture):
    """So a reader can see that the criterion was converted rather than rewritten,
    and against which unit the standard actually states it."""
    metric = _run(capture, _metamorfoze_shaped(), "lp/mm")
    assert metric.units == "lp/mm"
    assert metric.metadata["frequency_units"] == "lp/mm"
    assert metric.metadata["frequency_units_declared"] == "% of half-sampling freq"


def test_an_unconvertible_criterion_is_incomplete_rather_than_misjudged(capture):
    """Reading in cycles/px needs no scale, but carrying a lp/mm *threshold* into
    it does. With no scale available the honest answer is that the criterion
    cannot be brought across -- not a verdict against an unconverted number."""
    per_mm = profile_from_dict({
        "profile_id": "permm", "display_name": "permm",
        "metrics": [{
            "metric_id": "mtf50", "requires_capability": "sfr",
            "units": "lp/mm", "algorithm": {"frequency_units": "lp/mm"},
            "pass_condition": "value >= 2.5",
        }],
    })
    path, corners = capture
    result = analyze(path, target=edge_target(), profile=per_mm, corners=corners,
                     fiducial_extent_mm=None, frequency_units="cycles/px")
    metric = result.evaluation.results[0]
    assert metric.outcome == "incomplete"
    assert metric.passed is None
    assert "criterion that cannot be carried across" in metric.reason


def test_a_profile_with_no_criterion_converts_freely(capture):
    """Nothing to carry across, so an unjudged reading is available in any unit."""
    unjudged = profile_from_dict({
        "profile_id": "unjudged", "display_name": "unjudged",
        "metrics": [{"metric_id": "mtf50", "requires_capability": "sfr",
                     "units": "lp/mm", "algorithm": {"frequency_units": "lp/mm"}}],
    })
    path, corners = capture
    result = analyze(path, target=edge_target(), profile=unjudged, corners=corners,
                     fiducial_extent_mm=None, frequency_units="cycles/px")
    metric = result.evaluation.results[0]
    assert metric.outcome == "evaluated"
    assert metric.passed is None
    assert metric.value < 0.5
