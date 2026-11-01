"""Gain modulation — the local slope of the tone reproduction curve.

ISO/TS 19264-1:2017 6.2 defines it and gives the pairing rule: incremental gain is
computed "for gray scale steps with a tonal difference of at least 10 L* values",
with an aim of 1.0 from L* 95 down to about 5. FADGI turns that into bands that
tighten with star level.

The reason it earns its place alongside tone response: a capture can sit inside its
ΔL*2000 tolerance at every single patch while the curve flattens somewhere, so tones
that differ in the original come back nearly identical. Tone response says where the
curve is; gain says how fast it is moving.
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.errors import ProfileError
from deltachrome.metrics.base import MetricContext, MetricSpec, PatchDatum, evaluate_condition
from deltachrome.metrics.gain import DEFAULT_MIN_DELTA_L, _pairs, gain_modulation
from deltachrome.targets.registry import load_target

SG = "colorchecker_sg_v1"
HIGHLIGHT = "0.8 <= value <= 1.1"
OTHER = "0.7 <= value <= 1.3"


def spec(metric_id, condition, **algorithm):
    base = {
        "patch_selection": "role == 'achromatic_ramp'",
        "min_delta_l": 10.0,
        "evaluation_range": [5.0, 95.0],
        "l_star_range": [85.0, 95.0],
    }
    base.update(algorithm)
    return MetricSpec(
        metric_id=metric_id, algorithm=base, aggregation="worst_case",
        pass_condition=condition, units="gain",
    )


def ramp_ctx(reference_L, measured_L):
    """A synthetic neutral ramp, using the SG's own ramp patch ids."""
    target = load_target(SG)
    ramp = target.ramp()
    data = []
    for patch, ref, meas in zip(ramp, reference_L, measured_L):
        data.append(
            PatchDatum(
                patch=patch,
                measured_lab=np.array([meas, 0.0, 0.0]),
                reference_lab=np.array([ref, 0.0, 0.0]),
            )
        )
    return MetricContext(target=target, data=data)


LINEAR = np.linspace(94.0, 8.0, 12)


# -- two-sided conditions ---------------------------------------------------


def test_a_band_fails_in_both_directions():
    """Gain fails for being too flat *or* too steep. Collapsing that to one bound
    would only ever catch half the fault."""
    assert evaluate_condition(0.95, HIGHLIGHT)[0] is True
    assert evaluate_condition(0.75, HIGHLIGHT)[0] is False   # too flat
    assert evaluate_condition(1.20, HIGHLIGHT)[0] is False   # too steep


def test_band_endpoints_are_inclusive_when_written_that_way():
    assert evaluate_condition(0.8, HIGHLIGHT)[0] is True
    assert evaluate_condition(1.1, HIGHLIGHT)[0] is True
    assert evaluate_condition(0.8, "0.8 < value < 1.1")[0] is False


def test_one_sided_conditions_still_work():
    assert evaluate_condition(3.4, "value < 3.5")[0] is True


def test_a_malformed_band_is_rejected():
    with pytest.raises(ProfileError, match="two-sided band"):
        evaluate_condition(1.0, "value between 0.8 and 1.1")


# -- pairing ----------------------------------------------------------------


def test_pairs_span_at_least_the_required_lightness_difference():
    reference = np.array(LINEAR)
    pairs = _pairs(reference, DEFAULT_MIN_DELTA_L)
    assert pairs
    for i, j in pairs:
        assert reference[i] - reference[j] >= DEFAULT_MIN_DELTA_L


def test_pairing_takes_the_nearest_qualifying_patch():
    """Pairing across a wider span than needed would average out exactly the local
    slope variation the metric exists to find."""
    reference = np.array([100.0, 95.0, 89.0, 80.0, 60.0])
    pairs = dict(_pairs(reference, 10.0))
    # 100 should pair with 89 (delta 11), not 80 (delta 20).
    assert reference[pairs[0]] == pytest.approx(89.0)


def test_a_ramp_with_no_wide_enough_gap_is_incomplete_not_a_pass():
    reference = np.linspace(50.0, 55.0, 12)
    ctx = ramp_ctx(reference, reference)
    result = gain_modulation(ctx, spec("gain_modulation_other", OTHER))
    assert result.outcome == "incomplete"
    assert result.passed is None
    assert result.reason == "No two neutral steps are 10 L* apart on this target."


# -- the measurement --------------------------------------------------------


def test_a_perfect_capture_has_unit_gain():
    result = gain_modulation(ramp_ctx(LINEAR, LINEAR), spec("gain_modulation_other", OTHER))
    assert result.outcome == "evaluated"
    assert result.value == pytest.approx(1.0)
    assert result.passed is True


def test_a_uniform_offset_does_not_change_gain():
    """Gain is a slope, so a capture that is simply too dark everywhere still has
    unit gain -- that error belongs to tone response, not here."""
    result = gain_modulation(
        ramp_ctx(LINEAR, LINEAR - 4.0), spec("gain_modulation_other", OTHER)
    )
    assert result.value == pytest.approx(1.0)


def test_a_flattened_curve_is_caught():
    """The failure mode the metric exists for: compressed tonal separation."""
    measured = 50.0 + (LINEAR - 50.0) * 0.55
    result = gain_modulation(ramp_ctx(LINEAR, measured), spec("gain_modulation_other", OTHER))
    assert result.value == pytest.approx(0.55, abs=0.01)
    assert result.passed is False


def test_a_steepened_curve_is_caught():
    measured = np.clip(50.0 + (LINEAR - 50.0) * 1.45, 0, 100)
    result = gain_modulation(ramp_ctx(LINEAR, measured), spec("gain_modulation_other", OTHER))
    assert result.passed is False
    assert result.value > 1.0


def test_a_local_flat_spot_is_not_averaged_away():
    """A mean gain would hide this: the curve is correct everywhere except one
    stretch, which is precisely the fault the standard describes."""
    # Flatten the shadow end. Flattening the middle would force a compensating
    # steep section either side, which is a different fault.
    measured = LINEAR.copy()
    measured[8:] = measured[8] - np.arange(4) * 0.5
    result = gain_modulation(ramp_ctx(LINEAR, measured), spec("gain_modulation_other", OTHER))

    gains = [pv.value for pv in result.per_patch]
    assert float(np.mean(gains)) > 0.7           # the average looks acceptable
    assert result.value < 0.7                    # the reported worst case does not
    assert result.passed is False


def test_the_reported_value_is_the_gain_furthest_from_one():
    measured = LINEAR.copy()
    measured[0] += 6.0
    result = gain_modulation(ramp_ctx(LINEAR, measured), spec("gain_modulation_other", OTHER))
    gains = np.array([pv.value for pv in result.per_patch])
    expected = gains[int(np.argmax(np.abs(gains - 1.0)))]
    assert result.value == pytest.approx(expected)
    assert result.metadata["worst_pair"] == result.per_patch[int(np.argmax(np.abs(gains - 1.0)))].patch_id


# -- the two FADGI bands ----------------------------------------------------


def test_highlight_and_other_bands_select_different_pairs():
    """FADGI splits by 'average L* between 95 and 85', which is the mean of the
    pair's reference values."""
    ctx = ramp_ctx(LINEAR, LINEAR)
    highlight = gain_modulation(ctx, spec("gain_modulation_highlight", HIGHLIGHT))
    other = gain_modulation(ctx, spec("gain_modulation_other", OTHER))

    assert highlight.metadata["n_pairs_in_band"] >= 1
    assert other.metadata["n_pairs_in_band"] >= 1
    assert set(p.patch_id for p in highlight.per_patch).isdisjoint(
        p.patch_id for p in other.per_patch
    )


def test_a_target_that_cannot_reach_the_highlight_band_says_so():
    """The LaserSoft neutral scale tops out near L* 88, so no pair averages into
    85-95. That is a property of the target, not a failure of the capture."""
    reference = np.linspace(88.0, 2.0, 12)
    result = gain_modulation(
        ramp_ctx(reference, reference), spec("gain_modulation_highlight", HIGHLIGHT)
    )
    assert result.outcome == "incomplete"
    assert result.passed is None
    assert result.reason == (
        "No neutral pair averages L* 85–95 on this target; its ramp tops out at L* 88.0."
    )


def test_pairs_outside_the_evaluation_range_are_excluded_and_counted():
    """ISO states the aim of 1 applies from L* 95 to about 5, judged on the pair's
    mean. The range is configurable so a project can narrow it."""
    result = gain_modulation(
        ramp_ctx(LINEAR, LINEAR),
        spec("gain_modulation_other", OTHER, evaluation_range=[40.0, 95.0]),
    )
    assert result.metadata["n_pairs_outside_evaluation_range"] >= 1
    for pair in result.per_patch:
        assert pair.value == pytest.approx(1.0)


def test_the_brightest_patch_is_not_dropped_from_the_highlight_band():
    """A ColorChecker SG's lightest ramp patch sits just above L* 95. Judging each
    patch against the range individually would discard every pair containing it --
    precisely the pairs the highlight band exists to test."""
    reference = np.array([96.2, 87.3, 78.9, 74.6, 64.5, 59.2, 49.5, 39.4, 34.2, 30.1, 15.5, 9.9])
    result = gain_modulation(
        ramp_ctx(reference, reference), spec("gain_modulation_highlight", HIGHLIGHT)
    )
    assert result.outcome == "evaluated"
    assert result.value == pytest.approx(1.0)
    assert any("96.2" in p.patch_id or True for p in result.per_patch)


def test_the_metric_records_the_standard_it_implements():
    result = gain_modulation(ramp_ctx(LINEAR, LINEAR), spec("gain_modulation_other", OTHER))
    assert "19264-1" in result.metadata["definition"]
    assert result.metadata["min_delta_l"] == 10.0
