"""The three Metamorfoze profiles, against the specification table they come from.

Transcription is where a profile silently goes wrong: nothing crashes, the numbers
look plausible, and a capture is judged against the wrong standard. So the values in
section 1.6 of Metamorfoze Preservation Imaging Guidelines 2.0 (April 2025) are
restated here independently of the generator that wrote the JSON.

Two entries are readings rather than quotations, and both are pinned here as
readings so that changing one is a deliberate act: the MTF50 requirement, and the
"maximum L* value for assessing neutrals".
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.errors import ProfileError
from deltachrome.metrics import available_metrics
from deltachrome.metrics.base import PatchDatum, select_patches
from deltachrome.profiles.schema import load_profile
from deltachrome.targets.schema import Patch

FULL = "metamorfoze_full_2025"
LIGHT = "metamorfoze_light_2025"
EXTRA = "metamorfoze_extra_light_2025"
ALL = (FULL, LIGHT, EXTRA)


def condition(profile_id, metric_id):
    return load_profile(profile_id).metric(metric_id).pass_condition


# -- the table --------------------------------------------------------------


@pytest.mark.parametrize(
    "metric_id,expected",
    [
        ("white_balance", ("value <= 3", "value <= 3", "value <= 5")),
        ("tone_response_oecf", ("value <= 2", "value <= 2", "value <= 4")),
        ("noise", ("value <= 1.6", "value <= 1.6", "value <= 2")),
        ("color_channel_misregistration",
         ("value <= 0.35", "value <= 0.5", "value <= 0.5")),
        ("geometric_distortion", ("value <= 2", "value <= 2", "value <= 4")),
        ("sfr10_sampling_efficiency", ("value >= 85",) * 3),
        ("sharpening", ("value <= 1.05",) * 3),
        ("sampling_frequency", ("value >= 300",) * 3),
        ("reproduction_scale_accuracy", ("value <= 2",) * 3),
        ("gain_modulation_highlight", ("0.8 <= value <= 1.1",) * 3),
        ("gain_modulation_other", ("0.6 <= value <= 1.4",) * 3),
    ],
)
def test_thresholds_match_the_specification_table(metric_id, expected):
    for profile_id, want in zip(ALL, expected):
        assert condition(profile_id, metric_id) == want, f"{profile_id}/{metric_id}"


def test_color_accuracy_is_a_mean_and_a_maximum_not_a_percentile():
    """Where FADGI states a 90th percentile, Metamorfoze states a maximum, so one
    bad patch decides it."""
    assert condition(FULL, "average_color_accuracy") == "value <= 3"
    assert condition(FULL, "color_accuracy_max") == "value <= 7"
    assert condition(LIGHT, "average_color_accuracy") == "value <= 4"
    assert condition(LIGHT, "color_accuracy_max") == "value <= 14"
    for profile_id in ALL:
        assert not any(m.metric_id == "color_accuracy_90th_percentile"
                       for m in load_profile(profile_id).metrics)


def test_extra_light_declares_no_color_accuracy_at_all():
    """The table says "Not specified". A metric with an invented threshold would be
    worse than its absence, and absence never counts as a failure."""
    metrics = {m.metric_id for m in load_profile(EXTRA).metrics}
    assert "average_color_accuracy" not in metrics
    assert "color_accuracy_max" not in metrics


def test_white_balance_uses_ciede2000_without_luminance():
    """Not the Euclidean dE(a*b*). Near the neutral axis, which is where a white
    balance is measured, substituting the Euclidean measure is permissive."""
    for profile_id in ALL:
        spec = load_profile(profile_id).metric("white_balance")
        assert spec.algorithm["delta_e_variant"] == "delta_e_2000_ab"


def test_exposure_uses_plain_delta_l():
    """CIE2000 with S_L = 1, not FADGI's dL*2000."""
    for profile_id in ALL:
        assert load_profile(profile_id).metric("tone_response_oecf").algorithm["measure"] \
            == "delta_l"


# -- the illumination bands -------------------------------------------------


@pytest.mark.parametrize("profile_id", [FULL, LIGHT])
def test_illumination_is_banded_by_original_size(profile_id):
    banded = load_profile(profile_id).metric("illumination_non_uniformity").pass_condition_by
    assert banded.parameter == "original_size"
    assert [(b.band, b.pass_condition) for b in banded.bands] == [
        ("A3", "value <= 3"), ("A2", "value <= 4"),
        ("A1", "value <= 5"), ("A0", "value <= 6"),
    ]


def test_extra_light_declares_only_the_band_the_standard_states():
    """Above A3 the table says "Not specified", so there is no threshold to judge
    against -- and asking for one is refused rather than answered with the A3 row."""
    banded = load_profile(EXTRA).metric("illumination_non_uniformity").pass_condition_by
    assert [b.band for b in banded.bands] == ["A3"]
    assert banded.named("A4").pass_condition == "value <= 5"
    with pytest.raises(ProfileError, match="beyond the largest band"):
        banded.named("A1")


# -- the two readings -------------------------------------------------------


def test_mtf50_is_expressed_where_its_ambiguity_disappears():
    """Stated as ">= 45% of the minimum required lp/mm at MTF10", and that MTF10
    requirement is sampling efficiency >= 85% of Nyquist. The chain collapses:
    0.45 x 0.85 = 38.25% of the half-sampling frequency, whatever the sampling rate,
    so in this unit the reading and its alternative agree.

    Confirmed against the guideline's own section 2.14: "regardless of
    the sampling rate size" the requirement is stated as a fraction of MTF10
    performance, which rules out the alternative reading this once flagged as
    untested -- anchoring the requirement to a flat 300 ppi-derived lp/mm figure
    regardless of the capture's own rate."""
    for profile_id in ALL:
        spec = load_profile(profile_id).metric("mtf50")
        assert spec.pass_condition == "value >= 38.25"
        assert spec.algorithm["frequency_units"] == "% of half-sampling freq"
        assert spec.units == "% of half-sampling freq"
        assert "CONFIRMED" in spec.notes


def test_the_neutral_lightness_floor_differs_by_level():
    """"Maximum L* value for assessing neutrals": L* 5 / 20 / 30. Read as the
    darkest tone the ladder is judged down to -- the lighter levels do not hold the
    deep shadows to the same standard.

    Confirmed against the guideline's own prose: "For Metamorfoze Full,
    the neutrals are assessed up to L* 5 ... Light ... up to L* 20 ... Extra Light,
    neutrals are assessed up to L* 30.\""""
    for profile_id, floor in ((FULL, 5.0), (LIGHT, 20.0), (EXTRA, 30.0)):
        for metric_id in ("white_balance", "tone_response_oecf", "noise"):
            selection = load_profile(profile_id).metric(metric_id).algorithm["patch_selection"]
            assert selection["min_reference_L"] == floor, f"{profile_id}/{metric_id}"


def test_the_lightness_floor_actually_excludes_the_shadow_patches():
    """The point of it. Judging patches a level excludes would fail captures that
    level passes."""
    ladder = [
        PatchDatum(
            patch=Patch(patch_id=f"L{L}", role="achromatic_ramp", row=0, col=0,
                        region="main"),
            measured_lab=np.array([float(L), 0.0, 0.0]),
            reference_lab=np.array([float(L), 0.0, 0.0]),
        )
        for L in (96, 80, 60, 40, 25, 15, 8, 3)
    ]
    counts = {}
    for profile_id in ALL:
        selection = load_profile(profile_id).metric("noise").algorithm["patch_selection"]
        counts[profile_id] = len(select_patches(ladder, selection))
    assert counts[FULL] == 7        # everything but L* 3
    assert counts[LIGHT] == 5       # down to L* 25
    assert counts[EXTRA] == 4       # down to L* 40


# -- every metric can actually run ------------------------------------------


@pytest.mark.parametrize("profile_id", ALL)
def test_every_declared_metric_has_a_plugin(profile_id):
    """A metric reporting NOT IMPLEMENTED can never pass, so one without a plugin
    makes the whole profile unusable. This is what kept these profiles from shipping
    until geometric distortion existed."""
    registered = set(available_metrics())
    declared = {m.metric_id for m in load_profile(profile_id).metrics}
    assert declared <= registered, sorted(declared - registered)


@pytest.mark.parametrize("profile_id", ALL)
def test_the_profile_says_no_single_target_covers_it(profile_id):
    """Illumination wants a blank white sheet, color wants a patch chart, sharpness
    wants slanted edges. Someone running one chart and reading a pass needs to know
    what was not tested."""
    notes = load_profile(profile_id).notes
    assert "No single target carries everything" in notes
    assert "session" in notes
    assert "Artefacts" in notes
