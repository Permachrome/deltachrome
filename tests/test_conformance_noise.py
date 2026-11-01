"""File-level conformance and noise, and the FADGI 4-star profile that uses them.

FADGI's tables specify more than colorimetry. Every star level also fixes the
master format, bit depth, color space and sampling frequency. A profile that
evaluated only the color metrics and still printed "FADGI 4-star: PASS" for an
8-bit sRGB file would be overclaiming in exactly the way this project exists to
avoid.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from deltachrome.metrics.base import MetricContext, MetricSpec
from deltachrome.metrics.conformance import (
    bit_depth,
    color_space,
    master_file_format,
    sampling_frequency,
)
from deltachrome.pipeline import analyze
from deltachrome.imaging.loader import load_image
from deltachrome.profiles.schema import load_profile, profile_from_dict
from deltachrome.targets.registry import load_target
from fixtures import SyntheticChart, adobe_rgb_icc, write_jp2

SG = "colorchecker_sg_v1"
FADGI4 = "fadgi_4star_prints_photographs_2023"


@pytest.fixture
def chart_files(tmp_path):
    chart = SyntheticChart()
    reference = chart.write_reference(tmp_path / "ref.cgats.txt")
    image, corners = chart.render_capture(tmp_path / "capture.tif", skew=18)
    return chart, reference, image, corners


def ctx_for(image=None, placement=None):
    return MetricContext(
        target=load_target(SG),
        data=[],
        extras={k: v for k, v in (("image", image), ("placement", placement)) if v},
    )


# -- bit depth --------------------------------------------------------------


def test_bit_depth_passes_and_fails_against_the_star_level(chart_files):
    from deltachrome.imaging.loader import load_image

    _, _, image_path, _ = chart_files
    image = load_image(image_path)
    spec = MetricSpec(metric_id="bit_depth", pass_condition="value >= 16", units="bits")

    result = bit_depth(ctx_for(image), spec)
    assert result.value == 16.0 and result.passed is True

    strict = MetricSpec(metric_id="bit_depth", pass_condition="value >= 32")
    assert bit_depth(ctx_for(image), strict).passed is False


def test_bit_depth_is_judged_against_the_depths_the_table_permits(chart_files):
    """The profiles state 8 or 16 rather than a floor, because those are the two
    the tables name and a floor quietly admits everything above it."""
    from deltachrome.imaging.loader import load_image

    _, _, image_path, _ = chart_files
    image = load_image(image_path)  # 16-bit

    both = MetricSpec(metric_id="bit_depth", algorithm={"allowed": [8, 16]})
    result = bit_depth(ctx_for(image), both)
    assert result.passed is True
    assert result.condition == "8 or 16"

    eight_only = MetricSpec(metric_id="bit_depth", algorithm={"allowed": [8]})
    refused = bit_depth(ctx_for(image), eight_only)
    assert refused.passed is False
    assert refused.condition == "8"


def test_a_depth_is_written_as_a_whole_number(chart_files):
    """It is counted, not measured: "16.000" claims five figures of precision for
    a quantity that has two."""
    from deltachrome.imaging.loader import load_image

    _, _, image_path, _ = chart_files
    spec = MetricSpec(metric_id="bit_depth", algorithm={"allowed": [8, 16]})
    result = bit_depth(ctx_for(load_image(image_path)), spec)
    assert result.value == 16.0, "the number a session merges on is unchanged"
    assert result.value_text == "16"
    assert result.reading == "16"


def test_the_file_level_metrics_read_as_words_rather_than_as_numbers(chart_files):
    """A container and a colour space have no number at all, and a report that
    showed "-" for them would be saying it had not identified them."""
    from deltachrome.imaging.loader import load_image

    _, _, image_path, _ = chart_files
    image = load_image(image_path)

    container = master_file_format(
        ctx_for(image),
        MetricSpec(metric_id="master_file_format",
                   algorithm={"allowed": ["TIFF", "JPEG 2000"]}),
    )
    assert container.reading == "TIFF"
    # The permitted list as a person would write it, not as Python prints it.
    assert container.condition == "TIFF, JPEG 2000"

    space = color_space(
        ctx_for(image),
        MetricSpec(metric_id="color_space",
                   algorithm={"allowed": ["Adobe RGB (1998)"]}),
    )
    assert "Adobe RGB" in space.reading
    assert space.condition == "Adobe RGB (1998)"


def test_a_failing_bit_depth_is_the_verdict_and_nothing_more(tmp_path):
    """The failed row shows the depth read and the depths permitted, which is all
    an operator needs; a remedy note under it said the same thing again."""
    from deltachrome.imaging.loader import load_image

    pixels = (np.random.default_rng(1).uniform(0.2, 0.8, (32, 32, 3)) * 255).astype(np.uint8)
    path = write_jp2(tmp_path / "eight.jp2", pixels, icc=adobe_rgb_icc())
    spec = MetricSpec(metric_id="bit_depth", pass_condition="value >= 16")

    result = bit_depth(ctx_for(load_image(path)), spec)
    assert result.passed is False
    assert result.value_text == "8"
    assert result.notes == []


# -- color space and container --------------------------------------------


def test_color_space_matches_across_naming_variants(chart_files):
    """Profiles in the wild read 'eciRGB v2', 'ECI RGB v2' and 'ECIRGB_v2' for the
    same space, so matching cannot be literal."""
    from deltachrome.imaging.loader import load_image

    _, _, image_path, _ = chart_files
    image = load_image(image_path)  # tagged "Synthetic Adobe RGB (1998) [...]"

    allowed = MetricSpec(metric_id="color_space", algorithm={"allowed": ["Adobe RGB (1998)"]})
    assert color_space(ctx_for(image), allowed).passed is True

    denied = MetricSpec(metric_id="color_space", algorithm={"allowed": ["ProPhoto"]})
    result = color_space(ctx_for(image), denied)
    assert result.passed is False
    assert result.condition == "ProPhoto"


def test_master_file_format_is_checked(chart_files, tmp_path):
    from deltachrome.imaging.loader import load_image

    _, _, tif_path, _ = chart_files
    spec = MetricSpec(
        metric_id="master_file_format", algorithm={"allowed": ["TIFF", "JPEG 2000"]}
    )
    assert master_file_format(ctx_for(load_image(tif_path)), spec).passed is True

    pixels = (np.random.default_rng(2).uniform(0.2, 0.8, (32, 32, 3)) * 255).astype(np.uint8)
    jp2 = write_jp2(tmp_path / "m.jp2", pixels, icc=adobe_rgb_icc())
    assert master_file_format(ctx_for(load_image(jp2)), spec).passed is True

    narrow = MetricSpec(metric_id="master_file_format", algorithm={"allowed": ["TIFF"]})
    assert master_file_format(ctx_for(load_image(jp2)), narrow).passed is False


# -- sampling frequency -----------------------------------------------------


def test_sampling_frequency_is_incomplete_without_a_measured_fiducial_extent():
    """A target whose fiducial extent is unrecorded must not have its sampling rate
    derived from the card size instead -- that under-reports it, and the difference
    is enough to fail a level the capture actually meets."""
    from deltachrome.imaging.geometry import place_chart

    target = load_target(SG)
    target.fiducial_extent_mm = None
    placement = place_chart([(0, 0), (1000, 0), (1000, 700), (0, 700)], (1200, 900))
    ctx = MetricContext(target=target, data=[], extras={"placement": placement})

    result = sampling_frequency(
        ctx, MetricSpec(metric_id="sampling_frequency", pass_condition="value >= 594")
    )
    assert result.outcome == "incomplete"
    assert result.passed is None
    assert "fiducial_extent_mm" in result.reason
    assert "under-state" in result.reason


def test_the_sg_records_a_measured_fiducial_extent():
    """Read off the chart's own printed ruler on two different physical units."""
    target = load_target(SG)
    assert target.fiducial_extent_mm == pytest.approx((247.6, 176.1))

    raw = json.loads(target.source_path.read_text(encoding="utf-8"))
    provenance = raw["fiducial_extent_measured_from"]
    assert len(provenance["measurements"]) == 2
    # The two units were captured at very different sampling rates, so agreement
    # between them is not an artefact of one measurement.
    rates = sorted(m["sampling_rate_ppi"] for m in provenance["measurements"])
    assert rates[1] / rates[0] > 1.4
    for measurement in provenance["measurements"]:
        assert measurement["quad_mm"][0] == pytest.approx(247.6, abs=0.2)
        assert measurement["quad_mm"][1] == pytest.approx(176.1, abs=0.2)
    assert "CENTIMETERS" in provenance["detail"]
    assert "caliper" in provenance["caveat"]


def test_the_fiducial_extent_is_smaller_than_the_card():
    """The crosshairs sit inside the card edges. Conflating the two is exactly the
    error the separate field exists to prevent."""
    target = load_target(SG)
    assert target.physical_size_mm == (292.1, 203.2)
    assert target.fiducial_extent_mm[0] < target.physical_size_mm[0]
    assert target.fiducial_extent_mm[1] < target.physical_size_mm[1]
    # ~15% under-report if the card size were used by mistake.
    ratio = target.fiducial_extent_mm[0] / target.physical_size_mm[0]
    assert 0.80 < ratio < 0.90


def test_the_run_record_states_which_dimension_the_rate_came_from(chart_files):
    _, reference, image, corners = chart_files
    from deltachrome.pipeline import analyze

    data = analyze(
        image, target=SG, reference_path=reference, profile=FADGI4, corners=corners
    ).run.to_dict()
    assert data["geometry"]["sampling_rate_basis"] == "fiducial_extent_mm"
    assert data["geometry"]["sampling_rate_reference_mm"] == [247.6, 176.1]
    assert "sampling_rate_caveat" not in data["geometry"]


def test_sampling_frequency_is_computed_when_the_extent_is_known(chart_files):
    from deltachrome.imaging.geometry import place_chart

    target = load_target(SG)
    target.fiducial_extent_mm = (254.0, 178.0)
    placement = place_chart([(0, 0), (6000, 0), (6000, 4200), (0, 4200)], (6100, 4300))
    ctx = MetricContext(target=target, data=[], extras={"placement": placement})

    spec = MetricSpec(metric_id="sampling_frequency", pass_condition="value >= 594")
    result = sampling_frequency(ctx, spec)
    assert result.outcome == "evaluated"
    assert result.value == pytest.approx(6000 / (254.0 / 25.4), rel=0.02)
    assert result.passed is True
    assert any("not read from the file header" in n for n in result.notes)


# -- noise ------------------------------------------------------------------


def test_noise_is_measured_and_a_clean_render_is_quiet(chart_files):
    _, reference, image, corners = chart_files
    profile = load_profile(FADGI4)
    result = analyze(
        image, target=SG, reference_path=reference, profile=profile, corners=corners
    )
    noise = next(m for m in result.evaluation.results if m.metric_id == "noise")

    assert noise.outcome == "evaluated"
    # A synthetic chart is rendered flat, so the only spread is quantisation.
    assert noise.value < 0.5
    assert noise.metadata["noisiest_patch"]


def test_a_noisy_capture_fails_noise(chart_files, tmp_path):
    import tifffile

    from fixtures import adobe_rgb_icc

    chart, reference, image, corners = chart_files
    clean = tifffile.imread(image).astype(np.float64)
    rng = np.random.default_rng(5)
    noisy = np.clip(clean + rng.normal(0, 900, clean.shape), 0, 65535).astype(np.uint16)
    path = tmp_path / "noisy.tif"
    tifffile.imwrite(
        path, noisy, photometric="rgb",
        extratags=[(34675, "B", len(adobe_rgb_icc()), adobe_rgb_icc(), True)],
    )

    result = analyze(
        path, target=SG, reference_path=reference, profile=FADGI4, corners=corners
    )
    noise = next(m for m in result.evaluation.results if m.metric_id == "noise")
    assert noise.value > 1.0
    assert noise.passed is False


def test_near_zero_noise_warns_rather_than_failing(chart_files):
    """FADGI's lower limit is a tell, not a quality floor: a capture with no
    measurable noise has usually had noise reduction applied, which removes the fine
    detail the digitisation was meant to keep."""
    _, reference, image, corners = chart_files
    result = analyze(
        image, target=SG, reference_path=reference, profile=FADGI4, corners=corners
    )
    noise = next(m for m in result.evaluation.results if m.metric_id == "noise")

    assert noise.metadata["below_lower_limit"] is True
    assert any("noise reduction" in n for n in noise.notes)
    # A warning, not a failure.
    assert noise.passed is True


# -- the profile ------------------------------------------------------------


def test_four_star_thresholds_match_the_published_table():
    """Transcribed from FADGI 3rd Ed. section 3.8, pages 50-51, in the units the
    table itself states -- sampling efficiency and SFR50 as percentages, response at
    Nyquist and sharpening as modulation."""
    profile = load_profile(FADGI4)
    expected = {
        "average_color_accuracy": "value < 2",
        "color_accuracy_90th_percentile": "value < 4",
        "white_balance": "value < 2",
        "tone_response_oecf": "value < 1.5",
        "lightness_uniformity": "value < 1",
        "noise": "value < 1",
        "sampling_frequency": "value >= 594",
        "color_channel_misregistration": "value < 0.33",
        "sfr50": "45 <= value <= 65",
        "sfr10_sampling_efficiency": "value > 90",
        "sfr_response_at_nyquist": "value < 0.2",
        "sharpening": "value < 1.02",
        "reproduction_scale_accuracy": "value < 1",
    }
    for metric_id, condition in expected.items():
        assert profile.metric(metric_id).pass_condition == condition, metric_id
    # Bit depth is stated as the depths the table permits, not as a floor.
    assert profile.metric("bit_depth").algorithm["allowed"] == [16]


def test_four_star_is_stricter_than_three_star_everywhere_they_overlap():
    three = load_profile("fadgi_3star_documents_unbound_general_2023")
    four = load_profile(FADGI4)
    for metric_id in ("average_color_accuracy", "color_accuracy_90th_percentile",
                      "white_balance", "tone_response_oecf"):
        a = float(three.metric(metric_id).pass_condition.split("<")[1])
        b = float(four.metric(metric_id).pass_condition.split("<")[1])
        assert b < a, metric_id


def test_four_star_requires_16_bit_and_drops_srgb():
    profile = load_profile(FADGI4)
    allowed = profile.metric("color_space").algorithm["allowed"]
    assert not any("srgb" in a.lower() for a in allowed)
    assert profile.metric("bit_depth").algorithm["allowed"] == [16]


def test_every_four_star_metric_the_sg_supports_now_has_a_plugin(chart_files):
    """Gain modulation was the last gap. What remains unevaluated for a ColorChecker
    SG is genuinely inapplicable -- it has no slanted edges and no registration
    features -- not missing engine support."""
    _, reference, image, corners = chart_files
    result = analyze(
        image, target=SG, reference_path=reference, profile=FADGI4, corners=corners
    )
    assert result.evaluation.not_implemented == []

    by_id = {r.metric_id: r for r in result.evaluation.results}
    for metric_id in ("gain_modulation_highlight", "gain_modulation_other"):
        assert by_id[metric_id].outcome == "evaluated", metric_id
    for metric_id in ("sfr10_sampling_efficiency", "color_channel_misregistration"):
        assert by_id[metric_id].outcome == "not_applicable", metric_id


def test_gain_modulation_uses_a_two_sided_band_in_the_profile():
    """A ceiling would only catch a curve that is too steep. Too flat is the failure
    the metric was introduced for."""
    profile = load_profile(FADGI4)
    assert profile.metric("gain_modulation_highlight").pass_condition == "0.8 <= value <= 1.1"
    assert profile.metric("gain_modulation_other").pass_condition == "0.7 <= value <= 1.3"


def test_the_profile_records_what_it_does_not_model():
    """SFR50 is a two-sided range the schema cannot express yet; saying so in the
    profile is better than dropping the row silently."""
    profile = load_profile(FADGI4)
    assert "SFR50" in profile.notes
    assert "pages 50-51" in profile.standard_reference["thresholds_transcribed_from"]


# -- targets that cannot establish their own scale -------------------------
#
# A figure measured from one specimen is not a property of the model. Film targets
# are imaged onto stock and then cut to format, so neither the imaging scale across
# production batches nor the placement of the cut is guaranteed. Carrying one
# specimen's number in the target definition would quietly assert something about
# every other unit.

LS = "lasersoft_iso12641_2_advanced_v1"


def test_a_target_with_no_printed_scale_declares_that_it_has_none():
    target = load_target(LS)
    assert target.fiducial_extent_mm is None
    assert target.dimensional_reference["available"] is False
    assert "cut" in target.dimensional_reference["notes"]


def test_a_target_with_a_printed_scale_declares_where_it_is():
    target = load_target(SG)
    reference = target.dimensional_reference
    assert reference["available"] is True
    assert reference["kind"] == "printed_scale"
    assert "columns K-N" in reference["location"]


def test_a_declared_absence_of_scale_is_not_applicable():
    """This reversed a decision, and the reason is consistency within one report.

    The old answer was ``incomplete``, on the argument that the resolution
    requirement applies to the capture whatever chart was in front of it. That is
    true -- and it is true of *sharpness* too, which a chart carrying no slanted
    edges reports as N/A without anybody thinking the requirement has gone away.
    One metric holding the opposite convention made INCOMPLETE mean two things at
    once in a single report: "this should have been measured and was not" on some
    rows and "this chart cannot measure it" on others.

    What keeps the requirement in force is not this outcome. The reason still says
    it applies, a profile's coverage is reported, and a session distinguishes a
    metric no chart carried from one a chart carried and failed on.
    """
    from deltachrome.imaging.geometry import place_chart

    target = load_target(LS)
    placement = place_chart([(0, 0), (1000, 0), (1000, 700), (0, 700)], (1200, 900))
    ctx = MetricContext(target=target, data=[], extras={"placement": placement})

    result = sampling_frequency(
        ctx, MetricSpec(metric_id="sampling_frequency", pass_condition="value >= 1980")
    )
    assert result.outcome == "not_applicable"
    assert result.passed is None
    assert "carries no printed scale" in result.reason
    assert "still applies" in result.reason
    assert "--fiducial-extent" in result.reason


def test_a_scale_that_should_have_read_and_did_not_is_still_incomplete():
    """The distinction the change rests on. A target that says nothing about its
    own dimensions has not declared that it cannot be measured -- it has simply
    never been measured, and that is a gap to fill rather than a property of the
    chart."""
    from deltachrome.imaging.geometry import place_chart

    target = load_target(LS)
    object.__setattr__(target, "dimensional_reference", {})
    placement = place_chart([(0, 0), (1000, 0), (1000, 700), (0, 700)], (1200, 900))
    ctx = MetricContext(target=target, data=[], extras={"placement": placement})

    result = sampling_frequency(
        ctx, MetricSpec(metric_id="sampling_frequency", pass_condition="value >= 1980")
    )
    assert result.outcome == "incomplete"


def test_an_operator_measured_extent_unblocks_the_metric_for_that_unit():
    from deltachrome.imaging.geometry import place_chart

    target = load_target(LS)
    placement = place_chart([(0, 0), (6000, 0), (6000, 4200), (0, 4200)], (6100, 4300))
    ctx = MetricContext(
        target=target, data=[],
        extras={"placement": placement, "fiducial_extent_mm": (101.6, 71.1)},
    )
    result = sampling_frequency(
        ctx, MetricSpec(metric_id="sampling_frequency", pass_condition="value >= 1980")
    )
    assert result.outcome == "evaluated"
    assert result.metadata["extent_source"] == "operator-supplied"
    assert any("does not transfer to another unit" in n for n in result.notes)


def test_an_override_takes_precedence_over_the_definition():
    from deltachrome.imaging.geometry import place_chart

    target = load_target(SG)      # has a measured extent of its own
    placement = place_chart([(0, 0), (6000, 0), (6000, 4200), (0, 4200)], (6100, 4300))
    spec_ = MetricSpec(metric_id="sampling_frequency", pass_condition="value >= 1")

    from_definition = sampling_frequency(
        MetricContext(target=target, data=[], extras={"placement": placement}), spec_
    )
    overridden = sampling_frequency(
        MetricContext(target=target, data=[],
                      extras={"placement": placement, "fiducial_extent_mm": (123.8, 88.05)}),
        spec_,
    )
    assert from_definition.metadata["extent_source"] == "target definition"
    assert overridden.metadata["extent_source"] == "operator-supplied"
    # Half the extent, so twice the sampling rate.
    assert overridden.value == pytest.approx(from_definition.value * 2, rel=0.01)


@pytest.mark.parametrize(
    "text,expected",
    [("247.6x176.1", (247.6, 176.1)), ("100X70", (100.0, 70.0)), ("12,8", (12.0, 8.0))],
)
def test_the_extent_argument_accepts_the_obvious_spellings(text, expected):
    from deltachrome.cli import _parse_extent

    assert _parse_extent(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["247.6", "axb", "0x10", "-5x10"])
def test_a_malformed_extent_argument_is_refused(text):
    from deltachrome.cli import _parse_extent
    from deltachrome.errors import TargetError

    with pytest.raises(TargetError):
        _parse_extent(text)


def test_sampling_frequency_and_mtf50_cannot_disagree_about_the_scale(tmp_path):
    """They resolve it through the same helper on purpose. Two independent copies
    of the ruler-then-extent preference order would drift, and the symptom would be
    an lp/mm figure that quietly contradicts the ppi printed beside it."""
    from test_sfr import edge_target, rotated_square_capture

    path = rotated_square_capture(tmp_path)
    image = load_image(path)
    profile = profile_from_dict({
        "profile_id": "both", "display_name": "both",
        "metrics": [
            {"metric_id": "sampling_frequency", "requires_capability": None,
             "pass_condition": "value >= 0"},
            {"metric_id": "mtf50", "requires_capability": "sfr",
             "pass_condition": "value > 0", "algorithm": {"frequency_units": "lp/inch"}},
        ],
    })
    results = {r.metric_id: r for r in analyze(
        path, target=edge_target(), profile=profile,
        corners=[(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
        fiducial_extent_mm=(100.0, 100.0),
    ).evaluation.results}

    assert results["mtf50"].metadata["sampling_ppi"] == pytest.approx(
        results["sampling_frequency"].value, abs=0.01
    )


# -- telling a gamut overflow from underexposure ------------------------------


def _sampled(pixels, low=0.0005):
    """Run the flagging logic over a block of pixels, as the sampler would."""
    import numpy as np
    from deltachrome.imaging.sampling import _pinned_channels

    block = np.asarray(pixels, dtype=np.float64)
    return _pinned_channels(block, block.mean(axis=0), low)


def test_a_bright_patch_with_one_channel_at_the_floor_is_a_gamut_overflow():
    """A saturated yellow: red and green high, blue truncated at zero. Nothing is
    underexposed here and telling the operator to open up would make it worse."""
    assert _sampled([[0.89, 0.80, 0.0]] * 50) == ["B"]
    assert _sampled([[0.0, 0.72, 0.66]] * 50) == ["R"]


def test_a_saturated_red_is_not_read_as_underexposure():
    """The case that decides the test's shape. A saturated red has green at zero
    AND blue low, so requiring every other channel to be bright would classify it
    as a shadow problem. The test is on the BRIGHTEST other channel."""
    assert _sampled([[0.48, 0.0, 0.21]] * 50) == ["G"]


def test_a_patch_dark_in_every_channel_is_underexposure_not_gamut():
    """Nothing bright anywhere, so there is no colour too saturated to encode --
    the shadow simply was not recorded."""
    assert _sampled([[0.02, 0.0, 0.015]] * 50) == []
    assert _sampled([[0.0, 0.0, 0.0]] * 50) == []


def test_a_grayscale_capture_can_never_be_gamut_clipped():
    """One colour record has nothing to be out of gamut against, so the
    distinction does not exist and must not be claimed."""
    assert _sampled([[0.0]] * 50) == []


def _flags_for(profile_class):
    """Sample one saturated patch -- red pinned at zero, green and blue bright --
    from a capture whose embedded profile is of the given ICC class."""
    from types import SimpleNamespace

    import numpy as np
    from deltachrome.imaging.sampling import _sample_polygon

    pixels = np.zeros((40, 40, 3))
    pixels[..., 1], pixels[..., 2] = 0.72, 0.66
    image = SimpleNamespace(
        height=40, width=40, channels=3,
        window=lambda y0, y1, x0, x1: pixels[y0:y1, x0:x1],
        profile=SimpleNamespace(profile_class=profile_class),
    )
    square = np.array([[5.0, 5.0], [34.0, 5.0], [34.0, 34.0], [5.0, 34.0]])
    patch = SimpleNamespace(patch_id="X1", row=0, col=0)
    return _sample_polygon(image, square, patch, 1.0, 0.0005, measure_noise=False).flags


def test_a_working_space_capture_calls_a_pinned_channel_a_gamut_overflow():
    assert "gamut_clipped(R)" in _flags_for("mntr")


def test_a_device_space_capture_does_not_send_the_operator_to_a_wider_space():
    """A camera or scanner profile is the device's own response, not a space
    anything was encoded INTO, so there is no wider one to re-shoot in. The same
    pixels are flagged for what they are there: a channel that got no light."""
    for device_class in ("scnr", "cmra"):
        flags = _flags_for(device_class)
        assert "channel_floor(R)" in flags
        assert not any(f.startswith("gamut_clipped") for f in flags)


def test_the_colour_metric_explains_each_flag_and_its_remedy():
    """Overexposure and a gamut overflow both arrive as 'clipping' and are fixed at
    different controls, so the note has to separate them."""
    from deltachrome.metrics.difference import FLAG_EXPLANATIONS

    kinds = dict(FLAG_EXPLANATIONS)
    assert set(kinds) == {
        "gamut_clipped", "channel_floor", "clipped_high", "clipped_low",
        "mean_median_divergence",
    }
    assert "wider space" in kinds["gamut_clipped"]
    assert "no wider space" in kinds["channel_floor"]
    assert "Reduce exposure" in kinds["clipped_high"]
    assert "Increase exposure" in kinds["clipped_low"]
    # gamut_clipped must be attributed before clipped_low, which always accompanies
    # it -- otherwise the same patch is counted twice and reports a shadow problem
    # that is not there.
    order = [name for name, _ in FLAG_EXPLANATIONS]
    assert order.index("gamut_clipped") < order.index("clipped_low")
    assert order.index("channel_floor") < order.index("clipped_low")
