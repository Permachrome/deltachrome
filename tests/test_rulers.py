"""Reading a printed scale, and the traps in doing it.

A ruler is a comb, so measuring it is period estimation, and period estimation has
two classic ways of being confidently wrong: locking onto a harmonic, and letting
one stray mark bend the fit. A ruler read five times too coarse still produces a
beautiful straight line through every fifth tick, so neither failure announces
itself -- which is why they are tested here against combs of known pitch.
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.imaging.geometry import place_chart
from deltachrome.imaging.loader import load_image
from deltachrome.imaging.rulers import (
    RulerError,
    measure_ruler,
    read_rulers,
    sampling_rate,
)
from deltachrome.metrics.base import MetricContext, MetricSpec, get_metric
from deltachrome.targets.registry import load_target
from deltachrome.targets.schema import Fiducials, Region, TargetDefinition
from fixtures import adobe_rgb_icc, write_tiff_with_profile

METRIC = "reproduction_scale_accuracy"


def ruler_capture(
    path,
    *,
    pitch=40.0,
    divisions=30,
    major_every=5,
    margin=60,
    extra_marks=(),
    height=140,
    tick=26,
    long_tick=52,
    resolution=None,
    polarity="dark",
):
    """A synthetic rule: dark ticks at a known pitch, longer every ``major_every``.

    The long ticks are the point of the fixture. They are what a real rule draws and
    what makes the major-tick period, not the division, the strongest thing in the
    projected profile.

    ``polarity="light"`` inverts the whole thing into light ticks on a dark ground,
    which is what a *positive* transmissive chart prints. Same comb, same pitch, so
    the reading must not change.
    """
    ground, ink = (0.75, 0.2) if polarity == "dark" else (0.2, 0.85)
    width = int(margin * 2 + pitch * (divisions - 1)) + 1
    image = np.full((height, width), ground)
    image[20:23, :] = ink                                    # the rule's own baseline
    for i in range(divisions):
        x = int(round(margin + i * pitch))
        length = long_tick if (major_every and i % major_every == 0) else tick
        image[23:23 + length, x - 1:x + 2] = ink
    for x, length in extra_marks:
        image[23:23 + length, int(x) - 1:int(x) + 2] = ink
    rgb = np.dstack([image] * 3)
    encoded = np.clip(rgb, 0, 1) ** (1 / 2.19921875)
    write_tiff_with_profile(
        path, np.round(encoded * 65535).astype(np.uint16), adobe_rgb_icc(),
        resolution=resolution,
    )
    return path, width, height


def whole_frame(width, height):
    return place_chart(
        [(0, 0), (width, 0), (width, height), (0, height)], image_size=(width, height)
    )


def read(path, width, height, *, division_mm=1.0, band=(0.0, 0.15, 1.0, 0.55), **kw):
    image = load_image(path)
    return measure_ruler(
        image, whole_frame(width, height), bounds_relative=band,
        division_mm=division_mm, axis="u", **kw
    )


# -- the measurement is correct --------------------------------------------


@pytest.mark.parametrize("polarity", ["dark", "light"])
def test_a_rule_reads_the_same_whichever_way_its_ticks_go(tmp_path, polarity):
    """Every reflective chart here prints dark ticks on a light ground, which is what
    the reader originally assumed. A *positive* transmissive chart prints the
    opposite -- the 35mm microform target does -- and the failure is not a refusal:
    looking for dark ticks in a band of bright ones finds the noise between them and
    locks a confident 'pitch' onto it. Both ways must give the same number."""
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, divisions=26,
                               polarity=polarity)
    reading = read(path, w, h)
    assert reading.pitch_px == pytest.approx(40.0, abs=0.05)
    assert reading.residual_px < 1.0


@pytest.mark.parametrize("polarity", ["dark", "light"])
def test_polarity_can_be_declared_when_the_chart_is_known(tmp_path, polarity):
    """Auto-detection is the default because polarity belongs to the specimen -- one
    microform chart ships as both a positive and a negative through one set of
    declared bands. Declaring it stays available for a chart that is always one way."""
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, divisions=26,
                               polarity=polarity)
    assert read(path, w, h, polarity=polarity).pitch_px == pytest.approx(40.0, abs=0.05)


def test_declaring_the_wrong_polarity_is_refused_rather_than_answered(tmp_path):
    """The point of keeping the override honest: told to read light ticks in a band
    of dark ones, the reader must not find a period in the gaps between them."""
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, divisions=26)
    try:
        reading = read(path, w, h, polarity="light")
    except RulerError:
        return
    assert reading.pitch_px != pytest.approx(40.0, abs=1.0)


def test_an_unknown_polarity_is_a_programming_error(tmp_path):
    path, w, h = ruler_capture(tmp_path / "r.tif")
    with pytest.raises(RulerError, match="polarity must be"):
        read(path, w, h, polarity="inverted")



@pytest.mark.parametrize("pitch", [12.0, 25.5, 40.0, 78.8])
def test_a_known_pitch_is_recovered(tmp_path, pitch):
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=pitch, divisions=26)
    reading = read(path, w, h)
    assert reading.pitch_px == pytest.approx(pitch, abs=0.05)
    assert reading.residual_px < 1.0


def test_the_fundamental_is_found_and_not_the_major_tick_harmonic(tmp_path):
    """The trap this whole module is arranged around. Every fifth tick is longer and
    correlates more strongly, so a period estimate that takes the tallest peak reads
    the scale five times too coarse -- and fits it beautifully."""
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, divisions=41, major_every=5)
    reading = read(path, w, h)
    assert reading.pitch_px == pytest.approx(40.0, abs=0.05)
    assert reading.pitch_px < 100.0            # not 200, the five-division harmonic


def test_a_long_rule_at_a_fractional_pitch_is_not_mis_numbered(tmp_path):
    """A fine rule photographed at a non-integer pitch, long enough for a half-pixel
    period error to matter.

    Ticks are numbered by position over the estimated period, and that estimate
    comes from an integer autocorrelation lag -- so half a pixel of error starts
    mis-numbering once it accumulates to half a division. At 15.66 px that is the
    46th tick, which a 30-division fixture never reaches and the UTT's 179-division
    millimetre rule passes three times over. Ticks then collide on one index, get
    dropped as duplicates, and a rule straight to a tenth of a pixel is refused.
    """
    path, w, h = ruler_capture(
        tmp_path / "r.tif", pitch=15.66, divisions=180, major_every=10,
        tick=14, long_tick=30, height=90,
    )
    reading = read(path, w, h)
    assert reading.pitch_px == pytest.approx(15.66, abs=0.02)
    assert reading.divisions_used > 170
    assert reading.residual_px < 0.5


@pytest.mark.parametrize("major_every", [2, 4, 5, 10])
def test_any_major_tick_interval_a_rule_uses(tmp_path, major_every):
    path, w, h = ruler_capture(
        tmp_path / "r.tif", pitch=30.0, divisions=41, major_every=major_every
    )
    assert read(path, w, h).pitch_px == pytest.approx(30.0, abs=0.05)


def test_pixels_per_mm_and_ppi_follow_the_declared_division(tmp_path):
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, divisions=26)
    metric = read(path, w, h, division_mm=1.0)
    imperial = read(path, w, h, division_mm=25.4 / 16)
    assert metric.px_per_mm == pytest.approx(40.0, abs=0.05)
    assert metric.ppi == pytest.approx(40.0 * 25.4, rel=1e-3)
    # Same ticks, different declared division: the rate scales with it, which is
    # exactly why two rules on one chart agreeing is worth checking.
    assert imperial.px_per_mm == pytest.approx(40.0 / (25.4 / 16), abs=0.05)


def test_a_rotated_capture_reads_the_same(tmp_path):
    """The band is rectified through the placement, so a chart photographed askew is
    measured along the ruler's own direction rather than along the pixel grid."""
    import cv2

    pitch, angle, inset = 40.0, 3.0, 40
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=pitch, divisions=21, height=420)

    # An inset quad, so its corners stay inside the frame once rotated. The band is
    # expressed against that quad, which is how a real definition works anyway.
    quad = np.array(
        [[inset, inset], [w - inset, inset], [w - inset, h - inset], [inset, h - inset]],
        dtype=np.float64,
    )
    band = (
        (60 - inset) / (w - 2 * inset), (24 - inset) / (h - 2 * inset),
        (w - 60 - inset) / (w - 2 * inset), (70 - inset) / (h - 2 * inset),
    )
    straight = measure_ruler(
        load_image(path), place_chart([tuple(p) for p in quad], image_size=(w, h)),
        bounds_relative=band, division_mm=1.0, axis="u",
    )

    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    canvas = cv2.warpAffine(
        (np.clip(load_image(path).rgb, 0, 1) * 65535).astype(np.uint16), matrix, (w, h),
        borderValue=(48000, 48000, 48000),
    )
    rotated = write_tiff_with_profile(tmp_path / "rot.tif", canvas, adobe_rgb_icc())
    moved = np.column_stack([quad, np.ones(4)]) @ matrix.T

    reading = measure_ruler(
        load_image(rotated), place_chart([tuple(p) for p in moved], image_size=(w, h)),
        bounds_relative=band, division_mm=1.0, axis="u",
    )
    assert straight.pitch_px == pytest.approx(pitch, abs=0.1)
    assert reading.pitch_px == pytest.approx(straight.pitch_px, rel=0.01)


# -- and it refuses when it should -----------------------------------------


def test_a_stray_mark_is_rejected_rather_than_averaged_in(tmp_path):
    """A label, a speck, or a neighboring feature the band clips all look like
    ticks in projection. One is enough to bend the fitted line and, through it, the
    numbering of every tick at the far end."""
    clean, w, h = ruler_capture(tmp_path / "clean.tif", pitch=40.0, divisions=26, margin=100)
    expected = read(clean, w, h)

    # Placed in the clear run before the first tick, far enough from it to survive
    # peak finding and off the comb by more than a third of a division -- which is
    # exactly how a feature clipped by the band's edge presents itself.
    dirty, w2, h2 = ruler_capture(
        tmp_path / "dirty.tif", pitch=40.0, divisions=26, margin=100,
        extra_marks=((45, 40),),
    )
    reading = read(dirty, w2, h2)
    assert reading.pitch_px == pytest.approx(expected.pitch_px, abs=0.1)
    assert not any("comb" in w for w in reading.warnings), "rejected, and not announced"


def test_too_few_divisions_is_refused(tmp_path):
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, divisions=5, major_every=0)
    with pytest.raises(RulerError, match="fewer than"):
        read(path, w, h)


def test_a_band_with_no_ruler_in_it_is_refused(tmp_path):
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, divisions=26)
    with pytest.raises(RulerError):
        read(path, w, h, band=(0.05, 0.75, 0.95, 0.95))     # below the ticks


def test_a_band_too_small_to_read_is_refused(tmp_path):
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, divisions=26)
    with pytest.raises(RulerError, match="too small"):
        read(path, w, h, band=(0.4, 0.2, 0.41, 0.5))


# -- reducing several rulers -----------------------------------------------


def test_two_rulers_that_agree_report_their_agreement(tmp_path):
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, divisions=26)
    a = read(path, w, h, region_id="a")
    b = read(path, w, h, region_id="b")
    summary = sampling_rate([a, b])
    assert summary["ppi"] == pytest.approx(a.ppi, rel=1e-6)
    assert summary["agreement"] < 1e-6
    assert "warning" not in summary


def test_two_rulers_that_disagree_say_so(tmp_path):
    """The only check available that a scale was read as the chart intends. A rule
    misread by a factor still fits a straight line through its own ticks."""
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, divisions=26)
    a = read(path, w, h, division_mm=1.0, region_id="metric")
    b = read(path, w, h, division_mm=1.10, region_id="imperial")
    summary = sampling_rate([a, b])
    assert "warning" in summary
    assert "disagree" in summary["warning"]


# -- how fine a comb the reader can read ------------------------------------


def _synthetic_comb(pitch, divisions, tick_px, phase=0.37, pad=1.5):
    """A projected ruler profile with sub-pixel-exact tick coverage.

    Ticks are boxes of ``tick_px`` integrated analytically over each pixel rather
    than supersampled, so the antialiasing is exact and the only thing varying
    across a sweep is what is meant to vary. Built here rather than rendered
    through a TIFF because the question is about the estimator, and a render would
    add quantisation to the thing being measured.
    """
    span = pitch * (divisions - 1)
    n = int(np.ceil(span + 2 * pitch * pad)) + 1
    x = np.arange(n)
    ink = np.zeros(n)
    start = pitch * pad + phase
    for i in range(divisions):
        lo = start + i * pitch - tick_px / 2.0
        hi = lo + tick_px
        ink = np.maximum(
            ink, np.clip(np.minimum(x + 0.5, hi) - np.maximum(x - 0.5, lo), 0.0, None)
        )
    return 0.85 - 0.65 * ink


def _read_profile(profile):
    from deltachrome.imaging.rulers import _fit_to_comb, _tick_positions

    positions, period, _ = _tick_positions(profile)
    kept, index, _ = _fit_to_comb(positions, period)
    slope, _ = np.polyfit(index, kept, 1)
    return abs(float(slope))


@pytest.mark.parametrize("pitch", [8.0, 11.8, 15.7, 25.0, 40.0, 80.0, 125.0, 200.0])
def test_the_period_estimator_is_exact_from_eight_pixels_a_division_up(pitch):
    """This is why a ruler disagreement resolved as a
    finding about a chart rather than a bug here.

    Both ISA FADGI - ISO 19264 captures read their millimetre comb about 0.75%
    coarser than the eighth-inch comb printed on the same rule, at about a dozen
    pixels a division -- and the suspicion was that the estimator loses accuracy as
    the period closes on the tick width. It does not. On a comb of known pitch it is
    within 0.01% from 8 px a division to 60, and within 0.1% out to 200.
    """
    divisions = max(12, min(90, int(3000 / pitch)))
    measured = _read_profile(_synthetic_comb(pitch, divisions, max(1.0, pitch * 0.10)))
    tolerance = 0.0001 if pitch <= 60 else 0.001
    assert abs(measured - pitch) / pitch < tolerance


@pytest.mark.parametrize("tick_px", [1.0, 1.5, 2.0, 4.0])
def test_a_tick_narrower_than_a_pixel_does_not_bias_the_pitch(tick_px):
    """At 400 ppi a printed rule's tick line is one or two pixels wide, so it is
    barely sampled and its centroid rests on the antialiasing alone. That is the
    regime the FADGI - ISO 19264 charts are read in."""
    measured = _read_profile(_synthetic_comb(15.7, 90, tick_px))
    assert measured == pytest.approx(15.7, rel=0.0002)


# -- the chart's own scales disagreeing --------------------------------------


def test_scales_in_different_units_are_reported_apart_from_field_variation(tmp_path):
    """Two rules of the same unit at different places disagree because the
    magnification varies across the field -- that is distortion. A chart's metric and
    imperial engravings disagreeing is a property of the chart, shifts every rule of
    one family together, and is not distortion. Pooling them puts a printing error
    into the lens's figure."""
    from deltachrome.imaging.rulers import unit_agreement

    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, divisions=26)
    #: 1.008 rather than 1.0: one family reading 0.8% coarse, which is the size of
    #: the offset measured on both ISA FADGI - ISO 19264 captures.
    metric = [read(path, w, h, division_mm=1.0, region_id="ruler_a_mm", unit="mm"),
              read(path, w, h, division_mm=1.0, region_id="ruler_b_mm", unit="mm")]
    imperial = [read(path, w, h, division_mm=1.008, region_id="ruler_a_in", unit="in"),
                read(path, w, h, division_mm=1.008, region_id="ruler_b_in", unit="in")]
    summary = unit_agreement(metric + imperial)
    assert set(summary["by_unit"]) == {"mm", "in"}
    assert summary["disagreement"] == pytest.approx(0.008, abs=0.0005)
    assert summary["real"] is True

    # The same rules with no offset: nothing to report.
    same = unit_agreement(
        metric + [read(path, w, h, division_mm=1.0, region_id="ruler_a_in", unit="in")]
    )
    assert same["disagreement"] < 1e-6
    assert same["real"] is False


def test_one_unit_alone_has_no_disagreement_to_report(tmp_path):
    from deltachrome.imaging.rulers import unit_agreement

    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, divisions=26)
    summary = unit_agreement([read(path, w, h, region_id="ruler_a_mm", unit="mm")])
    assert summary["disagreement"] == 0.0
    assert summary["real"] is False


# -- the metric ------------------------------------------------------------


def _target_with_ruler(band=(0.0, 0.15, 1.0, 0.55), division_mm=1.0):
    return TargetDefinition(
        target_id="synthetic", display_name="synthetic ruler chart", manufacturer=None,
        target_type="test", physical_size_mm=None,
        capabilities=frozenset({"reproduction_scale"}), not_capable_of=frozenset(),
        fiducials=Fiducials(method="corners", corner_ids=[], corner_positions=[]),
        regions=[Region(region_id="rule", kind="scale", geometry={
            "unit": "mm", "division_mm": division_mm, "axis": "u",
            "bounds_relative": list(band),
        })],
        patches=[],
    )


def _run(image, readings, condition="value < 2"):
    target = _target_with_ruler()
    ctx = MetricContext(
        target=target, data=[], extras={"image": image, "rulers": readings}
    )
    spec = MetricSpec(
        metric_id=METRIC, requires_capability="reproduction_scale",
        aggregation="max", pass_condition=condition,
        units="% difference from header ppi",
    )
    return get_metric(METRIC)(ctx, spec)


def test_the_metric_is_incomplete_without_a_ruler(tmp_path):
    path, w, h = ruler_capture(tmp_path / "r.tif")
    result = _run(load_image(path), [])
    assert result.outcome == "incomplete"
    assert "no printed scale" in (result.reason or "").lower()


def test_the_metric_is_incomplete_when_the_file_claims_no_resolution(tmp_path):
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, divisions=26)
    image = load_image(path)
    assert image.header_ppi is None
    result = _run(image, [read(path, w, h)])
    assert result.outcome == "incomplete"
    assert "no resolution" in (result.reason or "")


def test_a_header_that_matches_the_ruler_passes(tmp_path):
    pitch = 40.0
    path, w, h = ruler_capture(
        tmp_path / "r.tif", pitch=pitch, divisions=26, resolution=pitch * 25.4
    )
    image = load_image(path)
    assert image.header_ppi == pytest.approx((pitch * 25.4, pitch * 25.4), rel=1e-3)
    result = _run(image, [read(path, w, h)])
    assert result.outcome == "evaluated"
    assert result.value < 0.5
    assert result.passed is True


def test_a_header_that_does_not_match_fails_and_says_which_way(tmp_path):
    """The fault the metric exists for: a resolution that is simply wrong, which
    nothing else in a workflow would notice."""
    path, w, h = ruler_capture(
        tmp_path / "r.tif", pitch=40.0, divisions=26, resolution=600.0
    )
    result = _run(load_image(path), [read(path, w, h)])
    assert result.passed is False
    assert result.value == pytest.approx(100.0 * (40.0 * 25.4 - 600.0) / 600.0, rel=0.02)
    assert result.metadata["signed_difference_pct"] > 0
    assert any("resolves finer than the file claims" in n for n in result.notes)


def test_the_reported_value_is_absolute_because_the_criterion_is_two_sided(tmp_path):
    """FADGI writes it as '<+/- 2%' while a profile condition is a ceiling. Reporting
    the signed value would pass every capture that resolves coarser than it claims."""
    path, w, h = ruler_capture(
        tmp_path / "r.tif", pitch=40.0, divisions=26, resolution=1500.0
    )
    result = _run(load_image(path), [read(path, w, h)])
    assert result.value > 0
    assert result.metadata["signed_difference_pct"] < 0
    assert result.passed is False


# -- sampling frequency prefers the measured rate --------------------------


def test_sampling_frequency_prefers_the_ruler_over_a_declared_extent(tmp_path):
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, divisions=26)
    reading = read(path, w, h)
    target = _target_with_ruler()
    ctx = MetricContext(
        target=target, data=[],
        extras={"image": load_image(path), "rulers": [reading],
                "placement": whole_frame(w, h), "fiducial_extent_mm": (1000.0, 1000.0)},
    )
    spec = MetricSpec(metric_id="sampling_frequency", pass_condition="value >= 400",
                      units="ppi")
    result = get_metric("sampling_frequency")(ctx, spec)
    assert result.outcome == "evaluated"
    assert result.metadata["source"] == "printed ruler"
    assert result.value == pytest.approx(reading.ppi, rel=1e-6)


# -- what the shipped definitions declare ----------------------------------


@pytest.mark.parametrize(
    "target_id,count",
    [
        ("isa_transmissive_4x5_v1", 4),
        ("isa_transmissive_35mm_v1", 4),
        ("isa_transmissive_120_v1", 4),
        ("isa_golden_thread_dlt_v1", 20),
    ],
)
def test_the_isa_targets_declare_independent_scales_in_both_units(target_id, count):
    """More than one rule per chart. They measure the same capture through
    separately engraved scales, and their agreement is the only check that either
    was read as the chart intends."""
    target = load_target(target_id)
    scales = [r for r in target.regions if r.kind == "scale"]
    assert len(scales) == count
    units = {r.geometry["unit"] for r in scales}
    assert units == {"mm", "in"}
    for region in scales:
        assert region.geometry["division_mm"] > 0
        assert len(region.geometry["bounds_relative"]) == 4


def test_the_golden_thread_measures_both_axes_at_every_module():
    """What makes geometric distortion measurable on it at all: a rule on all four
    sides of each of the five modules, so both axes are read -- twice, in different
    units -- at five field positions."""
    scales = [r for r in load_target("isa_golden_thread_dlt_v1").regions
              if r.kind == "scale"]
    by_axis_unit = {}
    for region in scales:
        key = (region.geometry["axis"], region.geometry["unit"])
        by_axis_unit[key] = by_axis_unit.get(key, 0) + 1
    assert by_axis_unit == {("u", "mm"): 5, ("u", "in"): 5,
                            ("v", "mm"): 5, ("v", "in"): 5}

    positions = {r.region_id.split("_")[1] for r in scales}
    assert positions == {"tl", "tr", "bl", "br", "center"}


# -- vertical rules ---------------------------------------------------------
#
# Every rule shipped before this point ran along ``u``, so the ``v`` branch was
# never exercised on a real chart. It was transposed: the destination quad sent the
# across-axis where the along-axis belonged, the strip came out rotated, and the
# projection then ran *across* the rule rather than along it. The ticks averaged
# out, the profile went flat, and every vertical rule reported "not on the rule" --
# a clean-looking refusal rather than a wrong number, which is why it survived.
#
# The ISA Golden Thread prints a rule on all four sides of every module, so the
# vertical ones are half the scale information the chart carries.


def vertical_ruler_capture(path, **kw):
    """The same synthetic rule, rotated a quarter turn."""
    import cv2

    from fixtures import adobe_rgb_icc, write_tiff_with_profile

    horizontal, width, height = ruler_capture(path, **kw)
    image = load_image(horizontal)
    rotated = cv2.rotate(image.rgb, cv2.ROTATE_90_CLOCKWISE)
    write_tiff_with_profile(
        path, np.round(np.clip(rotated, 0, 1) ** (1 / 2.19921875) * 65535).astype(np.uint16),
        adobe_rgb_icc(),
    )
    return path, rotated.shape[1], rotated.shape[0]


@pytest.mark.parametrize("pitch", [12.0, 25.5, 40.0, 78.8])
def test_a_vertical_rule_reads_the_same_pitch_as_a_horizontal_one(tmp_path, pitch):
    """The regression. Before the fix this raised "not on the rule" for every pitch."""
    path, w, h = vertical_ruler_capture(tmp_path / "v.tif", pitch=pitch, divisions=26)
    image = load_image(path)
    reading = measure_ruler(
        image, whole_frame(w, h),
        # The band the horizontal fixture uses, transposed with the image.
        bounds_relative=(0.45, 0.0, 0.85, 1.0),
        division_mm=1.0, axis="v",
    )
    assert reading.pitch_px == pytest.approx(pitch, abs=0.05)
    assert reading.residual_px < 1.0


def test_both_axes_agree_on_one_capture(tmp_path):
    """What makes a distortion measurement possible at all: the same physical pitch
    read along each axis has to come back the same when there is no distortion."""
    path, w, h = ruler_capture(tmp_path / "h.tif", pitch=40.0, divisions=26)
    horizontal = read(path, w, h).pitch_px

    vpath, vw, vh = vertical_ruler_capture(tmp_path / "v.tif", pitch=40.0, divisions=26)
    vertical = measure_ruler(
        load_image(vpath), whole_frame(vw, vh), bounds_relative=(0.45, 0.0, 0.85, 1.0),
        division_mm=1.0, axis="v",
    ).pitch_px

    assert vertical == pytest.approx(horizontal, rel=0.002)


def test_a_band_whose_marks_all_get_rejected_refuses_cleanly(tmp_path):
    """Outlier rejection can remove every located mark. That has to come back as a
    refusal, not an IndexError out of the middle of the fit -- a band scanning for a
    rule hits this case constantly, and a crash there is indistinguishable from a
    bug in the caller."""
    path, w, h = ruler_capture(
        tmp_path / "r.tif", pitch=40.0, divisions=3, major_every=0,
        extra_marks=((150, 30), (163, 30), (410, 30), (505, 30)),
    )
    with pytest.raises(RulerError):
        read(path, w, h)


# -- the band is searched for, not taken on trust ----------------------------
#
# A declared ``bounds_relative`` used to be the band that was read, which made
# every reading depend on a height tuned against whichever specimens were to
# hand. On the Golden Thread Object Level Target the window where all three known
# units read is about 0.02 quad-heights -- two or three pixels, because that
# chart's quad is ONE PATCH TALL -- and a third specimen missed it entirely and
# reported no scale at all.


def band_target(*, extent=None, regions=2, division_mm=1.0, band=(0.0, 0.15, 1.0, 0.55)):
    """A target declaring one or two rules over the whole frame."""
    return TargetDefinition(
        target_id="band_probe", display_name="Band probe", manufacturer=None,
        target_type="test", physical_size_mm=None,
        capabilities=frozenset({"reproduction_scale"}), not_capable_of=frozenset(),
        fiducials=Fiducials(method="corners", corner_ids=[], corner_positions=[]),
        fiducial_extent_mm=extent,
        regions=[
            Region(region_id=f"rule{i}", kind="scale", geometry={
                "unit": "mm", "division_mm": division_mm, "axis": "u",
                "bounds_relative": list(band),
            })
            for i in range(regions)
        ],
        patches=[],
    )


def test_a_rule_the_declared_band_misses_is_found_beside_it(tmp_path):
    """The reported failure, in miniature: the band is declared above the ticks
    and everything about the rule is otherwise fine."""
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, resolution=(600, 600))
    image = load_image(path)
    placement = whole_frame(w, h)
    # Declared well above the ticks, which start at row 23 of 140.
    missed = (0.0, 0.0, 1.0, 0.14)
    with pytest.raises(RulerError):
        measure_ruler(image, placement, bounds_relative=missed,
                      division_mm=1.0, axis="u")

    target = band_target(extent=(w / 40.0, h / 40.0), band=missed)
    readings, notes = read_rulers(image, placement, target)
    assert len(readings) == 2, notes
    assert all(abs(r.pitch_px - 40.0) < 0.5 for r in readings)
    assert all(r.bounds_used != missed for r in readings), "it moved off the declared band"
    assert all(r.bands_read > 1 for r in readings), "and reports how many agreed"


def test_a_band_that_already_reads_is_not_searched_at_all(tmp_path):
    """What keeps this affordable. The Device Level Target declares twenty rules
    and sweeping every one of them costs ten seconds to re-derive twenty answers
    that were already right."""
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, resolution=(600, 600))
    image = load_image(path)
    target = band_target(extent=(w / 40.0, h / 40.0))
    readings, _ = read_rulers(image, whole_frame(w, h), target)

    assert len(readings) == 2
    for reading in readings:
        assert reading.bounds_used == (0.0, 0.15, 1.0, 0.55)
        assert reading.bands_read == 1, "one read, not a sweep"


def test_a_search_with_nothing_to_check_against_is_not_run(tmp_path):
    """**An ungated search invents readings, and that is measured.** On a
    deliberately degraded placement of the Device Level Target it rescued six of
    twenty rules at 201, 243, 280, 291, 351 and 422 ppi on a chart whose real
    rate is 609 -- six perfectly regular combs found in whatever the band
    happened to cross. A refusal is a fact an operator can act on; a plausible
    wrong number is the failure this project exists to avoid."""
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, resolution=(600, 600))
    image = load_image(path)
    # No extent to predict a pitch from, and every rule declared where the ticks
    # are not -- so nothing read, so no consensus either.
    target = band_target(extent=None, band=(0.0, 0.0, 1.0, 0.14))
    readings, notes = read_rulers(image, whole_frame(w, h), target)

    assert readings == []
    assert len(notes) == 2
    assert "is not searched" in notes[0]


def test_a_rule_that_read_is_what_the_others_are_searched_against(tmp_path):
    """The gate that needs no declared extent, and the one most targets have:
    eleven of the twelve that declare rules declare more than one."""
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, resolution=(600, 600))
    image = load_image(path)
    good = Region(region_id="reads", kind="scale", geometry={
        "unit": "mm", "division_mm": 1.0, "axis": "u",
        "bounds_relative": [0.0, 0.15, 1.0, 0.55]})
    missed = Region(region_id="missed", kind="scale", geometry={
        "unit": "mm", "division_mm": 1.0, "axis": "u",
        "bounds_relative": [0.0, 0.0, 1.0, 0.14]})
    target = TargetDefinition(
        target_id="pair", display_name="Pair", manufacturer=None,
        target_type="test", physical_size_mm=None,
        capabilities=frozenset({"reproduction_scale"}), not_capable_of=frozenset(),
        fiducials=Fiducials(method="corners", corner_ids=[], corner_positions=[]),
        fiducial_extent_mm=None, regions=[good, missed], patches=[])

    readings, notes = read_rulers(image, whole_frame(w, h), target)
    found = {r.region_id: r for r in readings}
    assert "reads" in found
    assert "missed" in found, f"the rule that read should rescue the one that did not: {notes}"
    assert abs(found["missed"].pitch_px - found["reads"].pitch_px) < 0.5


def test_a_reading_a_factor_out_is_refused_even_though_it_is_regular(tmp_path):
    """The one check regularity cannot make. A millimetre comb read as
    centimetres fits a beautiful straight line through its own ticks and reports
    a rate ten times out; only something outside the located ticks can say so."""
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0, resolution=(600, 600))
    image = load_image(path)
    placement = whole_frame(w, h)
    # The synthetic comb is 40 px per 1 mm division, so the frame is w/40 mm
    # wide. Declared as 10 mm divisions it is perfectly regular and wrong by a
    # factor of ten.
    target = band_target(extent=(w / 40.0, h / 40.0), division_mm=10.0)
    readings, notes = read_rulers(image, placement, target)
    assert readings == [], "a factor-out reading must not be reported"
    assert notes

    # Without an extent and without a sibling that read, there is nothing to
    # check it against, and the old answer stands -- stated so the limit is
    # visible rather than assumed away.
    loose = band_target(extent=None, division_mm=10.0, regions=1)
    loose_readings, _ = read_rulers(image, placement, loose)
    assert len(loose_readings) == 1


def test_a_band_swept_off_the_capture_refuses_rather_than_crashing(tmp_path):
    """Latent until the band became something searched for: a declared band is
    inside the chart by construction, and a swept one is not."""
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0)
    image = load_image(path)
    with pytest.raises(RulerError, match="outside the capture"):
        measure_ruler(image, whole_frame(w, h),
                      bounds_relative=(0.0, 3.0, 1.0, 3.4), division_mm=1.0, axis="u")


# -- a scale printed as blocks rather than as ticks ---------------------------
#
# The Golden Thread charts print a two-row checkerboard above each rule, marking
# the same divisions as the rule beneath it with filled cells instead of thin
# marks. It is a far better feature to measure on, and on one capture here it is
# the only one of the two that can be read at all.


def block_capture(path, *, cell=60.0, cells=10, height=40, margin=30, rows=2,
                  resolution=None):
    """A row (or two complementary rows) of alternating light and dark cells."""
    width = int(margin * 2 + cell * cells)
    image = np.full((height * rows + 20, width), 0.75)
    for r in range(rows):
        top = 10 + r * height
        for i in range(cells):
            x0 = int(round(margin + i * cell))
            x1 = int(round(margin + (i + 1) * cell))
            dark = (i + r) % 2 == 0
            image[top:top + height, x0:x1] = 0.18 if dark else 0.82
    rgb = np.dstack([image] * 3)
    encoded = np.clip(rgb, 0, 1) ** (1 / 2.19921875)
    write_tiff_with_profile(
        path, np.round(encoded * 65535).astype(np.uint16), adobe_rgb_icc(),
        resolution=resolution,
    )
    return path, width, image.shape[0]


def test_a_block_pattern_measures_its_own_cell_pitch(tmp_path):
    path, w, h = block_capture(tmp_path / "b.tif", cell=60.0, cells=10, rows=1)
    reading = measure_ruler(
        load_image(path), whole_frame(w, h), bounds_relative=(0.0, 0.3, 1.0, 0.8),
        division_mm=10.0, axis="u", pattern="blocks",
    )
    assert abs(reading.pitch_px - 60.0) < 0.5
    assert reading.divisions_used >= 6


def test_a_block_fit_is_far_more_regular_than_a_tick_fit(tmp_path):
    """The whole argument for the second reader. A boundary between two filled
    areas is located from a band averaged down its whole height; a tick fills a
    few rows of the band it is found in, so grain moves its centroid."""
    blocks, w, h = block_capture(tmp_path / "b.tif", cell=60.0, cells=10, rows=1)
    reading = measure_ruler(
        load_image(blocks), whole_frame(w, h), bounds_relative=(0.0, 0.3, 1.0, 0.8),
        division_mm=10.0, axis="u", pattern="blocks",
    )
    assert reading.residual_px / reading.pitch_px < 0.01, (
        "a block fit is held to a tenth of the tick gate, and earns it"
    )


def test_a_band_across_both_rows_of_the_checkerboard_refuses(tmp_path):
    """They are complementary, so a band spanning both averages them into a flat
    line with no boundary in it. A refusal, and the band search moves off it."""
    path, w, h = block_capture(tmp_path / "b.tif", cell=60.0, cells=10, rows=2)
    with pytest.raises(RulerError):
        measure_ruler(
            load_image(path), whole_frame(w, h),
            # Straddling both rows, equally.
            bounds_relative=(0.0, 0.14, 1.0, 0.98),
            division_mm=10.0, axis="u", pattern="blocks",
        )


def test_a_block_reading_is_held_to_a_tenth_of_the_tick_gate(tmp_path):
    """Holding blocks to the tick gate admits readings an order of magnitude
    worse than a block fit ought to be. On one of the three Object Level units
    the declared band scraped in at 3.9%, half a percent off the rate, and
    dragged that chart's pooled reading with it."""
    from deltachrome.imaging.rulers import (
        MAX_BLOCK_RESIDUAL_FRACTION, MAX_RESIDUAL_FRACTION,
    )

    assert MAX_BLOCK_RESIDUAL_FRACTION < MAX_RESIDUAL_FRACTION / 3


def test_the_pattern_must_be_one_this_reader_knows(tmp_path):
    path, w, h = ruler_capture(tmp_path / "r.tif")
    with pytest.raises(RulerError, match="ticks' or 'blocks'"):
        measure_ruler(load_image(path), whole_frame(w, h),
                      bounds_relative=(0.0, 0.15, 1.0, 0.55),
                      division_mm=1.0, axis="u", pattern="dots")


def test_ticks_remain_the_default_so_every_target_reads_as_it_did(tmp_path):
    path, w, h = ruler_capture(tmp_path / "r.tif", pitch=40.0)
    plain = read(path, w, h)
    named = read(path, w, h, pattern="ticks")
    assert plain.pitch_px == named.pitch_px
