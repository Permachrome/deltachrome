"""Lightness uniformity, and the report that has to explain it honestly."""

from __future__ import annotations

import json

import numpy as np
import pytest

from deltachrome.metrics.base import MetricContext, MetricSpec, PatchDatum
from deltachrome.metrics.uniformity import lightness_uniformity
from deltachrome.pipeline import analyze
from deltachrome.reporting.console import render
from deltachrome.targets.registry import load_target
from fixtures import UNIFORMITY_IDS, SyntheticChart

SG = "colorchecker_sg_v1"
FADGI3 = "fadgi_3star_documents_unbound_general_2023"

UNIFORMITY_PROFILE = {
    "profile_id": "test_uniformity_profile",
    "display_name": "Test profile exercising lightness uniformity",
    "standard_reference": {"body": "none — test fixture"},
    "applicable_target_capabilities_required": ["color_accuracy"],
    "applicable_target_capabilities_optional": ["lightness_uniformity"],
    "metrics": [
        {
            "metric_id": "average_color_accuracy",
            "requires_capability": "color_accuracy",
            "algorithm": {"delta_e_variant": "delta_e_2000", "patch_selection": "all patches"},
            "aggregation": "mean",
            "pass_condition": "value < 3.5",
            "units": "dE2000",
        },
        {
            "metric_id": "lightness_uniformity",
            "requires_capability": "lightness_uniformity",
            "algorithm": {"patch_selection": {"roles": ["achromatic_border"]}},
            "aggregation": "max",
            "pass_condition": "value < 2",
            "units": "L*",
        },
    ],
    "overall_pass_rule": "all applicable metrics must pass",
}


@pytest.fixture
def uniformity_profile(tmp_path):
    path = tmp_path / "uniformity_profile.json"
    path.write_text(json.dumps(UNIFORMITY_PROFILE), encoding="utf-8")
    return path


@pytest.fixture
def chart_files(tmp_path):
    chart = SyntheticChart()
    reference = chart.write_reference(tmp_path / "ref.cgats.txt")
    image, corners = chart.render_capture(tmp_path / "capture.tif", skew=18)
    return chart, reference, image, corners


def make_data(entries):
    target = load_target(SG)
    return [
        PatchDatum(
            patch=target[pid].with_role("achromatic_border"),
            measured_lab=np.array(measured, dtype=float),
            reference_lab=np.array(reference, dtype=float),
        )
        for pid, measured, reference in entries
    ]


# -- the metric in isolation -----------------------------------------------


def test_a_perfectly_uniform_capture_reads_zero():
    data = make_data([(pid, (96.0, -0.9, 3.2), (96.0, -0.9, 3.2)) for pid in UNIFORMITY_IDS])
    spec = MetricSpec(metric_id="lightness_uniformity", aggregation="max", pass_condition="value < 2")
    result = lightness_uniformity(MetricContext(target=load_target(SG), data=data), spec)

    assert result.outcome == "evaluated"
    assert result.value == pytest.approx(0.0, abs=1e-9)
    assert result.passed is True


def test_the_raw_fadgi_ratio_is_reported_and_the_corrected_one_is_offered():
    """FADGI asks for standard deviation over mean, full stop. The chart's own
    spread inflates that, so the corrected figure is reported alongside -- but the
    headline value stays the one the standard defines."""
    references = [(96.0 + i * 0.3, -0.9, 3.2) for i in range(len(UNIFORMITY_IDS))]
    data = make_data([(pid, ref, ref) for pid, ref in zip(UNIFORMITY_IDS, references)])
    spec = MetricSpec(metric_id="lightness_uniformity", aggregation="max")
    result = lightness_uniformity(MetricContext(target=load_target(SG), data=data), spec)

    family = result.metadata["neutral_families"][0]
    # A perfect capture of a chart whose patches really do differ still shows spread.
    assert family["std_over_mean_pct"] > 0.5
    assert result.value == pytest.approx(family["std_over_mean_pct"])
    # ...but the camera contributed none of it.
    assert family["chart_corrected_std_over_mean_pct"] == pytest.approx(0.0, abs=1e-9)


def test_the_value_is_a_percentage_not_an_l_star_difference():
    """The units are 'standard deviation divided by mean', so a 1 L* spread at L*96
    is about 1%, not 1."""
    entries = [
        (pid, (96.0 + (1.0 if i % 2 else -1.0), -0.9, 3.2), (96.0, -0.9, 3.2))
        for i, pid in enumerate(UNIFORMITY_IDS)
    ]
    spec = MetricSpec(metric_id="lightness_uniformity", aggregation="max")
    result = lightness_uniformity(
        MetricContext(target=load_target(SG), data=make_data(entries)), spec
    )
    assert result.value == pytest.approx(100.0 * 1.0392 / 96.0, rel=0.02)
    assert "%" in result.units


def test_a_real_falloff_is_detected():
    """A lens vignette: corners darker than the center."""
    target = load_target(SG)
    grid = target.patch_grid
    entries = []
    for pid in UNIFORMITY_IDS:
        patch = target[pid]
        u, v = grid.center_relative(patch.row, patch.col)
        falloff = 6.0 * ((u - 0.5) ** 2 + (v - 0.5) ** 2)
        entries.append((pid, (96.0 - falloff, -0.9, 3.2), (96.0, -0.9, 3.2)))

    spec = MetricSpec(metric_id="lightness_uniformity", aggregation="max", pass_condition="value < 0.5")
    result = lightness_uniformity(MetricContext(target=target, data=make_data(entries)), spec)

    assert result.value > 0.5
    assert result.passed is False


def test_too_few_samples_is_incomplete_not_a_pass():
    data = make_data([(pid, (96.0, -0.9, 3.2), (96.0, -0.9, 3.2)) for pid in UNIFORMITY_IDS[:2]])
    spec = MetricSpec(metric_id="lightness_uniformity", aggregation="max", pass_condition="value < 2")
    result = lightness_uniformity(MetricContext(target=load_target(SG), data=data), spec)

    assert result.outcome == "incomplete"
    assert result.passed is None
    assert "at least" in result.reason


def test_sample_count_and_distribution_travel_with_the_result():
    """Confidence is reported as numbers rather than encoded in the metric's name,
    so a result resting on a few clustered points is visible as such."""
    data = make_data([(pid, (96.0, -0.9, 3.2), (96.0, -0.9, 3.2)) for pid in UNIFORMITY_IDS])
    spec = MetricSpec(metric_id="lightness_uniformity", aggregation="max")
    result = lightness_uniformity(MetricContext(target=load_target(SG), data=data), spec)

    assert result.metadata["sample_point_count"] == len(UNIFORMITY_IDS)
    distribution = result.metadata["sample_distribution"]
    assert distribution["on_perimeter"] == len(UNIFORMITY_IDS)
    assert distribution["covered_fraction_of_fiducial_quad"] > 0.7
    assert any("not the full camera frame" in n for n in result.notes)


# -- through the pipeline ---------------------------------------------------


def test_uniformity_runs_end_to_end(chart_files, uniformity_profile):
    _, reference, image, corners = chart_files
    result = analyze(
        image, target=SG, reference_path=reference, profile=uniformity_profile, corners=corners
    )
    metric = next(
        m for m in result.evaluation.results if m.metric_id == "lightness_uniformity"
    )
    assert metric.outcome == "evaluated"
    assert metric.value < 0.5
    assert metric.metadata["sample_point_count"] == len(UNIFORMITY_IDS)


# -- the report -------------------------------------------------------------


def test_report_states_the_verdict_and_the_provenance(chart_files):
    _, reference, image, corners = chart_files
    result = analyze(
        image, target=SG, reference_path=reference, profile=FADGI3, corners=corners
    )
    text = render(result)

    assert "RESULT: PASS" in text
    assert "ColorChecker Digital SG" in text
    assert "Synthetic Adobe RGB" in text
    assert "i1" not in text  # provenance comes from the file, not invented
    assert "M1" in text and "D50" in text
    assert "TONAL RAMP" in text


def test_report_shows_not_applicable_separately_from_failures(chart_files):
    _, reference, image, corners = chart_files
    result = analyze(
        image, target=SG, reference_path=reference, profile=FADGI3, corners=corners
    )
    text = render(result)

    assert "NOT EVALUATED" in text
    assert "sfr10_sampling_efficiency" in text
    assert "does not declare the capability" in text


def test_report_shows_the_color_management_path_it_used(chart_files):
    _, reference, image, corners = chart_files
    result = analyze(
        image, target=SG, reference_path=reference, profile=FADGI3, corners=corners
    )
    text = render(result)
    assert "Synthetic Adobe RGB" in text
    assert "Per-patch raw measurements are retained" in text


def test_report_names_the_failing_metric(chart_files, tmp_path):
    chart, reference, _, _ = chart_files
    image, corners = chart.render_capture(tmp_path / "warm.tif", gain=(1.12, 1.0, 0.88))
    result = analyze(
        image, target=SG, reference_path=reference, profile=FADGI3, corners=corners
    )
    text = render(result)

    assert "RESULT: FAIL" in text
    assert "white_balance" in text
    assert "FAIL" in text


def test_report_can_list_every_patch(chart_files):
    _, reference, image, corners = chart_files
    result = analyze(
        image, target=SG, reference_path=reference, profile=FADGI3, corners=corners
    )
    text = render(result, show_all_patches=True)
    assert "ALL PATCHES" in text
    for patch_id in ("A1", "N10", "K8"):
        assert patch_id in text


def test_report_shows_the_resolved_uniformity_set(chart_files, uniformity_profile):
    _, reference, image, corners = chart_files
    result = analyze(
        image, target=SG, reference_path=reference, profile=uniformity_profile, corners=corners
    )
    text = render(result)
    assert "LIGHTNESS UNIFORMITY SAMPLING" in text
    assert "uniformity set resolved from reference data" in text


# -- several neutral families ----------------------------------------------
#
# A real ColorChecker SG border is three interleaved neutral families -- near-white,
# mid-gray and near-black -- each repeated around the chart. Measured on a real unit:
# 14 at L*96, 16 at L*49, 14 at L*10. An earlier version kept only the largest group,
# which discarded 28 of 45 usable sample points and measured uniformity at one
# lightness instead of three.

FAMILY_LEVELS = [(96.0, -0.9, 3.2), (49.5, -0.4, 0.1), (10.2, -0.7, -2.0)]


def three_family_entries(falloff=None):
    """Uniformity patches at three lightness levels, spread around the chart."""
    target = load_target(SG)
    grid = target.patch_grid
    # The perimeter of a 14x10 grid is 2*14 + 2*10 - 4 = 44 patches.
    perimeter = [p for p in target.patches if grid.is_perimeter(p.row, p.col)]
    entries = []
    for index, patch in enumerate(perimeter):
        base = FAMILY_LEVELS[index % 3]
        u, v = grid.center_relative(patch.row, patch.col)
        shift = falloff(u, v) if falloff else 0.0
        entries.append(
            (patch.patch_id, (base[0] + shift, base[1], base[2]), base)
        )
    return entries


def make_multi(entries):
    target = load_target(SG)
    return [
        PatchDatum(
            patch=target[pid].with_role("achromatic_border"),
            measured_lab=np.array(measured, dtype=float),
            reference_lab=np.array(reference, dtype=float),
        )
        for pid, measured, reference in entries
    ]


def test_three_families_are_detected_and_kept_separate():
    data = make_multi(three_family_entries())
    spec = MetricSpec(metric_id="lightness_uniformity", aggregation="max")
    result = lightness_uniformity(MetricContext(target=load_target(SG), data=data), spec)

    families = result.metadata["neutral_families"]
    assert len(families) == 3
    assert sorted(f["reference_L_mean"] for f in families) == [10.2, 49.5, 96.0]
    assert sum(f["count"] for f in families) == 44


def test_families_are_not_pooled_into_one_mean():
    """Pooling families spanning L*10 to L*96 would report the chart's own design as
    camera non-uniformity -- a perfect capture would score ~36 L*."""
    data = make_multi(three_family_entries())
    spec = MetricSpec(metric_id="lightness_uniformity", aggregation="max")
    result = lightness_uniformity(MetricContext(target=load_target(SG), data=data), spec)

    assert result.value == pytest.approx(0.0, abs=1e-9)


def test_a_falloff_shows_up_in_every_family():
    data = make_multi(three_family_entries(
        falloff=lambda u, v: -6.0 * ((u - 0.5) ** 2 + (v - 0.5) ** 2)
    ))
    spec = MetricSpec(
        metric_id="lightness_uniformity", aggregation="max", pass_condition="value < 0.3"
    )
    result = lightness_uniformity(MetricContext(target=load_target(SG), data=data), spec)

    assert result.passed is False
    assert all(f["std_over_mean_pct"] > 0.3 for f in result.metadata["neutral_families"])


def test_the_dark_family_scores_far_worse_for_the_same_physical_error():
    """Standard deviation over mean is harsh on dark families because the
    denominator is small. The same 1 L* spread is ~1% at L*96 and ~10% at L*10 --
    which is exactly why the family choice has to be explicit and reported."""
    entries = []
    for i, (pid, measured, reference) in enumerate(three_family_entries()):
        shift = 1.0 if i % 2 else -1.0
        entries.append((pid, (measured[0] + shift, *measured[1:]), reference))
    spec = MetricSpec(metric_id="lightness_uniformity", aggregation="max")
    ctx = MetricContext(target=load_target(SG), data=make_multi(entries))

    families = {
        f["reference_L_mean"]: f["std_over_mean_pct"]
        for f in lightness_uniformity(ctx, spec).metadata["neutral_families"]
    }
    assert families[96.0] < 1.5
    assert families[10.2] > 8.0


def test_family_selection_is_explicit_and_defaults_to_the_lightest():
    entries = []
    for i, (pid, measured, reference) in enumerate(three_family_entries()):
        shift = 1.0 if i % 2 else -1.0
        entries.append((pid, (measured[0] + shift, *measured[1:]), reference))
    ctx = MetricContext(target=load_target(SG), data=make_multi(entries))

    default = lightness_uniformity(ctx, MetricSpec(metric_id="lightness_uniformity"))
    assert default.metadata["family_selection"] == "lightest"
    assert default.metadata["reported_family_reference_L"] == 96.0

    worst = lightness_uniformity(
        ctx,
        MetricSpec(metric_id="lightness_uniformity", algorithm={"family_selection": "worst"}),
    )
    assert worst.metadata["reported_family_reference_L"] == 10.2
    assert worst.value > default.value


def test_multi_family_coverage_is_recorded():
    """Every family stays in the record; which one was reported is not announced
    on every run, the chart repeating its neutrals by design."""
    data = make_multi(three_family_entries())
    spec = MetricSpec(metric_id="lightness_uniformity", aggregation="max")
    result = lightness_uniformity(MetricContext(target=load_target(SG), data=data), spec)
    assert len(result.metadata["neutral_families"]) == 3
    assert not any("families" in n for n in result.notes)


def test_the_worst_patch_table_names_the_metric_and_units_it_ranked(tmp_path):
    """It used to be headed "by dE2000" whatever it had actually sorted, which is
    wrong for every metric that is not a Delta E -- tone response is dL*2000."""
    from fixtures import SyntheticChart

    chart = SyntheticChart()
    image, corners = chart.render_capture(tmp_path / "c.tif", gain=(1.02, 1.0, 0.97))
    reference = chart.write_reference(tmp_path / "r.txt")
    text = render(analyze(
        image, target="colorchecker_sg_v1", reference_path=reference,
        profile="fadgi_3star_prints_photographs_2023", corners=corners,
        fiducial_extent_mm=(100.0, 70.0),
    ))
    assert "WORST PATCHES" in text
    assert "under average_color_accuracy" in text
