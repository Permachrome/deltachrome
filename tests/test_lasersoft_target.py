"""The LaserSoft ISO 12641-2 target, and profiles that judge nothing.

Second target definition, and the one that proves the engine generalises: 864
patches instead of 140, the opposite patch-id convention, a 96-step neutral scale
instead of 12, transmissive instead of reflective, and a CxF reference instead of
CGATS. None of that needed a metric plugin change.
"""

from __future__ import annotations

import json

import pytest

from deltachrome.metrics.base import MetricContext
from deltachrome.profiles.evaluator import evaluate_profile
from deltachrome.profiles.schema import load_profile
from deltachrome.targets.registry import load_target

LS = "lasersoft_iso12641_2_advanced_v1"
RAMP_COLUMNS = {31, 32, 33, 34}


def raw() -> dict:
    return json.loads(load_target(LS).source_path.read_text(encoding="utf-8"))


def test_the_grid_is_864_patches_addressed_row_letter_then_column_number():
    """The opposite convention to the ColorChecker SG, which is exactly why the id
    format is a property of the target rather than baked into the engine."""
    target = load_target(LS)
    grid = target.patch_grid
    assert (grid.rows, grid.columns) == (24, 36)
    assert len(target.patches) == 864
    assert grid.patch_id_format == "{row_label}{col_label}"
    assert grid.patch_id(0, 0) == "A1"
    assert grid.patch_id(23, 35) == "X36"
    assert target["X36"].row == 23 and target["X36"].col == 35


def test_ids_match_the_units_own_cxf_convention():
    ids = set(load_target(LS).patch_ids)
    assert {"A1", "A36", "X1", "X36", "Q9"} <= ids
    assert "1A" not in ids


def test_the_neutral_scale_is_96_steps_across_four_columns():
    """Derived from the measured reference, not from reading the target: columns
    31-34 interlock into one continuous serpentine ramp."""
    ramp = load_target(LS).ramp()
    assert len(ramp) == 96
    assert {int(p.patch_id[1:]) for p in ramp} == RAMP_COLUMNS
    assert [p.ramp_position for p in ramp] == list(range(1, 97))


def test_the_ramp_runs_light_to_dark_and_alternates_direction_by_column():
    """Column 34 ascends A->X while 31 descends, so the lightest and darkest steps
    sit in different columns. Assuming reading order would mis-plot the OECF."""
    ramp = load_target(LS).ramp()
    assert ramp[0].patch_id.startswith("X") and ramp[0].patch_id.endswith("34")
    assert ramp[-1].patch_id.endswith("31")


def test_geometry_is_measured_against_the_frame_corners():
    grid = load_target(LS).patch_grid
    assert grid.provisional is False
    measured = raw()["patch_grid"]["measured_from"]
    assert measured["mean_delta_e_2000_at_fit"] < 1.0
    assert "frame corners" in raw()["patch_grid"]["coordinate_frame"]


def test_the_grid_fits_inside_the_frame_with_margin_for_the_printed_labels():
    """The frame encloses the row and column labels as well as the patches, so the
    array has to be inset from it on every side."""
    grid = load_target(LS).patch_grid
    ou, ov = grid.grid_origin_relative
    eu, ev = grid.extent_relative
    assert ou > 0.01 and ov > 0.01
    assert ou + eu < 0.99 and ov + ev < 0.99


def test_patches_butt_together_so_pitch_equals_patch_size():
    """Unlike the SG, this target has no gaps between patches. The ROI inset is the
    only thing keeping a sample off its neighbor."""
    grid = load_target(LS).patch_grid
    assert grid.patch_size_relative == pytest.approx(grid.patch_spacing_relative)


def test_uniformity_is_not_claimed_and_the_reason_is_recorded():
    """The only repeated near-whites are the highlight ends of six adjacent color
    ramps, clustered in one corner -- not an areal-coverage uniformity sample."""
    target = load_target(LS)
    assert not target.supports("lightness_uniformity")
    assert "lightness_uniformity" in target.not_capable_of
    assert "clustered in one corner" in target.notes


def test_the_definition_records_that_the_target_is_transmissive():
    target = load_target(LS)
    assert target.target_type == "transmissive_color_chart"
    assert "Transparencies" in target.notes
    assert "not a FADGI conformance criterion" in target.notes


def test_the_transparency_profile_skips_what_this_target_cannot_do():
    target = load_target(LS)
    profile = load_profile("fadgi_4star_photographic_transparencies_4x5_and_larger_2023")
    evaluation = evaluate_profile(profile, target, MetricContext(target=target, data=[]))
    by_id = {r.metric_id: r for r in evaluation.results}

    assert by_id["lightness_uniformity"].outcome == "not_applicable"
    assert by_id["sfr10_sampling_efficiency"].outcome == "not_applicable"
    assert by_id["reproduction_scale_accuracy"].outcome == "not_applicable"
    # And the category specifies no color accuracy at all.
    assert "average_color_accuracy" not in by_id


# -- profiles that judge nothing -------------------------------------------


def test_a_profile_with_no_thresholds_reports_measured_not_pass():
    """Calling an unjudged run a PASS would imply something was tested and met."""
    profile = load_profile("measurement_only_v1")
    assert all(spec.pass_condition is None for spec in profile.metrics)
    assert profile.standard_reference["body"] == "none"
    assert "must never be described as conformance" in profile.standard_reference["notice"]


def test_measurement_only_metrics_render_as_measured_not_failed():
    from deltachrome.metrics.base import MetricResult

    result = MetricResult(metric_id="x", outcome="evaluated", value=1.0, passed=None)
    assert result.status == "MEASURED"
    assert MetricResult(metric_id="x", outcome="evaluated", value=1.0, passed=True).status == "PASS"
    assert MetricResult(metric_id="x", outcome="evaluated", value=1.0, passed=False).status == "FAIL"


def test_pass_still_requires_something_to_have_been_judged():
    from deltachrome.metrics.base import MetricResult
    from deltachrome.profiles.evaluator import ProfileEvaluation

    profile = load_profile("measurement_only_v1")
    unjudged = [MetricResult(metric_id="a", outcome="evaluated", value=1.0, passed=None)]
    assert ProfileEvaluation(profile=profile, results=unjudged).overall == "MEASURED"

    judged = unjudged + [
        MetricResult(metric_id="b", outcome="evaluated", value=1.0, passed=True)
    ]
    assert ProfileEvaluation(profile=profile, results=judged).overall == "PASS"

    failed = judged + [
        MetricResult(metric_id="c", outcome="evaluated", value=9.0, passed=False)
    ]
    assert ProfileEvaluation(profile=profile, results=failed).overall == "FAIL"


def test_the_measured_explanation_disclaims_conformance():
    from deltachrome.metrics.base import MetricResult
    from deltachrome.profiles.evaluator import ProfileEvaluation

    evaluation = ProfileEvaluation(
        profile=load_profile("measurement_only_v1"),
        results=[MetricResult(metric_id="a", outcome="evaluated", value=1.0, passed=None)],
    )
    assert "not a conformance claim" in evaluation.overall_explanation
