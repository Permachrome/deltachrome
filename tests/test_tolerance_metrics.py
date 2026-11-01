"""Highlight/Shadow Tolerance and Dynamic Range -- FADGI 2.4.15 and 2.4.16.

Both were absent from every profile that carries them, not merely unimplemented:
the profile JSON never declared the metric ids at all, so a row FADGI's own
table prints was silently missing from the report rather than showing
NOT IMPLEMENTED. These pin the two things that make the new readings honest --
that they are informative (``MEASURED``, never a verdict), and that Dynamic
Range reports a density span traceable to real per-patch numbers rather than an
invented threshold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deltachrome.metrics.base import MetricContext, MetricSpec, PatchDatum
from deltachrome.metrics.tolerance import (
    HIGHLIGHT_AIM_L,
    HIGHLIGHT_TOLERANCE_L,
    SHADOW_AIM_L,
    SHADOW_TOLERANCE_L,
    dynamic_range,
    highlight_tolerance,
    shadow_tolerance,
)
from deltachrome.targets.registry import load_target
from deltachrome.targets.schema import Patch


#: The local reference measurements this test reads. data/ is never committed,
#: so on a machine without it -- a build server -- the test skips rather than
#: fails, as every test needing local captures already does.
needs_local_references = pytest.mark.skipif(
    not Path("data/reference/ColorChecker_SG_M1.txt").is_file()
    or not Path("data/reference/T1910021.cxf").is_file(),
    reason="needs the local reference files in data/reference, which are not committed",
)

TARGET = load_target("isa_transmissive_4x5_v1")

RAMP_SELECTION = "role == 'achromatic_ramp'"


def ramp_data(steps):
    """``steps`` is (ramp_position, L*, density, flags) tuples, lightest first."""
    return [
        PatchDatum(
            patch=Patch(
                patch_id=f"L{position}", row=0, col=position,
                role="achromatic_ramp", ramp_position=position,
            ),
            measured_lab=np.array([L, 0.0, 0.0], dtype=float),
            reference_lab=np.array([L, 0.0, 0.0], dtype=float),
            reference_density=density,
            flags=list(flags),
        )
        for position, L, density, flags in steps
    ]


def spec(metric_id: str) -> MetricSpec:
    return MetricSpec(
        metric_id=metric_id,
        requires_capability="tonal_response",
        algorithm={"patch_selection": RAMP_SELECTION},
    )


# -- highlight / shadow tolerance ---------------------------------------------


def test_the_lightest_ramp_step_is_the_highlight():
    data = ramp_data([
        (1, 96.2, 0.05, ()), (2, 60.0, 0.9, ()), (3, 5.0, 2.2, ()),
    ])
    ctx = MetricContext(target=TARGET, data=data)
    result = highlight_tolerance(ctx, spec("highlight_tolerance"))
    assert result.outcome == "evaluated"
    assert result.value == pytest.approx(96.2)
    assert result.per_patch[0].patch_id == "L1"


def test_the_darkest_ramp_step_is_the_shadow():
    data = ramp_data([
        (1, 96.2, 0.05, ()), (2, 60.0, 0.9, ()), (3, 5.0, 2.2, ()),
    ])
    ctx = MetricContext(target=TARGET, data=data)
    result = shadow_tolerance(ctx, spec("shadow_tolerance"))
    assert result.value == pytest.approx(5.0)
    assert result.per_patch[0].patch_id == "L3"


def test_selection_is_by_declared_ramp_position_not_by_measured_lightness():
    """A patch's identity as the highlight or shadow end is what the target
    declares, not whichever happens to read lightest -- picking by measured
    value would beg the question this metric exists to check."""
    data = ramp_data([
        (1, 40.0, 1.5, ()),   # declared highlight, but reads dark
        (2, 90.0, 0.1, ()),   # declared shadow, but reads light
    ])
    ctx = MetricContext(target=TARGET, data=data)
    assert highlight_tolerance(ctx, spec("highlight_tolerance")).value == pytest.approx(40.0)
    assert shadow_tolerance(ctx, spec("shadow_tolerance")).value == pytest.approx(90.0)


def test_it_is_never_a_pass_or_a_fail():
    """FADGI's own words mark this row provisional/informational. A verdict
    here would be a claim the standard does not make."""
    data = ramp_data([(1, 96.0, 0.05, ())])
    ctx = MetricContext(target=TARGET, data=data)
    result = highlight_tolerance(ctx, spec("highlight_tolerance"))
    assert result.passed is None
    assert result.status == "MEASURED"


def test_the_aim_is_recorded_for_display_without_gating_anything():
    data = ramp_data([(1, HIGHLIGHT_AIM_L + HIGHLIGHT_TOLERANCE_L + 1, 0.05, ())])
    ctx = MetricContext(target=TARGET, data=data)
    result = highlight_tolerance(ctx, spec("highlight_tolerance"))
    assert result.metadata["fadgi_aim_L"] == HIGHLIGHT_AIM_L
    assert result.metadata["fadgi_tolerance_L"] == HIGHLIGHT_TOLERANCE_L
    assert result.metadata["within_fadgi_aim"] is False
    assert result.passed is None, "outside the aim is still not a fail -- informative only"


def test_shadow_aim_matches_fadgis_own_table():
    """96 +/- 2 and 2.7 +/- 1.5 are identical across every FADGI star level and
    every category that carries the row -- transcribed from the standard's own
    PDF pages, not per-profile."""
    assert HIGHLIGHT_AIM_L == 96.0 and HIGHLIGHT_TOLERANCE_L == 2.0
    assert SHADOW_AIM_L == 2.7 and SHADOW_TOLERANCE_L == 1.5


def test_no_declared_ramp_position_is_incomplete_not_a_guess():
    data = [
        PatchDatum(
            patch=Patch(patch_id="A1", row=0, col=0, role="achromatic_ramp"),
            measured_lab=np.array([50.0, 0.0, 0.0]),
            reference_lab=np.array([50.0, 0.0, 0.0]),
        )
    ]
    ctx = MetricContext(target=TARGET, data=data)
    result = highlight_tolerance(ctx, spec("highlight_tolerance"))
    assert result.outcome == "incomplete"


# -- dynamic range --------------------------------------------------------------


def test_dynamic_range_is_the_declared_density_span():
    data = ramp_data([
        (1, 96.0, 0.05, ()), (2, 60.0, 0.9, ()), (3, 5.0, 2.25, ()),
    ])
    ctx = MetricContext(target=TARGET, data=data)
    result = dynamic_range(ctx, spec("dynamic_range"))
    assert result.outcome == "evaluated"
    assert result.value == pytest.approx(2.20)
    assert result.metadata["d_max"] == pytest.approx(2.25)
    assert result.metadata["d_min"] == pytest.approx(0.05)
    assert result.passed is None
    assert result.status == "MEASURED"


def test_a_clipped_extreme_is_excluded_from_the_span():
    """FADGI's own words are 'the difference of the min/max density values an
    imaging system CAN capture' -- a patch the sampler flagged clipped is
    exactly one the system could not, so it must not set the span."""
    data = ramp_data([
        (1, 96.0, 0.05, ()),
        (2, 60.0, 0.9, ()),
        (3, 100.0, 2.25, ("clipped_high",)),  # the true shadow, but clipped
    ])
    ctx = MetricContext(target=TARGET, data=data)
    result = dynamic_range(ctx, spec("dynamic_range"))
    assert result.value == pytest.approx(0.85)  # 0.05 to 0.9, not to 2.25


def test_a_non_clipping_flag_does_not_exclude_a_patch():
    """A gamut or divergence flag is about colour fidelity, not about whether
    the tone itself was recorded -- only the two clipping flags exclude."""
    data = ramp_data([
        (1, 96.0, 0.05, ("mean_median_divergence",)),
        (2, 5.0, 2.2, ()),
    ])
    ctx = MetricContext(target=TARGET, data=data)
    result = dynamic_range(ctx, spec("dynamic_range"))
    assert result.value == pytest.approx(2.15)


def test_a_target_with_no_declared_density_reports_incomplete():
    """A reflective target's ramp carries no density column at all. Gated by
    ``requires_capability: tonal_response`` -- the same capability every ramp
    reading uses -- so a chart with no ramp at all reports N/A upstream in the
    evaluator; this is the case where the ramp exists but nothing in its
    reference states a density."""
    data = [
        PatchDatum(
            patch=Patch(patch_id="E5", row=0, col=0, role="achromatic_ramp",
                        ramp_position=1),
            measured_lab=np.array([90.0, 0.0, 0.0]),
            reference_lab=np.array([90.0, 0.0, 0.0]),
        )
    ]
    ctx = MetricContext(target=load_target("colorchecker_sg_v1"), data=data)
    result = dynamic_range(ctx, spec("dynamic_range"))
    assert result.outcome == "incomplete"
    assert "reflective" in result.reason.lower() or "no patch" in result.reason.lower()


def test_every_candidate_clipped_is_incomplete_with_its_own_reason():
    data = ramp_data([
        (1, 96.0, 0.05, ("clipped_high",)),
        (2, 5.0, 2.2, ("clipped_low",)),
    ])
    ctx = MetricContext(target=TARGET, data=data)
    result = dynamic_range(ctx, spec("dynamic_range"))
    assert result.outcome == "incomplete"
    assert "clipped" in result.reason.lower()


def test_reference_density_survives_the_pipelines_reference_reader():
    """The `D` column is read generically off ``PatchMeasurement.raw`` at the
    point where reference Lab is also read, for the real nominal file this
    target ships -- not a hand-built fixture standing in for the parser."""
    from deltachrome.pipeline import _reference_density
    from deltachrome.reference import read_reference

    reference = read_reference(TARGET.nominal_reference_path)
    densities = _reference_density(reference)
    assert densities["L1"] == pytest.approx(0.05)
    assert densities["L12"] == pytest.approx(2.25)
    assert len(densities) == 24  # every patch in the file states D


@needs_local_references
def test_a_reflective_reference_with_no_density_column_yields_nothing():
    from pathlib import Path

    from deltachrome.pipeline import _reference_density
    from deltachrome.reference import read_reference

    reference = read_reference(Path("data/reference/ColorChecker_SG_M1.txt"))
    assert _reference_density(reference) == {}


# -- the six FADGI categories that carry these rows ---------------------------

#: Transcribed from the FADGI 3rd Edition PDF's own tables, star levels 1-4 in
#: order -- the same figures the profile JSON files' own ``notes`` now quote.
#: microfilm and xray_film are the two categories whose numbers are NOT the
#: 3.5/3.8/3.9/4.0 every other category shares.
DYNAMIC_RANGE_BY_CATEGORY = {
    "microfilm": [3.5, 3.8, 3.9, 4.2],
    "photographic_negatives_35mm_to_4x5": [3.5, 3.8, 3.9, 4.0],
    "photographic_negatives_4x5_and_larger": [3.5, 3.8, 3.9, 4.0],
    "photographic_transparencies_35mm_to_4x5": [3.5, 3.8, 3.9, 4.0],
    "photographic_transparencies_4x5_and_larger": [3.5, 3.8, 3.9, 4.0],
    "xray_film": [3.5, 3.6, 3.9, 4.2],
}


@pytest.mark.parametrize("category", sorted(DYNAMIC_RANGE_BY_CATEGORY))
def test_every_fadgi_film_category_declares_the_three_new_metrics(category):
    """Before this, these two rows were not declared unimplemented -- they were
    not in the profile's ``metrics`` list at all, so the report showed nothing
    for a row FADGI's own table prints."""
    from deltachrome.profiles.schema import load_profile

    for star in (1, 2, 3, 4):
        profile = load_profile(f"fadgi_{star}star_{category}_2023")
        ids = {m.metric_id for m in profile.metrics}
        assert {"highlight_tolerance", "shadow_tolerance", "dynamic_range"} <= ids


@pytest.mark.parametrize("category", sorted(DYNAMIC_RANGE_BY_CATEGORY))
def test_the_dynamic_range_figure_matches_the_standards_own_table(category):
    from deltachrome.profiles.schema import load_profile

    for star, value in zip((1, 2, 3, 4), DYNAMIC_RANGE_BY_CATEGORY[category]):
        profile = load_profile(f"fadgi_{star}star_{category}_2023")
        spec = next(m for m in profile.metrics if m.metric_id == "dynamic_range")
        assert str(value) in spec.notes, (
            f"{category} {star}-star: expected {value} quoted in dynamic_range's "
            f"notes, got: {spec.notes}"
        )


@pytest.mark.parametrize("category", sorted(DYNAMIC_RANGE_BY_CATEGORY))
def test_none_of_the_three_carries_a_pass_condition(category):
    """FADGI marks all three informative-only. A ``pass_condition`` here would
    fail or pass a run on a criterion the standard does not impose."""
    from deltachrome.profiles.schema import load_profile

    profile = load_profile(f"fadgi_1star_{category}_2023")
    for metric_id in ("highlight_tolerance", "shadow_tolerance", "dynamic_range"):
        spec = next(m for m in profile.metrics if m.metric_id == metric_id)
        assert spec.pass_condition is None


def test_every_category_agrees_on_the_highlight_shadow_aim():
    """96 +/- 2 and 2.7 +/- 1.5 read identically in every one of the six
    categories' own tables -- checked here rather than assumed, since a
    profile-specific aim would need its own metric parameter and does not
    exist as things stand."""
    from deltachrome.profiles.schema import load_profile

    for category in DYNAMIC_RANGE_BY_CATEGORY:
        profile = load_profile(f"fadgi_1star_{category}_2023")
        highlight = next(m for m in profile.metrics if m.metric_id == "highlight_tolerance")
        shadow = next(m for m in profile.metrics if m.metric_id == "shadow_tolerance")
        assert "96" in highlight.notes and "2" in highlight.notes
        assert "2.7" in shadow.notes and "1.5" in shadow.notes
