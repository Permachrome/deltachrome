"""Illumination and white balance measured across the whole image plane.

Metamorfoze measures these on a frame-filling white sheet rather than on a chart,
and the distinction is the reason this exists: a chart sits in the middle of the
frame, so a chart-based uniformity figure describes the middle of the frame. The
corners -- where fall-off actually is -- are outside it.

The tests inject a known left-to-right ramp, because then the answer is arithmetic:
the spread is the difference between the two ends, and the extremes are where the
ends are.
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.imaging.color import delta_e_2000_ab
from deltachrome.imaging.loader import load_image
from deltachrome.metrics.base import MetricContext
from deltachrome.pipeline import analyze
from deltachrome.profiles.evaluator import evaluate_profile
from deltachrome.profiles.schema import load_profile, profile_from_dict
from deltachrome.targets.registry import load_target
from fixtures import render_flat_field

#: The edge-spot shortfall, which is inherent and documented -- see the module
#: docstring in metrics/flatfield.py. Tight enough that a real error would show.
EDGE_BIAS = 0.08

PROFILE = profile_from_dict({
    "profile_id": "flat_field_test",
    "display_name": "Flat field",
    "metrics": [
        {"metric_id": "illumination_non_uniformity", "requires_capability": "flat_field",
         "pass_condition": "value < 3", "units": "dL*"},
        {"metric_id": "white_balance_non_uniformity", "requires_capability": "flat_field",
         "pass_condition": "value < 3", "units": "dE2000(a*b*)"},
    ],
})


def run(path):
    image = load_image(path)
    corners = [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)]
    result = analyze(path, target="flat_field_v1", profile=PROFILE, corners=corners)
    return {r.metric_id: r for r in result.evaluation.results}


# -- the measurement --------------------------------------------------------


def test_a_known_lightness_ramp_comes_back(tmp_path):
    path = render_flat_field(tmp_path / "f.tif", lightness=(95.0, 92.0))
    result = run(path)["illumination_non_uniformity"]

    assert result.outcome == "evaluated"
    assert result.value == pytest.approx(3.0, abs=EDGE_BIAS)
    assert result.value < 3.0  # under-reported by the edge spots, never over
    # Only the horizontal position is determinate: the ramp is horizontal, so
    # every row holds the same lightness and which one wins is a rounding tie.
    assert result.metadata["brightest"]["at"][0] == 0.0
    assert result.metadata["darkest"]["at"][0] == 1.0


def test_a_known_color_ramp_comes_back(tmp_path):
    path = render_flat_field(tmp_path / "f.tif", b_star=(-2.0, 2.0))
    result = run(path)["white_balance_non_uniformity"]

    expected = float(delta_e_2000_ab(np.array([95.0, 0.0, -2.0]), np.array([95.0, 0.0, 2.0])))
    assert result.value == pytest.approx(expected, abs=0.15)
    # The extremes are the two ends of the ramp.
    left, right = result.metadata["worst_pair"]
    assert {left["at"][0], right["at"][0]} == {0.0, 1.0}
    assert left["b"] == pytest.approx(-right["b"], abs=0.05)


def test_an_even_field_measures_as_even(tmp_path):
    path = render_flat_field(tmp_path / "f.tif")
    results = run(path)
    assert results["illumination_non_uniformity"].value < 0.05
    assert results["white_balance_non_uniformity"].value < 0.05
    assert all(r.passed for r in results.values())


def test_the_grid_is_reported_and_adjustable(tmp_path):
    """Two runs at different grids are different measurements, so the grid travels
    with the result."""
    path = render_flat_field(tmp_path / "f.tif", lightness=(95.0, 92.0))
    profile = profile_from_dict({
        "profile_id": "coarse", "display_name": "c",
        "metrics": [{
            "metric_id": "illumination_non_uniformity",
            "requires_capability": "flat_field", "pass_condition": "value < 3",
            "algorithm": {"grid": 3},
        }],
    })
    image = load_image(path)
    result = analyze(
        path, target="flat_field_v1", profile=profile,
        corners=[(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
    ).evaluation.results[0]
    assert result.metadata["grid"] == 3
    assert result.metadata["spots"] == 9


# -- the failure that would otherwise flatter a bad capture -----------------


def test_a_clipped_field_is_flagged_rather_than_scored_as_perfect(tmp_path):
    """The dangerous one. A blown white sheet is uniform because it is clipped, so
    the measurement reads perfect exactly when the capture is worst."""
    path = render_flat_field(tmp_path / "f.tif", level=1.0)
    result = run(path)["illumination_non_uniformity"]

    assert result.value == pytest.approx(0.0, abs=1e-9)
    assert any("uniformity may read better than it is" in n for n in result.notes)
    assert all("clipped" in patch.flags for patch in result.per_patch)


def test_a_field_far_below_L95_is_called_out(tmp_path):
    """Metamorfoze asks for a sheet at L* ~95. A darker one compresses the same
    fall-off into a smaller Delta L*, so it would pass for the wrong reason."""
    path = render_flat_field(tmp_path / "f.tif", lightness=(50.0, 48.0))
    result = run(path)["illumination_non_uniformity"]
    assert any(n.startswith("White sheet reads L* ") and n.endswith("expected L* 95.")
               for n in result.notes)


# -- how it sits beside the chart-based metrics ----------------------------


def test_a_chart_cannot_measure_this_and_is_not_penalised_for_it():
    target = load_target("colorchecker_sg_v1")
    evaluation = evaluate_profile(PROFILE, target, MetricContext(target=target, data=[]))
    assert {r.outcome for r in evaluation.results} == {"not_applicable"}
    assert not evaluation.failures


def test_the_flat_field_cannot_measure_what_a_chart_does():
    """It has no patches and no edges, and says so, so a FADGI profile run against
    it reports N/A throughout rather than failing."""
    target = load_target("flat_field_v1")
    profile = load_profile("fadgi_3star_prints_photographs_2023")
    evaluation = evaluate_profile(profile, target, MetricContext(target=target, data=[]))
    assert not evaluation.failures
    assert {r.metric_id for r in evaluation.not_applicable} >= {
        "average_color_accuracy", "white_balance", "sfr50", "lightness_uniformity",
    }


def test_it_is_not_the_same_measurement_as_chart_lightness_uniformity():
    """Different formula, different area, different units. Sharing the concept name
    across the two would let a Metamorfoze figure be quoted as a FADGI one."""
    from deltachrome.metrics import available_metrics

    assert {"illumination_non_uniformity", "lightness_uniformity"} <= set(available_metrics())
    target = load_target("flat_field_v1")
    assert "lightness_uniformity" in target.not_capable_of
    assert "flat_field" in target.capabilities


def test_the_worst_patch_table_is_not_shown_for_absolute_readings(tmp_path):
    """It ranks deviations against a reference. A flat field's spots are absolute
    L* values, so ranking them put the brightest at the top under a heading calling
    them the worst -- and named a reference column that was empty."""
    from deltachrome.reporting.console import render

    path = render_flat_field(tmp_path / "f.tif", lightness=(95.0, 90.5))
    image = load_image(path)
    result = analyze(
        path, target="flat_field_v1", profile=PROFILE,
        corners=[(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
    )
    assert "WORST PATCHES" not in render(result)
