"""Metric infrastructure: selection, aggregation, thresholds, and outcome states."""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.errors import ProfileError
from deltachrome.metrics.base import (
    MetricContext,
    MetricSpec,
    PatchDatum,
    aggregate,
    evaluate_condition,
    select_patches,
)
from deltachrome.metrics.difference import patch_difference_metric
from deltachrome.targets.registry import load_target
from deltachrome.targets.schema import Patch


def make_data(specs):
    return [
        PatchDatum(
            patch=Patch(patch_id=pid, row=0, col=i, role=role),
            measured_lab=np.array(measured, dtype=float),
            reference_lab=np.array(reference, dtype=float),
        )
        for i, (pid, role, measured, reference) in enumerate(specs)
    ]


# -- aggregation ------------------------------------------------------------


def test_ninetieth_percentile_is_not_the_maximum():
    """FADGI's 90th percentile criterion is routinely misimplemented as a max,
    which is far stricter than the standard and fails compliant systems."""
    values = list(range(1, 101))
    p90, _ = aggregate(values, "percentile_90")
    assert p90 == pytest.approx(90.1)
    assert p90 < max(values)


def test_percentile_method_is_explicit_and_reported():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    linear, meta_linear = aggregate(values, "percentile_90", percentile_method="linear")
    nearest, meta_nearest = aggregate(values, "percentile_90", percentile_method="nearest_rank")

    assert linear == pytest.approx(9.1)
    assert nearest == 10.0  # an actually-measured value
    assert meta_linear["percentile_method"] == "linear"
    assert meta_nearest["percentile_method"] == "nearest_rank"


def test_max_per_patch_and_worst_edge_are_the_maximum():
    for name in ("max", "max_per_patch", "worst_edge"):
        assert aggregate([1.0, 9.0, 3.0], name)[0] == 9.0


def test_empty_aggregation_is_an_error():
    with pytest.raises(ProfileError, match="empty"):
        aggregate([], "mean")


def test_unknown_aggregation_is_an_error():
    with pytest.raises(ProfileError, match="Unknown aggregation"):
        aggregate([1.0], "average-ish")


# -- pass conditions --------------------------------------------------------


@pytest.mark.parametrize(
    "value,condition,expected",
    [
        (3.4, "value < 3.5", True),
        (3.5, "value < 3.5", False),
        (0.81, "value > 0.80", True),
        (0.80, "value > 0.80", False),
        (4.0, "value <= 4", True),
    ],
)
def test_pass_conditions(value, condition, expected):
    passed, _ = evaluate_condition(value, condition)
    assert passed is expected


def test_absent_condition_yields_none_not_a_pass():
    """An informational metric must not be silently counted as passing."""
    passed, threshold = evaluate_condition(99.0, None)
    assert passed is None and threshold is None


def test_malformed_condition_is_an_error():
    with pytest.raises(ProfileError, match="Unsupported pass_condition"):
        evaluate_condition(1.0, "value is fine")


def test_condition_is_not_executed_as_code():
    """Profiles are user-editable data files and must never be executable."""
    with pytest.raises(ProfileError):
        evaluate_condition(1.0, "__import__('os').system('true')")


# -- patch selection --------------------------------------------------------


def test_selection_forms():
    data = make_data(
        [
            ("A1", "achromatic_border", (96, 0, 0), (96, 0, 0)),
            ("E5", "achromatic_ramp", (90, 0, 0), (90, 0, 0)),
            ("B2", "chromatic", (50, 20, 20), (50, 20, 20)),
        ]
    )
    assert len(select_patches(data, "all patches")) == 3
    assert [d.patch_id for d in select_patches(data, "role == 'achromatic_ramp'")] == ["E5"]
    assert len(select_patches(data, "role in ['achromatic_ramp','chromatic']")) == 2
    assert [d.patch_id for d in select_patches(data, {"roles": ["achromatic_border"]})] == ["A1"]
    assert [d.patch_id for d in select_patches(data, {"patch_ids": ["B2"]})] == ["B2"]
    assert [d.patch_id for d in select_patches(data, ["A1", "B2"])] == ["A1", "B2"]


def test_unknown_selection_is_an_error():
    with pytest.raises(ProfileError, match="Unsupported patch_selection"):
        select_patches([], "role starts with x")


# -- the difference metric --------------------------------------------------


def test_white_balance_ignores_a_pure_tone_error():
    """A gray patch reproduced too dark is a tone problem, not a white balance
    one. Folding L* into the white balance number would hide which is which."""
    data = make_data(
        [("E5", "achromatic_ramp", (80.0, 0.5, -0.5), (95.0, 0.5, -0.5))]
    )
    ctx = MetricContext(target=load_target("colorchecker_sg_v1"), data=data)
    spec = MetricSpec(
        metric_id="white_balance",
        algorithm={"delta_e_variant": "delta_e_ab", "patch_selection": "all"},
        aggregation="max_per_patch",
        pass_condition="value < 4",
    )
    result = patch_difference_metric(ctx, spec)
    assert result.outcome == "evaluated"
    assert result.value == pytest.approx(0.0)
    assert result.passed is True


def test_worst_patch_is_recorded():
    data = make_data(
        [
            ("A1", "chromatic", (50, 0, 0), (50, 0, 0)),
            ("B1", "chromatic", (50, 0, 0), (60, 10, 10)),
        ]
    )
    ctx = MetricContext(target=load_target("colorchecker_sg_v1"), data=data)
    spec = MetricSpec(
        metric_id="average_color_accuracy",
        algorithm={"delta_e_variant": "delta_e_2000", "patch_selection": "all"},
        aggregation="mean",
        pass_condition="value < 3.5",
    )
    result = patch_difference_metric(ctx, spec)
    assert result.metadata["worst_patch"] == "B1"
    assert len(result.per_patch) == 2


def test_no_matching_patches_is_incomplete_not_a_pass():
    """A metric with nothing to measure must never report success."""
    ctx = MetricContext(target=load_target("colorchecker_sg_v1"), data=make_data([]))
    spec = MetricSpec(
        metric_id="white_balance",
        algorithm={"delta_e_variant": "delta_e_ab", "patch_selection": "all"},
        aggregation="max_per_patch",
        pass_condition="value < 4",
    )
    result = patch_difference_metric(ctx, spec)
    assert result.outcome == "incomplete"
    assert result.passed is None


def test_metric_without_a_named_formula_is_an_error():
    ctx = MetricContext(target=load_target("colorchecker_sg_v1"), data=make_data([]))
    spec = MetricSpec(metric_id="white_balance", algorithm={"patch_selection": "all"})
    with pytest.raises(ProfileError, match="does not name a difference formula"):
        patch_difference_metric(ctx, spec)
