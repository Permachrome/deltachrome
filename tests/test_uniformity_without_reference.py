"""Uniformity patches that carry no reference value.

Almost every metric here is a comparison against a measurement of the chart, so a
patch with no reference cannot take part. Lightness uniformity is the exception: it
is the spread of a set of nominally identical neutrals *about their own mean*, so
the set is its own reference.

That is not a theoretical exception. The ISA Golden Thread prints twenty
registration squares -- four at the corners of each of its five modules, reaching
every corner of the chart and its center -- and prints no values for any of them.
Requiring a reference would have left the best areal-coverage sample in the project
unusable.

The exception is deliberately narrow, because carrying reference-less patches into
every metric would also disarm a real safety check: a reference file for the wrong
model of chart currently leaves the patch data empty and the run refuses, and it can
only do that because a patch without a reference is normally dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deltachrome.errors import MissingReferenceError
from deltachrome.metrics.base import (
    ROLES_WITHOUT_REFERENCE,
    MetricContext,
    MetricSpec,
    PatchDatum,
    get_metric,
)
from deltachrome.targets.registry import load_target
from deltachrome.targets.schema import Patch
from fixtures import SyntheticChart

DEFINITIONS = Path(__file__).resolve().parents[1] / "src/deltachrome/targets/definitions"
PROFILE = "fadgi_3star_prints_photographs_2023"


def datum(patch_id, L, *, reference=None, role="achromatic_border",
          region="uniformity_center"):
    return PatchDatum(
        patch=Patch(patch_id=patch_id, row=0, col=0, role=role, region=region),
        measured_lab=np.array([L, 0.2, -0.3]),
        reference_lab=None if reference is None else np.array([reference, 0.0, 0.0]),
    )


def uniformity(data, **algorithm):
    target = load_target("isa_golden_thread_dlt_v1")
    spec = MetricSpec(
        metric_id="lightness_uniformity", requires_capability="lightness_uniformity",
        pass_condition="value < 3", units="%", algorithm=algorithm,
    )
    return get_metric("lightness_uniformity")(
        MetricContext(target=target, data=data), spec
    )


# -- the metric works without references ------------------------------------


def test_the_raw_fadgi_ratio_needs_no_reference_values():
    """Standard deviation of measured L* over its mean. Nothing else is required,
    which is the whole reason the exception exists."""
    values = [64.68, 66.01, 65.61, 65.86, 66.69, 66.41]
    result = uniformity([datum(f"U{i}", L) for i, L in enumerate(values)])

    array = np.array(values)
    assert result.outcome == "evaluated"
    assert result.value == pytest.approx(100.0 * array.std(ddof=1) / array.mean(), rel=1e-6)
    assert result.metadata["reference_values_available"] is False


def test_what_is_lost_without_references_is_named_rather_than_omitted():
    result = uniformity([datum(f"U{i}", 66.0 + 0.1 * i) for i in range(6)])
    family = result.metadata["neutral_families"][0]

    # No nominal lightness to split families on, and no way to separate the chart's
    # own spread from the capture's.
    assert family["chart_corrected_std_over_mean_pct"] is None
    assert family["reference_L_mean"] is None
    assert "no reference values" in result.metadata["family_selection"]


def test_a_referenced_set_still_gets_the_corrected_figure():
    """The SG's uniformity patches do have references, and nothing about that path
    changes."""
    data = [datum(f"U{i}", 66.0 + 0.1 * i, reference=66.0) for i in range(6)]
    result = uniformity(data)
    family = result.metadata["neutral_families"][0]
    assert result.metadata["reference_values_available"] is True
    assert family["chart_corrected_std_over_mean_pct"] is not None
    assert family["reference_L_mean"] == pytest.approx(66.0)


def test_the_perimeter_split_is_withheld_when_it_cannot_mean_anything():
    """Four squares at a module's corners are a 2x2, and every cell of a 2x2 is on
    its own edge. Reporting '20 on the perimeter, 0 interior' would read as a
    finding about the chart rather than as arithmetic."""
    target = load_target("isa_golden_thread_dlt_v1")
    squares = [p for p in target.patches if p.role == "achromatic_border"]
    data = [
        PatchDatum(patch=p, measured_lab=np.array([66.0 + 0.01 * i, 0.0, 0.0]),
                   reference_lab=None)
        for i, p in enumerate(squares)
    ]
    distribution = uniformity(data).metadata["sample_distribution"]
    assert "on_perimeter" not in distribution
    assert "no lattice" in distribution["perimeter_split"]
    # The samples reach beyond the fiducial quad, which is a property worth stating
    # rather than a fraction to be clamped.
    assert distribution["covered_fraction_of_fiducial_quad"] > 1.0
    assert distribution["centroid"] == [0.5, 0.5]


# -- and the exception stays narrow -----------------------------------------


def test_only_roles_that_do_not_need_a_reference_are_carried():
    assert ROLES_WITHOUT_REFERENCE == frozenset({"achromatic_border"})


@pytest.fixture
def chart_with_unreferenced_border(tmp_path):
    """An SG whose four corner patches are declared uniformity samples, with those
    four removed from the reference file."""
    chart = SyntheticChart()
    image, corners = chart.render_capture(tmp_path / "capture.tif")
    reference = chart.write_reference(tmp_path / "ref.txt")

    borders = ["A1", "A10", "N1", "N10"]
    kept = [
        line for line in reference.read_text(encoding="utf-8").splitlines()
        if line.split("\t")[0].split(" ")[0] not in borders
    ]
    trimmed = tmp_path / "trimmed.txt"
    trimmed.write_text("\n".join(kept) + "\n", encoding="utf-8")

    definition = json.loads((DEFINITIONS / "colorchecker_sg_v1.json").read_text("utf-8"))
    for patch in definition["patches"]:
        if patch["patch_id"] in borders:
            patch["role"] = "achromatic_border"
    path = tmp_path / "sg.json"
    path.write_text(json.dumps(definition), encoding="utf-8")
    return path, image, corners, trimmed, borders


def test_an_unreferenced_uniformity_patch_is_carried_and_said_to_be(
    chart_with_unreferenced_border,
):
    from deltachrome.pipeline import analyze

    target, image, corners, reference, borders = chart_with_unreferenced_border
    result = analyze(
        image, target=target, reference_path=reference, profile=PROFILE, corners=corners
    )
    carried = {r.metric_id: r for r in result.evaluation.results}
    assert carried["lightness_uniformity"].outcome == "evaluated"
    # Carried by design, and no longer announced on every run: a chart printing
    # no values for its uniformity squares is not something to warn about.
    assert not any("reference value" in w for w in result.warnings)


def test_reference_based_metrics_exclude_them_and_record_which(
    chart_with_unreferenced_border,
):
    """They are legitimate uniformity samples and impossible Delta E samples, and the
    report has to be able to say which patches were left out of which."""
    from deltachrome.pipeline import analyze

    target, image, corners, reference, borders = chart_with_unreferenced_border
    result = analyze(
        image, target=target, reference_path=reference, profile=PROFILE, corners=corners
    )
    delta_e = next(
        r for r in result.evaluation.results if r.metric_id == "average_color_accuracy"
    )
    assert set(delta_e.metadata["excluded_without_reference"]) == set(borders)
    assert all(pv.patch_id not in borders for pv in delta_e.per_patch)


def test_a_reference_file_for_the_wrong_chart_still_refuses(tmp_path):
    """The check the narrow exception protects. If every patch were carried without
    a reference, a wholly mismatched file would produce a full run of meaningless
    numbers instead of an error."""
    from deltachrome.pipeline import analyze

    chart = SyntheticChart()
    image, corners = chart.render_capture(tmp_path / "capture.tif")
    reference = tmp_path / "wrong.txt"
    reference.write_text(
        "CGATS.17\nNUMBER_OF_FIELDS\t4\nBEGIN_DATA_FORMAT\n"
        "SAMPLE_NAME\tLAB_L\tLAB_A\tLAB_B\nEND_DATA_FORMAT\n"
        "NUMBER_OF_SETS\t2\nBEGIN_DATA\nZZ1\t50\t0\t0\nZZ2\t60\t0\t0\nEND_DATA\n",
        encoding="utf-8",
    )
    with pytest.raises(MissingReferenceError, match="covers none of"):
        analyze(image, target="colorchecker_sg_v1", reference_path=reference,
                profile=PROFILE, corners=corners)
