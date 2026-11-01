"""End-to-end pipeline tests.

The reference values and the rendered image come from the same numbers, so a clean
render must return Delta E at the quantisation floor. That single assertion covers
ICC parsing, tone curves, the colorant matrix, the Lab conversion, the homography
and the ROI sampler at once: an error anywhere in that chain breaks it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from deltachrome.errors import DuplicateSampleError, MissingICCProfileError
from deltachrome.pipeline import analyze
from deltachrome.targets.registry import load_target
from deltachrome.targets.roles import resolve_roles
from deltachrome.reference.cgats import read_cgats
from fixtures import RAMP_IDS, UNIFORMITY_IDS, SyntheticChart

SG = "colorchecker_sg_v1"
FADGI3 = "fadgi_3star_documents_unbound_general_2023"


@pytest.fixture
def chart_files(tmp_path):
    chart = SyntheticChart()
    reference = chart.write_reference(tmp_path / "ref.cgats.txt")
    image, corners = chart.render_capture(tmp_path / "capture.tif", skew=18)
    return chart, reference, image, corners


def run(image, reference, corners, target_override=None, **kwargs):
    return analyze(
        image,
        target=target_override or SG,
        reference_path=reference,
        profile=FADGI3,
        corners=corners,
        **kwargs,
    )


# -- the round trip ---------------------------------------------------------


def test_exact_render_round_trips_at_the_quantisation_floor(chart_files):
    _, reference, image, corners = chart_files
    result = run(image, reference, corners)

    assert result.overall == "PASS"
    average = next(m for m in result.evaluation.results if m.metric_id == "average_color_accuracy")
    assert average.value < 0.02, "color management or sampling is losing precision"
    assert len(average.per_patch) == 140


def test_tonal_ramp_round_trips_exactly(chart_files):
    _, reference, image, corners = chart_files
    result = run(image, reference, corners)
    tone = next(m for m in result.evaluation.results if m.metric_id == "tone_response_oecf")
    assert tone.value < 0.02
    assert len(tone.per_patch) == 12


def test_perspective_skew_does_not_degrade_the_result(chart_files, tmp_path):
    chart, reference, _, _ = chart_files
    flat_image, flat_corners = chart.render_capture(tmp_path / "flat.tif", skew=0)
    skewed_image, skewed_corners = chart.render_capture(tmp_path / "skew.tif", skew=60)

    flat = run(flat_image, reference, flat_corners)
    skewed = run(skewed_image, reference, skewed_corners)
    for result in (flat, skewed):
        average = next(
            m for m in result.evaluation.results if m.metric_id == "average_color_accuracy"
        )
        assert average.value < 0.05


# -- injected errors --------------------------------------------------------


def test_a_channel_gain_error_shows_up_as_a_white_balance_failure(chart_files, tmp_path):
    chart, reference, _, _ = chart_files
    image, corners = chart.render_capture(tmp_path / "warm.tif", gain=(1.12, 1.0, 0.88))
    result = run(image, reference, corners)

    wb = next(m for m in result.evaluation.results if m.metric_id == "white_balance")
    assert wb.outcome == "evaluated"
    assert wb.passed is False
    assert result.overall == "FAIL"
    assert "white_balance" in result.evaluation.overall_explanation


def test_a_gamma_error_shows_up_in_tone_response(chart_files, tmp_path):
    chart, reference, _, _ = chart_files
    image, corners = chart.render_capture(tmp_path / "gamma.tif", gamma_error=1.18)
    result = run(image, reference, corners)

    tone = next(m for m in result.evaluation.results if m.metric_id == "tone_response_oecf")
    assert tone.value > 3.0
    assert tone.passed is False


def test_a_tone_error_alone_does_not_trip_white_balance(chart_files, tmp_path):
    """The two metrics must stay independent, or a report cannot say which fault
    the operator needs to fix."""
    chart, reference, _, _ = chart_files
    image, corners = chart.render_capture(tmp_path / "gamma.tif", gamma_error=1.18)
    result = run(image, reference, corners)

    wb = next(m for m in result.evaluation.results if m.metric_id == "white_balance")
    assert wb.value < 1.0


# -- role resolution --------------------------------------------------------


def test_roles_are_resolved_from_the_reference_file(chart_files):
    _, reference, _, _ = chart_files
    resolution = resolve_roles(load_target(SG), read_cgats(reference))

    assert resolution.counts["achromatic_ramp"] == 12
    assert set(resolution.uniformity_patches) == set(UNIFORMITY_IDS)
    assert resolution.uniformity_mean_lab[0] == pytest.approx(96.0, abs=0.2)

    # Randomly generated device colors can land near the neutral axis; those are
    # correctly 'achromatic_other' rather than 'chromatic'. What must hold is that
    # every patch got exactly one role and none were left unassigned.
    assert sum(resolution.counts.values()) == 140
    assert "unassigned" not in resolution.counts
    assert (
        resolution.counts["chromatic"] + resolution.counts.get("achromatic_other", 0)
        == 140 - 12 - len(UNIFORMITY_IDS)
    )


def test_ramp_order_is_derived_from_measured_lightness(chart_files):
    _, reference, _, _ = chart_files
    resolution = resolve_roles(load_target(SG), read_cgats(reference))
    ramp = resolution.target.ramp()
    lightness = [
        read_cgats(reference)[p.patch_id].lab[0] for p in ramp
    ]
    assert lightness == sorted(lightness, reverse=True)
    assert [p.patch_id for p in ramp] == RAMP_IDS


def test_a_scrambled_ramp_is_reordered_and_reported(tmp_path):
    """A ramp read backwards still passes a max-per-patch threshold while making
    the plotted OECF curve meaningless, so the order is derived, not assumed."""
    chart = SyntheticChart()
    path = chart.write_reference(tmp_path / "ref.txt")
    text = path.read_text()
    # Swap the reference values of the ramp's two ends.
    lines = text.splitlines()
    idx = {line.split()[0]: i for i, line in enumerate(lines) if line and line[0] in "ABCDEFGHIJKLMN"}
    lines[idx["E5"]], lines[idx["J6"]] = (
        "E5 " + lines[idx["J6"]].split(" ", 1)[1],
        "J6 " + lines[idx["E5"]].split(" ", 1)[1],
    )
    path.write_text("\n".join(lines) + "\n")

    resolution = resolve_roles(load_target(SG), read_cgats(path))
    assert any("re-derived from measured reference L*" in n for n in resolution.notes)
    ramp = resolution.target.ramp()
    assert ramp[0].patch_id == "J6"
    assert ramp[-1].patch_id == "E5"


# -- refusals ---------------------------------------------------------------


def test_a_duplicate_in_the_reference_file_stops_the_run(tmp_path):
    chart = SyntheticChart()
    reference = chart.write_reference(tmp_path / "ref.txt", duplicate="K8")
    image, corners = chart.render_capture(tmp_path / "cap.tif")
    with pytest.raises(DuplicateSampleError, match="K8"):
        run(image, reference, corners)


def test_an_image_without_an_icc_profile_stops_the_run(tmp_path, chart_files):
    import tifffile

    _, reference, _, corners = chart_files
    path = tmp_path / "bare.tif"
    tifffile.imwrite(path, np.zeros((1000, 1400, 3), dtype=np.uint16), photometric="rgb")
    with pytest.raises(MissingICCProfileError):
        run(path, reference, corners)


# -- the persisted record ---------------------------------------------------


def test_run_record_persists_raw_per_patch_data(chart_files, tmp_path):
    _, reference, image, corners = chart_files
    result = run(image, reference, corners)
    saved = result.run.save(tmp_path / "run.json")

    data = json.loads(saved.read_text())
    assert len(data["patches"]) == 140
    a1 = next(p for p in data["patches"] if p["patch_id"] == "A1")
    assert len(a1["rgb_mean"]) == 3
    assert a1["measured_lab"] and a1["reference_lab"]
    assert a1["pixel_count"] > 0


def test_run_record_captures_reproducibility_context(chart_files):
    _, reference, image, corners = chart_files
    data = run(image, reference, corners).run.to_dict()

    assert data["color_management"]["icc_profile"].startswith("Synthetic Adobe RGB")
    assert data["color_management"]["reference_illuminant"] == "D50"
    assert data["reference"]["measurement_mode"] == "M1"
    assert data["reference"]["provenance"]["INSTRUMENTATION"] == "synthetic"
    assert data["sampling"]["roi_inset"] == pytest.approx(0.55)
    assert data["geometry"]["effective_ppi"] > 0
    assert data["environment"]["deltachrome_version"]


def test_run_record_is_json_serialisable_with_no_nan(chart_files, tmp_path):
    _, reference, image, corners = chart_files
    text = run(image, reference, corners).run.to_json()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)


def test_percentile_method_override_is_recorded(chart_files):
    _, reference, image, corners = chart_files
    result = run(image, reference, corners, percentile_method="nearest_rank")
    metric = next(
        m for m in result.evaluation.results if m.metric_id == "color_accuracy_90th_percentile"
    )
    assert metric.metadata["percentile_method"] == "nearest_rank"


def test_run_options_are_written_into_the_record(chart_files):
    """These were read into ``context.extras`` at analysis time and never
    written back out, so a record built by ``analyze`` could not say what it had
    been overridden with -- a run made with ``--original-size`` could only be
    re-established under the profile's bare default."""
    _, reference, image, corners = chart_files
    result = run(
        image, reference, corners,
        percentile_method="nearest_rank", frequency_units="lp/mm",
        original_size="A2",
    )
    data = result.run.to_dict()
    assert data["percentile_method"] == "nearest_rank"
    assert data["frequency_units"] == "lp/mm"
    assert data["original_size"] == "A2"


def test_provisional_geometry_is_surfaced_as_a_warning(chart_files, tmp_path):
    """The SG's geometry is now measured, so this exercises the mechanism with a
    target that still declares itself provisional -- any newly added target will."""
    import json

    _, reference, image, corners = chart_files
    raw = json.loads(load_target(SG).source_path.read_text(encoding="utf-8"))
    raw["target_id"] = "provisional_probe"
    grid = next(r["patch_grid"] for r in raw["regions"] if r.get("patch_grid"))
    grid["provisional"] = True
    grid["provisional_reason"] = "spacing not measured from a physical unit"
    path = tmp_path / "provisional.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    result = run(image, reference, corners, target_override=str(path))
    assert any("provisional" in w for w in result.warnings)
    assert any("not measured from a physical unit" in w for w in result.warnings)


def test_measured_geometry_produces_no_provisional_warning(chart_files):
    _, reference, image, corners = chart_files
    result = run(image, reference, corners)
    assert not any("provisional" in w for w in result.warnings)

# -- LUT-based capture profiles --------------------------------------------


def test_a_capture_with_a_lut_profile_analyses_identically(chart_files, tmp_path):
    """A scanner-style LUT profile and the analytic profile describe the same
    transform here, so the two runs must agree to well inside the measurement's
    own precision. This is the end-to-end proof that the A2B path is correct."""
    from fixtures import lut_icc_from_matrix

    chart, reference, _, _ = chart_files
    lut_image, corners = chart.render_capture(
        tmp_path / "lut.tif", skew=18, icc=lut_icc_from_matrix(tag_type="mft2")
    )
    matrix_image, _ = chart.render_capture(tmp_path / "matrix.tif", skew=18)

    lut_result = run(lut_image, reference, corners)
    matrix_result = run(matrix_image, reference, corners)

    assert lut_result.image.profile.transform_kind == "lut"
    assert matrix_result.image.profile.transform_kind == "matrix_trc"

    for metric_id in ("average_color_accuracy", "white_balance", "tone_response_oecf"):
        lut_value = next(
            m for m in lut_result.evaluation.results if m.metric_id == metric_id
        ).value
        matrix_value = next(
            m for m in matrix_result.evaluation.results if m.metric_id == metric_id
        ).value
        assert lut_value == pytest.approx(matrix_value, abs=0.05), metric_id

    assert lut_result.overall == "PASS"


@pytest.mark.parametrize("tag_type", ["mft2", "mAB"])
def test_lut_capture_round_trips_at_the_encoding_floor(chart_files, tmp_path, tag_type):
    from fixtures import lut_icc_from_matrix

    chart, reference, _, _ = chart_files
    image, corners = chart.render_capture(
        tmp_path / f"lut_{tag_type}.tif", icc=lut_icc_from_matrix(tag_type=tag_type)
    )
    result = run(image, reference, corners)

    average = next(
        m for m in result.evaluation.results if m.metric_id == "average_color_accuracy"
    )
    assert average.value < 0.05
    assert result.overall == "PASS"


def test_the_run_record_says_which_icc_structure_was_evaluated(chart_files, tmp_path):
    """Which structure was used is part of the result. Two runs of the same capture
    through different transforms are different measurements."""
    from fixtures import lut_icc_from_matrix

    chart, reference, _, _ = chart_files
    image, corners = chart.render_capture(
        tmp_path / "lut.tif", icc=lut_icc_from_matrix(tag_type="mAB")
    )
    data = run(image, reference, corners).run.to_dict()

    cm = data["color_management"]
    assert cm["transform_kind"] == "lut"
    assert cm["lut"]["tag"] == "A2B1"
    assert cm["lut"]["tag_type"] == "mAB"
    assert cm["lut"]["pcs_encoding"] == "xyz_u1f15"
    assert cm["lut"]["clut_grid"] == [9, 9, 9]
    assert "lookup pipeline" in cm["method"]


def test_a_perceptual_only_profile_warns_in_the_report(chart_files, tmp_path):
    """Perceptual tables carry gamut mapping. If that is all a profile offers, the
    report has to say the numbers include it."""
    from deltachrome.reporting.console import render
    from fixtures import lut_icc_from_matrix

    chart, reference, _, _ = chart_files
    image, corners = chart.render_capture(
        tmp_path / "perceptual.tif", icc=lut_icc_from_matrix(tag=b"A2B0")
    )
    result = run(image, reference, corners)
    text = render(result)

    assert "COLOR MANAGEMENT NOTES" in text
    assert "perceptual" in text


def test_every_patch_value_records_where_it_was_measured(chart_files):
    """A number's position is the difference between "the illumination is uneven"
    and "the left side is dark". Only edges carried one until now, which is why
    the edge field was the only metric that could be drawn as a field."""
    _, reference, image, corners = chart_files
    result = run(image, reference, corners)

    metric = next(
        m for m in result.run.metrics
        if m["metric_id"] == "average_color_accuracy"
    )
    placed = [p for p in metric["per_patch"] if p.get("position_relative")]
    assert len(placed) == len(metric["per_patch"])
    for patch in placed:
        u, v = patch["position_relative"]
        assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0


def test_the_recorded_position_is_the_patch_the_reader_would_point_at(chart_files):
    """A1 is the corner the placement's top-left was put on, so it must land in
    the top-left quarter of the chart and not somewhere the grid merely indexes."""
    _, reference, image, corners = chart_files
    result = run(image, reference, corners)
    metric = next(
        m for m in result.run.metrics
        if m["metric_id"] == "average_color_accuracy"
    )
    by_id = {p["patch_id"]: p["position_relative"] for p in metric["per_patch"]}
    assert by_id["A1"][0] < 0.25 and by_id["A1"][1] < 0.25
    assert by_id["N10"][0] > 0.75 and by_id["N10"][1] > 0.75


def test_a_reading_that_has_no_single_place_does_not_invent_one(chart_files):
    """Gain modulation spans a pair of patches. Putting it at either would put a
    mark on a field map where nothing was measured."""
    from deltachrome.metrics.base import PatchValue

    assert PatchValue(
        patch_id="x", role="r", value=1.0,
        measured_lab=(0.0, 0.0, 0.0), reference_lab=None,
    ).position_relative is None


def test_the_placement_maps_both_ways(chart_files):
    """Painting a mark and asking what is under the cursor must not be able to
    disagree about where a point is."""
    import numpy as np

    _, reference, image, corners = chart_files
    result = run(image, reference, corners)
    from deltachrome.imaging.geometry import place_chart

    placement = place_chart(
        corners,
        image_size=(result.run.image["width"], result.run.image["height"]),
    )
    points = np.array([[0.0, 0.0], [0.37, 0.81], [1.0, 1.0]])
    back = placement.to_relative(placement.to_pixel(points))
    assert np.allclose(back, points, atol=1e-9)


def test_each_patch_records_the_region_it_was_sampled_from(chart_files):
    """The inset quadrilateral, not the nominal patch. The inset is a parameter
    and perspective makes it a quadrilateral, so outlining the nominal patch
    would claim an area larger than the one that was measured."""
    _, reference, image, corners = chart_files
    record = run(image, reference, corners).run

    patch = next(p for p in record.patches if p.patch_id == "K3")
    assert patch.roi_px is not None and len(patch.roi_px) == 4

    xs = [p[0] for p in patch.roi_px]
    ys = [p[1] for p in patch.roi_px]
    # Skewed capture: the quadrilateral is not axis-aligned.
    assert xs[0] != xs[3] or ys[0] != ys[1]

    # And it is the inset region, so smaller than the patch's own spacing.
    from deltachrome.targets.registry import load_target

    grid = load_target(SG).patch_grid
    nominal = grid.patch_size_relative[0] * record.geometry["chart_area_px"] ** 0.5
    assert max(xs) - min(xs) < nominal


def test_a_patch_that_could_not_be_sampled_records_no_region(chart_files):
    """Better nothing than a rectangle drawn where no read happened."""
    from deltachrome.results.models import PatchRecord

    record = PatchRecord(
        patch_id="X1", row=0, col=0, role="chromatic", ramp_position=None,
        rgb_mean=[], rgb_median=[], rgb_std=[], measured_lab=None,
        reference_lab=None, pixel_count=0, clipped_high_fraction=float("nan"),
        clipped_low_fraction=float("nan"), flags=["not_sampled"],
    )
    assert record.roi_px is None


def test_how_the_chart_was_placed_and_named_is_in_the_record(chart_files):
    """A placement nobody chose has to be visible in the archive. It is a field of
    the geometry now, where it was a warning on every automatic run."""
    from deltachrome.pipeline import rescore

    chart, reference, image, corners = chart_files
    plain = analyze(image, target=SG, reference_path=reference, profile=FADGI3,
                    corners=corners)
    assert plain.run.geometry["placed"] == "by_hand"
    assert plain.run.geometry["identified"] is False

    found = analyze(image, target=SG, reference_path=reference, profile=FADGI3,
                    corners=corners, placed="automatic", identified=True)
    assert found.run.geometry["placed"] == "automatic"
    assert found.run.geometry["identified"] is True
    assert not any("automatically" in w for w in found.warnings)

    again = rescore(found, "fadgi_4star_prints_photographs_2023")
    assert again.run.geometry["placed"] == "automatic"
    assert again.run.geometry["identified"] is True

    with pytest.raises(ValueError, match="placed must be one of"):
        analyze(image, target=SG, reference_path=reference, profile=FADGI3,
                corners=corners, placed="by_robot")


def test_a_chart_named_from_a_narrowed_set_says_which_set(chart_files):
    """A chart named from five candidates is a different claim from one named
    from twenty-three, so the record says which -- and says nothing at all in the
    ordinary case, where every installed target was searched."""
    from deltachrome.pipeline import rescore

    chart, reference, image, corners = chart_files
    plain = analyze(image, target=SG, reference_path=reference, profile=FADGI3,
                    corners=corners, placed="automatic", identified=True)
    assert "searched_targets" not in plain.run.geometry

    owned = ["it8_7_2_reflective_v1", "colorchecker_sg_v1"]
    narrowed = analyze(image, target=SG, reference_path=reference, profile=FADGI3,
                       corners=corners, placed="automatic", identified=True,
                       searched_targets=owned)
    assert narrowed.run.geometry["searched_targets"] == sorted(owned)
    again = rescore(narrowed, "fadgi_4star_prints_photographs_2023")
    assert again.run.geometry["searched_targets"] == sorted(owned)

    from deltachrome.reporting.console import render
    assert "among 2 owned target(s)" in render(narrowed)
