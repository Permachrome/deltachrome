"""Target and profile definitions, and the guarantees they carry."""

from __future__ import annotations


import pytest

from deltachrome.errors import ProfileError, TargetError
from deltachrome.metrics.base import MetricContext
from deltachrome.profiles.evaluator import evaluate_profile
from pathlib import Path

from deltachrome.profiles.schema import available_profiles, load_profile
from deltachrome.targets.registry import available_targets, load_target

SG = "colorchecker_sg_v1"
FADGI3 = "fadgi_3star_documents_unbound_general_2023"
FADGI3_PRINTS = "fadgi_3star_prints_photographs_2023"


def test_sg_definition_is_complete_and_consistent():
    target = load_target(SG)
    assert len(target.patches) == 140
    assert len(set(target.patch_ids)) == 140
    grid = target.patch_grid
    assert grid is not None and grid.rows == 10 and grid.columns == 14


def test_sg_carries_no_reference_values():
    """A definition is a template for a model of chart. Reference values belong to
    a physical unit and must arrive from an imported measurement file."""
    target = load_target(SG)
    raw = target.source_path.read_text(encoding="utf-8")
    assert "reference_value" not in raw
    assert target.reference_data["required"] is True


def test_sg_ramp_is_the_twelve_fadgi_gray_patches():
    """Membership is the documented fact (FADGI OpenDICE names E5-J6)."""
    ramp = load_target(SG).ramp()
    assert {p.patch_id for p in ramp} == {
        "E5", "F5", "G5", "H5", "I5", "J5", "E6", "F6", "G6", "H6", "I6", "J6"
    }
    assert [p.ramp_position for p in ramp] == list(range(1, 13))


def test_sg_ramp_order_interleaves_across_its_two_rows():
    """Measured on a real unit: the ramp is not sequential within each row. Row 5
    left-to-right carries the odd steps and row 6 left-to-right the even steps in
    reverse. The definition records what was measured, and the engine still
    re-derives the order at analysis time so a different unit is reported."""
    ramp = load_target(SG).ramp()
    assert [p.patch_id for p in ramp] == [
        "E5", "J6", "F5", "I6", "G5", "H6", "H5", "G6", "I5", "F6", "J5", "E6"
    ]
    row5 = [p.ramp_position for p in ramp if p.patch_id.endswith("5")]
    row6 = [p.ramp_position for p in ramp if p.patch_id.endswith("6")]
    assert row5 == [1, 3, 5, 7, 9, 11]
    assert row6 == [2, 4, 6, 8, 10, 12]


def test_capabilities_and_not_capable_of_do_not_overlap():
    target = load_target(SG)
    assert not (target.capabilities & target.not_capable_of)
    assert not target.supports("sfr")


def test_unknown_target_lists_what_is_available():
    with pytest.raises(TargetError, match="Installed targets"):
        load_target("no_such_target")


def test_fadgi_profile_uses_percentile_not_max():
    profile = load_profile(FADGI3)
    spec = profile.metric("color_accuracy_90th_percentile")
    assert spec.aggregation == "percentile_90"
    assert spec.pass_condition == "value < 7"


def test_fadgi_white_balance_is_ab_only_on_the_ramp():
    """CIEDE2000 with the L* component eliminated, as FADGI 2.4.4 defines it and
    OpenDICE computes it -- not the Euclidean distance the tables' label suggests."""
    spec = load_profile(FADGI3).metric("white_balance")
    assert spec.algorithm["delta_e_variant"] == "delta_e_2000_ab"
    assert spec.algorithm["patch_selection"] == "role == 'achromatic_ramp'"
    assert spec.aggregation == "max_per_patch"


def test_fadgi_tone_response_uses_delta_l_2000():
    spec = load_profile(FADGI3).metric("tone_response_oecf")
    assert spec.algorithm["measure"] == "delta_l_2000"


def test_forking_a_profile_drops_the_standard_claim():
    """A user may edit thresholds, but the resulting document must not present
    itself as the standard it no longer matches."""
    profile = load_profile(FADGI3)
    fork = profile.fork("house_standard_2026")
    assert fork.profile_id == "house_standard_2026"
    assert fork.local_criteria
    # And the file says nothing about what it was forked from.
    assert FADGI3 not in str(fork.standard_reference)
    assert "must not be described as conforming" in fork.standard_reference["notice"]
    assert fork.locked_fields == ()


def test_fork_must_change_the_id():
    with pytest.raises(ProfileError, match="different profile_id"):
        load_profile(FADGI3).fork(FADGI3)


def test_metrics_the_target_cannot_measure_are_not_applicable_not_failures():
    target = load_target(SG)
    profile = load_profile(FADGI3)
    evaluation = evaluate_profile(profile, target, MetricContext(target=target, data=[]))

    by_id = {r.metric_id: r for r in evaluation.results}
    assert by_id["sfr10_sampling_efficiency"].outcome == "not_applicable"
    assert by_id["color_channel_misregistration"].outcome == "not_applicable"
    assert by_id["sfr10_sampling_efficiency"].passed is None


def test_a_missing_plugin_is_distinct_from_not_applicable():
    """Conflating these would let a report claim full coverage of a standard while
    silently skipping a metric the target could actually measure.

    Every metric the shipped profiles declare now has a plugin, so this is tested
    against a profile that declares one Deltachrome has never heard of -- which is
    also the case that matters in practice, since a user profile may name anything.
    """
    from deltachrome.metrics.base import MetricSpec

    profile = load_profile(FADGI3_PRINTS)
    profile.metrics.append(
        MetricSpec(metric_id="a_metric_with_no_plugin", pass_condition="value < 1")
    )
    target = load_target(SG)
    evaluation = evaluate_profile(profile, target, MetricContext(target=target, data=[]))

    result = next(
        r for r in evaluation.results if r.metric_id == "a_metric_with_no_plugin"
    )
    assert result.outcome == "not_implemented"
    assert evaluation.overall == "INCOMPLETE"
    assert "no plugin" in evaluation.overall_explanation


def test_every_metric_the_shipped_profiles_declare_has_a_plugin():
    """The gap is closed. A profile that reports NOT IMPLEMENTED can never report a
    pass, so any metric a shipped profile names without a plugin behind it makes
    that whole profile unusable -- which is worth catching here rather than in a
    report."""
    from deltachrome.metrics import available_metrics
    from deltachrome.profiles.schema import available_profiles

    registered = set(available_metrics())
    for profile_id in available_profiles():
        declared = {spec.metric_id for spec in load_profile(profile_id).metrics}
        missing = sorted(declared - registered)
        assert not missing, f"{profile_id} declares {missing} with no plugin"


def test_registries_find_the_built_ins():
    assert SG in available_targets()
    assert FADGI3 in available_profiles()


# -- the generated FADGI profile set ---------------------------------------
#
# Star levels are generated from transcribed conformance tables rather than
# hand-written, so a level cannot drift from its neighbors by a typo.

FADGI_2023 = [p for p in available_profiles() if p.endswith("_2023")]


def test_every_star_level_exists_for_each_generated_category():
    for slug in ("prints_photographs", "photographic_transparencies_4x5_and_larger"):
        levels = sorted(
            load_profile(p).standard_reference["star_level"]
            for p in FADGI_2023
            if p.endswith(f"{slug}_2023") and "documents" not in p
        )
        assert levels == [1, 2, 3, 4], slug


@pytest.mark.parametrize(
    "metric_id",
    ["tone_response_oecf", "white_balance", "lightness_uniformity", "sampling_frequency"],
)
def test_thresholds_tighten_monotonically_with_star_level(metric_id):
    """A higher star level must never be more permissive. This is the check that
    catches a transcription slip in a table of sixty numbers."""
    for slug in ("prints_photographs", "photographic_transparencies_4x5_and_larger"):
        profiles = sorted(
            (load_profile(p) for p in FADGI_2023 if p.endswith(f"{slug}_2023")),
            key=lambda p: p.standard_reference["star_level"],
        )
        values = []
        for profile in profiles:
            condition = profile.metric(metric_id).pass_condition
            values.append(float(condition.split()[-1]))
        if metric_id == "sampling_frequency":
            assert values == sorted(values), (slug, metric_id, values)
        else:
            assert values == sorted(values, reverse=True), (slug, metric_id, values)


def test_transparency_profiles_specify_no_color_accuracy_metric():
    """FADGI notes film is not scene referred, so there is no unambiguous original to
    judge colorimetric accuracy against. Inventing a threshold to fill the gap would
    be exactly the overclaiming this project refuses."""
    for profile_id in FADGI_2023:
        profile = load_profile(profile_id)
        if "Transparencies" not in profile.standard_reference["material_category"]:
            continue
        ids = {m.metric_id for m in profile.metrics}
        assert "average_color_accuracy" not in ids
        assert "color_accuracy_90th_percentile" not in ids
        assert "color_accuracy" not in profile.required_capabilities
        assert "NO color accuracy metric" in profile.notes


def test_reflective_profiles_do_specify_color_accuracy():
    profile = load_profile("fadgi_4star_prints_photographs_2023")
    ids = {m.metric_id for m in profile.metrics}
    assert {"average_color_accuracy", "color_accuracy_90th_percentile"} <= ids


def test_transparency_profiles_are_marked_provisional_by_the_standard():
    profile = load_profile("fadgi_3star_photographic_transparencies_4x5_and_larger_2023")
    assert profile.standard_reference["provisional"] is True
    assert "provisional" in profile.notes


def test_every_profile_records_where_its_numbers_came_from():
    for profile_id in FADGI_2023:
        ref = load_profile(profile_id).standard_reference
        assert ref["body"] == "FADGI"
        assert ref["document_version"]
        assert ref["material_category"]
        assert ref.get("thresholds_transcribed_from") or ref.get("source_url")


def test_profiles_can_be_selected_by_body_material_and_level():
    """A standard is a three-way choice, and picking the wrong material category is
    consequential -- the transparency tables have no color accuracy metric."""
    matches = [
        p for p in FADGI_2023
        if load_profile(p).standard_reference.get("star_level") == 4
        and "Transparencies" in load_profile(p).standard_reference["material_category"]
    ]
    # Two transparency categories, split by format size -- and they are not
    # interchangeable: the smaller one asks for exactly twice the sampling rate, so
    # a 35mm original judged by the 4x5 table would pass at half the resolution it
    # needs.
    assert sorted(matches) == [
        "fadgi_4star_photographic_transparencies_35mm_to_4x5_2023",
        "fadgi_4star_photographic_transparencies_4x5_and_larger_2023",
    ]


def test_the_generator_reproduces_the_checked_in_profiles():
    """The JSON is generated from tools/fadgi_tables_2023.json. If the two drift,
    the checked-in thresholds no longer have a traceable source."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen", Path(__file__).resolve().parents[1] / "tools/generate_fadgi_profiles.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    tables = gen.load_tables()
    checked = 0
    for section, table in tables.items():
        slug = gen.SLUGS[section]
        for i, star in enumerate(table["stars"] or [None]):
            profile_id = (
                f"fadgi_{star}star_{slug}_2023" if star else f"fadgi_{slug}_2023"
            )
            profile = load_profile(profile_id)
            for metric_id, expected in [
                ("tone_response_oecf", _condition(table, "tone_response", i, "value < {}")),
                ("white_balance", _condition(table, "white_balance", i, "value < {}")),
                ("sampling_frequency", _condition(table, "resolution_ppi", i, "value >= {}")),
            ]:
                if expected is None:
                    continue
                assert profile.metric(metric_id).pass_condition == expected, (
                    f"{profile_id}.{metric_id}"
                )
                checked += 1
    assert checked > 100, f"only {checked} thresholds checked; the loop is not running"


def test_the_generator_reproduces_every_checked_in_profile_exactly(tmp_path):
    """The threshold test above checks three figures a profile, and two corrections
    drifted past it: bit depth was made a list of permitted depths by hand in every
    shipped file, and three informative metrics were added by hand to twenty-four
    of them, while the generator still wrote neither. Regenerating would have
    reverted both silently. So every file is compared whole."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen", Path(__file__).resolve().parents[1] / "tools/generate_fadgi_profiles.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    shipped = gen.OUT
    gen.OUT = tmp_path

    written = gen.build()
    assert len(written) == 55
    for path in written:
        assert path.read_text(encoding="utf-8") == (shipped / path.name).read_text(
            encoding="utf-8"
        ), path.name


def _condition(table, key, i, template):
    values = table.get(key)
    return None if values is None else template.format(f"{values[i]:g}")


def test_every_material_category_the_standard_defines_is_modelled():
    """Eleven of fifteen were missing until the tables were read out of the PDF.
    A category that is not modeled is not a gap someone notices -- they pick the
    nearest one that exists, which is how a transparency gets judged by a table
    that has no color accuracy in it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen", Path(__file__).resolve().parents[1] / "tools/generate_fadgi_profiles.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    sections = set(gen.load_tables())
    assert len(sections) == 15
    modeled = {
        load_profile(p).standard_reference.get("section")
        for p in available_profiles()
        if load_profile(p).standard_reference.get("body") == "FADGI"
    }
    assert sections <= modeled, f"not modeled: {sorted(sections - modeled)}"


def test_a_category_with_no_performance_levels_is_still_a_profile():
    """Documents (Unbound): Modern Textual Records has no star levels at all --
    one specification. A profile scheme that assumed four levels would have had
    to leave the category out or invent three."""
    profile = load_profile("fadgi_documents_unbound_modern_textual_2023")
    reference = profile.standard_reference
    assert "star_level" not in reference
    assert reference["quality_level"] == "Single level"
    assert profile.metric("tone_response_oecf").pass_condition == "value < 3"


def test_the_categories_that_omit_a_star_level_omit_the_lowest():
    """Bound Volumes: Rare and Special Materials prints a 1-Star column and leaves
    every cell in it empty, so its rows align to the stricter end. Reading them as
    1/2/3-star would state 4-star requirements as though they were 2-star."""
    for slug in ("bound_volumes_rare_special", "documents_unbound_manuscripts_rare_special"):
        assert f"fadgi_1star_{slug}_2023" not in available_profiles()
        for star in (2, 3, 4):
            assert f"fadgi_{star}star_{slug}_2023" in available_profiles()
        # 4-star tone response is the strictest value the table prints.
        assert load_profile(
            f"fadgi_4star_{slug}_2023"
        ).metric("tone_response_oecf").pass_condition == "value < 1.5"


def test_a_guideline_sharing_nothing_with_the_chart_says_so_rather_than_failing():
    """Every shipped standard judges the file itself, so a run producing no value
    at all is a guideline and a chart with nothing in common -- the measurement-
    only guideline on a chart with no colour patches -- and not a run where
    everything went wrong. It used to say "no metric produced a value" and then
    "every metric that did run met its threshold", about none."""
    from deltachrome.metrics.base import MetricResult
    from deltachrome.profiles.evaluator import ProfileEvaluation
    from deltachrome.profiles.schema import load_profile

    evaluation = ProfileEvaluation(
        profile=load_profile("measurement_only_v1"),
        results=[MetricResult(metric_id="average_color_accuracy",
                              outcome="not_applicable")],
    )
    assert evaluation.overall == "INCOMPLETE"
    assert evaluation.overall_explanation.startswith(
        "Not a pass and not a failure: nothing this guideline judges can be "
        "measured on this chart."
    )
    assert "did run" not in evaluation.overall_explanation
