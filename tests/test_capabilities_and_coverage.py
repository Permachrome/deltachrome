"""What a target can measure, and how much of a profile a run actually covered.

Two rules, and they pull in opposite directions.

A metric the target cannot measure must never count against it -- not at the metric
level, not in the roll-up. Charts are partial by design and sites routinely cover
the rest with a second target in the same session, so penalising a ColorChecker SG
for having no slanted edges would be penalising it for being a ColorChecker SG.

But a pass over part of a standard is a partial result, and a report that does not
say which part is inviting someone to quote it as the whole. So absence costs
nothing and is stated everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deltachrome.errors import ProfileError, TargetError
from deltachrome.metrics.base import MetricContext
from deltachrome.profiles.evaluator import evaluate_profile, required_capability_note
from deltachrome.profiles.schema import available_profiles, load_profile, profile_from_dict
from deltachrome.targets.registry import available_targets, load_target
from deltachrome.targets.schema import KNOWN_CAPABILITIES, target_from_dict

DEFINITIONS = Path(__file__).resolve().parents[1] / "src/deltachrome/targets/definitions"
SG = "colorchecker_sg_v1"
PRINTS4 = "fadgi_4star_prints_photographs_2023"


def sg_definition(**changes):
    data = json.loads((DEFINITIONS / "colorchecker_sg_v1.json").read_text("utf-8"))
    data.update(changes)
    return data


# -- the vocabulary is closed ----------------------------------------------


def test_an_unknown_capability_is_refused_rather_than_ignored():
    """The failure it prevents is silent and looks like a target limitation: a name
    no profile recognizes disables every metric that would have matched, and the run
    then passes on whatever is left."""
    # DO NOT AMERICANISE THIS STRING. It is the typo being guarded against: the
    # vocabulary is American throughout, so a British spelling is the plausible
    # near-miss that would silently disable every colour metric.
    with pytest.raises(TargetError, match="unknown capability"):
        target_from_dict(sg_definition(capabilities=["colour_accuracy"]))
    with pytest.raises(TargetError, match="teleportation"):
        target_from_dict(sg_definition(capabilities=["teleportation"]))


def test_disclaiming_an_unknown_capability_is_refused_too():
    """A typo in not_capable_of is just as bad: it neither disables anything nor
    records anything, and reads as a considered statement about the chart."""
    with pytest.raises(TargetError, match="unknown capability"):
        target_from_dict(sg_definition(not_capable_of=["sfr_measurment"]))


def test_the_error_lists_what_is_valid():
    with pytest.raises(TargetError) as caught:
        target_from_dict(sg_definition(capabilities=["nonsense"]))
    for capability in ("color_accuracy", "sfr", "reproduction_scale"):
        assert capability in str(caught.value)


def test_a_profile_naming_an_unknown_capability_is_refused():
    """Checked from the other side as well. A profile requiring something no target
    can declare makes its metric permanently N/A, which reads as every target being
    unsuitable rather than as a typo."""
    data = {
        "profile_id": "broken",
        "metrics": [{"metric_id": "sfr50", "requires_capability": "sharpness"}],
    }
    with pytest.raises(ProfileError, match="unknown capability"):
        profile_from_dict(data)


def test_everything_shipped_uses_the_known_vocabulary():
    for target_id in available_targets():
        target = load_target(target_id)
        assert (target.capabilities | target.not_capable_of) <= KNOWN_CAPABILITIES
    for profile_id in available_profiles():
        profile = load_profile(profile_id)
        named = {s.requires_capability for s in profile.metrics if s.requires_capability}
        assert named <= KNOWN_CAPABILITIES


# -- absence costs nothing --------------------------------------------------


def test_metrics_the_target_cannot_measure_do_not_count_either_way():
    """The rule that makes partial targets usable at all."""
    target = load_target(SG)
    evaluation = evaluate_profile(
        load_profile(PRINTS4), target, MetricContext(target=target, data=[])
    )
    absent = evaluation.not_applicable
    assert {r.metric_id for r in absent} >= {
        "sfr10_sampling_efficiency", "sfr50", "sharpening",
        "color_channel_misregistration",
    }
    # Reproduction scale is NOT among them any more. The SG prints a millimetre
    # scale, so the chart can measure it; whether a given capture did is a
    # question for that run, and the answer there is incomplete rather than N/A.
    assert "reproduction_scale_accuracy" not in {r.metric_id for r in absent}
    assert all(r.passed is None for r in absent)
    assert all(not r.counts_toward_overall for r in absent)
    assert not any(r in evaluation.failures for r in absent)


def test_a_required_capability_the_target_lacks_does_not_fail_the_run():
    """A profile naming a capability as required is describing the category, not
    imposing a condition on the chart. Sites cover the gap with another target."""
    target = load_target(SG)
    profile = load_profile(PRINTS4)
    profile.required_capabilities = frozenset(profile.required_capabilities | {"sfr"})
    evaluation = evaluate_profile(profile, target, MetricContext(target=target, data=[]))
    assert evaluation.capability_report["required_missing"] == ["sfr"]
    assert not any(r.metric_id.startswith("sfr") for r in evaluation.failures)

    note = required_capability_note(profile, target)
    assert note is not None
    assert "Nothing here fails on that" in note
    assert "second target" in note


def test_no_note_when_the_target_carries_what_the_profile_requires():
    target = load_target(SG)
    assert required_capability_note(load_profile(PRINTS4), target) is None


# -- but absence is always stated ------------------------------------------


def test_coverage_counts_what_was_and_was_not_reached():
    target = load_target(SG)
    evaluation = evaluate_profile(
        load_profile(PRINTS4), target, MetricContext(target=target, data=[])
    )
    coverage = evaluation.coverage
    assert coverage["declared"] == len(evaluation.results)
    assert coverage["not_applicable"] == len(evaluation.not_applicable)
    assert "sfr50" in coverage["not_measurable_by_this_target"]
    assert coverage["judged"] + coverage["not_applicable"] + coverage["incomplete"] \
        + coverage["measured_unjudged"] == coverage["declared"]


def test_the_coverage_line_reads_as_a_sentence():
    target = load_target(SG)
    evaluation = evaluate_profile(
        load_profile(PRINTS4), target, MetricContext(target=target, data=[])
    )
    assert "not measurable by this target" in evaluation.coverage_line


def test_a_pass_says_how_much_of_the_profile_it_covered(tmp_path):
    """The hole this closes: 'All 12 judged metrics met their thresholds' reads as
    conformance when six of eighteen were never tested."""
    from deltachrome.pipeline import analyze
    from fixtures import SyntheticChart

    chart = SyntheticChart()
    image, corners = chart.render_capture(tmp_path / "c.tif")
    reference = chart.write_reference(tmp_path / "r.txt")
    result = analyze(
        image, target=SG, reference_path=reference,
        profile="fadgi_1star_prints_photographs_2023", corners=corners,
        # The synthetic chart is rendered small; without an extent to derive from,
        # sampling frequency fails and the pass path is never reached.
        fiducial_extent_mm=(100.0, 70.0),
    )
    assert result.overall == "PASS"
    explanation = result.evaluation.overall_explanation
    assert "cannot measure were not tested" in explanation
    assert "sfr50" in explanation
    assert result.run.profile["coverage"]["not_applicable"] > 0
    assert "not measurable by this target" in result.run.profile["coverage_line"]
