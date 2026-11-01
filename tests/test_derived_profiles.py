"""Local specifications derived from a standard.

Projects are rarely uniformly one star level. "Four star for 35mm photographs,
except sampling frequency, which we can set at 2000 ppi" is a real and reasonable
thing to want -- and it is not FADGI 4-star. So the result carries the parent as
provenance rather than as a claim, records exactly where it departs, and costs the
run its conformance claim whichever way the threshold moved.

The departure list is the point. "Custom profile" tells a reader nothing; the
sentence worth keeping is the one naming what was agreed and what was not.
"""

from __future__ import annotations

import json

import pytest

from deltachrome.errors import ProfileError
from deltachrome.metrics.base import MetricContext
from deltachrome.profiles.evaluator import evaluate_profile
from deltachrome.profiles.schema import (
    available_profiles,
    compare_conditions,
    is_local,
    load_profile,
    user_profile_dir,
)
from deltachrome.targets.registry import load_target

BASE = "fadgi_4star_photographic_transparencies_35mm_to_4x5_2023"
PRINTS3 = "fadgi_3star_prints_photographs_2023"


@pytest.fixture
def local_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DELTACHROME_PROFILE_DIR", str(tmp_path / "profiles"))
    from deltachrome.profiles import schema

    schema._index.cache_clear()
    yield tmp_path / "profiles"
    schema._index.cache_clear()


# -- deriving ---------------------------------------------------------------


def test_the_worked_example(local_dir):
    """Four star for 35mm, except sampling frequency at 2000 ppi."""
    derived = load_profile(BASE).derive(
        "house_35mm", thresholds={"sampling_frequency": "value >= 2000"}
    )
    assert derived.metric("sampling_frequency").pass_condition == "value >= 2000"
    # Everything else is the parent's.
    assert derived.metric("sfr50").pass_condition == \
        load_profile(BASE).metric("sfr50").pass_condition

    assert derived.local_criteria
    assert derived.departures == [{
        "metric_id": "sampling_frequency", "change": "threshold",
        "from": "value >= 3960", "to": "value >= 2000", "direction": "relaxed",
    }]
    assert "relaxed from 'value >= 3960' to 'value >= 2000'" in \
        derived.departure_summary
    # And the rest of the table came across unexamined, which is the fact the
    # parent id was really carrying: one threshold was decided, seventeen were
    # handed over.
    assert "sampling_frequency" not in derived.inherited
    assert "sfr50" in derived.inherited
    assert len(derived.inherited) == len(derived.metrics) - 1


def test_a_metric_can_be_omitted_entirely(local_dir):
    derived = load_profile(PRINTS3).derive("no_noise", omit=["noise"])
    assert not any(m.metric_id == "noise" for m in derived.metrics)
    assert derived.departures[0]["change"] == "omitted"
    assert derived.departures[0]["direction"] == "removed"


def test_an_algorithm_can_be_overridden_and_is_merged(local_dir):
    derived = load_profile(PRINTS3).derive(
        "worst_edges", algorithms={"sfr50": {"edge_aggregation": "worst"}}
    )
    spec = derived.metric("sfr50")
    assert spec.algorithm["edge_aggregation"] == "worst"
    # Merged, not replaced: the parent's other settings survive.
    parent = load_profile(PRINTS3).metric("sfr50").algorithm
    for key, value in parent.items():
        if key != "edge_aggregation":
            assert spec.algorithm[key] == value


def test_overriding_a_metric_the_profile_does_not_have_is_refused():
    with pytest.raises(ProfileError, match="no metric"):
        load_profile(BASE).derive("bad", thresholds={"average_color_accuracy": "value < 3"})


def test_a_derivation_that_changes_nothing_is_refused():
    """It would be a copy claiming less than the original for no reason."""
    with pytest.raises(ProfileError, match="would change nothing"):
        load_profile(BASE).derive("pointless")


# -- direction, which is reported but does not decide anything ---------------


@pytest.mark.parametrize(
    "before,after,expected",
    [
        ("value < 3.5", "value < 5", "relaxed"),
        ("value < 3.5", "value < 2", "tightened"),
        ("value > 90", "value > 80", "relaxed"),
        ("value > 80", "value > 90", "tightened"),
        ("value >= 2940", "value >= 2000", "relaxed"),
        ("40 <= value <= 75", "30 <= value <= 95", "relaxed"),
        ("30 <= value <= 95", "40 <= value <= 75", "tightened"),
        ("value < 2", "30 <= value <= 95", "changed"),
        ("value < 2", "value < 2", "unchanged"),
    ],
)
def test_threshold_direction(before, after, expected):
    assert compare_conditions(before, after) == expected


def test_tightening_still_voids_the_conformance_claim(local_dir):
    """A stricter profile is no more the standard than a looser one, and
    'stricter, so it still counts' is easy to game and awkward to explain."""
    derived = load_profile(BASE).derive(
        "stricter", thresholds={"sampling_frequency": "value >= 5000"}
    )
    assert derived.departures[0]["direction"] == "tightened"

    target = load_target("isa_transmissive_35mm_v1")
    evaluation = evaluate_profile(derived, target, MetricContext(target=target, data=[]))
    assert evaluation.conformance_claim is False
    assert any("local guideline" in c for c in evaluation.conformance_caveats)


def test_both_reasons_can_apply_and_are_named_separately(local_dir):
    """A nominal reference and a derived profile are independent reasons, and a run
    can carry both."""
    from deltachrome.reference.cgats import MANUFACTURER_NOMINAL

    derived = load_profile(BASE).derive(
        "house", thresholds={"sampling_frequency": "value >= 2000"}
    )
    target = load_target("isa_transmissive_35mm_v1")
    evaluation = evaluate_profile(
        derived, target,
        MetricContext(target=target, data=[], reference_basis=MANUFACTURER_NOMINAL),
    )
    caveats = evaluation.conformance_caveats
    assert len(caveats) == 2
    assert any("nominal values" in c for c in caveats)
    assert any("local guideline" in c for c in caveats)
    assert "Advisory, not a conformance result" in evaluation.overall_explanation


def test_a_standard_profile_keeps_its_claim():
    target = load_target("isa_transmissive_35mm_v1")
    evaluation = evaluate_profile(
        load_profile(BASE), target, MetricContext(target=target, data=[])
    )
    assert evaluation.conformance_claim is True
    assert evaluation.conformance_caveats == []


# -- saving and finding again -----------------------------------------------


def test_a_derived_profile_saves_and_loads_by_id(local_dir):
    derived = load_profile(BASE).derive(
        "house_35mm", thresholds={"sampling_frequency": "value >= 2000"},
        display_name="House standard",
    )
    path = derived.save()
    assert path == local_dir / "house_35mm.json"

    loaded = load_profile("house_35mm")
    assert loaded.display_name == "House standard"
    assert loaded.local_criteria
    assert loaded.metric("sampling_frequency").pass_condition == "value >= 2000"
    # The parent, the departure list and the sentence describing it are gone from
    # the file. What survives is which criteria were never chosen.
    assert BASE not in path.read_text()
    assert loaded.departures == []
    assert loaded.inherited == derived.inherited
    assert is_local("house_35mm")
    assert not is_local(BASE)


def test_a_saved_profile_round_trips_every_metric(local_dir):
    base = load_profile(PRINTS3)
    derived = base.derive("round_trip", thresholds={"noise": "value < 9"})
    derived.save()
    loaded = load_profile("round_trip")

    assert [m.metric_id for m in loaded.metrics] == [m.metric_id for m in derived.metrics]
    for spec in derived.metrics:
        other = loaded.metric(spec.metric_id)
        assert other.pass_condition == spec.pass_condition
        assert other.requires_capability == spec.requires_capability
        assert other.algorithm == spec.algorithm


def test_a_user_profile_may_shadow_a_shipped_id(local_dir):
    """Allowed on purpose: a site that decided its own 'fadgi_3star_...' means
    something particular should get what it decided, and is_local says which
    answered."""
    load_profile(PRINTS3).derive(PRINTS3 + "_x", thresholds={"noise": "value < 9"})
    shadow = load_profile(PRINTS3).derive("shadow", thresholds={"noise": "value < 9"})
    shadow.profile_id = PRINTS3
    shadow.save()

    from deltachrome.profiles import schema

    schema._index.cache_clear()
    assert is_local(PRINTS3)
    assert load_profile(PRINTS3).metric("noise").pass_condition == "value < 9"


def test_the_user_profile_directory_is_overridable(local_dir):
    assert user_profile_dir() == local_dir


def test_saving_elsewhere_writes_where_asked(tmp_path, local_dir):
    derived = load_profile(BASE).derive("elsewhere", omit=["sharpening"])
    path = derived.save(tmp_path / "shared")
    assert path == tmp_path / "shared" / "elsewhere.json"
    assert json.loads(path.read_text())["profile_id"] == "elsewhere"


# -- what severing must not break ---------------------------------------------


def test_a_guideline_saved_before_the_parent_was_severed_still_reads_as_local(
    local_dir,
):
    """A profile file is a user's file, and a rename must not make one unreadable.

    The old shape carried ``derived_from``, ``original_standard_reference``,
    ``departures`` and ``summary``. Only one fact in there was ever read -- that
    the criteria are local -- and reading it wrong is not a cosmetic failure: it
    would hand a conformance claim to a run that has not earned one, which is the
    single direction this must never fail in.
    """
    import json

    local_dir.mkdir(parents=True, exist_ok=True)
    path = local_dir / "old_shape.json"
    path.write_text(json.dumps({
        "$schema_version": 1,
        "profile_id": "old_shape",
        "display_name": "Our 4-star",
        "standard_reference": {
            "derived_from": BASE,
            "original_standard_reference": {"body": "FADGI", "star_level": 4},
            "departures": [{"metric_id": "noise", "change": "threshold",
                            "from": "value < 1", "to": "value < 3",
                            "direction": "relaxed"}],
            "summary": "FADGI 4-star, with noise relaxed",
        },
        "metrics": [
            {"metric_id": "noise", "requires_capability": "tonal_response",
             "aggregation": "max", "pass_condition": "value < 3",
             "units": "std dev"},
            {"metric_id": "white_balance", "requires_capability": "white_balance",
             "aggregation": "max_per_patch", "pass_condition": "value < 4",
             "units": "dE(a*b*)"},
        ],
    }), encoding="utf-8")

    loaded = load_profile(path)
    assert loaded.local_criteria
    # Its departure list still reads, because it is still in that file.
    assert loaded.departures[0]["metric_id"] == "noise"
    # And its inherited set is reconstructed from those departures rather than
    # coming back empty. Exact, not a guess: the old shape names what was
    # touched, so the complement is what was not. Empty would have meant "every
    # criterion here was chosen deliberately" the next time it was saved, which
    # is the flattering direction.
    assert loaded.inherited == ["white_balance"]


def test_a_run_record_written_before_the_parent_was_severed_still_reads_as_local():
    """Same failure one layer out. A year of saved runs carries ``derived_from``
    in their profile block; new ones carry ``local_criteria``."""
    from deltachrome.profiles.evaluator import criteria_are_local

    assert criteria_are_local({"derived_from": "fadgi_3star_prints_photographs_2023"})
    assert criteria_are_local({"local_criteria": True})
    assert not criteria_are_local({"local_criteria": False})
    assert not criteria_are_local({})
    assert not criteria_are_local(None)


def test_inherited_is_what_the_operator_never_chose(local_dir):
    """The fact the parent id was really carrying, and the reason severing it
    needed something in its place.

    Moving one threshold on a FADGI profile carries every other criterion across
    unexamined -- eighteen of them, algorithm parameters included -- and nothing
    on the builder screen says so, because from the operator's side they changed
    one number.
    """
    derived = load_profile(BASE).derive(
        "house", thresholds={"sampling_frequency": "value >= 2000"}
    )
    assert len(derived.inherited) == len(derived.metrics) - 1
    assert "sampling_frequency" not in derived.inherited

    # Amending narrows it and never grows it: a criterion decided in an earlier
    # session does not become inherited again by being left alone in this one.
    again = derived.derive(
        "house", thresholds={"sfr50": "value >= 0.4"},
        inherited=derived.inherited, allow_identical=True,
    )
    assert "sampling_frequency" not in again.inherited
    assert "sfr50" not in again.inherited
    assert set(again.inherited) == set(derived.inherited) - {"sfr50"}


def test_nothing_in_a_saved_guideline_names_what_it_was_built_from(local_dir):
    """Swept rather than asserted key by key, because the leaks were not in the
    keys anybody would think to check.

    Two survived the obvious fix. The profile's own ``notes`` carried the
    parent's transcription note -- "Thresholds read from FADGI 3rd Edition, May
    2023, section 3.8 (Prints and Photographs), pages 50-51" -- verbatim into a
    guideline that had just moved one of those thresholds, so the file asserted
    something untrue about itself. And every metric carried its own, six of which
    name the body outright.
    """
    derived = load_profile(BASE).derive(
        "swept", thresholds={"sampling_frequency": "value >= 2000"},
        display_name="Our 35mm",
    )
    raw = derived.save().read_text(encoding="utf-8")

    parent = load_profile(BASE)
    for trace in (
        BASE, "derived_from", "original_standard_reference", "departures",
        "summary", parent.display_name, str(parent.body),
        str(parent.standard_reference.get("material_category")),
    ):
        assert trace not in raw, f"{trace!r} survives in a saved guideline"

    # And what it does say is that the criteria are local, plus the list of the
    # ones nobody chose.
    assert set(json.loads(raw)["standard_reference"]) == {
        "local", "notice", "inherited",
    }


def test_an_added_metric_takes_its_shape_from_the_guideline_s_own_vocabulary():
    """The trap severing the parent could have walked into.

    A metric needs a capability, an aggregation and a unit to run, and the bodies
    disagree: Metamorfoze's exposure is a plain `dL*` where FADGI's is `dL*2000`,
    and the two are not the same number away from mid-grey. The builder used to ask the parent's `body`
    which vocabulary to offer. With no parent it asks the guideline's own metrics,
    which is inference -- but inference from the object's own contents rather than
    from a table it was once beside.
    """
    from deltachrome.profiles.catalogue import vocabulary_of

    # It recovers the answer on every shipped profile that names a real body.
    # `measurement_only_v1` states `body: "none"` and judges nothing; its metric
    # shapes are the ordinary ones, so the honest answer for it is the majority
    # vocabulary rather than its own placeholder -- which is what it gets.
    for profile_id in available_profiles():
        profile = load_profile(profile_id)
        if profile.body in (None, "none"):
            continue
        assert vocabulary_of(profile) == profile.body, profile_id

    # And on a guideline, which states no body at all.
    derived = load_profile("metamorfoze_full_2025").derive(
        "house_mm", thresholds={"noise": "value < 5"},
    )
    assert derived.body is None
    assert vocabulary_of(derived) == "Metamorfoze"


def test_a_hand_written_local_profile_inherits_nothing(local_dir):
    """The case the reconstruction must not swallow. A local profile stating
    neither ``inherited`` nor ``departures`` was written by hand, line by line,
    so an empty list is the true answer rather than a missing one."""
    import json

    local_dir.mkdir(parents=True, exist_ok=True)
    path = local_dir / "by_hand.json"
    path.write_text(json.dumps({
        "profile_id": "by_hand",
        "display_name": "By Hand",
        "standard_reference": {"local": True},
        "metrics": [{"metric_id": "noise", "requires_capability": "tonal_response",
                     "aggregation": "max", "pass_condition": "value < 3",
                     "units": "std dev"}],
    }), encoding="utf-8")

    loaded = load_profile(path)
    assert loaded.local_criteria
    assert loaded.inherited == []
