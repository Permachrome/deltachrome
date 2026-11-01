"""A department's own written standard, and being able to meet it.

A run loses its conformance claim two ways, and until now they were one bit:

* the **reference** is not a measurement of the chart in front of the camera, so
  the specimen's own print tolerance sits inside every delta-E;
* the **criteria** are a local specification derived from a standard rather than
  the standard itself.

Folding them together meant a department that used the guideline builder -- which
exists so a site can write down the specification it works to -- could never see an
unqualified verdict, however well the capture was measured. Which taught the
qualification to mean nothing: a tone that is always on is a tone nobody reads.

They are not the same kind of thing. A nominal reference is a fact about the
evidence and no decision makes it untrue. A derived guideline is a decision,
recorded, named on the report and listed departure by departure -- and the run
genuinely met it. So the verdict stands and says what it is a verdict *about*.

``conformance_claim`` itself is unchanged and still withheld for either, because
conformance means what the standard says it means.
"""

from __future__ import annotations

import pytest

from deltachrome.profiles.evaluator import ProfileEvaluation, advisory_line
from deltachrome.profiles.schema import load_profile
from deltachrome.reference.cgats import (
    MANUFACTURER_NOMINAL,
    PER_UNIT_MEASUREMENT,
    POPULATION_MEAN,
)

STANDARD = "fadgi_3star_bound_volumes_general_2023"


def evaluation(*, derived: bool, basis: str) -> ProfileEvaluation:
    profile = load_profile(STANDARD)
    if derived:
        profile = profile.derive(
            "house_spec", thresholds={"white_balance": "value < 5"}
        )
    return ProfileEvaluation(profile=profile, results=[], reference_basis=basis)


# -- the two caveats are different things --------------------------------------


def test_a_local_specification_alone_does_not_make_a_run_advisory():
    """The correction. The chart was measured, so the reading stands; what it is
    not is a claim against the standard the specification came from."""
    local = evaluation(derived=True, basis=PER_UNIT_MEASUREMENT)
    assert local.evidence_is_sound is True
    assert local.criteria_caveat is not None
    assert local.reference_caveat is None
    assert local.conformance_claim is False, "still not a conformance test"
    assert local.verdict_line == local.overall
    assert "ADVISORY" not in local.verdict_line


def test_a_nominal_reference_makes_a_run_advisory_with_or_without_one():
    """The half that must not move. A run is only as good as what it was compared
    with, and on nominal data that is the model rather than the chart."""
    for derived in (False, True):
        nominal = evaluation(derived=derived, basis=MANUFACTURER_NOMINAL)
        assert nominal.evidence_is_sound is False
        assert nominal.reference_caveat is not None
        assert nominal.verdict_line.endswith("(ADVISORY)")


def test_a_batch_mean_is_evidence_about_the_model_and_so_still_advisory():
    """Centred in its population and still not a measurement of this chart."""
    mean = evaluation(derived=False, basis=POPULATION_MEAN)
    assert mean.evidence_is_sound is False
    assert mean.verdict_line.endswith("(ADVISORY)")


def test_a_standard_run_on_a_measured_chart_is_qualified_by_nothing():
    clean = evaluation(derived=False, basis=PER_UNIT_MEASUREMENT)
    assert clean.conformance_claim is True
    assert clean.conformance_caveats == []
    assert clean.verdict_line == clean.overall


# -- and the record does not change --------------------------------------------


def test_the_caveats_the_record_stores_are_unchanged():
    """``conformance_caveats`` is what a year of saved runs carries and what a
    reopened one shows. Splitting how the caveats are *derived* must not change
    how many there are or which fact each one carries."""
    both = evaluation(derived=True, basis=MANUFACTURER_NOMINAL)
    assert len(both.conformance_caveats) == 2
    assert both.conformance_caveats == [both.reference_caveat, both.criteria_caveat]
    assert "nominal values" in both.conformance_caveats[0]
    assert "local guideline" in both.conformance_caveats[1]
    assert both.conformance_claim is False


def test_the_criteria_caveat_names_the_guideline_and_not_its_parent():
    """A saved guideline is its own specification. Naming the standard it was
    derived from puts a reader back in touch with a table the operator may have
    started from only because it was convenient -- and the departure sentence that
    used to follow it ("noise relaxed from 'value < 3' to 'value < 9'") means
    nothing except beside that table. The record keeps both."""
    both = evaluation(derived=True, basis=PER_UNIT_MEASUREMENT)
    caveat = both.criteria_caveat
    assert caveat is not None
    assert STANDARD not in caveat
    assert "local guideline" in caveat


# -- what a reader is told -----------------------------------------------------


def test_the_caveat_line_says_which_kind_of_qualification_this_is():
    local = advisory_line(reference_basis=PER_UNIT_MEASUREMENT, local_criteria=True)
    nominal = advisory_line(reference_basis=MANUFACTURER_NOMINAL, local_criteria=False)
    assert local is not None and nominal is not None
    assert local.startswith("Local guideline:")
    assert STANDARD not in local
    assert nominal.startswith("Advisory:")
    # A run carrying both is advisory: the weaker fact governs.
    assert advisory_line(
        reference_basis=MANUFACTURER_NOMINAL, local_criteria=True
    ).startswith("Advisory:")


def test_the_explanation_leads_with_the_kind_rather_than_always_with_advisory():
    local = evaluation(derived=True, basis=PER_UNIT_MEASUREMENT)
    assert local.overall_explanation.startswith("Not a conformance test against")
    nominal = evaluation(derived=False, basis=MANUFACTURER_NOMINAL)
    assert nominal.overall_explanation.startswith("Advisory, not a conformance result")


@pytest.mark.parametrize("claim,sound,expected", [
    (True, True, ""),
    (False, True, "AGAINST A LOCAL GUIDELINE"),
    (False, False, "ADVISORY"),
])
def test_the_console_verdict_qualifies_itself_by_kind(claim, sound, expected):
    """And it says what the verdict is AGAINST rather than that it was met --
    "MEETS A LOCAL GUIDELINE" beside the word FAIL says the opposite of what
    happened."""
    from deltachrome.reporting.console import _verdict_suffix

    suffix = _verdict_suffix(claim=claim, evidence_sound=sound)
    assert expected in suffix
    assert "MEETS" not in suffix
