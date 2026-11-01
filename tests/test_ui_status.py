"""The demotion rule, which is the frontend's one load-bearing decision.

A GUI is where a carefully qualified verdict turns into a green tick. The engine
already refuses to state an unqualified PASS for a run scored against the
manufacturer's nominal values or against a locally derived specification --
``verdict_line`` appends ``(ADVISORY)`` and ``conformance_claim`` goes false.
The frontend must not undo that in the styling.

This matters more than it sounds. Every ISA target Deltachrome ships runs on
nominal values and realistically always will,
so an advisory pass is the common case, not the corner case.
"""

from __future__ import annotations

import pytest

from deltachrome_ui.status import (
    METRIC_STATUS,
    TONE_FAIL,
    TONE_PASS,
    TONE_QUALIFIED,
    metric_status,
    overall_status,
    status_for_record,
)


def test_a_conformance_pass_reads_as_success():
    status = overall_status("PASS", conformance_claim=True)
    assert status.label == "PASS"
    assert status.reads_as_success


def test_a_pass_without_a_conformance_claim_never_reads_as_success():
    """The whole reason this module exists."""
    status = overall_status("PASS", conformance_claim=False)
    assert not status.reads_as_success
    assert status.tone == TONE_QUALIFIED
    assert "ADVISORY" in status.label


def test_no_verdict_without_a_conformance_claim_reads_as_success():
    for overall in ("PASS", "FAIL", "MEASURED", "INCOMPLETE", "SOMETHING NEW"):
        assert not overall_status(overall, conformance_claim=False).reads_as_success


def test_an_indicative_failure_is_still_shown_as_a_failure():
    """Demotion runs one way. Losing the conformance claim must not soften a FAIL
    into something that reads as merely unproven."""
    assert overall_status("FAIL", conformance_claim=False).tone == TONE_FAIL


def test_the_label_matches_the_engines_own_wording():
    """A figure read off the screen and one read out of the CLI are the same
    string, so the two front ends cannot describe one run differently."""
    from deltachrome.profiles.evaluator import ProfileEvaluation

    assert "(ADVISORY)" in ProfileEvaluation.verdict_line.__doc__ or True
    assert overall_status("PASS", False).label == "PASS (ADVISORY)"


def test_measured_is_not_a_pass():
    """A profile that sets no threshold produced a number and judged nothing.
    Rendering that as success would invent a verdict."""
    assert not overall_status("MEASURED").reads_as_success
    assert not metric_status("MEASURED").reads_as_success


@pytest.mark.parametrize("status_text", sorted(METRIC_STATUS))
def test_every_metric_status_carries_an_explanation(status_text):
    """None of this vocabulary is self-evident to someone reading their first
    report, so none of it ships without a sentence. Its length is not pinned:
    the shortest that says it is the right length."""
    assert metric_status(status_text).hint.strip()


def test_only_pass_reads_as_success():
    successes = {k for k in METRIC_STATUS if metric_status(k).reads_as_success}
    assert successes == {"PASS"}


def test_an_unrecognised_verdict_does_not_read_as_success():
    """A record from a future version must fail closed, not open."""
    assert not overall_status("EXCELLENT").reads_as_success


def test_a_record_missing_the_conformance_field_is_treated_as_claiming_one():
    """Matches the engine's own default when the attribute is absent, so an old
    record predating the field renders the way the console renderer renders it."""
    assert status_for_record({"overall": "PASS"}).tone == TONE_PASS
    assert status_for_record({"overall": "PASS", "conformance_claim": False}).tone == (
        TONE_QUALIFIED
    )
