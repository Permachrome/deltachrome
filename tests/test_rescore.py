"""Judging a capture that has already been read, against a different profile.

The question is ordinary and used to cost a whole re-analysis: *it failed at four
star -- does it pass at three?* Or the plainer version, which is choosing the
wrong material category and finding out from the report.

What these tests pin is the one claim that makes it safe. A re-score reuses the
**reading** and re-runs the **judgement**, and those are different things: every
metric plugin runs again on the new profile's own algorithm parameters, so a
profile that reduces a field of edges differently gets its own answer rather than
the previous profile's answer compared against a new threshold. What it may never
do is change what the capture said.
"""

from __future__ import annotations

import pytest

from deltachrome.pipeline import analyze, rescore
from fixtures import SyntheticChart

SG = "colorchecker_sg_v1"
FADGI3 = "fadgi_3star_documents_unbound_general_2023"
FADGI4 = "fadgi_4star_documents_unbound_general_2023"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rescore")
    chart = SyntheticChart()
    reference = chart.write_reference(tmp / "ref.cgats.txt")
    image, corners = chart.render_capture(tmp / "capture.tif", skew=18)
    return analyze(
        image, target=SG, reference_path=reference, profile=FADGI3, corners=corners
    )


def test_the_readings_survive_a_change_of_profile(run):
    """The measurement is what the capture said, and no profile has a say in it."""
    again = rescore(run, FADGI4)
    assert again.run.patches == run.run.patches
    assert again.run.geometry == run.run.geometry
    assert again.run.color_management == run.run.color_management


def test_the_verdict_is_the_new_profiles(run):
    again = rescore(run, FADGI4)
    assert again.run.profile["profile_id"] == FADGI4
    assert run.run.profile["profile_id"] == FADGI3
    conditions = {
        m["metric_id"]: m.get("condition") for m in again.run.metrics
    }
    was = {m["metric_id"]: m.get("condition") for m in run.run.metrics}
    # Four star is stricter than three on colour accuracy, so at least one
    # threshold must actually have moved -- otherwise this test would pass on a
    # re-score that quietly did nothing.
    assert conditions["average_color_accuracy"] != was["average_color_accuracy"]


def test_it_is_a_new_run_rather_than_an_edit_of_the_old_one(run):
    """Two profiles applied to one capture are two findings, and the archive
    holds both. Overwriting the first would make the library's history depend on
    which question happened to be asked last."""
    again = rescore(run, FADGI4)
    assert again.run.run_id != run.run.run_id
    assert run.run.profile["profile_id"] == FADGI3
    assert run.evaluation.profile.profile_id == FADGI3


def test_the_measurement_travels_so_it_can_be_judged_a_third_time(run):
    """Chaining must not degrade. A re-score of a re-score is judged from the
    same reading as the first, not from the second's output."""
    four = rescore(run, FADGI4)
    back = rescore(four, FADGI3)
    assert back.run.overall == run.run.overall
    assert [m["value"] for m in back.run.metrics] == [
        m["value"] for m in run.run.metrics
    ]


def test_one_profiles_coverage_note_does_not_carry_into_the_next(run):
    """The warnings list is copied rather than appended to.

    A note about the reference file belongs to the reading and must survive every
    verdict made from it; a note about what a profile requires and this chart
    cannot measure belongs to that profile and must not.
    """
    warnings = list(run.run.warnings)
    for _ in range(3):
        rescore(run, FADGI4)
    assert list(run.run.warnings) == warnings
    assert len(rescore(run, FADGI3).run.warnings) == len(warnings)


def test_a_run_read_back_off_disk_refuses_rather_than_guessing():
    """A saved record holds every number and no pixels.

    It can be reopened and combined into a session, and it cannot be judged
    again -- because judging runs the metrics, and the metrics need the capture.
    Saying so is the difference between a refusal and a wrong answer.
    """
    from deltachrome.pipeline import AnalysisResult

    stub = AnalysisResult.__new__(AnalysisResult)
    stub.measurement = None
    with pytest.raises(ValueError, match="saved record"):
        rescore(stub, FADGI4)


def test_a_run_option_belonging_to_the_run_is_reapplied(tmp_path):
    """``--percentile-method`` is a convention of the run, not of the profile.

    A re-score that dropped it would answer the new question under a different
    convention from the run it replaces, and nothing on the report would say so.
    """
    chart = SyntheticChart()
    reference = chart.write_reference(tmp_path / "ref.cgats.txt")
    image, corners = chart.render_capture(tmp_path / "capture.tif", skew=18)
    first = analyze(
        image, target=SG, reference_path=reference, profile=FADGI3,
        corners=corners, percentile_method="nearest_rank",
    )
    again = rescore(first, FADGI4)
    metric = next(
        m for m in again.run.metrics
        if m["metric_id"] == "color_accuracy_90th_percentile"
    )
    assert metric["metadata"]["percentile_method"] == "nearest_rank"
