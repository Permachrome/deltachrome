"""Several captures judged together.

The workflow this exists for: a ColorChecker SG has no slanted edges, a dedicated
resolution chart has no color patches, and a site shoots both in one sitting
because the question is whether the *system* meets the specification -- not whether
either chart did on its own.

So the rules under test are all about combining. Nothing beneath a session changes:
each capture is analyzed exactly as it would be alone.
"""

from __future__ import annotations

import json

import pytest

from deltachrome.errors import ProfileError
from deltachrome.imaging.loader import load_image
from deltachrome.session import (
    PER_CAPTURE_METRICS,
    CaptureOutcome,
    SessionCapture,
    analyze_session,
    load_session_file,
    session_from_records,
)
from fixtures import SyntheticChart
from test_sfr import edge_target, rotated_square_capture

PROFILE = "fadgi_1star_prints_photographs_2023"


@pytest.fixture
def two_captures(tmp_path):
    """One chart with color and no edges, one with edges and no color."""
    chart = SyntheticChart()
    image, corners = chart.render_capture(tmp_path / "sg.tif")
    reference = chart.write_reference(tmp_path / "sg.txt")
    color = SessionCapture(
        image=image, target="colorchecker_sg_v1", corners=corners,
        reference=reference, fiducial_extent_mm=(100.0, 70.0), label="SG",
    )

    edge_path = rotated_square_capture(tmp_path)
    edge_image = load_image(edge_path)
    edges = SessionCapture(
        image=edge_path, target=edge_target(),
        corners=[(0, 0), (edge_image.width, 0),
                 (edge_image.width, edge_image.height), (0, edge_image.height)],
        label="edges",
    )
    return color, edges


# -- a target with nothing to compare --------------------------------------


def test_a_chart_with_no_patches_can_be_analysed_at_all(tmp_path):
    """A dedicated resolution chart is slanted edges and a scale and nothing else.
    Requiring reference data for it would be requiring data about nothing, and it
    made the whole class of target unusable."""
    from deltachrome.pipeline import analyze

    path = rotated_square_capture(tmp_path)
    image = load_image(path)
    result = analyze(
        path, target=edge_target(), profile=PROFILE,
        corners=[(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
    )
    assert result.run.patches == []
    assert any(r.metric_id == "sfr50" and r.outcome == "evaluated"
               for r in result.evaluation.results)
    assert any(r.metric_id == "average_color_accuracy" and r.outcome == "not_applicable"
               for r in result.evaluation.results)


def test_a_reference_supplied_for_a_patchless_target_is_reported_as_unused(tmp_path):
    from deltachrome.pipeline import analyze

    chart = SyntheticChart()
    reference = chart.write_reference(tmp_path / "ref.txt")
    path = rotated_square_capture(tmp_path)
    image = load_image(path)
    result = analyze(
        path, target=edge_target(), reference_path=reference, profile=PROFILE,
        corners=[(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
    )
    assert any("was not read" in w for w in result.warnings)


# -- coverage is unioned ----------------------------------------------------


def test_two_partial_captures_cover_more_than_either_alone(two_captures):
    color, edges = two_captures
    session = analyze_session([color, edges], profile=PROFILE)

    by_id = {m.metric_id: m for m in session.metrics}
    assert by_id["average_color_accuracy"].summary()["covered_by"] == ["SG"]
    assert by_id["sfr50"].summary()["covered_by"] == ["edges"]
    # Neither chart reaches both; together they do.
    assert by_id["average_color_accuracy"].summary()["not_measurable_by"] == ["edges"]
    assert by_id["sfr50"].summary()["not_measurable_by"] == ["SG"]


def test_what_no_capture_reached_is_named(two_captures):
    """The one thing a session can say that a single run cannot: this gap is a
    choice about which charts were shot, and another target would close it."""
    color, edges = two_captures
    session = analyze_session([color, edges], profile=PROFILE)

    assert "color_channel_misregistration" in session.uncovered
    assert not session.covers_profile
    assert "measured by no capture" in session.explanation
    assert "adding a target that carries the feature" in session.explanation


def test_absence_still_never_fails(two_captures):
    """Same rule as a single run, deliberately. A session that penalised uncovered
    metrics would penalise a site for the charts it happens to own."""
    color, edges = two_captures
    session = analyze_session([color, edges], profile=PROFILE)
    uncovered = {m.metric_id for m in session.metrics if not m.covered}
    assert uncovered
    assert not any(m.metric_id in uncovered for m in session.failures)


# -- overlap ---------------------------------------------------------------


def test_where_both_measure_a_metric_both_are_reported(two_captures):
    color, edges = two_captures
    session = analyze_session([color, edges], profile=PROFILE)
    depth = next(m for m in session.metrics if m.metric_id == "bit_depth")
    assert set(depth.summary()["covered_by"]) == {"SG", "edges"}
    assert len(depth.summary()["values"]) == 2


def test_a_failure_on_either_capture_governs(two_captures):
    """A mean would pass a system one chart says is failing."""
    color, edges = two_captures
    session = analyze_session([color, edges], profile="fadgi_4star_prints_photographs_2023")
    for metric in session.failures:
        assert metric.status == "FAIL"
        assert metric.governing[1].passed is False
    if session.failures:
        assert session.overall == "FAIL"


# -- file properties are not disagreements ---------------------------------


def test_per_capture_metrics_are_not_reported_as_disagreements(two_captures):
    """Two captures differ in sampling rate by design -- a reflective chart is shot
    at the object plane, a transmissive one at high magnification -- so calling that
    a disagreement would bury the real ones."""
    color, edges = two_captures
    session = analyze_session([color, edges], profile=PROFILE)
    for metric in session.metrics:
        if metric.metric_id in PER_CAPTURE_METRICS:
            assert metric.disagreement is None
            assert metric.per_capture is True


def test_a_system_metric_measured_twice_reports_its_spread(two_captures):
    color, edges = two_captures
    session = analyze_session([color, edges], profile=PROFILE)
    shared = [m for m in session.metrics
              if len(m.summary()["values"]) > 1 and not m.per_capture]
    for metric in shared:
        values = list(metric.summary()["values"].values())
        assert metric.disagreement == pytest.approx(max(values) - min(values))


# -- the record and the file -----------------------------------------------


def test_the_session_record_carries_every_capture_and_the_verdict(two_captures):
    color, edges = two_captures
    data = analyze_session([color, edges], profile=PROFILE).to_dict()
    assert [c["name"] for c in data["captures"]] == ["SG", "edges"]
    assert data["overall"] in {"PASS", "FAIL", "INCOMPLETE", "MEASURED"}
    assert data["covers_profile"] is False
    assert data["uncovered"]
    assert len(data["metrics"]) >= 18


def test_a_session_file_resolves_paths_against_itself(tmp_path):
    """So a session stays valid when it is re-run from another directory -- which is
    the point of writing one down."""
    chart = SyntheticChart()
    image, corners = chart.render_capture(tmp_path / "sg.tif")
    chart.write_reference(tmp_path / "sg.txt")
    (tmp_path / "session.json").write_text(json.dumps({
        "profile": PROFILE,
        "captures": [{
            "image": "sg.tif", "target": "colorchecker_sg_v1", "reference": "sg.txt",
            "corners": " ".join(f"{x},{y}" for x, y in corners),
        }],
    }), encoding="utf-8")

    captures, profile_id, _ = load_session_file(tmp_path / "session.json")
    assert profile_id == PROFILE
    assert captures[0].image == tmp_path / "sg.tif"
    assert captures[0].reference == tmp_path / "sg.txt"
    assert len(captures[0].corners) == 4


def test_a_session_file_missing_what_it_needs_says_so(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"profile": PROFILE}), encoding="utf-8")
    with pytest.raises(ProfileError, match="needs 'profile' and 'captures'"):
        load_session_file(path)

    path.write_text(json.dumps({
        "profile": PROFILE, "captures": [{"image": "a.tif"}]
    }), encoding="utf-8")
    with pytest.raises(ProfileError, match=r"missing \['corners', 'target'\]"):
        load_session_file(path)


def test_corners_can_be_given_as_text_or_as_pairs(tmp_path):
    path = tmp_path / "s.json"
    for corners in ("1,2 3,4 5,6 7,8", [[1, 2], [3, 4], [5, 6], [7, 8]]):
        path.write_text(json.dumps({
            "profile": PROFILE,
            "captures": [{"image": "a.tif", "target": "t", "corners": corners}],
        }), encoding="utf-8")
        captures, _, _ = load_session_file(path)
        assert captures[0].corners == [(1, 2), (3, 4), (5, 6), (7, 8)]


def test_a_session_needs_at_least_one_capture():
    with pytest.raises(ProfileError, match="at least one capture"):
        analyze_session([], profile=PROFILE)


# -- combining runs that have already been saved ----------------------------
#
# The workflow this is for: analyze each chart as it comes, then ask at the end
# whether the *system* passed. Every number a session reduces was written into
# each record when the run was made, so combining them needs no pixels and no
# re-measurement -- which is what makes it the frontend's path and not a second
# capture pipeline.


def _records(captures, profile=PROFILE):
    """Analyze each capture and return its record as JSON would deliver it."""
    from deltachrome.pipeline import analyze

    out = []
    for capture in captures:
        run = analyze(
            capture.image, target=capture.target, reference_path=capture.reference,
            profile=profile, corners=capture.corners,
            fiducial_extent_mm=capture.fiducial_extent_mm,
        )
        out.append(json.loads(json.dumps(run.run.to_dict())))
    return out


def test_a_session_rebuilt_from_records_matches_one_built_from_runs(two_captures):
    """The load-bearing claim. If these could drift apart there would be two
    session verdicts for one set of captures, and no way to know which was quoted."""
    color, edges = two_captures
    live = analyze_session([color, edges], profile=PROFILE)
    rebuilt = session_from_records(_records([color, edges]))

    assert rebuilt.overall == live.overall
    assert rebuilt.covers_profile == live.covers_profile
    assert rebuilt.conformance_claim == live.conformance_claim
    assert rebuilt.uncovered == live.uncovered
    assert rebuilt.incomplete == live.incomplete
    assert rebuilt.coverage_line == live.coverage_line

    # Statuses metric for metric. Names differ -- a session file's labels are not
    # persisted in a run record -- so the comparison is on ids.
    def statuses(session):
        return {m.metric_id: m.status for m in session.metrics}

    assert statuses(rebuilt) == statuses(live)


def test_the_readings_come_back_too_and_not_only_the_aggregates(two_captures):
    """A session that could show only the aggregate would be unable to say *where*
    two charts disagree, which is most of what a session is for."""
    color, edges = two_captures
    rebuilt = session_from_records(_records([color, edges]))
    colour = next(m for m in rebuilt.metrics
                  if m.metric_id == "average_color_accuracy")
    _, result = colour.governing
    assert len(result.per_patch) == 140
    assert result.units and result.condition


def test_runs_judged_against_different_profiles_are_refused(two_captures):
    """The mistake that is easiest to make from a library holding a year of runs
    against a dozen profiles -- and it is not a visible one: pooling readings scored
    against different thresholds gives a verdict belonging to no standard."""
    color, edges = two_captures
    records = (_records([color], profile=PROFILE)
               + _records([edges], profile="fadgi_4star_prints_photographs_2023"))
    with pytest.raises(ProfileError, match="different profiles"):
        session_from_records(records)


def test_an_indicative_run_is_not_upgraded_by_being_read_back(two_captures):
    """A saved run has neither its profile object nor its reference file to hand, so
    the claim has to be honoured as recorded rather than re-derived. Re-deriving it
    would default to True and quietly promote every nominal-reference run."""
    color, _ = two_captures
    record = _records([color])[0]
    record["conformance_claim"] = False
    session = session_from_records([record])
    assert session.conformance_claim is False


def test_a_session_needs_at_least_one_record():
    with pytest.raises(ProfileError, match="at least one capture"):
        session_from_records([])


def test_a_record_naming_no_profile_needs_one_supplied(two_captures):
    color, _ = two_captures
    record = _records([color])[0]
    record["profile"] = {}
    with pytest.raises(ProfileError, match="names the profile"):
        session_from_records([record])
    assert session_from_records([record], profile=PROFILE).overall


def test_duplicate_capture_names_are_kept_apart(two_captures):
    """Two runs of one capture against one profile is an ordinary thing to have in a
    library -- a placement refined and re-run. Collapsing them onto one name would
    make the coverage columns lie about which reading came from where."""
    color, _ = two_captures
    records = _records([color]) * 2
    session = session_from_records(records)
    assert len({c.name for c in session.captures}) == 2


# -- the two kinds of gap ---------------------------------------------------


def test_a_metric_no_chart_carries_is_separated_from_one_that_failed(two_captures):
    """Both leave a metric unmeasured and they call for opposite actions: buy a
    target, or fix the capture. Telling an operator to buy a chart when the
    placement was wrong is the mistake this split exists to prevent."""
    color, edges = two_captures
    session = analyze_session([color, edges], profile=PROFILE)
    assert set(session.not_carried) <= set(session.uncovered)
    assert not (set(session.not_carried) & set(session.incomplete))
    for metric_id in session.not_carried:
        assert "adding a target" in session.explanation
    for metric_id in session.incomplete:
        assert metric_id not in session.explanation.split("measured by no capture")[-1]


# -- reaching it from the command line --------------------------------------


def test_the_cli_combines_saved_runs_without_a_session_file(two_captures, tmp_path, capsys):
    """The terminal has to reach what the frontend reaches. A session file describes
    captures to *measure*; --from-runs combines measurements already made, which is
    the same question asked of work already done."""
    from deltachrome.cli import main

    color, edges = two_captures
    paths = []
    for index, record in enumerate(_records([color, edges])):
        path = tmp_path / f"run{index}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(str(path))

    # Exit 3 is INCOMPLETE: these two charts between them do not cover the whole
    # profile, which is the ordinary case and the reason sessions exist.
    assert main(["session", "--from-runs", *paths]) in (0, 2, 3)
    out = capsys.readouterr().out
    assert "session of 2 capture(s)" in out
    assert "METRICS ACROSS CAPTURES" in out


def test_the_cli_refuses_a_session_file_and_saved_runs_at_once(tmp_path, capsys):
    """They answer the same question from opposite ends and combining them has no
    meaning -- one describes captures to measure, the other measurements made."""
    from deltachrome.cli import main

    assert main(["session", str(tmp_path / "s.json"),
                 "--from-runs", str(tmp_path / "r.json")]) != 0
    assert "not both" in capsys.readouterr().err


def test_the_cli_asks_for_one_or_the_other(capsys):
    from deltachrome.cli import main

    assert main(["session"]) != 0
    assert "--from-runs" in capsys.readouterr().err


def test_records_of_one_capture_can_be_named(tmp_path):
    """The default cannot serve a board: every chart of one capture carries the
    same filename, so it names them ``board.tif``, ``board.tif (1)``,
    ``board.tif (2)`` -- which distinguishes them and identifies none of them.

    ``labels`` is the same parameter ``session_from_results`` already takes, and
    the two are described as twins, so the asymmetry was the accident.
    """
    from deltachrome.session import session_from_records

    def record(run_id: str) -> dict:
        return {
            "run_id": run_id,
            "image": {"filename": "board.tif"},
            "profile": {"profile_id": "fadgi_3star_documents_unbound_general_2023"},
            "target": {"target_id": "t", "display_name": "T"},
            "metrics": [],
            "overall": "PASS",
        }

    records = [record("a"), record("b"), record("c")]

    bare = session_from_records(records)
    assert [c.name for c in bare.captures] == [
        "board.tif", "board.tif (1)", "board.tif (2)"
    ]

    named = session_from_records(
        records, labels=["Target 1", "Target 2", "Target 3"])
    assert [c.name for c in named.captures] == ["Target 1", "Target 2", "Target 3"]

    # Short or missing labels fall back rather than raising.
    partial = session_from_records(records, labels=["Target 1"])
    assert partial.captures[0].name == "Target 1"
    assert partial.captures[1].name == "board.tif"
