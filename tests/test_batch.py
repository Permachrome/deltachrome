"""A folder of captures, judged one at a time.

The happy path is the least of it. A batch is unattended, so what is worth pinning
is what happens when a capture is not a capture, when the run stops halfway, and
when a hundred readings agree closely enough that ordinary statistics start calling
every one of them remarkable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deltachrome import batch as B
from deltachrome.metrics.base import MetricResult


def capture(name: str, **readings: float) -> B.BatchCapture:
    """A measured capture carrying the given readings, with FADGI-ish criteria."""
    conditions = {
        "sfr50": ("40 <= value <= 75", "% of half-sampling freq"),
        "gain_modulation_highlight": ("0.7 <= value <= 1.2", "gain"),
        "average_color_accuracy": ("value < 3.5", "dE2000"),
        "sampling_frequency": ("value >= 294", "ppi"),
        "master_file_format": (None, None),
    }
    results = []
    for metric_id, value in readings.items():
        condition, units = conditions.get(metric_id, ("value < 100", None))
        results.append(MetricResult(
            metric_id=metric_id, outcome="evaluated", value=value, units=units,
            passed=True, condition=condition,
        ))
    return B.BatchCapture(
        path=Path(name), outcome="measured", overall="PASS", results=results
    )


# -- what a batch can say that its captures cannot -----------------------------


def test_a_reading_unlike_the_rest_of_its_own_batch_is_reported():
    """The whole reason a batch summary is not just a list of verdicts. Every frame
    here meets the requirement, so no per-image verdict mentions the two that are
    measurably softer -- and softening across a session is exactly what an
    unattended run of production captures is shot to catch."""
    captures = [capture(f"good{i}.tif", sfr50=65.0 + (i % 3) * 0.4) for i in range(18)]
    captures.append(capture("soft.tif", sfr50=52.0))
    found = B.outliers(captures)
    assert [o.capture for o in found] == ["soft.tif"]
    assert found[0].metric_id == "sfr50"
    assert found[0].passed is True, "it met the requirement; that is the point"
    assert "batch median" in found[0].describe()


def test_a_difference_too_small_to_act_on_is_not_reported_however_unusual():
    """Standard deviations measure unusualness and say nothing about importance.

    Incremental gain across twenty captures of one chart reads 0.971 with a spread
    of 0.003, so a frame at 0.941 is eleven standard deviations out and has moved
    six percent of the way across a criterion running from 0.7 to 1.2. Reporting it
    buries the findings that matter under ten that do not.
    """
    captures = [
        capture(f"c{i}.tif", gain_modulation_highlight=0.971 + (i % 2) * 0.002)
        for i in range(19)
    ]
    captures.append(capture("odd.tif", gain_modulation_highlight=0.941))
    assert B.outliers(captures) == []
    # The same reading against a criterion it has actually moved across is kept.
    assert B.outliers(captures, share=0.01)


def test_a_metric_with_no_spread_produces_no_outliers():
    """Bit depth, the container format, a sampling rate on a locked-down rig. Every
    capture agrees exactly, the deviation is zero, and dividing by it would call
    every reading infinitely unusual."""
    captures = [capture(f"c{i}.tif", sampling_frequency=600.0) for i in range(20)]
    assert B.outliers(captures) == []


def test_a_batch_too_small_to_have_a_distribution_reports_nothing():
    """Below about ten, the median absolute deviation is a statement about which
    sample happened to land in the middle. Silence is not a claim that there are no
    outliers; it is a refusal to invent them."""
    captures = [capture(f"c{i}.tif", sfr50=65.0) for i in range(5)]
    captures.append(capture("soft.tif", sfr50=20.0))
    assert B.outliers(captures) == []


def test_a_criterion_naming_no_number_is_judged_on_statistics_alone():
    """A list of permitted colour spaces has no width to weigh a deviation
    against, so there is nothing to hold the reading to beyond its own rarity."""
    assert B._criterion_span(None, 5.0) is None
    assert B._criterion_span("value in ['TIFF', 'JPEG 2000']", 5.0) is None
    # The real shapes of the two non-numeric criteria this project ships. Both
    # carry numbers, and a loose search made the first a threshold of two thousand.
    assert B._criterion_span("TIFF, JPEG 2000, PDF/A", 5.0) is None
    assert B._criterion_span("8 or 16", 8.0) is None
    assert B._criterion_span("0.7 <= value <= 1.2", 0.9) == pytest.approx(0.5)
    assert B._criterion_span("value < 3.5", 2.4) == pytest.approx(1.1)
    assert B._criterion_span("value >= 294", 2083.0) == pytest.approx(1789.0)


# -- what is in the folder -----------------------------------------------------


def test_only_images_are_taken_and_nothing_else_is_called_a_failure(tmp_path):
    """A folder of captures routinely holds a spreadsheet, an ICC profile and a
    .DS_Store. None of those is a capture that failed."""
    for name in ("a.tif", "b.JP2", "c.png", "notes.xlsx", ".DS_Store", "profile.icc"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.tif").write_bytes(b"x")

    assert [p.name for p in B.captures_in(tmp_path)] == ["a.tif", "b.JP2", "c.png"]
    assert [p.name for p in B.captures_in(tmp_path, recursive=True)] == [
        "a.tif", "b.JP2", "c.png", "d.tif"
    ]


def test_a_folder_that_is_not_one_says_so(tmp_path):
    from deltachrome.errors import DeltachromeError

    with pytest.raises(DeltachromeError):
        B.captures_in(tmp_path / "nope")


# -- the run itself ------------------------------------------------------------


def test_a_capture_that_cannot_be_read_does_not_end_the_run(tmp_path):
    """Ninety-nine results and one stated absence, never eleven and a traceback."""
    folder = tmp_path / "captures"
    folder.mkdir()
    for name in ("one.tif", "two.tif", "three.tif"):
        (folder / name).write_bytes(b"not an image at all")

    result = B.run_batch(
        B.captures_in(folder),
        target="colorchecker_sg_v1",
        profile="fadgi_3star_bound_volumes_general_2023",
        into=tmp_path / "out",
    )
    assert len(result.captures) == 3
    assert result.measured == []
    assert len(result.refused) == 3
    for refused in result.refused:
        assert refused.reason, "a refusal without a reason is not a refusal"
    assert "3 could not be measured" in result.summary_line()


def test_the_index_is_written_as_the_run_goes_and_not_at_the_end(tmp_path):
    """A run that dies at frame ninety has to leave a readable account of the
    eighty-nine, and an index written once at the end is the one thing a crash is
    guaranteed to take with it."""
    folder = tmp_path / "captures"
    folder.mkdir()
    for name in ("a.tif", "b.tif", "c.tif"):
        (folder / name).write_bytes(b"x")
    out = tmp_path / "out"
    seen: list[int] = []

    def watch(number, total, path, capture):
        index = json.loads((out / B.INDEX_NAME).read_text())
        seen.append(len(index["captures"]))

    B.run_batch(
        B.captures_in(folder), target="colorchecker_sg_v1",
        profile="fadgi_3star_bound_volumes_general_2023", into=out, on_progress=watch,
    )
    assert seen == [1, 2, 3], "the index grows with the run"


def test_stopping_is_recorded_rather_than_looking_like_a_short_folder(tmp_path):
    folder = tmp_path / "captures"
    folder.mkdir()
    for name in ("a.tif", "b.tif", "c.tif"):
        (folder / name).write_bytes(b"x")
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    result = B.run_batch(
        B.captures_in(folder), target="colorchecker_sg_v1",
        profile="fadgi_3star_bound_volumes_general_2023",
        into=tmp_path / "out", should_stop=stop,
    )
    assert result.stopped_early is True
    assert len(result.captures) == 2
    assert "stopped before the end" in result.summary_line()
    assert json.loads((tmp_path / "out" / B.INDEX_NAME).read_text())["stopped_early"]


def test_a_resumed_run_does_not_repeat_what_it_already_measured(tmp_path):
    """Read from the index rather than from the records present: a record written
    for a capture that was later renamed should not count as that capture's
    result."""
    out = tmp_path / "out"
    out.mkdir()
    (out / B.INDEX_NAME).write_text(json.dumps({
        "captures": [
            {"capture": str(tmp_path / "captures" / "a.tif"),
             "outcome": "measured", "record": "a.json"},
            {"capture": str(tmp_path / "captures" / "b.tif"),
             "outcome": "refused", "record": None},
        ]
    }))
    done = B.already_done(out)
    assert list(done) == [str(tmp_path / "captures" / "a.tif")]
    assert done[str(tmp_path / "captures" / "a.tif")] == "a.json"


def test_an_index_that_cannot_be_read_is_not_a_reason_to_refuse_to_start(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / B.INDEX_NAME).write_text("{ truncated")
    assert B.already_done(out) == {}


# -- what the record is --------------------------------------------------------


def test_a_record_is_named_after_its_capture_and_stays_unique():
    """The first thing anyone does with a batch folder is look for the record
    belonging to a frame they are suspicious of, and a run id does not help them."""
    taken: set[str] = set()
    assert B._record_name(Path("/x/NMNH-001.tif"), taken) == "NMNH-001.json"
    assert B._record_name(Path("/y/NMNH-001.jp2"), taken) == "NMNH-001_2.json"
    assert B._record_name(Path("/z/NMNH-001.png"), taken) == "NMNH-001_3.json"


def test_the_batch_claims_conformance_only_when_every_capture_can(tmp_path):
    measured = capture("a.tif")
    measured.conformance_claim = True
    other = capture("b.tif")
    other.conformance_claim = False
    result = B.BatchResult(
        profile=None, target_id="t", target_name="T", directory=tmp_path,
        captures=[measured], created_utc="now",
    )
    assert result.conformance_claim is True
    result.captures.append(other)
    assert result.conformance_claim is False


def test_a_batch_with_nothing_measured_claims_nothing(tmp_path):
    result = B.BatchResult(
        profile=None, target_id="t", target_name="T", directory=tmp_path,
        captures=[B.BatchCapture(path=Path("a.tif"), outcome="refused", reason="no")],
        created_utc="now",
    )
    assert result.conformance_claim is False


def test_a_resumed_run_summarises_the_whole_folder_and_not_just_this_invocation(tmp_path):
    """A resumed run reports twenty measured; it must also report which of them
    failed and how they differ from each other. Leaving carried-over captures blank
    made a batch where every frame failed read as a batch where none did -- wrong
    in the direction that matters, and avoidable, because everything needed is in
    the record already.
    """
    out = tmp_path / "out"
    out.mkdir()
    record = out / "a.json"
    record.write_text(json.dumps({
        "run_id": "x", "overall": "FAIL", "conformance_claim": False,
        "warnings": ["something to carry"],
        "metrics": [{
            "metric_id": "sfr50", "outcome": "evaluated", "value": 41.0,
            "units": "% of half-sampling freq", "passed": True,
            "condition": "40 <= value <= 75",
        }],
    }))
    carried = B._carried_over(tmp_path / "captures" / "a.tif", record)
    assert carried.measured is True
    assert carried.overall == "FAIL"
    assert carried.conformance_claim is False
    assert carried.reading("sfr50") == pytest.approx(41.0)
    assert carried.warnings == ["something to carry"]


def test_a_carried_over_capture_whose_record_is_gone_is_still_measured(tmp_path):
    """The index says it was measured. Re-measuring would contradict the index and
    inventing readings would be worse, so it carries as measured and says nothing
    about what it read."""
    carried = B._carried_over(tmp_path / "a.tif", tmp_path / "missing.json")
    assert carried.measured is True
    assert carried.overall is None and carried.results == []


def test_a_batch_cannot_be_asked_to_guess_its_target():
    r"""Identification is off in a batch, and this is where that is *enforced*.

    The module says why: on a frame that is mostly subject the search returns a
    confident wrong answer, and unattended that is a wrong answer per capture rather
    than one bad suggestion an operator checks. What makes it unreachable is the
    signature -- ``target`` keyword-only with no default -- so there is no call that
    leaves the chart to be worked out. It reached the desktop entry point once by
    exactly this route, an omission rather than a decision, and every test passed
    because every test supplied one.

    Asserted on the signature rather than by calling, because the property is that
    the call does not *exist*.
    """
    import inspect

    parameter = inspect.signature(B.run_batch).parameters["target"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty, (
        "a default -- None included -- would put a confident wrong answer behind "
        "a forgotten argument"
    )


def test_a_board_in_the_first_frame_is_reported_for_the_run(monkeypatch):
    r"""A board in the first frame is said, once per batch.

    A batch measures one chart per frame, and a folder of boards would otherwise
    be measured one card a frame in silence. So the search runs once, on the first
    capture, and what it finds is a fact about the run rather than about that
    capture -- one search against a whole batch. Asking every capture was an
    option and is deferred past the beta: on the QA-62 board's JPEG 2000
    master it is 7.9 s a frame against a 10 s decode.
    """
    from deltachrome.imaging import detect as D

    monkeypatch.setattr(
        D, "extra_charts", lambda image, target, measured: ["a", "b", "c"],
    )
    notes: list[str] = []
    seen: list[list[str]] = []

    def fake_analyze(image, **kwargs):
        seen.append(list(kwargs.get("extra_warnings", [])))
        raise RuntimeError("stop -- the note is written before this")

    for number in (1, 2):
        notes.clear()
        seen.clear()
        B._one(
            Path("x.tif"), target_def=object(), profile_def=object(), folder=Path("."),
            corners=None, reference_path=None, reference_library=None, taken=set(),
            number=number, total=9, run_options={}, batch_notes=notes,
            load_image=lambda _p: object(),
            analyze=fake_analyze,
            capture_record=lambda runs: runs,
            detect_placement=lambda _i, _t: type(
                "D", (), {"found": True, "corners": [(0, 0)], "reason": "", "notes": []}
            )(),
            detect_placement_near=lambda *a, **k: None,
        )
        assert bool(notes) is (number == 1), "once for the run, not once a capture"
        assert not any(w in ("a", "b", "c") for w in seen[0]), (
            "a capture's own warnings do not carry it"
        )
        if number == 1:
            said = notes[0]

    assert "4 charts" in said, "three unmeasured plus the one that was"
    assert "first capture" in said, "it says how far it looked"
    assert "Turn on" not in said, "it must not point at a control that is gone"


def test_a_failing_first_frame_search_never_costs_the_capture(monkeypatch):
    """The capture has measured correctly by the time this runs, and an advisory
    search raising must not refuse it."""
    from deltachrome.imaging import detect as D

    def explode(*_a, **_k):
        raise ValueError("the search broke")

    monkeypatch.setattr(D, "extra_charts", explode)
    assert B._first_frame_notice(object(), object(), [(0, 0)]) is None


# -- a folder that fails ------------------------------------------------------
#
# The batch had only ever met synthetic bad files. A folder of real
# captures made bad -- painted out, cropped, exposed wrongly, turned, a truncated
# TIFF, a different chart at a different size -- found four refusals nobody could
# read and two placements measured confidently on something that was not the chart.


def _found(corners=((0, 0), (10, 0), (10, 10), (0, 10))):
    return type("D", (), {"found": True, "corners": list(corners), "reason": "",
                          "notes": [], "extent": object()})()


def _refused(reason, extent=object()):
    return type("D", (), {"found": False, "corners": [], "reason": reason,
                          "notes": [], "extent": extent})()


def _run_one(**overrides):
    measured: list = []

    def fake_analyze(image, **kwargs):
        measured.append(True)
        raise RuntimeError("stop here")

    arguments = dict(
        target_def=object(), profile_def=object(), folder=Path("."),
        corners=None, reference_path=None, reference_library=None, taken=set(),
        number=2, total=2, run_options={},
        load_image=lambda _p: object(), analyze=fake_analyze,
        capture_record=lambda runs: runs,
        detect_placement=lambda _i, _t: _found(),
        detect_placement_near=lambda *a, **k: _found(),
    )
    arguments.update(overrides)
    return B._one(Path("x.tif"), **arguments), measured


def test_a_placement_whose_own_ramp_runs_backwards_is_refused_not_measured(monkeypatch):
    """On a board of QA-62 cards the finder returned a quad of marks from several
    cards and the batch measured it: SFR50 16%, a FAIL describing nothing. Its
    ladder read -0.26 where every real placement reads +1.00."""
    from deltachrome.imaging import detect as D

    monkeypatch.setattr(D, "ramp_agreement", lambda *a, **k: -0.26)
    capture, measured = _run_one()
    assert not measured
    assert capture.outcome == "refused"
    assert "not on the chart" in capture.reason and "-0.26" in capture.reason


@pytest.mark.parametrize("reading", [1.0, None])
def test_a_ramp_that_reads_in_order_or_not_at_all_is_measured(monkeypatch, reading):
    """It refuses only on evidence. A chart declaring no ramp is measured as before."""
    from deltachrome.imaging import detect as D

    monkeypatch.setattr(D, "ramp_agreement", lambda *a, **k: reading)
    _capture, measured = _run_one()
    assert measured


def test_a_ramp_that_cannot_be_read_is_not_a_refusal(monkeypatch):
    from deltachrome.imaging import detect as D

    def explode(*_a, **_k):
        raise ValueError("no ramp here")

    monkeypatch.setattr(D, "ramp_agreement", explode)
    _capture, measured = _run_one()
    assert measured


def test_a_seeded_refusal_says_where_it_looked():
    """'Only 0 colored patches found' reads as a fault in the chart; in a batch it
    far more often means the chart is not where the seed says."""
    capture, _ = _run_one(
        corners=[(0, 0), (1, 0), (1, 1), (0, 1)],
        detect_placement_near=lambda *a, **k: _refused("Only 0 colored patches found."),
    )
    assert capture.reason == (
        "Searched where the seed placement puts the chart. Only 0 colored patches found."
    )


def test_a_batch_refusal_does_not_carry_a_class_name():
    from deltachrome.errors import ImageError

    def damaged(_p):
        raise ImageError("x.tif could not be read.")

    capture, _ = _run_one(load_image=damaged)
    assert capture.reason == "x.tif could not be read."
