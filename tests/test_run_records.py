"""Reading a run record back.

The project's raw-data-first rule says per-measurement values are persisted so an
old capture can be re-scored without re-shooting it. That was only half true: the
engine could write a record and nothing could read one back into the objects the
report renders. These cover the return trip, and in particular the one thing that
must survive it -- a verdict that cannot be quoted as conformance.
"""

from __future__ import annotations

from pathlib import Path

from deltachrome.results.models import (
    SCHEMA_VERSION,
    AnalysisRun,
    EdgeRecord,
    PatchRecord,
    file_fingerprint,
)


def minimal(**overrides):
    record = {
        "run_id": "abc123", "created_utc": "2026-01-01T00:00:00+00:00",
        "image": {"path": "/m/c.tif", "filename": "c.tif"},
        "target": {"target_id": "t"}, "reference": {"basis": "measured"},
        "profile": {"profile_id": "p"}, "geometry": {}, "sampling": {},
        "color_management": {}, "patches": [], "edges": [],
        "edge_field_summary": {}, "registration": [], "registration_summary": {},
        "rulers": {}, "metrics": [], "overall": "PASS", "overall_explanation": "",
    }
    record.update(overrides)
    return record


def _run(run_id: str) -> AnalysisRun:
    """One ordinary run record, for the capture tests below."""
    return AnalysisRun(**minimal(run_id=run_id))


def test_a_record_survives_the_round_trip(tmp_path):
    patch = PatchRecord(
        patch_id="A1", row=0, col=0, role="chromatic", ramp_position=None,
        rgb_mean=[0.5, 0.4, 0.3], rgb_median=[0.5, 0.4, 0.3],
        rgb_std=[0.01, 0.01, 0.01], measured_lab=[50.0, 1.0, -2.0],
        reference_lab=[51.0, 1.2, -2.1], pixel_count=900,
        clipped_high_fraction=0.0, clipped_low_fraction=0.0,
        flags=["mean_median_divergence"], roi_px=[[1, 2], [3, 2], [3, 4], [1, 4]],
    )
    run = AnalysisRun(**minimal(patches=[patch]) | {"patches": [patch]})
    path = run.save(tmp_path / "run.json")

    back = AnalysisRun.load(path)
    assert back.run_id == run.run_id
    assert back.overall == "PASS"
    assert len(back.patches) == 1
    # Rebuilt as records, not left as dictionaries: the report reaches for
    # attributes, and a dict would fail at the point of rendering rather than here.
    assert isinstance(back.patches[0], PatchRecord)
    assert back.patches[0].flags == ["mean_median_divergence"]
    assert back.patches[0].roi_px[2] == [3, 4]


def test_the_reason_a_verdict_cannot_be_quoted_survives():
    """The whole point of persisting the caveats. A reopened run showing
    'PASS (ADVISORY)' that cannot say why is worse than one that will not open:
    the reader sees a qualification with no way to weigh it."""
    caveats = [
        "judged against the manufacturer's nominal values for this model of chart, "
        "not against a measurement of the chart itself"
    ]
    run = AnalysisRun.from_dict(minimal(
        conformance_claim=False, conformance_caveats=caveats))
    assert run.conformance_claim is False
    assert run.conformance_caveats == caveats


def test_an_older_record_opens_rather_than_raising():
    """Schema 3 predates both new fields. These files are the archive; a library
    that refuses to list its own history is worse than one that lists it thinly."""
    old = minimal(schema_version=3)
    old.pop("registration_summary")
    run = AnalysisRun.from_dict(old)
    assert run.overall == "PASS"
    assert run.conformance_caveats == []
    assert run.registration_summary == {}


def test_a_field_from_a_newer_engine_is_dropped_not_fatal():
    run = AnalysisRun.from_dict(minimal(some_future_field={"x": 1}))
    assert run.overall == "PASS"
    assert not hasattr(run, "some_future_field")


def test_edges_rebuild_with_their_curves():
    edge = {
        "edge_id": "e1", "region": "r", "orientation": "vertical",
        "center_px": [10.0, 20.0], "center_relative": [0.1, 0.2],
        "bounds_px": [0, 0, 10, 10], "edge_angle_deg": 5.0,
        "phase_coverage_px": 4.0, "fit_residual_px": 0.1, "rows_used": 100,
        "linearized": True, "binning": "distance",
        "sampling_efficiency_pct": 90.0, "sfr50_pct": 55.0,
        "mtf_at_nyquist": 0.2, "peak_modulation": 1.0, "peak_frequency": 0.05,
        "overshoot_pct": 0.0,
        "mtf_curve": {"frequency_cyc_px": [0.0, 0.25], "mtf": [1.0, 0.6]},
    }
    run = AnalysisRun.from_dict(minimal(edges=[edge]))
    assert isinstance(run.edges[0], EdgeRecord)
    assert run.edges[0].mtf_curve["mtf"] == [1.0, 0.6]


def test_the_schema_version_was_raised_with_the_new_fields():
    assert SCHEMA_VERSION >= 4


def test_run_options_survive_the_round_trip():
    """``original_size``, ``percentile_method`` and ``frequency_units`` were read
    into ``context.extras`` at analysis time and never written back out -- so a
    record built by ``analyze`` could not say what it had been overridden with,
    and a reopened run could only be re-established under the profile's bare
    default. They are top-level fields now, and a record from before this simply
    omits them rather than claiming a wrong default."""
    run = AnalysisRun.from_dict(minimal(
        original_size="A2", percentile_method="nearest_rank",
        frequency_units="lp/mm",
    ))
    assert run.original_size == "A2"
    assert run.percentile_method == "nearest_rank"
    assert run.frequency_units == "lp/mm"

    restored = AnalysisRun.from_dict(run.to_dict())
    assert restored.original_size == "A2"
    assert restored.percentile_method == "nearest_rank"
    assert restored.frequency_units == "lp/mm"

    older = AnalysisRun.from_dict(minimal())
    assert older.original_size is None
    assert older.percentile_method is None
    assert older.frequency_units is None


def test_a_metrics_written_form_survives_the_record(tmp_path):
    """``value_text`` is how a reopened run knows its master was JPEG 2000 and
    not 0.000. A record written before it simply omits the key, and a metric
    without one is formatted from its number exactly as before."""
    from deltachrome.metrics.base import MetricResult

    entry = {"metric_id": "master_file_format", "outcome": "evaluated",
             "value": None, "value_text": "JPEG 2000", "passed": True,
             "condition": "TIFF, JPEG 2000"}
    assert MetricResult.from_record(entry).reading == "JPEG 2000"

    older = {"metric_id": "noise", "outcome": "evaluated", "value": 1.5}
    assert MetricResult.from_record(older).value_text is None
    assert MetricResult.from_record(older).reading == "1.500"


# -- recognizing a capture that moved ---------------------------------------


def test_a_fingerprint_identifies_the_same_file_after_a_rename(tmp_path):
    original = tmp_path / "capture.tif"
    original.write_bytes(b"\x00\x01" * 5000)
    before = file_fingerprint(original)
    moved = tmp_path / "elsewhere" / "renamed.tif"
    moved.parent.mkdir()
    original.rename(moved)
    assert file_fingerprint(moved) == before


def test_different_captures_fingerprint_differently(tmp_path):
    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    a.write_bytes(b"\x00" * 4096)
    b.write_bytes(b"\x00" * 4095 + b"\x01")
    assert file_fingerprint(a) != file_fingerprint(b)


def test_a_missing_file_fingerprints_to_nothing_rather_than_raising(tmp_path):
    """Called while building a record; a capture read from a stream that has since
    gone is not a reason to lose the measurements."""
    assert file_fingerprint(tmp_path / "absent.tif") is None


def test_the_fingerprint_does_not_read_the_whole_file(tmp_path):
    """A preservation master runs to hundreds of megabytes and a library lists
    dozens of runs. Reading everything to notice a file moved is not affordable."""
    from deltachrome.results.models import FINGERPRINT_CHUNK

    big = tmp_path / "big.bin"
    big.write_bytes(b"\x00" * (FINGERPRINT_CHUNK * 3))
    reads: list[int] = []
    real_open = type(big).open

    def counting_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        real_read = handle.read

        def read(size=-1):
            data = real_read(size)
            reads.append(len(data))
            return data

        handle.read = read
        return handle

    import pathlib

    pathlib.Path.open = counting_open
    try:
        assert file_fingerprint(big) is not None
    finally:
        pathlib.Path.open = real_open
    assert sum(reads) <= FINGERPRINT_CHUNK * 2


# -- the archive outlives the vocabulary ------------------------------------


def test_a_record_written_before_the_spelling_settled_still_opens():
    """Renaming a field in the code cannot make a year of saved runs unreadable,
    and the failure would not have been graceful: the unknown key is dropped, the
    required argument is then missing, and the record raises a TypeError naming a
    field the file does not contain."""
    edge = {
        "edge_id": "e0", "region": "r", "orientation": "vertical",
        "centre_px": [10.0, 20.0], "centre_relative": [0.1, 0.2],
        "bounds_px": [0, 0, 10, 10], "edge_angle_deg": 5.0,
        "phase_coverage_px": 4.0, "fit_residual_px": 0.1, "rows_used": 100,
        "linearised": True, "binning": "distance",
        "sampling_efficiency_pct": 90.0, "sfr50_pct": 55.0,
        "mtf_at_nyquist": 0.2, "peak_modulation": 1.0, "peak_frequency": 0.05,
        "overshoot_pct": 0.0,
    }
    run = AnalysisRun.from_dict(minimal(edges=[edge], schema_version=4))
    assert run.edges[0].center_px == [10.0, 20.0]
    assert run.edges[0].center_relative == [0.1, 0.2]
    assert run.edges[0].linearized is True


def test_a_target_definition_written_before_the_spelling_settled_still_loads():
    """A definition is a file a user can write. One using the older key would
    otherwise declare a region with no geometry, which reads as a chart that
    prints no such feature rather than as a file that needs editing."""
    from deltachrome.targets.schema import _geometry_from_dict

    assert _geometry_from_dict({"centre": [0.5, 0.5], "half_size": 0.1}) == {
        "center": [0.5, 0.5], "half_size": 0.1,
    }
    assert _geometry_from_dict({"center": [0.5, 0.5]}) == {"center": [0.5, 0.5]}


def test_the_schema_version_records_the_rename():
    assert SCHEMA_VERSION >= 5


# -- one record per capture ---------------------------------------------------


def test_a_capture_record_holds_its_charts_whole_and_unaltered():
    """The container is additive. Every chart inside is an ordinary run record,
    serialised exactly as it always was, so re-scoring takes one, the report renders
    one and session_from_records combines them -- all through code that already
    exists. Widening the run itself would have changed the shape of every record ever
    saved to accommodate an arrangement of charts."""
    from deltachrome.results.models import CaptureRecord, SCHEMA_VERSION

    runs = [_run(f"chart-{n}") for n in range(3)]
    record = CaptureRecord(
        capture_id="cap-1", created_utc="2026-01-01T00:00:00+00:00",
        image={"filename": "board.tif"}, profile={"profile_id": "p"},
        charts=runs, overall="PASS",
    )
    data = record.to_dict()

    assert len(data["charts"]) == 3
    for chart, run in zip(data["charts"], runs):
        assert chart["schema_version"] == SCHEMA_VERSION
        assert chart == run.to_dict(), "a chart's record must be byte-for-byte its own"


def test_a_capture_record_round_trips():
    from deltachrome.results.models import CaptureRecord

    record = CaptureRecord(
        capture_id="cap-1", created_utc="2026-01-01T00:00:00+00:00",
        image={"filename": "board.tif"}, profile={"profile_id": "p"},
        charts=[_run("a"), _run("b")], overall="FAIL",
        conformance_caveats=["nominal values"], warnings=["something"],
    )
    back = CaptureRecord.from_dict(record.to_dict())

    assert back.capture_id == "cap-1"
    assert [c.run_id for c in back.charts] == ["a", "b"]
    assert back.overall == "FAIL"
    assert back.conformance_caveats == ["nominal values"]


def test_a_record_written_before_captures_existed_still_opens():
    """A year of saved runs must keep opening. Given a bare run record, a capture is
    the one chart it describes -- so nothing downstream has to know which shape a
    file on disk is until it looks."""
    from deltachrome.results.models import CaptureRecord

    run = _run("old-run")
    capture = CaptureRecord.from_dict(run.to_dict())

    assert capture.capture_id == "old-run"
    assert len(capture.charts) == 1
    assert capture.charts[0].to_dict() == run.to_dict()


def test_the_library_reads_both_shapes_without_branching():
    """A capture carries run_id, target and geometry at its top level precisely so
    that everything which already knows how to summarise a run keeps working on one
    unchanged. The only thing a row has to ask is how many charts are in there."""
    from deltachrome.results.models import CaptureRecord
    from deltachrome_ui.library import summarise

    single = summarise(_run("one").to_dict(), Path("one.json"))
    assert single.chart_count == 1

    board = CaptureRecord(
        capture_id="cap", created_utc="", image={"filename": "board.tif"},
        profile={"display_name": "FADGI 3-star"},
        charts=[_run("a"), _run("b")], overall="PASS",
    )
    row = summarise(board.to_dict(), Path("cap.json"))
    assert row.chart_count == 2
    assert row.run_id == "cap"
    assert row.image_name == "board.tif"


def test_the_target_column_names_the_models_not_the_count():
    """"4 x Applied Image QA-62 + Golden Thread Object Level" says what was shot;
    "5 charts" says only that something was."""
    from deltachrome.results.models import CaptureRecord

    def run_of(name):
        run = _run(name)
        run.target = {"display_name": name}
        return run

    record = CaptureRecord(
        capture_id="c", created_utc="", image={}, profile={},
        charts=[run_of("QA-62"), run_of("QA-62"), run_of("Object Level")],
        overall="PASS",
    )
    assert record.target_summary == "2 x QA-62 + Object Level"


def test_a_capture_record_says_what_its_charts_were_scored_against():
    """It did not, and the library's Basis column read *Unstated* on every capture
    record because of it -- a run scored against nominal values looked like a run
    that declined to say what it was scored against. The keys a run reader looks
    for are carried at the top level and this was the one that was missed."""
    from deltachrome.results.models import CaptureRecord

    def chart(basis: str) -> AnalysisRun:
        return AnalysisRun(**minimal() | {
            "reference": {"basis": basis, "path": "ref.txt"}
        })

    one = CaptureRecord(
        capture_id="c", created_utc="now", image={}, profile={},
        charts=[chart("manufacturer_nominal")], overall="PASS",
    )
    assert one.to_dict()["reference"]["basis"] == "manufacturer_nominal"
    assert one.to_dict()["reference"]["path"] == "ref.txt"


def test_charts_scored_against_different_references_report_the_weakest():
    """The same rule ``conformance_claim`` already applies one level up: a capture
    is quotable only if all of it is, so the basis it reports is the one that
    governs it."""
    from deltachrome.results.models import CaptureRecord

    def chart(basis: str) -> AnalysisRun:
        return AnalysisRun(**minimal() | {"reference": {"basis": basis}})

    record = CaptureRecord(
        capture_id="c", created_utc="now", image={}, profile={},
        charts=[chart("per_unit_measurement"), chart("manufacturer_nominal")],
        overall="PASS",
    )
    assert record.to_dict()["reference"]["basis"] == "manufacturer_nominal"

    both_measured = CaptureRecord(
        capture_id="c", created_utc="now", image={}, profile={},
        charts=[chart("per_unit_measurement"), chart("population_mean")],
        overall="PASS",
    )
    assert both_measured.to_dict()["reference"]["basis"] == "population_mean"


# -- what one reading is in ---------------------------------------------------


def test_a_readings_own_unit_survives_the_record():
    """Four metrics read in one quantity and are judged in another: lightness
    uniformity's readings are L* differences under a coefficient of variation,
    geometric distortion's are each rule's ppi under a percent. A readings table
    headed with the aggregate's unit mislabels every row, so the record says."""
    from deltachrome.metrics.base import MetricResult

    entry = {"metric_id": "geometric_distortion", "outcome": "evaluated",
             "value": 0.369, "units": "%", "reading_units": "ppi"}
    result = MetricResult.from_record(entry)
    assert result.reading_unit == "ppi"
    assert result.units == "%"

    run = AnalysisRun.from_dict(minimal(metrics=[entry]))
    assert AnalysisRun.from_dict(run.to_dict()).metrics[0]["reading_units"] == "ppi"

    # Where the readings are in the aggregate's unit the key is null, and the
    # reading falls back to it.
    same = {"metric_id": "noise", "outcome": "evaluated", "value": 1.5,
            "units": "std dev of L*", "reading_units": None}
    assert MetricResult.from_record(same).reading_unit == "std dev of L*"


def test_a_record_from_before_schema_9_is_given_its_reading_units():
    """The record is the archive. Records written before the key existed are
    filled in on read, so a year of lightness uniformity runs does not reopen
    with its readings labelled as a percent."""
    from deltachrome.metrics.base import MetricResult

    older = minimal(schema_version=8, metrics=[
        {"metric_id": "lightness_uniformity", "outcome": "evaluated",
         "value": 0.9, "units": "% (std dev / mean L*)"},
        {"metric_id": "noise", "outcome": "evaluated", "value": 1.5,
         "units": "std dev of L*"},
    ])
    metrics = AnalysisRun.from_dict(older).metrics
    assert metrics[0]["reading_units"] == "dL* from family mean"
    assert metrics[1]["reading_units"] is None
    assert MetricResult.from_record(metrics[1]).reading_unit == "std dev of L*"


def test_the_migration_says_what_the_plugins_write():
    """LEGACY_READING_UNITS describes what four plugins wrote before the record
    could say so. If a plugin's unit moves, the migration must not go on
    describing the old one."""
    from deltachrome.metrics import distortion, flatfield, uniformity
    from deltachrome.results.models import LEGACY_READING_UNITS

    assert LEGACY_READING_UNITS == {
        "lightness_uniformity": uniformity.READING_UNITS,
        "geometric_distortion": distortion.READING_UNITS,
        "illumination_non_uniformity": flatfield.READING_UNITS,
        "white_balance_non_uniformity": flatfield.READING_UNITS,
    }
