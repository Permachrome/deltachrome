"""Reading saved runs back.

Two rules pull in opposite directions here and both matter.

Reading is **forgiving**: a record is JSON on disk that may have been written by
an older or newer engine, so a missing field is a blank cell and an unparseable
file is a visible bad row. A run that silently vanished from the library because
its JSON was truncated would be worse than one shown as broken -- the entire
point of persisting raw per-patch measurements is that they can be relied on to
still be there.

The **verdict** is not forgiving. It goes through ``deltachrome_ui.status``,
which refuses to show an unqualified pass for a run that did not earn one, and a
record from the future fails closed.
"""

from __future__ import annotations

import json
from pathlib import Path

from deltachrome_ui.library import read_run, scan_runs
from deltachrome_ui.status import TONE_PASS, TONE_QUALIFIED


def write(path: Path, record: dict) -> Path:
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def minimal(**overrides) -> dict:
    record = {
        "run_id": "abc123",
        "created_utc": "2026-08-20T20:15:21+00:00",
        "image": {"filename": "capture.tif", "path": "/masters/capture.tif"},
        "target": {"display_name": "ColorChecker Digital SG"},
        "profile": {"display_name": "FADGI 3-Star"},
        "geometry": {"corners_px": [[1, 2], [3, 4], [5, 6], [7, 8]]},
        "overall": "PASS",
        "conformance_claim": True,
        "warnings": [],
    }
    record.update(overrides)
    return record


def test_a_complete_record_summarises(tmp_path):
    run = read_run(write(tmp_path / "r.json", minimal()))
    assert run.image_name == "capture.tif"
    assert run.target_name == "ColorChecker Digital SG"
    assert run.status.tone == TONE_PASS
    assert run.corners == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
    assert run.is_readable


def test_an_indicative_run_is_marked_as_one_in_the_list_itself(tmp_path):
    """Not tucked into a detail view nobody opens."""
    run = read_run(write(tmp_path / "r.json", minimal(conformance_claim=False)))
    assert run.status.tone == TONE_QUALIFIED
    assert "ADVISORY" in run.status.label


def test_a_record_missing_everything_optional_still_lists(tmp_path):
    run = read_run(write(tmp_path / "r.json", {"overall": "FAIL"}))
    assert run.is_readable
    assert run.target_name == "—"
    assert run.corners is None


def test_truncated_json_becomes_a_visible_row_not_a_crash(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"run_id": "abc", "over', encoding="utf-8")
    run = read_run(path)
    assert not run.is_readable
    assert run.image_name == "broken.json"
    assert not run.status.reads_as_success


def test_a_json_file_that_is_not_a_record_is_rejected(tmp_path):
    run = read_run(write(tmp_path / "list.json", []))  # type: ignore[arg-type]
    assert not run.is_readable


def test_malformed_corners_are_dropped_rather_than_half_read(tmp_path):
    """Three corners is not a placement, and two of four is worse than none:
    seeding a canvas from it would silently move one handle."""
    for bad in ([[1, 2], [3, 4], [5, 6]], "1,2 3,4", [[1, 2], [3, 4], [5, 6], "x"]):
        run = read_run(write(tmp_path / "r.json", minimal(geometry={"corners_px": bad})))
        assert run.corners is None, bad


def test_runs_are_listed_newest_first(tmp_path):
    write(tmp_path / "a.json", minimal(created_utc="2026-01-01T00:00:00+00:00"))
    write(tmp_path / "b.json", minimal(created_utc="2026-08-01T00:00:00+00:00"))
    write(tmp_path / "c.json", minimal(created_utc="2026-04-01T00:00:00+00:00"))
    assert [r.path.stem for r in scan_runs(tmp_path)] == ["b", "c", "a"]


def test_scanning_a_directory_that_does_not_exist_is_empty_not_an_error(tmp_path):
    assert scan_runs(tmp_path / "nope") == []


def test_non_json_files_are_ignored(tmp_path):
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    write(tmp_path / "r.json", minimal())
    assert len(scan_runs(tmp_path)) == 1


def test_an_unparseable_timestamp_still_displays(tmp_path):
    run = read_run(write(tmp_path / "r.json", minimal(created_utc="whenever")))
    assert run.created_display == "whenever"


def test_a_real_engine_record_round_trips(tmp_path):
    """Against the actual AnalysisRun shape rather than a hand-written stand-in,
    so a schema change shows up here instead of as a blank library."""
    from deltachrome.results.models import AnalysisRun, environment_info, now_utc

    run = AnalysisRun(
        run_id="deadbeef", created_utc=now_utc(),
        image={"filename": "x.tif", "path": "/m/x.tif", "width": 10, "height": 10},
        target={"display_name": "ISA Golden Thread"},
        reference={}, profile={"display_name": "FADGI 3-Star"},
        geometry={"corners_px": [[0, 0], [9, 0], [9, 9], [0, 9]]},
        sampling={}, color_management={}, patches=[], edges=[],
        edge_field_summary={}, registration=[], registration_summary={},
        rulers={}, metrics=[], overall="PASS", overall_explanation="",
        conformance_claim=False, warnings=["nominal reference"],
        environment=environment_info(),
    )
    path = run.save(tmp_path / "real.json")
    summary = read_run(path)
    assert summary.is_readable
    assert summary.target_name == "ISA Golden Thread"
    assert summary.warning_count == 1
    assert summary.corners == [(0.0, 0.0), (9.0, 0.0), (9.0, 9.0), (0.0, 9.0)]
    assert not summary.status.reads_as_success  # conformance_claim was False


def test_a_record_carries_the_ids_needed_to_reuse_it(tmp_path):
    """Display names are for reading; ids are what let a run be *reused* — the
    target and profile restored rather than remembered and re-picked."""
    run = read_run(write(tmp_path / "r.json", minimal(
        target={"target_id": "colorchecker_sg_v1", "display_name": "ColorChecker SG"},
        profile={"profile_id": "fadgi_3star_prints_photographs_2023",
                 "display_name": "FADGI 3-Star"},
    )))
    assert run.target_id == "colorchecker_sg_v1"
    assert run.profile_id == "fadgi_3star_prints_photographs_2023"


def test_reusable_ids_are_absent_rather_than_invented_when_the_record_lacks_them(tmp_path):
    run = read_run(write(tmp_path / "r.json", {"overall": "PASS"}))
    assert run.target_id is None and run.profile_id is None


# -- reading a capture again, from its own record --


def test_a_complete_record_can_be_read_again():
    """Everything ``analyze`` needs, read the same tolerant way
    ``command_for_record`` already reads it."""
    from deltachrome_ui.library import establish_kwargs

    kwargs = establish_kwargs(minimal(
        target={"target_id": "colorchecker_sg_v1", "display_name": "ColorChecker SG"},
        profile={"profile_id": "fadgi_3star_prints_photographs_2023",
                 "display_name": "FADGI 3-Star"},
        reference={"path": "/refs/sg.txt"},
        sampling={"roi_inset": 0.55},
        original_size="A2", percentile_method="nearest_rank",
        frequency_units="lp/mm",
    ))
    assert kwargs == {
        "target": "colorchecker_sg_v1",
        "profile": "fadgi_3star_prints_photographs_2023",
        "corners": [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)],
        "reference_path": "/refs/sg.txt",
        "roi_inset": 0.55,
        "original_size": "A2",
        "percentile_method": "nearest_rank",
        "frequency_units": "lp/mm",
    }


def test_a_record_with_no_reference_path_is_still_readable():
    """A run scored on manufacturer nominal values has no file to point at --
    ``analyze`` resolves the nominal fallback itself, the same as a fresh run."""
    from deltachrome_ui.library import establish_kwargs

    kwargs = establish_kwargs(minimal(
        target={"target_id": "isa_golden_thread_dlt_v1"},
        profile={"profile_id": "metamorfoze_full_2025"},
        reference={"basis": "manufacturer_nominal"},
    ))
    assert kwargs["reference_path"] is None


def test_a_record_with_no_placement_cannot_be_read_again():
    """No amount of re-decoding supplies a corner placement nobody recorded."""
    from deltachrome_ui.library import establish_kwargs

    assert establish_kwargs(minimal(
        target={"target_id": "colorchecker_sg_v1"},
        profile={"profile_id": "fadgi_3star_prints_photographs_2023"},
        geometry={},
    )) is None


def test_a_record_with_no_target_or_profile_id_cannot_be_read_again():
    """Older than the ids being carried at all -- a record this bare has
    nothing ``establish_kwargs`` can supply."""
    from deltachrome_ui.library import establish_kwargs

    assert establish_kwargs(minimal(target={}, profile={})) is None
