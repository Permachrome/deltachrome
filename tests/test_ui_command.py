"""The command line a run can be reproduced from.

Placement alone was never reproducible: four numbers without the target, the
profile and the reference do not identify a measurement. These check that what
lands on the clipboard is a whole invocation, that it can survive the paths this
sector actually uses, and -- the one that matters -- that it never invents a
piece it does not have.
"""

from __future__ import annotations

import shlex

import pytest

from deltachrome_ui.command import analyze_command, command_for_record, corners_argument

CORNERS = [(1025.0, 1921.4), (6966.0, 1935.0), (6954.2, 6164.0), (1008.0, 6151.0)]


def test_the_whole_invocation_is_emitted_not_just_the_placement():
    command = analyze_command(
        image_path="capture.tif",
        target_id="colorchecker_sg_v1",
        profile_id="fadgi_4star_prints_photographs_2023",
        corners=CORNERS,
        reference_path="data/reference/ColorChecker_SG_M1.txt",
    )
    parts = shlex.split(command)
    assert parts[:3] == ["deltachrome", "analyze", "capture.tif"]
    assert "--target" in parts and "colorchecker_sg_v1" in parts
    assert "--profile" in parts
    assert "--reference" in parts
    assert parts[parts.index("--corners") + 1].startswith("1025.0,1921.4")


def test_a_path_with_spaces_survives_the_round_trip():
    """The normal case here: masters live in directories named after collections."""
    command = analyze_command(
        image_path="/Volumes/Masters/Coll 12/ISA DLT.tif",
        target_id="isa_golden_thread_dlt_v1", profile_id="metamorfoze_full_2025",
    )
    assert shlex.split(command)[2] == "/Volumes/Masters/Coll 12/ISA DLT.tif"


def test_a_missing_piece_is_omitted_rather_than_guessed():
    """A command short a --reference fails the moment it runs, which is a better
    outcome than one carrying a plausible path to the wrong chart's data."""
    command = analyze_command(
        image_path="capture.tif", target_id="t", profile_id="p",
    )
    assert "--reference" not in command
    assert "--corners" not in command


def test_an_option_left_alone_does_not_appear():
    """Same rule as the setup screen's: emitting a default overwrites the
    profile's own choice with a value that merely happens to match today."""
    command = analyze_command(
        image_path="c.tif", target_id="t", profile_id="p",
        options={"original_size": None, "percentile_method": "nearest_rank"},
    )
    assert "--original-size" not in command
    assert "--percentile-method nearest_rank" in command


def test_a_library_is_offered_only_when_no_explicit_file_was_chosen():
    both = analyze_command(
        image_path="c.tif", target_id="t", profile_id="p",
        reference_path="unit.txt", reference_library="data/reference",
    )
    assert "--reference-library" not in both
    assert "--reference unit.txt" in both

    library_only = analyze_command(
        image_path="c.tif", target_id="t", profile_id="p",
        reference_library="data/reference",
    )
    assert "--reference-library data/reference" in library_only


def test_refinement_is_part_of_what_produced_the_placement():
    assert "--refine-placement" in analyze_command(
        image_path="c.tif", target_id="t", profile_id="p", refined=True)
    assert "--refine-placement" not in analyze_command(
        image_path="c.tif", target_id="t", profile_id="p")


def test_the_extent_option_is_written_in_the_syntax_the_flag_parses():
    from deltachrome.inputs import parse_extent

    command = analyze_command(
        image_path="c.tif", target_id="t", profile_id="p",
        options={"fiducial_extent_mm": (247.6, 176.1)},
    )
    parts = shlex.split(command)
    written = parts[parts.index("--fiducial-extent") + 1]
    assert parse_extent(written) == pytest.approx((247.6, 176.1))


def test_a_saved_record_reproduces_itself():
    """The library's copy button reads the record, which stores what the run was
    judged as and against precisely so it can be repeated."""
    record = {
        "image": {"path": "/masters/sg.jp2"},
        "target": {"target_id": "colorchecker_sg_v1"},
        "profile": {"profile_id": "fadgi_4star_prints_photographs_2023"},
        "reference": {"path": "/refs/sg.txt"},
        "geometry": {"corners_px": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]},
        "sampling": {"roi_inset": 0.55},
    }
    parts = shlex.split(command_for_record(record))
    assert "/masters/sg.jp2" in parts
    assert "colorchecker_sg_v1" in parts
    assert "/refs/sg.txt" in parts
    assert "--roi-inset" in parts and "0.55" in parts
    assert parts[parts.index("--corners") + 1] == "1.0,2.0 3.0,4.0 5.0,6.0 7.0,8.0"


def test_a_real_saved_run_reproduces_its_run_options():
    """The hand-built dict above only ever exercised ``roi_inset``.
    ``original_size``, ``percentile_method`` and ``frequency_units`` are read
    off ``AnalysisRun.to_dict()`` -- a real record, not a fixture that happens
    to carry the keys the reader expects."""
    from deltachrome.results.models import AnalysisRun

    run = AnalysisRun(
        run_id="abc123", created_utc="2026-01-01T00:00:00+00:00",
        image={"path": "/masters/sg.jp2"},
        target={"target_id": "colorchecker_sg_v1"},
        reference={"path": "/refs/sg.txt"},
        profile={"profile_id": "metamorfoze_full_2025"},
        geometry={"corners_px": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]},
        sampling={"roi_inset": 0.55},
        color_management={}, patches=[], edges=[], edge_field_summary={},
        registration=[], registration_summary={}, rulers={}, metrics=[],
        overall="PASS", overall_explanation="",
        original_size="A2", percentile_method="nearest_rank",
        frequency_units="lp/mm",
    )
    parts = shlex.split(command_for_record(run.to_dict()))
    assert "--original-size" in parts and "A2" in parts
    assert "--percentile-method" in parts and "nearest_rank" in parts
    assert "--frequency-units" in parts and "lp/mm" in parts


def test_a_nominal_run_is_not_given_a_reference_it_never_had():
    """An ISA run scored on manufacturer nominal values has no file to point at.
    Inventing one would turn an advisory result into a claimed conformance."""
    record = {
        "image": {"path": "/masters/isa.tif"},
        "target": {"target_id": "isa_golden_thread_dlt_v1"},
        "profile": {"profile_id": "metamorfoze_full_2025"},
        "reference": {"basis": "manufacturer_nominal"},
        "geometry": {}, "sampling": {},
    }
    assert "--reference" not in command_for_record(record)


def test_corners_are_written_at_the_precision_the_parser_reads():
    from deltachrome.inputs import parse_corners

    text = corners_argument(CORNERS)
    parsed = parse_corners(text)
    for (px, py), (qx, qy) in zip(parsed, CORNERS):
        assert px == pytest.approx(qx, abs=0.05)
        assert py == pytest.approx(qy, abs=0.05)


# -- opening a saved run -----------------------------------------------------


def test_a_saved_run_renders_the_report_without_the_capture(tmp_path):
    """Opening a run must not cost a decode of the master. Everything the report
    shows is in the record; what is missing is the pixels behind a mark, which
    the detail view says plainly rather than treating as a broken measurement."""
    from deltachrome_ui.library import load_result

    record = {
        "run_id": "r1", "created_utc": "2026-01-01T00:00:00+00:00",
        "image": {"path": "/gone/c.tif", "filename": "c.tif"},
        "target": {"target_id": "t", "display_name": "T"},
        "reference": {"basis": "measured"},
        "profile": {"profile_id": "p", "display_name": "P"},
        "geometry": {}, "sampling": {}, "color_management": {},
        "patches": [], "edges": [], "edge_field_summary": {},
        "registration": [], "registration_summary": {}, "rulers": {},
        "metrics": [], "overall": "PASS", "overall_explanation": "",
        "conformance_claim": True, "conformance_caveats": [],
    }
    path = tmp_path / "r1.json"
    path.write_text(__import__("json").dumps(record), encoding="utf-8")

    result = load_result(path)
    assert result.evaluation.overall == "PASS"
    assert result.evaluation.conformance_claim is True
    assert result.run.target["display_name"] == "T"


def test_an_old_record_without_caveats_still_says_why_it_is_qualified(tmp_path):
    """Schema 3 stored the claim but not the reasons. Rather than show an
    unexplained ADVISORY, recover what the record does carry."""
    import json

    from deltachrome_ui.library import load_result

    record = {
        "run_id": "r2", "created_utc": "", "image": {"filename": "c.tif"},
        "target": {}, "reference": {"basis": "manufacturer_nominal"},
        "profile": {"profile_id": "p"}, "geometry": {}, "sampling": {},
        "color_management": {}, "patches": [], "edges": [],
        "edge_field_summary": {}, "registration": [], "registration_summary": {},
        "rulers": {}, "metrics": [], "overall": "PASS",
        "overall_explanation": "", "conformance_claim": False,
        "schema_version": 3,
    }
    path = tmp_path / "r2.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    result = load_result(path)
    assert result.evaluation.conformance_claim is False
    assert any("nominal values" in c for c in result.evaluation.conformance_caveats)


def test_a_qualified_saved_run_never_reads_as_a_clean_pass(tmp_path):
    """The rule the whole frontend is built around, checked at the point a saved
    record re-enters it."""
    import json

    from deltachrome_ui.library import load_result
    from deltachrome_ui.status import overall_status

    record = {
        "run_id": "r3", "created_utc": "", "image": {}, "target": {},
        "reference": {"basis": "manufacturer_nominal"}, "profile": {},
        "geometry": {}, "sampling": {}, "color_management": {}, "patches": [],
        "edges": [], "edge_field_summary": {}, "registration": [],
        "registration_summary": {}, "rulers": {}, "metrics": [],
        "overall": "PASS", "overall_explanation": "", "conformance_claim": False,
        "conformance_caveats": ["nominal values"],
    }
    path = tmp_path / "r3.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    result = load_result(path)
    status = overall_status(
        result.evaluation.overall, result.evaluation.conformance_claim
    )
    assert status.reads_as_success is False
    assert "ADVISORY" in status.label
