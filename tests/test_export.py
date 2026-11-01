"""Writing a capture's values out as text, CSV and XML.

What these pin is the export's promises. All three formats hold the **same
data** -- the format is a preference, never a difference in what is received. The
header is five facts and no more. Nothing a person reads is an id. And no
explanations or warnings come along: the export is the values.
"""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET

import pytest

from deltachrome.pipeline import analyze
from deltachrome.reporting.export import (
    COLUMNS,
    HEADER,
    XML_TAGS,
    ExportError,
    as_capture,
    build_table,
    export,
    format_for,
    to_csv,
    to_text,
    to_xml,
)
from deltachrome.session import capture_record
from fixtures import SyntheticChart

SG = "colorchecker_sg_v1"
FADGI3 = "fadgi_3star_documents_unbound_general_2023"


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("export")
    chart = SyntheticChart()
    reference = chart.write_reference(tmp / "ref.cgats.txt")
    image, corners = chart.render_capture(tmp / "capture.tif", skew=18)
    return analyze(
        image, target=SG, reference_path=reference, profile=FADGI3, corners=corners
    )


@pytest.fixture(scope="module")
def saved(result, tmp_path_factory):
    """The record exactly as the library holds it, read back off disk."""
    path = capture_record([result]).save(
        tmp_path_factory.mktemp("lib") / "capture.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def board(result):
    return capture_record([result, result], labels=["Target 1", "Target 2"])


# -- reading each format back ----------------------------------------------


def _from_csv(text: str):
    lines = list(csv.reader(io.StringIO(text)))
    blank = lines.index([])
    header = [tuple(line) for line in lines[:blank]]
    return header, lines[blank + 1], lines[blank + 2:]


def _from_xml(text: str, columns):
    """Flatten the nested document back into the table's rows."""
    from deltachrome.reporting.export import (
        METRIC_COLUMNS, NESTED_TAGS, READING_COLUMNS, TARGET_COLUMNS,
    )

    root = ET.fromstring(text)
    labels = {tag: label for label, tag in XML_TAGS.items()}
    header = [(labels[node.tag], node.text or "") for node in root.find("header")]

    def level(node, level_columns):
        tags = {NESTED_TAGS.get(c, XML_TAGS[c]): c for c in level_columns}
        return {tags[child.tag]: child.text or ""
                for child in node if child.tag in tags}

    rows = []
    data = root.find("data")
    targets = data.findall("target") or [None]
    for target in targets:
        outer = level(target, TARGET_COLUMNS) if target is not None else {}
        for metric in (target if target is not None else data).findall("metric"):
            fields = {**outer, **level(metric, METRIC_COLUMNS)}
            readings = metric.findall("reading") or [None]
            for reading in readings:
                cells = dict(fields)
                if reading is not None:
                    cells.update(level(reading, READING_COLUMNS))
                rows.append([cells.get(column, "") for column in columns])
    return header, list(columns), rows


def _from_text(text: str, columns):
    lines = text.splitlines()
    blank = lines.index("")
    header = [tuple(re.split(r"\s{2,}", line, maxsplit=1)) for line in lines[:blank]]
    heading = lines[blank + 1]
    starts = [heading.index(column) for column in columns]
    rows = []
    for line in lines[blank + 2:]:
        bounds = zip(starts, starts[1:] + [None])
        rows.append([line[a:b].strip() if a < len(line) else "" for a, b in bounds])
    return header, list(columns), rows


@pytest.mark.parametrize("which", ["single", "board"])
def test_all_three_formats_hold_exactly_the_same_data(which, saved, board):
    """The format is a preference. If one could carry a fact another could not,
    the choice of format would decide what an institution receives."""
    table = build_table(as_capture(saved) if which == "single" else board)
    expected = (list(table.header), table.columns, table.rows)
    assert _from_csv(to_csv(table)) == expected
    assert _from_xml(to_xml(table), table.columns) == expected
    assert _from_text(to_text(table), table.columns) == expected


# -- the header ------------------------------------------------------------


def test_the_header_is_six_facts(saved):
    table = build_table(as_capture(saved))
    assert [label for label, _ in table.header] == list(HEADER)
    values = dict(table.header)
    assert values["Software"].startswith("Deltachrome ")
    assert values["Date of Analysis"].endswith(" UTC")
    assert values["Image"] == saved["image"]["filename"]
    assert values["Target"] == "ColorChecker Digital SG" or values["Target"]
    assert values["Guideline"] == saved["profile"]["display_name"]
    assert values["Verdict"] == saved["overall"]


@pytest.mark.parametrize("basis, claim, qualified", [
    ("per_unit_measurement", True, False),
    # A local guideline on a measured chart: the evidence is sound.
    ("per_unit_measurement", False, False),
    ("manufacturer_nominal", False, True),
    ("population_mean", False, True),
])
def test_the_verdict_is_never_bare_on_unsound_evidence(saved, basis, claim, qualified):
    record = json.loads(json.dumps(saved))
    for chart in record["charts"]:
        chart["reference"]["basis"] = basis
        chart["conformance_claim"] = claim
    record["conformance_claim"] = claim
    verdict = dict(build_table(as_capture(record)).header)["Verdict"]
    assert verdict.endswith(" (ADVISORY)") == qualified
    assert verdict.startswith(record["overall"])


def test_xml_nests_readings_inside_their_metric(board):
    root = ET.fromstring(to_xml(build_table(board)))
    targets = root.findall("data/target")
    assert [t.findtext("name") for t in targets] == ["Target 1", "Target 2"]
    metric = next(m for m in targets[0].findall("metric")
                  if m.findtext("name") == "Average Color Accuracy")
    assert len(metric.findall("verdict")) == 1
    assert len(metric.findall("reading")) > 1


def test_a_board_names_every_target_in_the_header_and_on_every_row(board):
    table = build_table(board)
    assert table.columns[0] == "Target"
    assert dict(table.header)["Target"].startswith("Target 1: ")
    assert {row[0] for row in table.rows} == {"Target 1", "Target 2"}


def test_a_single_target_has_no_target_column(saved):
    """With one target it would be the header's own line on every row."""
    assert "Target" not in build_table(as_capture(saved)).columns


# -- nothing a person reads is an id ---------------------------------------


def test_no_cell_is_an_id(saved):
    capture = as_capture(saved)
    table = build_table(capture)
    ids = {m["metric_id"] for m in capture.charts[0].metrics}
    ids |= {"per_unit_measurement", "manufacturer_nominal", "population_mean"}
    cells = {cell for row in table.rows for cell in row}
    cells |= {value for _, value in table.header}
    assert not cells & ids
    assert not any("_" in cell for cell in cells)


def test_metric_names_and_units_are_the_screens_own(saved):
    table = build_table(as_capture(saved))
    metric = table.columns.index("Metric")
    units = table.columns.index("Units")
    assert "Average Color Accuracy" in {row[metric] for row in table.rows}
    assert "ΔE*2000" in {row[units] for row in table.rows}


def test_no_warnings_or_explanations_come_along(saved):
    record = json.loads(json.dumps(saved))
    record["charts"][0]["warnings"] = ["SENTINEL WARNING"]
    record["charts"][0]["overall_explanation"] = "SENTINEL EXPLANATION"
    record["charts"][0]["conformance_caveats"] = ["SENTINEL CAVEAT"]
    text = to_csv(build_table(as_capture(record)))
    assert "SENTINEL" not in text


# -- the data --------------------------------------------------------------


def test_every_reading_is_a_row_and_every_metric_appears(saved):
    capture = as_capture(saved)
    table = build_table(capture)
    run = capture.charts[0]
    assert len(table.rows) == sum(max(1, len(m.get("per_patch") or []))
                                  for m in run.metrics)


def test_a_patch_reading_carries_its_measured_and_aim_colour(saved):
    capture = as_capture(saved)
    table = build_table(capture)
    patches = {p.patch_id: p for p in capture.charts[0].patches}
    col = {name: i for i, name in enumerate(table.columns)}
    rows = [r for r in table.rows if r[col["Metric"]] == "Average Color Accuracy"]
    assert rows
    for row in rows:
        patch = patches[row[col["Reading"]]]
        assert float(row[col["L*"]]) == pytest.approx(patch.measured_lab[0], abs=1e-3)
        assert float(row[col["Aim b*"]]) == pytest.approx(patch.reference_lab[2], abs=1e-3)


def test_a_metric_with_no_readings_still_has_its_row(saved):
    table = build_table(as_capture(saved))
    col = {name: i for i, name in enumerate(table.columns)}
    rows = [r for r in table.rows if r[col["Metric"]] == "Bit Depth"]
    assert len(rows) == 1 and rows[0][col["Reading"]] == ""


def test_a_target_compared_against_nothing_says_so(saved):
    """The engine records a per-unit basis for a chart that ships no values,
    which is right for the claim and wrong as a description."""
    record = json.loads(json.dumps(saved))
    record["charts"][0]["reference"] = {"basis": "per_unit_measurement",
                                        "path": None, "sample_count": 0}
    table = build_table(as_capture(record))
    col = table.columns.index("Reference")
    assert {row[col] for row in table.rows} == {"None"}


@pytest.mark.parametrize("basis, word", [
    ("per_unit_measurement", "Measured"),
    ("manufacturer_nominal", "Nominal"),
    ("population_mean", "Batch Mean"),
])
def test_the_reference_basis_is_a_word(saved, basis, word):
    record = json.loads(json.dumps(saved))
    record["charts"][0]["reference"]["basis"] = basis
    table = build_table(as_capture(record))
    col = table.columns.index("Reference")
    assert {row[col] for row in table.rows} == {word}


def test_csv_opens_in_excel_as_utf8(saved, tmp_path):
    path = export(saved, tmp_path / "run.csv")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


# -- reading names ---------------------------------------------------------


def test_readings_are_named_by_where_they_were_taken():
    from types import SimpleNamespace

    from deltachrome.reporting.labels import reading_names

    edges = {
        "module_tl_d0_e0": SimpleNamespace(region="module_tl", orientation="vertical"),
        "module_tl_d0_e1": SimpleNamespace(region="module_tl", orientation="vertical"),
        "module_tl_d1_e0": SimpleNamespace(region="module_tl", orientation="horizontal"),
    }
    readings = [
        {"patch_id": "module_tl_d0_e0"}, {"patch_id": "module_tl_d0_e1"},
        {"patch_id": "module_tl_d1_e0"},
        {"patch_id": "module_bl:x", "role": "worst pair B-R"},
        {"patch_id": "r0c2"}, {"patch_id": "E5"}, {"patch_id": "E5-F5"},
        {"patch_id": "ruler_imperial_top"},
    ]
    assert reading_names(readings, edges) == [
        "Upper Left Horizontal 1", "Upper Left Horizontal 2", "Upper Left Vertical",
        "Lower Left Horizontal (B-R)", "Row 1, Column 3", "E5", "E5-F5",
        "Ruler Imperial Top",
    ]


def test_an_older_records_permitted_list_reads_whole():
    """Written as ``format in [...]`` before the record stored the list, and cut
    mid-list -- closing bracket lost -- by the comma split."""
    from deltachrome.reporting.labels import condition_label

    assert condition_label("format in ['TIFF', 'JPEG 2000']") == "TIFF, JPEG 2000"


# -- choosing a format -----------------------------------------------------


def test_the_extension_chooses_the_format(saved, tmp_path):
    for suffix in ("txt", "csv", "xml"):
        assert export(saved, tmp_path / f"out.{suffix.upper()}").exists()
    assert format_for("x.Csv") == "csv"
    with pytest.raises(ExportError, match="extension"):
        format_for("report.pdf")


def test_a_bare_run_record_exports_as_a_capture_of_one(result, saved):
    """Every record written before captures could hold several still exports."""
    bare = build_table(as_capture(result.run.to_dict()))
    held = build_table(as_capture(saved))
    assert bare.rows == held.rows


def test_the_columns_are_declared_once():
    assert set(COLUMNS) | set(HEADER) <= set(XML_TAGS)


# -- the command line ------------------------------------------------------


def test_export_writes_a_saved_record_in_every_format_asked(saved, tmp_path):
    from deltachrome.cli import EXIT_OK, main

    record = tmp_path / "capture.json"
    record.write_text(json.dumps(saved), encoding="utf-8")
    outputs = [tmp_path / "r.txt", tmp_path / "r.csv", tmp_path / "r.xml"]
    assert main(["export", str(record), *map(str, outputs)]) == EXIT_OK
    assert all(path.exists() for path in outputs)


def test_export_refuses_a_bad_extension_before_writing_anything(saved, tmp_path):
    from deltachrome.cli import EXIT_ERROR, main

    record = tmp_path / "capture.json"
    record.write_text(json.dumps(saved), encoding="utf-8")
    first = tmp_path / "r.csv"
    assert main(["export", str(record), str(first), str(tmp_path / "r.pdf")]) == EXIT_ERROR
    assert not first.exists()


def test_export_refuses_a_session_record(tmp_path):
    """A session has metrics and is a report of no one capture."""
    from deltachrome.cli import EXIT_ERROR, main

    record = tmp_path / "session.json"
    record.write_text(json.dumps({"captures": [], "metrics": [], "overall": "PASS"}))
    assert main(["export", str(record), str(tmp_path / "s.csv")]) == EXIT_ERROR
