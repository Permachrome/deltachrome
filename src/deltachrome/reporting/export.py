"""A capture's measured values, written out as text, CSV or XML.

**Data, not a report.** What somebody exports is the numbers: which metric, what
it read, what it was held to, and the readings it was reduced from. The verdict's
explanations, the run's warnings, the caveat sentences and the provenance blocks
are the report's, and they stay on screen and in the record.

**One table, three spellings of it.** :func:`build_table` assembles a short header
and a list of rows, and the three writers do nothing but lay that table out. No
format can carry a fact another lacks, because none of them is given anything
else -- which format is used is a preference of the person or the institution,
and must never decide what they receive.

**Everything a person reads is a word, never an id.** ``white_balance`` is *White
Balance*, ``dE(a*b*)`` is *ΔE\\*ab*, ``manufacturer_nominal`` is *Nominal*, an
edge is *Upper Left Horizontal 2* rather than ``module_tl_d0_e1``. The words are
the screens' own, from :mod:`deltachrome.reporting.labels`, so the file and the
report a person has open beside it agree. A patch keeps the label printed on the
chart, that being what a person can find on the card. The ids stay in the JSON
record, which is what anything importing these results should read.

**It reads the record.** A fresh run and the same run reopened from the library
both have one, so they export identically. The unit is the capture, as it is for
Save: a frame holding several targets exports every one of them, each row saying
which.
"""

from __future__ import annotations

import csv
import io
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from deltachrome import __version__
from deltachrome.errors import DeltachromeError
from deltachrome.metrics.base import MetricResult
from deltachrome.reporting.labels import (
    basis_label,
    condition_label,
    metric_title,
    reading_names,
    reading_unit,
    unit_label,
)
from deltachrome.results.models import CaptureRecord

#: File extension to format. The extension is the whole of how a format is
#: chosen, on the command line and in the save dialog alike.
FORMATS = {".txt": "txt", ".csv": "csv", ".xml": "xml"}

#: Decimal places, as the report writes a reading. The file and the screen are
#: read side by side, and a figure that changes precision between them reads as a
#: different quantity.
DECIMALS = 3

#: The header, in order: what made the numbers, when, from which file, of which
#: target, against which guideline, and the answer. The guideline's thresholds
#: are not here because they are in the data, row by row.
HEADER = ("Software", "Date of Analysis", "Image", "Target", "Guideline", "Verdict")

#: The data columns, in order. ``Target`` leads only where the capture holds more
#: than one -- with one, it would be the header's own line on every row.
COLUMNS = (
    "Target", "Metric", "Verdict", "Value", "Units", "Condition", "Reference",
    "Reading", "Reading Value", "Reading Units",
    "L*", "a*", "b*", "Aim L*", "Aim a*", "Aim b*",
)

#: An XML element name for every header field and column. Structural rather than
#: read -- a tag cannot hold a space or an asterisk -- and fixed, so a system
#: ingesting these can key on them.
XML_TAGS = {
    "Software": "software", "Date of Analysis": "date_of_analysis",
    "Image": "image", "Target": "target", "Guideline": "guideline",
    "Metric": "metric", "Verdict": "verdict", "Value": "value", "Units": "units",
    "Condition": "condition", "Reference": "reference", "Reading": "reading",
    "Reading Value": "reading_value", "Reading Units": "reading_units",
    "L*": "l_star", "a*": "a_star", "b*": "b_star",
    "Aim L*": "aim_l_star", "Aim a*": "aim_a_star", "Aim b*": "aim_b_star",
}


class ExportError(DeltachromeError):
    """An export that cannot be written as asked."""


@dataclass
class ExportTable:
    """Everything an export holds, before it is given a format."""

    header: list[tuple[str, str]]
    columns: list[str]
    rows: list[list[str]] = field(default_factory=list)


def format_for(path: str | Path) -> str:
    """The format a path's extension names, or a refusal naming the ones there are."""
    suffix = Path(path).suffix.lower()
    if suffix not in FORMATS:
        raise ExportError(
            f"{Path(path).name}: cannot tell what to export from the extension "
            f"'{suffix or '(none)'}'. Use one of "
            + ", ".join(sorted(FORMATS)) + "."
        )
    return FORMATS[suffix]


def export(record: Mapping[str, Any] | CaptureRecord, path: str | Path,
           fmt: str | None = None) -> Path:
    """Write ``record`` to ``path`` in the format its extension names.

    ``record`` is a capture record or a bare run record, as a dictionary or as a
    :class:`CaptureRecord` -- every shape a saved analysis has had reads the same.
    """
    path = Path(path)
    fmt = fmt or format_for(path)
    writers = {"txt": to_text, "csv": to_csv, "xml": to_xml}
    if fmt not in writers:
        raise ExportError(f"Unknown export format '{fmt}'.")
    text = writers[fmt](build_table(as_capture(record)))
    # A CSV gets a byte-order mark, which is what makes Excel read UTF-8 rather
    # than the platform's code page: guideline names carry typographic quotes and
    # units a ΔE, and without it both arrive garbled in the application most of
    # these files will be opened in.
    encoding = "utf-8-sig" if fmt == "csv" else "utf-8"
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" so the CSV writer's own \r\n is not doubled on Windows.
    with path.open("w", encoding=encoding, newline="") as handle:
        handle.write(text)
    return path


def as_capture(record: Mapping[str, Any] | CaptureRecord) -> CaptureRecord:
    if isinstance(record, CaptureRecord):
        return record
    return CaptureRecord.from_dict(record)


# -- the table ---------------------------------------------------------------


def build_table(capture: CaptureRecord) -> ExportTable:
    """The header and every row, as strings, ready for any of the three formats."""
    several = len(capture.charts) > 1
    columns = list(COLUMNS if several else COLUMNS[1:])
    table = ExportTable(header=_header(capture), columns=columns)
    for number, run in enumerate(capture.charts, start=1):
        reference = _reference(run.reference or {})
        patches = {p.patch_id: p for p in run.patches}
        edges = {getattr(e, "edge_id", None): e for e in run.edges}
        for entry in run.metrics:
            metric = [
                metric_title(str(entry.get("metric_id") or "")),
                _status(entry),
                _metric_value(entry),
                unit_label(entry.get("units")),
                condition_label(entry.get("condition")),
                reference,
            ]
            if several:
                metric.insert(0, f"Target {number}")
            readings = list(entry.get("per_patch") or [])
            if not readings:
                table.rows.append(metric + [""] * (len(columns) - len(metric)))
                continue
            unit = reading_unit(entry)
            for reading, name in zip(readings, reading_names(readings, edges)):
                patch = patches.get(str(reading.get("patch_id") or ""))
                measured = patch.measured_lab if patch is not None else None
                aim = patch.reference_lab if patch is not None else None
                table.rows.append(
                    metric + [name, _number(reading.get("value")), unit]
                    + _lab(measured) + _lab(aim)
                )
    return table


def _header(capture: CaptureRecord) -> list[tuple[str, str]]:
    environment = capture.environment or {}
    if not environment and capture.charts:
        environment = capture.charts[0].environment or {}
    version = environment.get("deltachrome_version") or __version__
    names = [str((run.target or {}).get("display_name")
                 or (run.target or {}).get("target_id") or "")
             for run in capture.charts]
    if len(names) > 1:
        target = "; ".join(f"Target {n}: {name}" for n, name in enumerate(names, 1))
    else:
        target = names[0] if names else ""
    values = (
        f"Deltachrome {version}",
        _date(capture.created_utc),
        str((capture.image or {}).get("filename") or ""),
        target,
        str((capture.profile or {}).get("display_name")
            or (capture.profile or {}).get("profile_id") or ""),
        _verdict(capture),
    )
    return list(zip(HEADER, values))


def _verdict(capture: CaptureRecord) -> str:
    """The capture's verdict as it may be quoted -- never bare on unsound evidence.

    The evaluator's ``verdict_line`` rule, read from a record: ``PASS
    (ADVISORY)`` wherever the reference is not a measurement of the chart in
    front of the camera. A capture of several targets carries the weakest of
    their bases, which is the rule its own claim already follows. Where the
    record states no basis -- one written before it was stored -- the claim
    decides, which withholds rather than grants.
    """
    basis = (capture.to_dict().get("reference") or {}).get("basis")
    sound = (basis == "per_unit_measurement") if basis else capture.conformance_claim
    return capture.overall if sound else f"{capture.overall} (ADVISORY)"


def _date(stamp: str) -> str:
    """``2026-08-23T19:01:02+00:00`` -> ``2026-08-23 19:01:02 UTC``."""
    try:
        moment = datetime.fromisoformat(str(stamp))
    except ValueError:
        return str(stamp or "")
    if moment.utcoffset():
        return moment.isoformat(sep=" ")
    return moment.strftime("%Y-%m-%d %H:%M:%S") + " UTC"


def _reference(reference: Mapping[str, Any]) -> str:
    """What the target's readings were compared against, in a word.

    ``None`` for a target that ships no aim values and was given no file -- the
    QA-62 compares nothing, and the engine records it as a per-unit basis
    because nothing it reports leans on a nominal value. True for the claim and
    false as a description: a reader seeing *Measured* would look for a
    measurement file that does not exist.
    """
    if not reference.get("path") and not reference.get("sample_count"):
        return "None"
    return basis_label(reference.get("basis"))


def _status(entry: Mapping[str, Any]) -> str:
    """The metric's verdict as the report prints it, from what the record stored."""
    stored = entry.get("status")
    if stored:
        return str(stored)
    return MetricResult.from_record(entry).status


def _metric_value(entry: Mapping[str, Any]) -> str:
    """The reading, as a word or a whole number where the engine says it is one."""
    text = entry.get("value_text")
    return str(text) if text else _number(entry.get("value"))


def _number(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return ""
    return f"{x:.{DECIMALS}f}"


def _lab(values) -> list[str]:
    if not values:
        return ["", "", ""]
    return [_number(v) for v in list(values)[:3]]


# -- the three formats -------------------------------------------------------


def to_csv(table: ExportTable) -> str:
    """The header as label-value lines, a blank line, then the table."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerows(table.header)
    writer.writerow([])
    writer.writerow(table.columns)
    writer.writerows(table.rows)
    return buffer.getvalue()


def to_text(table: ExportTable) -> str:
    """The same, as aligned columns a person can read in any text editor."""
    label_width = max(len(label) for label, _ in table.header)
    lines = [f"{label:<{label_width}}  {value}".rstrip() for label, value in table.header]
    lines.append("")
    widths = [
        max([len(column)] + [len(row[i]) for row in table.rows])
        for i, column in enumerate(table.columns)
    ]
    for row in [table.columns] + table.rows:
        lines.append("  ".join(
            cell.ljust(width) for cell, width in zip(row, widths)
        ).rstrip())
    return "\n".join(lines) + "\n"


#: Which columns describe the target, which the metric, which one reading -- the
#: three levels the XML nests. Each level's own name column becomes its ``name``
#: element, and a reading's value and units are its ``value`` and ``units``.
TARGET_COLUMNS = ("Target",)
METRIC_COLUMNS = ("Metric", "Verdict", "Value", "Units", "Condition", "Reference")
READING_COLUMNS = ("Reading", "Reading Value", "Reading Units",
                   "L*", "a*", "b*", "Aim L*", "Aim a*", "Aim b*")
NESTED_TAGS = {"Target": "name", "Metric": "name", "Reading": "name",
               "Reading Value": "value", "Reading Units": "units"}


def to_xml(table: ExportTable) -> str:
    """The same, nested: a reading inside its metric, a metric inside its target.

    Built from the table and nothing else, so it holds what the CSV holds and the
    nesting only stops a metric's own fields being written once per reading. A
    ``target`` level appears only where the capture holds several, exactly as the
    table's Target column does.
    """
    root = ET.Element("deltachrome_export")
    header = ET.SubElement(root, "header")
    for label, value in table.header:
        ET.SubElement(header, XML_TAGS[label]).text = value
    data = ET.SubElement(root, "data")

    index = {column: i for i, column in enumerate(table.columns)}
    several = "Target" in index
    target_node = metric_node = None
    target_key = metric_key = None
    for row in table.rows:
        cells = lambda columns: [(c, row[index[c]]) for c in columns if c in index]
        if several and row[index["Target"]] != target_key:
            target_key, metric_key = row[index["Target"]], None
            target_node = _cells(ET.SubElement(data, "target"), cells(TARGET_COLUMNS))
        key = tuple(value for _, value in cells(METRIC_COLUMNS))
        if key != metric_key:
            metric_key = key
            metric_node = _cells(
                ET.SubElement(target_node if several else data, "metric"),
                cells(METRIC_COLUMNS),
            )
        reading = cells(READING_COLUMNS)
        if any(value for _, value in reading):
            _cells(ET.SubElement(metric_node, "reading"), reading)
    ET.indent(root, space="  ")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            + ET.tostring(root, encoding="unicode") + "\n")


def _cells(parent: ET.Element, cells) -> ET.Element:
    for column, value in cells:
        # An empty cell is an absent element rather than an empty one, which a
        # parser would read as the empty string where nothing was measured.
        if value:
            ET.SubElement(parent, NESTED_TAGS.get(column, XML_TAGS[column])).text = value
    return parent
