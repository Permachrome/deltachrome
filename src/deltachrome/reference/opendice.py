"""OpenDICE target profiles as reference files.

OpenDICE -- the Library of Congress's open FADGI analysis tool -- reads its aim
values from a "target profile": a tab-separated table with one header line and no
other metadata. FADGI's website distributes these, so an operator moving across
from OpenDICE arrives holding them:

    <tab>L<tab>a <tab>b<tab>D
    Patch1<tab>38.07<tab>11.74<tab>13.51<tab>
    Patch10<tab>95.93<tab>-1.03<tab>2.84<tab>0.04

Three header spellings are in circulation (``L a b D``, ``L* a* b* D`` and ``D``
alone, the last for film targets whose aims are a density ladder), so the header is
matched on what it names rather than on its exact text.

What the format does not say, and what is done about it:

* **No illuminant or observer.** Nothing is invented. The engine already reads a
  file that states neither as D50 and 2°, which is FADGI's convention and what
  OpenDICE assumes.
* **Patch names.** OpenDICE writes ``Patch12`` where the chart prints ``12``; the
  prefix is OpenDICE's own convention and is removed here. Everything else is left
  to :func:`deltachrome.reference.sample_ids.reconcile_sample_ids`, which is where a
  target declares how a file's names map onto its own.
* **A name may repeat.** OpenDICE's UTT profile names a colour patch ``B1`` and,
  ninety rows later, the first step of the bottom grey scale ``B1`` as well; which
  is which is settled only by position. Every row is kept, in order, in
  :attr:`MeasurementSet.row_names`, and a repeat that the target does not resolve by
  a declared row order is refused exactly as a CGATS duplicate is.
* **Density only.** A row with no L*a*b* takes its L* from the density, anchored so
  the file's lightest step reads L* 96 -- the same convention the film targets'
  shipped nominal values use (:func:`deltachrome.imaging.color.density_to_lightness`).
  OpenDICE anchors at a perfect white instead; the difference is a rendering intent,
  not a measurement, and the density itself is kept in ``D`` beside it.

Where a row carries both, its stated L* is used and its density is kept only as
density. OpenDICE judges tone against the density converted back to L*, which
loses the precision the file already had: near white, 0.01 D is about 0.9 L*.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from deltachrome.errors import CGATSParseError
from deltachrome.imaging.color import density_to_lightness
from deltachrome.reference.cgats import MeasurementSet, PatchMeasurement

#: Recorded in the measurement set's metadata so a run says what it was given.
SOURCE_FORMAT_KEY = "DELTACHROME_SOURCE_FORMAT"
SOURCE_FORMAT = "OpenDICE target profile"

#: Recorded where any L* was derived from density, saying how.
LIGHTNESS_FROM_DENSITY_KEY = "DELTACHROME_LIGHTNESS_FROM_DENSITY"

_COLUMNS = {"l": "L", "a": "a", "b": "b", "d": "D"}
_PREFIX = re.compile(r"^Patch\s*#?\s*")


def _header(line: str) -> list[str] | None:
    """The header's column names, or None if this is not an OpenDICE header."""
    cells = line.rstrip("\r\n").split("\t")
    if len(cells) < 2 or cells[0].strip():
        return None
    names = [cell.strip().rstrip("*").strip().lower() for cell in cells[1:]]
    if not names or any(name not in _COLUMNS for name in names):
        return None
    columns = [_COLUMNS[name] for name in names]
    if len(set(columns)) != len(columns):
        return None
    if not ({"L", "a", "b"} <= set(columns) or "D" in columns):
        return None
    return columns


def looks_like_opendice(head: str) -> bool:
    """Whether the start of a file is an OpenDICE target profile."""
    first = head.lstrip("﻿").splitlines()[0] if head.strip() else ""
    return _header(first) is not None


def _number(text: str, *, row: int, column: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise CGATSParseError(
            f"Row {row} of the OpenDICE profile has {text!r} under {column}, which is "
            "not a number."
        ) from exc


def parse_opendice(text: str) -> MeasurementSet:
    """Parse an OpenDICE target profile into a :class:`MeasurementSet`."""
    lines = text.lstrip("﻿").splitlines()
    columns = _header(lines[0]) if lines else None
    if columns is None:
        raise CGATSParseError("This does not look like an OpenDICE target profile.")

    rows: list[tuple[str, dict[str, float | None]]] = []
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        cells = line.split("\t")
        name = _PREFIX.sub("", cells[0].strip())
        if not name:
            raise CGATSParseError(f"Row {number} of the OpenDICE profile names no patch.")
        values = {
            column: _number(cells[i + 1] if i + 1 < len(cells) else "", row=number,
                            column=column)
            for i, column in enumerate(columns)
        }
        rows.append((name, values))
    if not rows:
        raise CGATSParseError("The OpenDICE profile lists no patches.")

    densities = [v["D"] for _, v in rows if v.get("D") is not None]
    anchor = min(densities) if densities else None
    derived = False

    samples: dict[str, PatchMeasurement] = {}
    seen: dict[str, int] = {}
    for name, values in rows:
        triplet = tuple(values.get(c) for c in ("L", "a", "b"))
        density = values.get("D")
        if all(v is not None for v in triplet):
            lab = tuple(float(v) for v in triplet)
        elif any(v is not None for v in triplet):
            raise CGATSParseError(
                f"Patch {name} of the OpenDICE profile gives part of an L*a*b* value."
            )
        elif density is not None:
            L = float(density_to_lightness(np.array([density]), anchor_density=anchor)[0])
            lab = (L, 0.0, 0.0)
            derived = True
        else:
            raise CGATSParseError(
                f"Patch {name} of the OpenDICE profile has neither L*a*b* nor density."
            )

        raw: dict[str, float | str] = {"SAMPLE_NAME": name}
        raw.update({k: v for k, v in zip(("LAB_L", "LAB_A", "LAB_B"), lab)})
        if density is not None:
            raw["D"] = density
        # A repeated name keeps its row under a key of its own, so nothing is lost
        # before a declared order can say which patch it is.
        seen[name] = seen.get(name, 0) + 1
        key = name if seen[name] == 1 else f"{name} (occurrence {seen[name]})"
        samples[key] = PatchMeasurement(sample_name=key, lab=lab, raw=raw)

    metadata = {SOURCE_FORMAT_KEY: SOURCE_FORMAT}
    if derived:
        metadata[LIGHTNESS_FROM_DENSITY_KEY] = (
            f"L* derived from density, lightest step ({anchor:g} D) at L* 96"
        )
    return MeasurementSet(
        samples=samples,
        metadata=metadata,
        fields=["SAMPLE_NAME", "LAB_L", "LAB_A", "LAB_B", *(["D"] if densities else [])],
        row_names=[name for name, _ in rows],
    )


def read_opendice(path: str | Path) -> MeasurementSet:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    result = parse_opendice(text)
    result.source_path = path
    return result
