"""CGATS.17 reference-measurement parser.

Design rules, all of them consequences of the primer:

* Field order is read from ``BEGIN_DATA_FORMAT`` / ``END_DATA_FORMAT``. We never
  assume a fixed column layout -- different instruments and different versions of
  i1Profiler emit different column sets in different orders.
* Whichever of Lab / XYZ / spectral is present is stored as-is. Conversion happens
  on demand, from the highest-fidelity representation available.
* Duplicate ``SAMPLE_NAME`` entries are detected and refused, never silently
  overwritten. This is not hypothetical: the reference file for this project's own
  ColorChecker SG listed K8 twice with conflicting values, the second entry being
  a mislabelled K9. A last-write-wins parser would have quietly poisoned every
  Delta E result involving that patch.
* Measurement provenance metadata is preserved and surfaced in reports, because
  "which illuminant / which observer / which instrument" is part of the result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence


from deltachrome.errors import CGATSParseError, DuplicateSampleError

# Provenance keywords worth carrying into the report. Anything else in the file
# header is still kept in `MeasurementSet.metadata`; this list only drives which
# fields get promoted in human-facing output.
#: What a population mean has to state to be believed: its held-out error against
#: the units it was built from, and those units' own spread. Held out because a
#: mean scored against its own members is fitted to what it is tested on.
HELD_OUT_KEY = "DELTACHROME_NOMINAL_HELD_OUT_DE"
POPULATION_SPREAD_KEY = "DELTACHROME_NOMINAL_POPULATION_SPREAD_DE"
POPULATION_UNITS_KEY = "DELTACHROME_NOMINAL_POPULATION_UNITS"

PROVENANCE_KEYWORDS: tuple[str, ...] = (
    "ORIGINATOR",
    "DESCRIPTOR",
    "CREATED",
    "INSTRUMENTATION",
    "MEASUREMENT_SOURCE",
    "ILLUMINATION_NAME",
    "OBSERVER_ANGLE",
    "ILLUMINANT",
    "OBSERVER",
    "DELTACHROME_SPECTRA_INTEGRATED",
    "SERIAL",
    "CALIBRATION_DATE",
    "FILTER",
    "WEIGHTING_FUNCTION",
    "DEVCALSTD",
    "MANUFACTURER",
    "TARGET_INSTRUMENT",
    "DELTACHROME_REFERENCE_BASIS",
    "DELTACHROME_NOMINAL_SOURCE",
    HELD_OUT_KEY,
    POPULATION_SPREAD_KEY,
    POPULATION_UNITS_KEY,
)

#: What the reference numbers actually are. Not cosmetic: a Delta E against a
#: measurement of *this* chart is the capture system's error, while a Delta E
#: against the manufacturer's aim values for the *model* also contains whatever
#: this specimen's print differs from the aim by. The two are different quantities
#: and the difference has to survive into the report.
PER_UNIT_MEASUREMENT = "per_unit_measurement"
MANUFACTURER_NOMINAL = "manufacturer_nominal"

#: A mean of many individually measured units of one model, **demonstrated to sit
#: inside that model's own population**. Between the two above and genuinely so: it
#: is not a measurement of the chart in front of the camera, so it earns no
#: conformance claim, and it is not a printed aim either -- the error it introduces
#: is smaller than the difference between two real charts of the model.
#:
#: That last clause is the whole of the definition and it is checked rather than
#: taken on trust: see :attr:`MeasurementSet.basis`. A file may not simply declare
#: itself centred.
POPULATION_MEAN = "population_mean"

#: Every basis this version understands. Anything else a file declares is treated
#: as nominal -- see :attr:`MeasurementSet.basis` for why that direction.
KNOWN_BASES = (PER_UNIT_MEASUREMENT, POPULATION_MEAN, MANUFACTURER_NOMINAL)


_SAMPLE_NAME_FIELDS = ("SAMPLE_NAME", "SAMPLEID", "SAMPLE_ID", "SampleID", "SAMPLE")
_LAB_FIELDS = {"L": ("LAB_L",), "a": ("LAB_A",), "b": ("LAB_B",)}
_XYZ_FIELDS = {"X": ("XYZ_X",), "Y": ("XYZ_Y",), "Z": ("XYZ_Z",)}
_RGB_FIELDS = {"R": ("RGB_R",), "G": ("RGB_G",), "B": ("RGB_B",)}

# i1Profiler writes SPECTRAL_NM_380; other tools write nm380, SPECTRAL_380, R_380.
# SPEC_380 is the OpenQualia Measurement File Standard's own preferred spelling,
# and it is the one every OQM file published by DT Nexus uses -- so without it a
# 130-patch chart's 36 measured bands per patch were read as no spectra at all.
_SPECTRAL_RE = re.compile(
    r"^(?:SPECTRAL_NM_|SPECTRAL_|SPEC_|NM_|NM|R_)(\d{3}(?:\.\d+)?)$", re.IGNORECASE
)

_TOKEN_RE = re.compile(r'"([^"]*)"|(\S+)')


def _tokenize(line: str) -> list[str]:
    """Split a CGATS line, honouring double-quoted values that contain spaces."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(line):
        quoted, bare = match.group(1), match.group(2)
        tokens.append(quoted if quoted is not None else bare)
    return tokens


def _strip_comment(line: str) -> str:
    """Remove a trailing ``#`` comment that is not inside a quoted string."""
    out: list[str] = []
    in_quote = False
    for ch in line:
        if ch == '"':
            in_quote = not in_quote
        elif ch == "#" and not in_quote:
            break
        out.append(ch)
    return "".join(out)


def _maybe_number(text: str) -> float | str:
    try:
        return float(text)
    except ValueError:
        return text


@dataclass(frozen=True)
class PatchMeasurement:
    """One measured sample from a reference file.

    Only the representations actually present in the file are populated. Nothing
    is back-filled with a default -- an absent representation stays ``None`` so
    callers can tell "not measured" from "measured as zero".
    """

    sample_name: str
    lab: tuple[float, float, float] | None = None
    xyz: tuple[float, float, float] | None = None
    rgb: tuple[float, float, float] | None = None
    spectral: dict[float, float] | None = None
    raw: dict[str, float | str] = field(default_factory=dict)

    @property
    def has_colorimetry(self) -> bool:
        return self.lab is not None or self.xyz is not None or bool(self.spectral)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.lab is not None:
            L, a, b = self.lab
            return f"<PatchMeasurement {self.sample_name} Lab=({L:.2f}, {a:.2f}, {b:.2f})>"
        return f"<PatchMeasurement {self.sample_name}>"


@dataclass
class CGATSTable:
    """One ``BEGIN_DATA``/``END_DATA`` block plus the header that preceded it."""

    fields: list[str]
    rows: list[list[float | str]]
    metadata: dict[str, str]

    def column(self, name: str) -> list[float | str]:
        try:
            idx = self.fields.index(name)
        except ValueError as exc:
            raise CGATSParseError(f"No such field {name!r}; file has {self.fields}") from exc
        return [row[idx] for row in self.rows]


@dataclass
class MeasurementSet:
    """Parsed reference data, keyed by sample name.

    This is the object the rest of the engine consumes. It is deliberately not a
    plain dict: the provenance metadata travels with the numbers, because a Delta E
    computed against a D65 measurement and one computed against a D50 measurement
    are different quantities that would otherwise look identical in a report.
    """

    samples: dict[str, PatchMeasurement]
    metadata: dict[str, str]
    fields: list[str]
    source_path: Path | None = None
    #: Which of the file's tables these samples came from, 1-based.
    table_index: int = 1
    #: One descriptor per table the file holds, so a caller -- or a future UI -- can
    #: offer the choice rather than having to re-parse to discover it exists.
    tables_available: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: Each row's patch name exactly as the file gave it, in file order, for a
    #: format that may repeat a name and leaves position to say which is which.
    #: Empty for every format that names each patch once.
    row_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __contains__(self, name: object) -> bool:
        return name in self.samples

    def __iter__(self) -> Iterator[PatchMeasurement]:
        return iter(self.samples.values())

    def __getitem__(self, name: str) -> PatchMeasurement:
        return self.samples[name]

    def get(self, name: str) -> PatchMeasurement | None:
        return self.samples.get(name)

    @property
    def illuminant(self) -> str | None:
        """Which illuminant the file's colorimetry assumes.

        Two spellings, because two families of tool write these files. i1Profiler
        and the CGATS files this project started from write ``ILLUMINATION_NAME``;
        the OpenQualia standard requires the shorter ``ILLUMINANT`` beside
        ``OBSERVER`` whenever a file states Lab or XYZ. Reading only the first
        meant every OQM file reported no illuminant and fell through to the D50
        default -- correct for the files published so far, and correct by luck
        rather than by reading what they say.
        """
        return self.metadata.get("ILLUMINATION_NAME") or self.metadata.get("ILLUMINANT")

    @property
    def observer(self) -> str | None:
        return self.metadata.get("OBSERVER_ANGLE") or self.metadata.get("OBSERVER")

    @property
    def instrument(self) -> str | None:
        return self.metadata.get("INSTRUMENTATION") or self.metadata.get("TARGET_INSTRUMENT")

    @property
    def basis(self) -> str:
        """What these numbers are: one of :data:`KNOWN_BASES`.

        Defaults to a per-unit measurement when nothing is declared, because that is
        what an ordinary reference file from a spectrophotometer is. Anything that
        *is* declared and is not recognised falls to ``manufacturer_nominal``.

        **That direction matters and it used to run the other way.** The test was
        ``MANUFACTURER_NOMINAL if declared == MANUFACTURER_NOMINAL else
        PER_UNIT_MEASUREMENT``, so every unrecognised value became a per-unit
        measurement and earned a full conformance claim -- which meant adding any
        new basis to this module would silently have promoted the files carrying it.
        A basis this version does not understand cannot be quoted as conformance to
        anything, so it withholds rather than grants.

        **A population mean is earned, not asserted.** A file declaring itself one
        must also state its held-out error and its population's own spread, and the
        first must be smaller than the second -- otherwise the mean sits further
        from an average chart than two charts sit from each other, which makes it a
        worse predictor of the chart in front of you than a randomly chosen other
        unit would be. That is not hypothetical: the Golden Thread Object Level
        Target's printed table reads 0.853 against a 0.610 unit-to-unit spread,
        where the 40-unit mean that replaced it reads 0.457. Same model, same
        format, opposite answers. A file failing the check is nominal.
        """
        declared = (self.metadata.get("DELTACHROME_REFERENCE_BASIS") or "").strip().lower()
        if not declared:
            return PER_UNIT_MEASUREMENT
        if declared == PER_UNIT_MEASUREMENT:
            return PER_UNIT_MEASUREMENT
        if declared == POPULATION_MEAN and self.population_is_centred:
            return POPULATION_MEAN
        return MANUFACTURER_NOMINAL

    @property
    def held_out_de(self) -> float | None:
        """How far this mean sits from a unit it was not built from, in dE2000."""
        return _number(self.metadata.get(HELD_OUT_KEY))

    @property
    def population_spread_de(self) -> float | None:
        """How far two units of this model sit from each other, in dE2000."""
        return _number(self.metadata.get(POPULATION_SPREAD_KEY))

    @property
    def population_units(self) -> int | None:
        """How many measured units the mean was built from."""
        value = _number(self.metadata.get(POPULATION_UNITS_KEY))
        return None if value is None else int(value)

    @property
    def population_is_centred(self) -> bool:
        """Whether the stated figures put this mean inside its own population."""
        held_out, spread = self.held_out_de, self.population_spread_de
        if held_out is None or spread is None or spread <= 0:
            return False
        return held_out < spread

    @property
    def is_nominal(self) -> bool:
        """Whether these numbers describe the model rather than *this* chart.

        True for a population mean as well as for a printed table. The distinction
        between those two is real and is carried by :attr:`basis`; what this asks is
        the cruder question every caller of it wants -- was the chart in front of the
        camera itself measured.
        """
        return self.basis != PER_UNIT_MEASUREMENT

    @property
    def nominal_caveat(self) -> str | None:
        return self.metadata.get("DELTACHROME_NOMINAL_CAVEAT")

    @property
    def measurement_mode(self) -> str | None:
        """The M-mode (M0/M1/M2) if the file records one, else ``None``."""
        return _measurement_mode(self.metadata)

    def provenance(self) -> dict[str, str]:
        """The subset of metadata worth printing in a report header."""
        return {k: self.metadata[k] for k in PROVENANCE_KEYWORDS if k in self.metadata}

    def coverage(self, sample_names: Sequence[str]) -> tuple[list[str], list[str]]:
        """Split ``sample_names`` into (present, missing) against this set."""
        present = [n for n in sample_names if n in self.samples]
        missing = [n for n in sample_names if n not in self.samples]
        return present, missing


def read_cgats(
    path: str | Path, *, encoding: str = "utf-8-sig", table: int | str | None = None
) -> MeasurementSet:
    """Read and parse a CGATS file from disk."""
    path = Path(path)
    try:
        text = path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        # i1Profiler occasionally emits latin-1 in the DESCRIPTOR field.
        text = path.read_text(encoding="latin-1")
    result = parse_cgats(text, table=table)
    result.source_path = path
    return result



def _measurement_mode(metadata: dict[str, str]) -> str | None:
    for key in ("MEASUREMENT_SOURCE", "FILTER", "DESCRIPTOR"):
        value = metadata.get(key)
        if value:
            match = re.search(r"\bM[0123]\b", value)
            if match:
                return match.group(0)
    return None


def _describe_table(number: int, table: CGATSTable) -> dict[str, Any]:
    return {
        "table": number,
        "rows": len(table.rows),
        "measurement_mode": _measurement_mode(table.metadata),
        "descriptor": table.metadata.get("DESCRIPTOR"),
        "fields": list(table.fields),
    }


def _choose_table(
    tables: list[CGATSTable], available: list[dict[str, Any]], wanted: int | str | None
) -> int:
    """Index of the table to use. A request must be honoured exactly or refused,
    never silently fall back.

    The default is the one table measured under **M1** where there is one, and the
    first otherwise. M1 is the condition ISO 13655 defines for a D50 viewing
    environment, which is what every comparison here is made under; a file
    exporting M0, M1 and M2 side by side lists M0 first, so "the first" used to
    choose the condition least like the one being judged.
    """
    if wanted is None:
        m1 = [d["table"] for d in available if (d["measurement_mode"] or "") == "M1"]
        return m1[0] - 1 if len(m1) == 1 else 0
    if isinstance(wanted, int) or (isinstance(wanted, str) and wanted.strip().isdigit()):
        number = int(wanted)
        if not 1 <= number <= len(tables):
            raise CGATSParseError(
                f"Table {number} was requested but the file holds {len(tables)}. "
                f"Available: {_table_listing(available)}."
            )
        return number - 1
    wanted_mode = str(wanted).strip().upper()
    matches = [d["table"] for d in available if (d["measurement_mode"] or "") == wanted_mode]
    if not matches:
        raise CGATSParseError(
            f"No table in this file is measured under {wanted_mode}. "
            f"Available: {_table_listing(available)}."
        )
    if len(matches) > 1:
        raise CGATSParseError(
            f"{len(matches)} tables in this file are measured under {wanted_mode} "
            f"(tables {matches}); name one by number instead."
        )
    return matches[0] - 1


def _table_listing(available: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"table {d['table']} ({d['rows']} row{'' if d['rows'] == 1 else 's'}"
        + (f", {d['measurement_mode']}" if d["measurement_mode"] else "")
        + (f", {d['descriptor']}" if d["descriptor"] else "")
        + ")"
        for d in available
    )


def _table_choice_warnings(
    tables: list[CGATSTable],
    available: list[dict[str, Any]],
    index: int,
    requested: int | str | None,
) -> list[str]:
    """Say what was chosen, and how much the choice could have mattered.

    Nearly every reference file holds one table, so this is silent in the ordinary
    case. Where it is not, the thing worth knowing is not that several tables exist
    but whether they were measured under *different conditions* -- two tables of one
    M1 pass are a formatting detail, while M0 and M1 of the same chart are different
    numbers, and picking between them changes every Delta E in the report.
    """
    if len(tables) == 1:
        declared = tables[0].metadata.get("NUMBER_OF_TABLES")
        if declared and declared.strip() not in ("1", "1.0"):
            return [
                f"The file's header declares NUMBER_OF_TABLES {declared} but only one "
                "BEGIN_DATA block was found. The header and the file disagree; the "
                "data found is what was used."
            ]
        return []

    # A table somebody named needs no announcing, and tables of one condition are a
    # formatting detail. What is worth a line is a default chosen between
    # measurement conditions, since that choice changes every Delta E.
    modes = sorted({d["measurement_mode"] for d in available} - {None})
    if requested is not None or len(modes) < 2:
        return []
    chosen = available[index]["measurement_mode"] or f"table {index + 1}"
    return [
        f"Reference file holds several measurements ({', '.join(modes)}); "
        f"{chosen} used."
    ]


def _number(value: object) -> float | None:
    """A declared figure as a float, or ``None`` where it is absent or unreadable."""
    if value is None:
        return None
    try:
        return float(str(value).strip().strip('"'))
    except (TypeError, ValueError):
        return None


def parse_cgats(text: str, *, table: int | str | None = None) -> MeasurementSet:
    """Parse CGATS text into a :class:`MeasurementSet`.

    ``table`` picks one of a multi-table file's tables, either by 1-based number or
    by measurement condition (``"M1"``). The default is the table measured under M1
    where there is exactly one, and the first otherwise -- nearly every reference
    file holds one table -- and it is announced whenever the default chose between
    measurement conditions.
    """
    tables = _parse_tables(text)
    if not tables:
        raise CGATSParseError(
            "No BEGIN_DATA/END_DATA block found. This does not look like a CGATS file."
        )

    warnings: list[str] = []
    available = [_describe_table(i, t) for i, t in enumerate(tables, start=1)]
    index = _choose_table(tables, available, table)
    warnings.extend(_table_choice_warnings(tables, available, index, table))
    table_index = index + 1
    table = tables[index]

    name_field = _find_field(table.fields, _SAMPLE_NAME_FIELDS)
    if name_field is None:
        raise CGATSParseError(
            f"No sample-name field found. Looked for {list(_SAMPLE_NAME_FIELDS)}, "
            f"file declares {table.fields}."
        )

    lab_idx = _triplet_indices(table.fields, _LAB_FIELDS)
    xyz_idx = _triplet_indices(table.fields, _XYZ_FIELDS)
    rgb_idx = _triplet_indices(table.fields, _RGB_FIELDS)
    spectral_idx = _spectral_indices(table.fields)
    name_idx = table.fields.index(name_field)

    if lab_idx is None and xyz_idx is None and not spectral_idx:
        raise CGATSParseError(
            "File contains no colorimetric data (no Lab, no XYZ, no spectral fields). "
            f"Fields present: {table.fields}."
        )

    samples: dict[str, PatchMeasurement] = {}
    duplicates: dict[str, list[dict[str, float | str]]] = {}

    for row in table.rows:
        raw = dict(zip(table.fields, row))
        name = str(row[name_idx]).strip()
        # Numeric sample ids come back as floats from the tokenizer; 12.0 -> "12".
        if isinstance(row[name_idx], float) and float(row[name_idx]).is_integer():
            name = str(int(row[name_idx]))

        measurement = PatchMeasurement(
            sample_name=name,
            lab=_triplet(row, lab_idx),
            xyz=_triplet(row, xyz_idx),
            rgb=_triplet(row, rgb_idx),
            spectral={nm: float(row[i]) for nm, i in spectral_idx} or None,
            raw=raw,
        )

        if name in samples:
            duplicates.setdefault(name, [samples[name].raw]).append(raw)
            continue
        samples[name] = measurement

    if duplicates:
        raise DuplicateSampleError(_duplicate_report(duplicates))

    declared = table.metadata.get("NUMBER_OF_SETS")
    if declared is not None:
        try:
            if int(float(declared)) != len(table.rows):
                warnings.append(
                    f"The file says it holds {int(float(declared))} patches; "
                    f"{len(table.rows)} were found."
                )
        except ValueError:
            warnings.append(f"The file's patch count is not a number: {declared!r}")

    return MeasurementSet(
        samples=samples,
        metadata=table.metadata,
        fields=table.fields,
        table_index=table_index,
        tables_available=available,
        warnings=warnings,
    )


def _duplicate_report(duplicates: dict[str, list[dict[str, float | str]]]) -> str:
    """Build an actionable error message for duplicate sample names.

    Distinguishes benign exact repeats from conflicting ones, and prints the
    fields that actually differ, so the operator can identify which entry is the
    mislabelled one rather than just being told "there is a duplicate".
    """
    lines = [
        f"Reference file contains {len(duplicates)} duplicated SAMPLE_NAME "
        f"value(s). Refusing to guess which is correct."
    ]
    for name, entries in sorted(duplicates.items()):
        differing = sorted(
            {
                key
                for key in entries[0]
                if len({_fmt(e.get(key)) for e in entries}) > 1
            }
        )
        if not differing:
            lines.append(f"  {name}: {len(entries)} identical entries.")
            continue
        lines.append(f"  {name}: {len(entries)} entries differing in {', '.join(differing)}")
        for n, entry in enumerate(entries, start=1):
            detail = "  ".join(f"{key}={_fmt(entry.get(key))}" for key in differing)
            lines.append(f"    #{n}: {detail}")
    lines.append(
        "Fix the file (or correct the mislabelled entry) and re-import. "
        "Known real-world case: a ColorChecker SG file listing K8 twice, where the "
        "second entry was a mislabelled K9 -- confirmable against neighboring "
        "column values."
    )
    return "\n".join(lines)


def _fmt(value: float | str | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _find_field(fields: Sequence[str], candidates: Sequence[str]) -> str | None:
    lowered = {f.lower(): f for f in fields}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _triplet_indices(
    fields: Sequence[str], spec: dict[str, tuple[str, ...]]
) -> tuple[int, int, int] | None:
    found = []
    for candidates in spec.values():
        name = _find_field(fields, candidates)
        if name is None:
            return None
        found.append(fields.index(name))
    return (found[0], found[1], found[2])


def _spectral_indices(fields: Sequence[str]) -> list[tuple[float, int]]:
    out: list[tuple[float, int]] = []
    for idx, name in enumerate(fields):
        match = _SPECTRAL_RE.match(name)
        if match:
            out.append((float(match.group(1)), idx))
    return sorted(out)


def _triplet(
    row: Sequence[float | str], idx: tuple[int, int, int] | None
) -> tuple[float, float, float] | None:
    if idx is None:
        return None
    try:
        return (float(row[idx[0]]), float(row[idx[1]]), float(row[idx[2]]))
    except (TypeError, ValueError):
        return None


def _parse_tables(text: str) -> list[CGATSTable]:
    """Walk the file line by line, accumulating header keywords and data blocks.

    Keywords are scoped. Everything before the first table is the file's own header
    and belongs to every table in it; everything after a table's ``END_DATA``
    describes the *next* table. Accumulating them all into one dict, as an earlier
    version did, gave the second table of a multi-table file the first table's
    provenance as well as its own -- which matters most exactly where multi-table
    files matter most, when the tables are measurements under different conditions
    and the header is what says which is which.
    """
    global_metadata: dict[str, str] = {}
    global_repeated: dict[str, list[str]] = {}
    metadata: dict[str, str] = global_metadata
    repeated: dict[str, list[str]] = global_repeated
    header_closed = False
    declared_keywords: set[str] = set()
    tables: list[CGATSTable] = []

    fields: list[str] | None = None
    rows: list[list[float | str]] | None = None
    mode: str | None = None
    pending_fields: list[str] = []

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        upper = line.upper()

        if upper == "BEGIN_DATA_FORMAT":
            if not header_closed:
                header_closed = True
                metadata, repeated = {}, {}
            mode, pending_fields = "format", []
            continue
        if upper == "END_DATA_FORMAT":
            if not pending_fields:
                raise CGATSParseError(f"Empty DATA_FORMAT block ending at line {lineno}.")
            fields, mode = pending_fields, None
            continue
        if upper == "BEGIN_DATA":
            if fields is None:
                raise CGATSParseError(
                    f"BEGIN_DATA at line {lineno} without a preceding "
                    "BEGIN_DATA_FORMAT block; column meanings are unknowable."
                )
            mode, rows = "data", []
            continue
        if upper == "END_DATA":
            if rows is None or fields is None:
                raise CGATSParseError(f"END_DATA at line {lineno} without BEGIN_DATA.")
            tables.append(
                CGATSTable(
                    fields=list(fields),
                    rows=rows,
                    metadata={**global_metadata, **metadata},
                )
            )
            metadata, repeated = {}, {}
            mode, rows = None, None
            continue

        if mode == "format":
            pending_fields.extend(_tokenize(line))
            continue

        if mode == "data":
            assert rows is not None and fields is not None
            values = [_maybe_number(tok) for tok in _tokenize(line)]
            if len(values) != len(fields):
                raise CGATSParseError(
                    f"Line {lineno} has {len(values)} values but DATA_FORMAT declares "
                    f"{len(fields)} fields ({fields}). Line: {line!r}"
                )
            rows.append(values)
            continue

        # Header region: KEYWORD "value", or a bare declaration line.
        tokens = _tokenize(line)
        if not tokens:
            continue
        key = tokens[0]
        if key.upper() == "KEYWORD" and len(tokens) > 1:
            declared_keywords.add(tokens[1])
            continue
        name = key.upper() if key.isupper() or "_" in key else key
        value = " ".join(tokens[1:]) if len(tokens) > 1 else ""
        # A keyword may legitimately appear more than once -- i1Profiler writes two
        # WEIGHTING_FUNCTION lines, one for the illuminant and one for the observer.
        # Overwriting would silently discard half the measurement's provenance.
        repeated.setdefault(name, []).append(value)
        metadata[name] = "; ".join(repeated[name])

    if mode is not None:
        raise CGATSParseError(f"File ends inside an unterminated {mode.upper()} block.")

    return tables
