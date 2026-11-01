"""A library of per-unit reference files, indexed by chart identifier.

The problem this solves: reference measurements are *per physical unit*, and two
charts of the same model look identical in a photograph. Pointing a capture at the
wrong unit's reference produces numbers that look entirely reasonable and mean
nothing. Relying on an operator to remember which chart was in the frame is exactly
the kind of silent failure this project refuses elsewhere.

So a chart's own machine-readable identifier does the pairing. Both ColorChecker SG
units here carry a QR code (``CCSG21``, ``CCSG43``); the LaserSoft ISO 12641-2
target carries a barcode whose value is also the ``TARGET_SERIAL`` inside its CxF.

Identifiers come from three places, in descending order of authority:

1. **Inside the file** -- CxF vendor tags (``Serial``), or a ``TARGET_SERIAL`` /
   ``DELTACHROME_TARGET_ID`` keyword in a CGATS header. Self-describing and
   survives renaming.
2. **The filename stem** -- naming a file ``CCSG21.txt`` is the zero-effort path.
3. **A manifest** -- explicit aliases in ``deltachrome-references.json``, for files
   that carry no identifier and cannot be renamed.

A match is never applied silently: :meth:`ReferenceLibrary.resolve` returns what it
found and why, and the caller confirms.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from deltachrome.errors import MissingReferenceError
from deltachrome.imaging.codes import normalize_identifier

MANIFEST_NAME = "deltachrome-references.json"
REFERENCE_SUFFIXES = (".txt", ".cgats", ".cgt", ".cxf", ".xml", ".it8")

_CGATS_ID_KEYWORDS = ("DELTACHROME_TARGET_ID", "TARGET_SERIAL", "SERIAL", "TARGET_ID")
_CXF_TAG = re.compile(r'<Tag\s+Name="(?:\d+_)?([^"]+)"\s+Value="([^"]*)"', re.I)
_CXF_DESC = re.compile(r"<Description>([^<]*)</Description>", re.I)

#: A LaserSoft Imaging serial as printed beside the barcode: a letter for the
#: product line, a two-digit year, a month, and the unit or batch number.
_LASERSOFT_SERIAL = re.compile(r"^([RTE])(\d)(\d)(0[1-9]|1[0-2])(\d{2,})$")
#: The digit that stands for each product line in the barcode. R is a reflective
#: IT8.7/2, T an ISO 12641-2 transparency, E an IT8.7/1 transparency.
_LASERSOFT_LINE = {"R": "0", "T": "1", "E": "2"}


def lasersoft_barcode(serial: str) -> str | None:
    """What a LaserSoft chart's barcode carries, from the serial printed beside it.

    **The barcode is not the serial.** It is the same fields rewritten: the product
    letter becomes a digit, the year loses its decade, and twenty is added to the
    month. Measured bar by bar on all four LaserSoft charts here, and the rule
    reproduces every one with nothing fitted to any single chart:

    ========== ========== =========
    printed    rule       bars read
    ========== ========== =========
    T1910021   1 9 30 021 1930021
    T1710030   1 7 30 030 1730030
    R100213    0 0 22 13  002213
    E180106    2 8 21 06  282106
    ========== ========== =========

    It was taken for a misread for as long as only the ISO 12641-2 was looked at,
    because there the one changed digit is the month's tens -- "the third digit
    reads 3 where the card prints 1" -- and the bars do encode a 3: the same card
    draws its other two 1s as 1s. Why LaserSoft encode it this way is not known;
    an internal production code is the likely reading.

    **It runs forward only.** Going back from a barcode to a serial needs the
    decade the barcode dropped, so a capture's code is never turned into a
    serial; each bank file's serial is turned into the barcode it would carry,
    and the two are compared. Returns ``None`` for anything not shaped like a
    LaserSoft serial.
    """
    match = _LASERSOFT_SERIAL.match(serial.strip().upper())
    if not match:
        return None
    line, _decade, year, month, unit = match.groups()
    return f"{_LASERSOFT_LINE[line]}{year}{int(month) + 20}{unit}"


@dataclass
class ReferenceEntry:
    """One reference file in the library and the identifiers it answers to."""

    path: Path
    identifiers: set[str] = field(default_factory=set)
    serial: str | None = None
    description: str | None = None
    target_type: str | None = None
    format: str = "cgats"
    aliases: set[str] = field(default_factory=set)
    #: Barcodes this file's serial implies under a maker's own encoding -- see
    #: :func:`lasersoft_barcode`. Derived rather than stated, so it is kept apart
    #: from :attr:`identifiers`, which a bank listing shows as what the file says.
    encoded: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        return self.path.name

    def describe(self) -> str:
        bits = [self.name]
        if self.serial:
            bits.append(f"serial {self.serial}")
        if self.target_type:
            bits.append(self.target_type)
        if self.description:
            bits.append(self.description[:60])
        return "  |  ".join(bits)


@dataclass
class Resolution:
    """The outcome of trying to pair a capture with a reference."""

    entry: ReferenceEntry | None
    matched_code: str | None
    reason: str
    candidates: list[ReferenceEntry] = field(default_factory=list)
    ambiguous: bool = False

    @property
    def path(self) -> Path | None:
        return self.entry.path if self.entry else None


def peek_identity(path: Path, *, head_bytes: int = 16384) -> ReferenceEntry:
    """Read just enough of a reference file to identify it.

    Deliberately does not parse the whole file: a CxF with 864 spectral patches is
    a megabyte of XML, and indexing a library should not cost a full parse per file.
    """
    raw = path.open("rb").read(head_bytes)
    try:
        head = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        head = raw.decode("latin-1", errors="replace")

    entry = ReferenceEntry(path=path)
    stem = normalize_identifier(path.stem)
    if stem:
        entry.identifiers.add(stem)

    from deltachrome.reference import sniff_format

    if sniff_format(head) == "cxf":
        entry.format = "cxf"
        for name, value in _CXF_TAG.findall(head):
            key = name.strip().upper()
            if key in ("SERIAL", "SERIALNUMBER") and value:
                entry.serial = value
                entry.identifiers.add(normalize_identifier(value))
            elif key == "TARGETTYPE":
                entry.target_type = value
        match = _CXF_DESC.search(head)
        if match:
            entry.description = match.group(1).strip()
        return _with_encoded(entry)

    entry.format = "cgats"
    for line in head.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith(("BEGIN_DATA", "BEGIN_DATA_FORMAT")):
            break
        parts = stripped.replace("\t", " ").split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts[0].upper(), parts[1].strip().strip('"').strip()
        if key in _CGATS_ID_KEYWORDS and value:
            entry.serial = entry.serial or value
            entry.identifiers.add(normalize_identifier(value))
        elif key == "DESCRIPTOR" and value:
            entry.description = value
    return _with_encoded(entry)


def lasersoft_serials(code: str) -> list[str]:
    """The printed serials a LaserSoft barcode could have come from.

    More than one, because the barcode keeps only the last digit of the year:
    ``1730030`` is ``T1710030`` or ``T2710030``. Used to tell an operator which
    file to look for, never to pair -- pairing runs forward, from the bank's
    serials, where nothing is ambiguous.
    """
    match = re.fullmatch(r"([012])(\d)(\d\d)(\d{2,})", code.strip())
    if not match:
        return []
    line, year, month, unit = match.groups()
    if not 1 <= int(month) - 20 <= 12:
        return []
    letter = {digit: letter for letter, digit in _LASERSOFT_LINE.items()}[line]
    return [f"{letter}{decade}{year}{int(month) - 20:02d}{unit}" for decade in "12"]


def _serial_hint(codes: Iterable[str]) -> str:
    """A sentence naming the LaserSoft serial a 2-of-5 barcode stands for, if any."""
    for code in sorted(codes, key=len, reverse=True):
        serials = lasersoft_serials(code)
        if serials:
            return (f" If this is a LaserSoft chart, barcode {code} is printed as "
                    f"serial {serials[0]} (or {serials[1]}, the barcode not recording "
                    "the decade); add that unit's file to the bank.")
    return ""


def _with_encoded(entry: ReferenceEntry) -> ReferenceEntry:
    """Add the barcodes the file's serial, or failing that its name, implies."""
    for source in (entry.serial, entry.path.stem):
        code = lasersoft_barcode(source) if source else None
        if code:
            entry.encoded.add(normalize_identifier(code))
    return entry


class ReferenceLibrary:
    """An indexed directory of per-unit reference files."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise MissingReferenceError(f"Reference library is not a directory: {directory}")
        self.manifest_path = self.directory / MANIFEST_NAME
        self.entries: list[ReferenceEntry] = []
        self._index: dict[str, list[ReferenceEntry]] = {}
        self.reload()

    # -- building -----------------------------------------------------------

    def reload(self) -> None:
        self.entries = []
        for path in sorted(self.directory.iterdir()):
            if path.name == MANIFEST_NAME or not path.is_file():
                continue
            if path.suffix.lower() not in REFERENCE_SUFFIXES:
                continue
            try:
                self.entries.append(peek_identity(path))
            except OSError:
                continue

        for code, target in self._manifest().items():
            key = normalize_identifier(code)
            entry = next((e for e in self.entries if e.path.name == target), None)
            if entry is not None:
                entry.identifiers.add(key)
                entry.aliases.add(code)

        self._index = {}
        self._encoded_index: dict[str, list[ReferenceEntry]] = {}
        for entry in self.entries:
            for identifier in entry.identifiers:
                self._index.setdefault(identifier, []).append(entry)
            for code in entry.encoded:
                self._encoded_index.setdefault(code, []).append(entry)

    def manifest(self) -> dict:
        """The whole manifest, or an empty one.

        Read and written whole rather than key by key, because this application
        is not the only thing entitled to put something in it: a manifest a user
        hand-wrote may carry keys this version has never heard of, and rewriting
        one binding must not delete them.
        """
        if not self.manifest_path.exists():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MissingReferenceError(f"{self.manifest_path}: invalid JSON -- {exc}") from exc
        return data if isinstance(data, dict) else {}

    def _manifest(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in self.manifest().get("bindings", {}).items()}

    def _write_manifest(self, data: dict) -> Path:
        """Write the manifest, or remove it when there is nothing left to say.

        An empty manifest and no manifest are the same state to every reader, and
        leaving a file behind saying ``{}`` makes an untouched library look
        configured.
        """
        if any(data.values()):
            self.manifest_path.write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        elif self.manifest_path.exists():
            self.manifest_path.unlink()
        self.reload()
        return self.manifest_path

    def added_on(self, filename: str) -> str | None:
        """When a file entered this library, if this application put it there.

        An ISO 8601 string written at the time of the copy, rather than the
        file's own mtime: a copy that preserves the timestamp reports when the
        *measurement* was made, and one that does not is changed by anything that
        touches the file. Neither answers "when did this arrive here", which is
        the question a bank listing is asked.
        """
        value = self.manifest().get("added", {}).get(filename)
        return str(value) if value else None

    # -- querying -----------------------------------------------------------

    def find(self, code: str) -> list[ReferenceEntry]:
        return list(self._index.get(normalize_identifier(code), []))

    def resolve(
        self, codes: Iterable[str], *, standard_2of5: Iterable[str] = ()
    ) -> Resolution:
        """Pair a set of decoded codes with a reference file.

        An ambiguous match -- one code matching several files, or several codes each
        matching a different file -- is reported as ambiguous rather than resolved by
        picking one. Guessing here would defeat the whole point of the pairing.

        ``standard_2of5`` names the codes that were read as Standard 2 of 5, which
        is LaserSoft's symbology. Only those, and only when nothing matches as
        read, are compared against the barcodes the bank's LaserSoft serials imply
        (:func:`lasersoft_barcode`) -- so a chart that pairs today pairs the same
        way, and a code read off some other chart is never reinterpreted.
        """
        codes = [c for c in codes if c]
        if not codes:
            return Resolution(
                entry=None,
                matched_code=None,
                reason="No identification code was decoded from the capture.",
            )

        hits: list[tuple[str, ReferenceEntry]] = []
        for code in codes:
            for entry in self.find(code):
                hits.append((code, entry))
        encoded = False
        if not hits:
            for code in standard_2of5:
                for entry in self._encoded_index.get(normalize_identifier(code), []):
                    hits.append((code, entry))
            encoded = bool(hits)

        if not hits:
            return Resolution(
                entry=None,
                matched_code=None,
                reason=(
                    f"Decoded {', '.join(codes)}, but no reference file in "
                    f"{self.directory} answers to any of them. Name the file after the "
                    "code, add a TARGET_SERIAL keyword to it, or bind it explicitly."
                    + _serial_hint(standard_2of5)
                ),
                candidates=list(self.entries),
            )

        distinct = {entry.path for _, entry in hits}
        if len(distinct) > 1:
            return Resolution(
                entry=None,
                matched_code=None,
                ambiguous=True,
                reason=(
                    f"Decoded {', '.join(codes)}, which matches "
                    f"{len(distinct)} different reference files. Refusing to choose."
                ),
                candidates=[entry for _, entry in hits],
            )

        code, entry = hits[0]
        reason = f"Code {code!r} matched {entry.name}."
        if encoded:
            serial = entry.serial or entry.path.stem
            reason = (f"Barcode {code!r} is LaserSoft's encoding of serial "
                      f"{serial}, and matched {entry.name}.")
        return Resolution(
            entry=entry,
            matched_code=code,
            reason=reason,
            candidates=[entry],
        )

    # -- editing ------------------------------------------------------------

    def bind(self, code: str, filename: str) -> Path:
        """Record an explicit code to file binding in the manifest.

        This is what a chart whose printed code disagrees with its own file needs:
        a file stating no serial -- i1Profiler writes none -- answers to nothing a
        chart prints, and binding the printed code pairs the two without renaming
        a file whose name is the thing a person reads. (The LaserSoft's barcode,
        which looked like the same case, is an encoding of its serial and pairs
        without one -- :func:`lasersoft_barcode`.)
        """
        target = self.directory / filename
        if not target.exists():
            raise MissingReferenceError(f"No such reference file in the library: {filename}")
        code = code.strip()
        if not code:
            raise MissingReferenceError("A binding needs a code to bind.")
        data = self.manifest()
        data.setdefault("bindings", {})[code] = target.name
        return self._write_manifest(data)

    def unbind(self, code: str) -> Path:
        """Drop a code binding. Codes the file states for itself are untouched.

        Which is why this cannot remove every identifier: a serial inside the file
        and a filename stem are facts about the file, and the only way to stop
        those answering is to change the file or its name.
        """
        data = self.manifest()
        bindings = data.get("bindings", {})
        bindings.pop(code, None)
        data["bindings"] = bindings
        return self._write_manifest(data)

    def add_file(self, source: str | Path) -> ReferenceEntry:
        """Copy a reference file into the library and record when it arrived.

        **Copied rather than referenced.** A library of paths into other people's
        folders is a library that decays: the measurement a run was judged against
        must still be there in a year, and a path is not a promise. The cost is
        two copies that can drift, and it is the right way round -- this one
        cannot be edited out from under a record.
        """
        source = Path(source)
        if source.suffix.lower() not in REFERENCE_SUFFIXES:
            raise MissingReferenceError(
                f"{source.name}: not a reference file this reads. Known kinds are "
                f"{', '.join(REFERENCE_SUFFIXES)}."
            )
        destination = self._free_name(source.name)
        self.directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        data = self.manifest()
        data.setdefault("added", {})[destination.name] = (
            datetime.now().astimezone().isoformat(timespec="seconds")
        )
        self._write_manifest(data)
        return next(e for e in self.entries if e.path.name == destination.name)

    def add_measurement(
        self,
        content: bytes,
        filename: str,
        *,
        bind: str | None = None,
        origin: str | None = None,
        note: str | None = None,
    ) -> ReferenceEntry:
        """Put a measurement this application fetched into the library.

        The counterpart to :meth:`add_file` for bytes that never were a file on
        this machine -- an OpenQualia label's measurement, downloaded from the
        host the chart itself named.

        ``bind`` is the identifier the file is recorded as answering to, and it
        is **the unit that was asked for, not the serial the file states**. That
        distinction is not hypothetical: asking the standard's own example host
        for ``DT-AR-2020013`` returns a file whose ``SERIAL`` keyword reads
        ``DT-AR-2020014``, and which is a different measurement from the real
        014 file. Indexing on what the file says about itself would file that
        measurement under another chart's serial and pair it with the wrong
        capture ever after -- the exact failure this whole module exists to
        prevent. What the file says is still read and still indexed by
        :func:`peek_identity`; this adds the requested unit alongside it, so a
        disagreement is visible rather than resolved.

        ``origin`` records where it came from, so the bank can say that a file
        was fetched rather than chosen by hand, and from which host.
        """
        safe = Path(filename).name
        if not safe or safe.startswith("."):
            raise MissingReferenceError(f"{filename!r} is not a usable filename.")
        if Path(safe).suffix.lower() not in REFERENCE_SUFFIXES:
            safe = f"{safe}.txt"
        destination = self._free_name(safe)
        self.directory.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

        data = self.manifest()
        data.setdefault("added", {})[destination.name] = (
            datetime.now().astimezone().isoformat(timespec="seconds")
        )
        if origin:
            data.setdefault("origin", {})[destination.name] = origin
        # A caveat about the file itself, kept with it rather than shown once and
        # lost: a serial disagreement is a fact about this copy that matters every
        # time it is paired, not only on the day it arrived.
        if note:
            data.setdefault("notes", {})[destination.name] = note
        if bind:
            data.setdefault("bindings", {})[bind.strip()] = destination.name
        self._write_manifest(data)
        return next(e for e in self.entries if e.path.name == destination.name)

    def note_of(self, filename: str) -> str | None:
        """A caveat recorded against this file when it was fetched, if any."""
        value = self.manifest().get("notes", {}).get(filename)
        return str(value) if value else None

    def origin_of(self, filename: str) -> str | None:
        """Where a file came from, if this application fetched it.

        ``None`` for a file the operator added, which is most of them -- the
        answer to "where did this come from" is then the folder they took it
        from, which the library deliberately does not remember.
        """
        value = self.manifest().get("origin", {}).get(filename)
        return str(value) if value else None

    def _free_name(self, name: str) -> Path:
        """A name in this directory that is not taken.

        Never overwrites. Two units' files can legitimately arrive under one name
        -- an instrument that writes ``Measurement.txt`` every time is the normal
        case, not the odd one -- and silently replacing the first would swap the
        reference under every run that had paired with it.
        """
        candidate = self.directory / name
        if not candidate.exists():
            return candidate
        stem, suffix = candidate.stem, candidate.suffix
        for n in range(2, 1000):
            candidate = self.directory / f"{stem}-{n}{suffix}"
            if not candidate.exists():
                return candidate
        raise MissingReferenceError(f"Cannot find a free name for {name}.")

    def remove_file(self, filename: str) -> None:
        """Take a file out of the library, with its bindings and its arrival date.

        It is deleted rather than moved aside. The library holds a copy by
        construction, so the original is where it came from -- which is the one
        thing the run library, whose records cannot be regenerated, cannot say
        about its own rows.
        """
        target = self.directory / filename
        if target.parent.resolve() != self.directory.resolve():
            raise MissingReferenceError(f"{filename} is not in this library.")
        target.unlink(missing_ok=True)
        data = self.manifest()
        data["bindings"] = {
            code: name for code, name in data.get("bindings", {}).items()
            if name != filename
        }
        data.get("added", {}).pop(filename, None)
        data.get("origin", {}).pop(filename, None)
        data.get("notes", {}).pop(filename, None)
        self._write_manifest(data)
