"""Per-unit reference measurement data.

Reference files arrive as CGATS or CxF3 depending on the vendor, or as an OpenDICE
target profile from somebody moving across from OpenDICE. :func:`read_reference`
dispatches on content so format stays an input concern and nothing downstream has to
know which one a measurement came from.
"""

from pathlib import Path

from deltachrome.reference.cgats import (
    MANUFACTURER_NOMINAL,
    PER_UNIT_MEASUREMENT,
    CGATSTable,
    MeasurementSet,
    PatchMeasurement,
    parse_cgats,
    read_cgats,
)
from deltachrome.reference.cxf import parse_cxf, read_cxf
from deltachrome.reference.opendice import looks_like_opendice, parse_opendice, read_opendice
from deltachrome.reference.spectral import fill_from_spectra

__all__ = [
    "MANUFACTURER_NOMINAL",
    "PER_UNIT_MEASUREMENT",
    "CGATSTable",
    "MeasurementSet",
    "PatchMeasurement",
    "parse_cgats",
    "parse_cxf",
    "parse_opendice",
    "read_cgats",
    "read_cxf",
    "read_opendice",
    "read_reference",
]

_CXF_SUFFIXES = {".cxf", ".xml"}


def sniff_format(head: str) -> str | None:
    """Identify a reference format from the start of the file, or None if unclear."""
    stripped = head.lstrip()
    if stripped.startswith("<?xml") or stripped.startswith("<CxF") or "CxF3" in head:
        return "cxf"
    upper = head.upper()
    if upper.lstrip().startswith("CGATS") or "BEGIN_DATA_FORMAT" in upper:
        return "cgats"
    if looks_like_opendice(head):
        return "opendice"
    return None


def read_reference(path: str | Path, *, table: int | str | None = None) -> MeasurementSet:
    """Read a reference measurement file: CGATS, CxF3 or an OpenDICE target profile.

    Content decides, in both directions. Vendors ship CxF under .xml and, less
    often, CGATS under .cxf -- trusting the extension over the bytes turns a
    readable file into a parse error.

    ``table`` selects one table of a multi-table CGATS file, by 1-based number or by
    measurement condition. CxF carries one measurement set per file, so naming a
    table for one is an error rather than a no-op.
    """
    path = Path(path)
    head = path.open("rb").read(4096).decode("utf-8-sig", errors="replace")

    detected = sniff_format(head)
    if detected == "cxf" or (
        detected is None and path.suffix.lower() in _CXF_SUFFIXES
    ):
        if table is not None:
            raise ValueError(
                f"A table was requested but {path.name} is CxF3, which holds one "
                "measurement set per file. Tables are a CGATS concept."
            )
        reference = read_cxf(path)
    elif detected == "opendice":
        if table is not None:
            raise ValueError(
                f"A table was requested but {path.name} is an OpenDICE target profile, "
                "which holds one table. Tables are a CGATS concept."
            )
        reference = read_opendice(path)
    else:
        reference = read_cgats(path, table=table)
    # A file may carry spectra and no colorimetry at all -- the LaserSoft ISO
    # 12641-2's own per-unit measurement is 864 patches of exactly that -- so the
    # integration happens here, once, rather than in each consumer.
    fill_from_spectra(reference)
    return reference
