"""Pairing a capture with the right reference file.

Reference data is per physical unit. Two charts of the same model are
indistinguishable in a photograph, and measuring against the wrong unit's file
produces numbers that look entirely reasonable and mean nothing -- so the
question "which file goes with this capture" deserves better than the operator's
memory. Where a chart carries a printed code, this reads it and either pairs the
capture automatically or says loudly that the file it was handed disagrees.

This lived in the CLI until a second front end needed it. That was the wrong
place for two reasons: policy reachable by only one front end is policy the two
can disagree about, and the CLI version loaded the image a second time to scan
it -- so every ``analyze`` decoded the master twice, which on a JPEG 2000 master
is around 25 seconds of pure repetition.

**Nominal fallback is deliberately not decided here.** The pipeline already owns
that policy in full (:func:`deltachrome.pipeline._reference_source`), including
which of three different situations produced a missing file. Re-deciding it here
would be a second copy free to drift. When no file is named, this says so and
lets the pipeline apply its own rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from deltachrome.errors import MissingReferenceError
from deltachrome.imaging.loader import LoadedImage, load_image
from deltachrome.reference.library import ReferenceEntry, ReferenceLibrary, peek_identity
from deltachrome.targets.schema import TargetDefinition

#: An explicit file was named and is used as given.
EXPLICIT = "explicit"
#: The chart's printed code was read and matched against a library.
PAIRED = "paired"
#: No file was named; the pipeline applies its own fallback policy.
DEFERRED = "deferred"


@dataclass(frozen=True)
class ReferenceChoice:
    """Which reference file a run should use, and how that was decided."""

    #: ``None`` means nothing was named -- not that nothing will be used. The
    #: pipeline may still fall back to the target's manufacturer nominal values.
    path: Path | None
    basis: str
    #: Informational, worth showing before the run starts.
    notes: list[str] = field(default_factory=list)
    #: Serious enough to travel with the result. A code mismatch lands here.
    warnings: list[str] = field(default_factory=list)
    scan: object | None = None
    #: The capture's printed code answers to nothing the named file answers to.
    #: A field rather than a search of the warning's wording, which it used to be
    #: until rewording the warning silently turned it off.
    code_mismatch: bool = False


def identity_with_manifest(path: Path) -> ReferenceEntry:
    """Identify a reference file, honouring any manifest bindings beside it.

    A file's aliases usually live in the library manifest rather than inside the
    file, so identifying it in isolation would under-report what it answers to
    and produce a false mismatch warning against a correctly paired file.
    """
    try:
        library = ReferenceLibrary(path.parent)
    except Exception:  # noqa: BLE001 -- a missing or unreadable manifest is not fatal
        return peek_identity(path)
    entry = next((e for e in library.entries if e.path.resolve() == path.resolve()), None)
    return entry or peek_identity(path)


def resolve_reference(
    *,
    target: TargetDefinition,
    image: LoadedImage | str | Path | None = None,
    explicit_path: str | Path | None = None,
    library_dir: str | Path | None = None,
    verify_code: bool = True,
    scan: object | None = None,
) -> ReferenceChoice:
    """Decide which reference file pairs with this capture.

    ``image`` accepts an already-decoded :class:`~deltachrome.imaging.loader.LoadedImage`
    and is only consulted when a scan is actually required -- which is the whole
    point of the parameter. A caller that already has the master in memory (the
    desktop frontend does) pays nothing, and a run that needs no scan never
    touches the file at all.

    ``scan`` goes one step further for a caller that has already *read the codes*.
    The desktop frontend now scans while it is working out which chart the capture
    holds, so that it can show the pairing before anything is measured rather than
    raising at the operator after a wait -- and without this it would then scan the
    same pixels again at run time. A scan is a fact about the capture, not about
    the run, so handing one in cannot change the answer; the same result is
    returned on ``ReferenceChoice.scan`` either way.
    """
    notes: list[str] = []
    warnings: list[str] = []
    explicit = Path(explicit_path) if explicit_path is not None else None
    library_path = Path(library_dir) if library_dir is not None else None

    if explicit is None and library_path is None:
        # Nothing named, so nothing to decide here: the pipeline's rule covers
        # three cases this does not need to know about.
        return ReferenceChoice(path=None, basis=DEFERRED, notes=notes)

    needs_scan = library_path is not None or (explicit is not None and verify_code)
    if scan is None and needs_scan:
        scan = _scan(image, tiled=library_path is not None)
    elif not needs_scan:
        scan = None

    if explicit is not None:
        mismatch = False
        if scan is not None and scan.codes:
            entry = identity_with_manifest(explicit)
            # Everything the file answers to: its stem, any serial it states, and
            # every code bound to it in the bank's manifest -- and, for a code read
            # as Standard 2 of 5, the barcode LaserSoft would print for its serial.
            # But only a claim can *disagree*: a serial, a bound code, or a
            # file named as nothing but a code. A descriptive filename is
            # evidence when it matches and none when it does not -- an
            # i1Profiler export states no serial at all, so a file measured on
            # this very chart was reported as not matching it.
            named = {explicit.stem} if re.fullmatch(r"[A-Za-z0-9]+", explicit.stem) else set()
            stated = sorted(({entry.serial} - {None}) | entry.aliases | named)
            mismatch = bool(stated) and not (
                entry.identifiers & {c.normalized for c in scan.codes}
                or entry.encoded & {c.normalized for c in scan.codes
                                    if c.symbology == "Code25"}
            )
            if mismatch:
                warnings.append(
                    f"Chart's printed code {', '.join(scan.texts)} doesn't match "
                    f"the reference file's {', '.join(stated)}."
                )
        return ReferenceChoice(
            path=explicit, basis=EXPLICIT, notes=notes, warnings=warnings, scan=scan,
            code_mismatch=mismatch,
        )

    assert scan is not None  # library_path is set, so a scan was required
    resolution = ReferenceLibrary(library_path).resolve(
        scan.texts, standard_2of5=getattr(scan, "standard_2of5", ()))
    if resolution.path is None:
        # Where the chart's code is an OpenQualia label it has already told us
        # where its measurement is published, so "bind a code to a filename" is
        # the wrong next step to offer: there is a better one, and an operator
        # sent looking through the bank would not find out about it.
        labels = getattr(scan, "labels", [])
        if labels:
            label = labels[0]
            raise MissingReferenceError(
                f"{resolution.reason}\nThis chart's code is an OpenQualia label: a "
                f"measurement of {label.describe()} is published at {label.host}. "
                "Open the capture in the desktop application and press Fetch "
                "Measurement to download it into the bank, or add the file by hand."
            )
        raise MissingReferenceError(
            f"{resolution.reason}\nList what the library holds, and bind a code to a "
            "filename, to pair this capture with its own unit's file."
        )
    notes.append(f"Reference paired automatically: {resolution.reason}")
    return ReferenceChoice(
        path=resolution.path, basis=PAIRED, notes=notes, warnings=warnings, scan=scan
    )


def _scan(image: LoadedImage | str | Path | None, *, tiled: bool):
    from deltachrome.imaging.codes import scan_codes

    if image is None:
        raise ValueError(
            "Pairing by the chart's printed code needs the capture, but no image "
            "was supplied to resolve_reference()."
        )
    loaded = image if isinstance(image, LoadedImage) else load_image(image)
    return scan_codes(loaded, tiled=tiled)
