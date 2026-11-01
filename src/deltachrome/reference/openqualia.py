"""OpenQualia labels: reading one off a chart, and fetching what it points at.

An OQ Label is a QR code or Data Matrix printed on -- or beside -- a target,
encoding a plain HTTPS URL that resolves to that *physical unit's* measurement
file. It is the answer to the problem :mod:`deltachrome.reference.library` was
built around: reference data is per unit, two charts of one model are
indistinguishable in a photograph, and the chart itself is the only witness to
which unit it is. A label turns "which file goes with this capture" from a
question about the operator's memory into a question about the pixels.

The standard is small. A label carries three query parameters --
``TargetID``, ``Manufacturer``, ``TargetType`` -- and client software appends a
fourth, ``AccessMode=ActiveMeasurement``, to be handed the measurement rather
than a web page meant for somebody who scanned the code with a phone. What comes
back is an OQM file, which is ANSI CGATS.17 with several normally-optional
keywords made mandatory, so :func:`deltachrome.reference.read_reference` already
reads it and nothing here parses colour.

**The URL is used exactly as the chart printed it.** That is the one rule this
module is built around, and it is not fastidiousness. Measured against the DT
host that serves the standard's own example:

* ``TargetID`` is case-sensitive -- ``TV24017`` resolves and ``tv24017`` is a
  404 -- while ``Manufacturer`` and ``TargetType`` are folded for case and
  whitespace, as the standard says they should be.
* Dashes are not folded either, and the host's own web interface uses different
  spellings from its API: ``OLT-1x`` appears in the browsable URL and 404s
  against the measurement endpoint, where ``OLT1x`` resolves.

So a label is never rebuilt out of its parts, canonicalised, or matched against
a table of names Deltachrome keeps. It is parsed only far enough to *report*
what it says and to bind the file it returns; the bytes on the wire are the
chart's own URL with one parameter added. The standard asks for that outright --
"OpenQualia-compatible software should use all provided parameters in the URL
used to download the measurement file" -- because a label may carry vendor
parameters this application has never heard of.

**Nothing here reaches the network unless a caller asks it to.** Parsing is
pure; :func:`fetch_measurement` is the only function that opens a socket, and
the frontend calls it from a control the operator presses. A capture is not
consent to make an outbound request.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace

from deltachrome.errors import ReferenceError
from deltachrome.imaging.codes import normalize_identifier

#: The parameter client software adds, and the value that asks for the file
#: rather than the human-facing page. Absent from the label itself by design:
#: the bare URL is meant to be useful to somebody who scans it with a phone.
ACCESS_MODE = "AccessMode"
ACTIVE_MEASUREMENT = "ActiveMeasurement"

#: The three parameters a label must carry. Matched case-insensitively when
#: *reading* a label -- a self-served label made by hand is entitled to spell
#: them however the URL standard allows -- but never rewritten when fetching.
REQUIRED_PARAMS = ("TargetID", "Manufacturer", "TargetType")

#: How much of a response is read before giving up. An OQM file with spectra for
#: a 130-patch chart is around 60 kB; a megabyte is two decades of headroom and
#: still refuses to stream something that is not a measurement file.
MAX_BYTES = 4 * 1024 * 1024

#: Long enough for a slow connection to a live host, short enough that an
#: operator does not think the application has hung.
DEFAULT_TIMEOUT = 20.0

#: Only these schemes are fetched. The standard requires ``https``; a label
#: offering anything else is either a misread or something that should not be
#: opened because a chart in a photograph asked for it.
_ALLOWED_SCHEMES = ("https",)

#: A serial is written on a target and typed by people. The standard limits it
#: to letters, digits and dashes.
_TARGET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,63}$")


#: How a label came about. It changes nothing about the request and everything
#: about how a refusal should read: told "the code was read correctly" after
#: typing a serial by hand, an operator goes looking for a code that was never
#: involved.
FROM_CHART = "chart"
FROM_SERIAL = "serial"


@dataclass(frozen=True)
class OpenQualiaLabel:
    """A decoded OQ Label: what it says, and the URL it said it with.

    ``url`` is the label verbatim. Every other field is read out of it for
    display and for binding, and none of them is used to rebuild it.
    """

    url: str
    target_id: str
    manufacturer: str
    target_type: str
    #: :data:`FROM_CHART` when this was decoded off a capture, :data:`FROM_SERIAL`
    #: when an operator typed the serial.
    source: str = FROM_CHART

    @property
    def measurement_url(self) -> str:
        """The label's URL with ``AccessMode=ActiveMeasurement``.

        Any existing ``AccessMode`` is replaced -- a label carrying
        ``Interactive`` would otherwise fetch a web page -- and every other
        character is left exactly as the chart printed it.

        **Edited as text, not re-encoded.** Parsing the query into pairs and
        rebuilding it looks tidier and quietly rewrites the URL: ``%20``
        becomes ``+``, and the order of parameters is at the mercy of a dict.
        Both are legal and neither is what the chart says, and this is a host
        that distinguishes ``OLT1x`` from ``OLT-1x`` and ``TV24017`` from
        ``tv24017``. The one thing this property may change is the parameter
        the standard says client software adds.
        """
        head, sep, fragment = self.url.partition("#")
        base, has_query, query = head.partition("?")
        kept = [
            piece
            for piece in query.split("&")
            if piece and piece.partition("=")[0].strip().lower() != ACCESS_MODE.lower()
        ]
        kept.append(f"{ACCESS_MODE}={ACTIVE_MEASUREMENT}")
        return f"{base}?{'&'.join(kept)}{sep}{fragment}"

    @property
    def host(self) -> str:
        return urllib.parse.urlsplit(self.url).netloc

    def describe(self) -> str:
        return f"{self.manufacturer} {self.target_type}, unit {self.target_id}"

    def suggested_filename(self) -> str:
        """A filename to fall back on when the server names no file.

        Built from what the label says rather than from the file's own header,
        for the reason :func:`fetch_measurement` records: a served file's
        ``SERIAL`` can disagree with the unit that was asked for.
        """
        stem = "_".join(
            part for part in (self.manufacturer, self.target_type, self.target_id) if part
        )
        return f"{_safe_stem(stem) or 'openqualia'}.oqm.txt"


def parse_label(text: str) -> OpenQualiaLabel | None:
    """Read a decoded code as an OQ Label, or ``None`` if it is not one.

    Deliberately total: every code a chart carries is offered here, and most of
    them -- a bare serial, a vendor's own barcode -- are not labels. Returning
    ``None`` is the ordinary outcome and not a failure.
    """
    if not text:
        return None
    candidate = text.strip()
    try:
        parts = urllib.parse.urlsplit(candidate)
    except ValueError:
        return None
    if parts.scheme.lower() not in _ALLOWED_SCHEMES or not parts.netloc:
        return None

    # Parameter *names* are matched case-insensitively: the standard fixes their
    # spelling for labels a member issues, and a self-served label -- which the
    # standard explicitly provides for, built by hand on top of a Dropbox link --
    # is only held to the URL standard.
    found: dict[str, str] = {}
    for key, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=True):
        lowered = key.lower()
        for name in REQUIRED_PARAMS:
            if lowered == name.lower() and value and name not in found:
                found[name] = value
    if len(found) != len(REQUIRED_PARAMS):
        return None
    if not _TARGET_ID.match(found["TargetID"]):
        return None

    return OpenQualiaLabel(
        url=candidate,
        target_id=found["TargetID"],
        manufacturer=found["Manufacturer"],
        target_type=found["TargetType"],
    )


def is_label(text: str) -> bool:
    """Whether a decoded code is an OQ Label."""
    return parse_label(text) is not None


def _identity(target) -> tuple[str, str, str] | None:
    """``(host, manufacturer, target_type)`` from a target's definition, or ``None``.

    **The host is part of the identity.** OpenQualia is a label standard and not a
    repository: there is no central index, and each member serves its own units
    from wherever it chooses. A serial alone therefore says nothing about where to
    look, and a default host would offer every target a lookup at one member's
    server whether or not that member publishes the model. So a definition names
    the host it was verified against, and one that names none is offered no
    lookup -- the same answer as a target nobody publishes, and for the same
    reason: there is nowhere known to look.
    """
    identity = getattr(target, "openqualia", None) or {}
    host = str(identity.get("host") or "").strip()
    manufacturer = identity.get("manufacturer")
    target_type = identity.get("target_type")
    if not (host and manufacturer and target_type):
        return None
    parts = urllib.parse.urlsplit(host)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES or not parts.netloc or parts.query:
        return None
    return host, str(manufacturer), str(target_type)


def label_for_serial(target, serial: str) -> OpenQualiaLabel | None:
    """Build a label for a unit of ``target`` by serial, or ``None``.

    The counterpart to reading a label off the chart, for the two cases that
    leaves out: a chart that prints a serial and no label -- every Golden Thread
    made before 2025 -- and filling a bank before a shoot rather than during one.

    ``None`` where the target's definition does not say where and how the model
    is published, which is most targets and is not a failure: there is nowhere
    known to look.

    **The identity comes from the definition, never from the operator.** That is
    the whole point: `TargetType` is the one field somebody would have to guess,
    the host offers no listing to guess from, and it distinguishes spellings that
    look interchangeable -- its own browsable slug ``OLT-1x`` 404s where
    ``OLT1x`` resolves. A serial is written on the chart; a target type and the
    server that publishes it are not.
    """
    identity = _identity(target)
    serial = (serial or "").strip()
    if identity is None or not serial or not _TARGET_ID.match(serial):
        return None
    host, manufacturer, target_type = identity
    label = parse_label(
        f"{host}?Manufacturer={urllib.parse.quote(manufacturer, safe='')}"
        f"&TargetType={urllib.parse.quote(target_type, safe='')}"
        f"&TargetID={urllib.parse.quote(serial, safe='')}"
    )
    return None if label is None else replace(label, source=FROM_SERIAL)


def publishes_measurements(target) -> bool:
    """Whether a target declares where its units' measurements are published."""
    return _identity(target) is not None


#: Keywords a measurement file may state its own serial under, most specific
#: first. The OpenQualia standard requires ``SERIAL``; the others are what the
#: wider CGATS world writes.
_SERIAL_KEYWORDS = ("DELTACHROME_TARGET_ID", "TARGET_SERIAL", "SERIAL", "TARGET_ID")
_SERIAL_RE = re.compile(
    r"^\s*(?:%s)\s+\"?([^\"\r\n]+?)\"?\s*$" % "|".join(_SERIAL_KEYWORDS),
    re.IGNORECASE | re.MULTILINE,
)


def stated_serial(content: bytes) -> str | None:
    """The serial a measurement file states for itself, if it states one."""
    head = content[:16384].decode("utf-8-sig", errors="replace")
    for line in head.splitlines():
        if line.strip().upper().startswith(("BEGIN_DATA", "BEGIN_DATA_FORMAT")):
            break
        match = _SERIAL_RE.match(line)
        if match and match.group(1).strip():
            return match.group(1).strip()
    return None


@dataclass(frozen=True)
class FetchedMeasurement:
    """What came back from a label's URL."""

    label: OpenQualiaLabel
    content: bytes
    filename: str
    #: The URL actually read, after any redirects. The standard's own example
    #: host redirects one domain to another, so this is routinely not the label.
    final_url: str

    @property
    def text(self) -> str:
        return self.content.decode("utf-8-sig", errors="replace")

    @property
    def stated_serial(self) -> str | None:
        """What the file says its own serial is, which is not always the unit
        that was asked for."""
        return stated_serial(self.content)

    @property
    def serial_disagreement(self) -> str | None:
        """A warning when the file names a different unit than was requested.

        The one genuine defect found in around 140 published measurements read
        from this host: asking for NGT2 unit ``DT-AR-2020013`` returns a file
        whose ``SERIAL`` reads ``DT-AR-2020014``, and it is a different
        measurement from the real 014 file -- 0.52 dE2000 apart over 130
        patches, a different length, a different patch ordering. So one of the
        two is mislabelled inside and nothing here can say which.

        Worth catching because it is exact and has no judgement in it, unlike
        every other check on reference data. Two other files that looked
        anomalous turned out to be perfectly good measurements of later
        production batches, which is why this compares identifiers and not
        colours.
        """
        stated = self.stated_serial
        if stated is None:
            return None
        if normalize_identifier(stated) == normalize_identifier(self.label.target_id):
            return None
        return (
            f"{self.label.host} was asked for unit {self.label.target_id} and returned a "
            f"measurement that states its own serial as {stated}. One of the two is "
            "mislabelled and this cannot say which. The file is filed under the unit that "
            f"was requested ({self.label.target_id}), so pairing will work as expected -- "
            "but if the chart in front of you is not that unit, every color number "
            "measured against this file is wrong in a way nothing else will reveal. "
            "Check the serial printed on the chart against both."
        )


def fetch_measurement(
    label: OpenQualiaLabel,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    opener: object | None = None,
) -> FetchedMeasurement:
    """Retrieve the measurement file a label points at.

    The only function in Deltachrome that opens a network connection, and it is
    called from a control the operator pressed. ``opener`` exists so tests can
    exercise every path without one.

    Raises :class:`~deltachrome.errors.ReferenceError` with the reason on any
    failure, including a response that is not a measurement file: a host that
    answers a missing unit with a friendly HTML page would otherwise put an
    error document in the reference bank under a serial number.
    """
    url = label.measurement_url
    request = urllib.request.Request(
        url,
        headers={"Accept": "text/plain, */*", "User-Agent": _user_agent()},
    )
    open_url = getattr(opener, "open", None) or urllib.request.urlopen
    try:
        with open_url(request, timeout=timeout) as response:
            final_url = getattr(response, "url", url) or url
            if urllib.parse.urlsplit(final_url).scheme.lower() not in _ALLOWED_SCHEMES:
                raise ReferenceError(
                    f"{label.describe()}: the label redirected to {final_url!r}, which "
                    "is not https. Nothing was downloaded."
                )
            content = response.read(MAX_BYTES + 1)
            filename = _filename_from(response) or label.suggested_filename()
    except urllib.error.HTTPError as exc:
        raise ReferenceError(_http_reason(label, exc)) from exc
    except urllib.error.URLError as exc:
        raise ReferenceError(
            f"Could not reach {label.host} to fetch the measurement for "
            f"{label.describe()}: {exc.reason}. The chart's code was read correctly; "
            "this is a network problem, not a pairing one."
        ) from exc
    except OSError as exc:
        raise ReferenceError(
            f"Could not fetch the measurement for {label.describe()}: {exc}"
        ) from exc

    if len(content) > MAX_BYTES:
        raise ReferenceError(
            f"{label.describe()}: the response from {label.host} is larger than "
            f"{MAX_BYTES // (1024 * 1024)} MB, which no measurement file is. "
            "Nothing was saved."
        )
    if not _looks_like_measurement(content):
        raise ReferenceError(
            f"{label.host} answered for {label.describe()}, but what came back is not "
            "a measurement file -- it carries no CGATS data block. The unit may not be "
            "published, or the host may have returned a web page. Nothing was saved."
        )
    return FetchedMeasurement(
        label=label, content=content, filename=filename, final_url=final_url
    )


def _http_reason(label: OpenQualiaLabel, exc: urllib.error.HTTPError) -> str:
    if exc.code == 404:
        # Said differently depending on where the request came from, because the
        # two have different likely causes and different next steps. A code read
        # off a chart is almost certainly right, so the useful thing to say is
        # that the pairing is not at fault; a serial typed by hand is the thing
        # most likely to be wrong, and saying "the code was read correctly" there
        # sends somebody looking for a code that was never involved.
        cause = (
            "The code was read correctly, so this is not a pairing failure -- the "
            "unit's file may be private to an organisation, or may never have been "
            "published."
            if label.source == FROM_CHART else
            "Check the serial against the one printed on the chart. Failing that, "
            "the unit's file may be private to an organisation, or may never have "
            "been published."
        )
        return (
            f"{label.host} has no published measurement for {label.describe()}. "
            f"{cause} A measurement supplied with the chart can be added to the "
            "bank by hand."
        )
    return (
        f"{label.host} refused the request for {label.describe()} "
        f"({exc.code} {exc.reason})."
    )


def _looks_like_measurement(content: bytes) -> bool:
    """Whether the bytes are a CGATS-family measurement file.

    Checked on the response rather than trusted from the content type, because
    the failure this guards against is a host answering with something friendly:
    an HTML page saved under a serial number would be found by the bank, offered
    for pairing, and fail at parse time in the middle of a run.
    """
    head = content[:8192].decode("utf-8-sig", errors="replace").upper()
    return "BEGIN_DATA_FORMAT" in head or "BEGIN_DATA" in head


def _filename_from(response: object) -> str | None:
    """The filename the server suggests, if it suggests a safe one."""
    headers = getattr(response, "headers", None)
    raw = headers.get("Content-Disposition") if headers is not None else None
    if not raw:
        return None
    # RFC 5987 form first (``filename*=utf-8''DT%20NGT2%20...``), which is what
    # the standard's own example host sends, then the plain form.
    match = re.search(r"filename\*\s*=\s*[^']*'[^']*'([^;]+)", raw, re.I)
    if match:
        name = urllib.parse.unquote(match.group(1).strip())
    else:
        match = re.search(r'filename\s*=\s*"([^"]+)"|filename\s*=\s*([^;]+)', raw, re.I)
        if not match:
            return None
        name = (match.group(1) or match.group(2) or "").strip()
    return _safe_filename(name)


def _safe_filename(name: str) -> str | None:
    """Reduce a server-supplied name to something that cannot escape a directory.

    The name is chosen by a remote host, so it is treated as a suggestion and
    not as a path: any directory part is discarded outright rather than
    sanitised, which is the only handling that cannot be talked around.
    """
    name = name.replace("\\", "/").split("/")[-1].strip()
    if not name or name in (".", "..") or name.startswith("."):
        return None
    cleaned = re.sub(r"[^A-Za-z0-9 ._-]+", "_", name).strip(" .")
    return cleaned[:120] or None


def _safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")[:100]


def _user_agent() -> str:
    from deltachrome import __version__

    return f"Deltachrome/{__version__} (+OpenQualia label client)"
