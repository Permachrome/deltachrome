"""Reading identification codes off a capture.

Charts increasingly carry a machine-readable identifier -- both ColorChecker SG
units here have a QR code, and the LaserSoft ISO 12641-2 target has a barcode. That
identifier is the reliable way to pair a capture with the *per-unit* reference file
it must be measured against, which otherwise depends on an operator remembering
which chart was in the frame.

Two things make this harder than a normal barcode scan:

* **Scale.** A QR occupying 300 px in an 11600 px master vanishes when the image is
  downsampled far enough to scan quickly. So detection runs a cheap whole-image
  pass first and only then tiles the full-resolution image.
* **Decoder coverage.** OpenCV reads QR and a few 1D symbologies (EAN/UPC); it does
  not read Code 39, Code 128 or Data Matrix. ``zxing-cpp`` covers far more and
  installs as a self-contained wheel, so it is used when present. Decoders are
  tried in order and the layer degrades to "nothing found" rather than failing.

Nothing here is authoritative on its own: a decoded code is a *suggestion* that the
caller confirms against a reference library. Deltachrome never silently measures
against a reference the operator did not choose or confirm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from deltachrome.imaging.loader import LoadedImage, to_luma

#: Codes are identifiers, not free text. Anything wilder than this is far more
#: likely to be a misread than a real chart serial.
_PLAUSIBLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.:/-]{1,63}$")


def _is_plausible(text: str) -> bool:
    """Whether a decoded string is worth keeping.

    Two shapes qualify, and they had to be separated. A chart serial is short
    and tame, which is what :data:`_PLAUSIBLE` describes. An **OpenQualia
    label** is a URL, and fails that pattern twice over -- the standard's own
    example is 108 characters against a 64-character cap, and ``?``, ``&`` and
    ``=`` are outside its character class. So every OQ Label was decoded
    correctly and then dropped here, one line before anything could use it.

    A label is admitted by being *parseable as one* rather than by widening the
    pattern: it must be an https URL carrying the three parameters the standard
    requires. That is a far narrower gate than any regex over free text, which
    matters because this filter is what stops a decoder's misread of chart
    content becoming a chart identifier.
    """
    if _PLAUSIBLE.match(text):
        return True
    from deltachrome.reference.openqualia import is_label

    return is_label(text)


@dataclass
class DetectedCode:
    """One decoded identification code, with where and how it was found."""

    text: str
    symbology: str
    decoder: str
    points: np.ndarray | None = None
    source: str = "whole-image"

    @property
    def normalized(self) -> str:
        return normalize_identifier(self.text)

    @property
    def label(self):
        """This code as an :class:`~deltachrome.reference.openqualia.OpenQualiaLabel`.

        ``None`` for the ordinary case of a bare serial. A label is worth
        distinguishing because it carries somewhere to *get* the measurement,
        where a serial only says which one is wanted.
        """
        from deltachrome.reference.openqualia import parse_label

        return parse_label(self.text)

    @property
    def bounds(self) -> tuple[int, int, int, int] | None:
        """Where the code sits, in the capture's own pixels, or ``None``.

        The coordinates matter beyond curiosity: a printed code block is a
        region of high-contrast structure, and the chart search offers it as a
        candidate chart -- the ColorChecker SG's own QR is put forward as a
        second target in its frame. A caller that has decoded a code knows what
        that region is and can say so, which no property of the pixels could.

        ``None`` where the decoder gave no position. Only OpenCV's QR reader and
        zxing-cpp do; the 2-of-5 path decodes a scanline and has no vertical
        extent to report.
        """
        if self.points is None or len(self.points) == 0:
            return None
        flat = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)
        return (int(flat[:, 0].min()), int(flat[:, 1].min()),
                int(np.ceil(flat[:, 0].max())), int(np.ceil(flat[:, 1].max())))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<DetectedCode {self.text!r} {self.symbology} via {self.decoder}>"


@dataclass
class CodeScanResult:
    codes: list[DetectedCode] = field(default_factory=list)
    decoders_used: list[str] = field(default_factory=list)
    decoders_unavailable: list[str] = field(default_factory=list)
    tiles_scanned: int = 0

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.codes]

    @property
    def standard_2of5(self) -> list[str]:
        """The codes read as Standard 2 of 5 -- LaserSoft's symbology, whose
        barcodes encode the printed serial rather than spell it
        (:func:`deltachrome.reference.library.lasersoft_barcode`)."""
        return [c.text for c in self.codes if c.symbology == "Code25"]

    @property
    def labels(self) -> list:
        """Every decoded code that is an OpenQualia label.

        Usually empty, and at most one in practice: the standard requires one
        working label per target and says an error should be raised if two
        resolve, precisely because two labels on one chart means two claims
        about which physical unit it is.
        """
        return [label for label in (c.label for c in self.codes) if label is not None]

    @property
    def boxes(self) -> list[tuple[int, int, int, int]]:
        """Every decoded code's place in the capture, for callers that care."""
        return [box for box in (c.bounds for c in self.codes) if box is not None]

    def describe(self) -> str:
        if not self.codes:
            missing = (
                f" ({', '.join(self.decoders_unavailable)} not installed)"
                if self.decoders_unavailable
                else ""
            )
            return (
                f"No identification code decoded after {self.tiles_scanned} tile(s)"
                f"{missing}."
            )
        return "; ".join(f"{c.text} [{c.symbology}]" for c in self.codes)


def normalize_identifier(text: str) -> str:
    """Fold a code or filename into a comparable key.

    Chart serials get written as ``CCSG21``, ``CC-SG-21`` and ``ccsg 21`` by
    different people and different tools; matching has to survive that.

    An **OpenQualia label folds to the unit it names**, not to its own URL.
    A label is a hundred characters of host, path and query, of which exactly
    one field -- ``TargetID`` -- identifies the physical chart; the rest says
    where that unit's measurement is published. Folding the whole string would
    make a label match nothing a file could ever be called, so every caller
    would need to know about labels separately. Doing it here means one
    place knows, and the library index, the manifest bindings and the
    duplicate-code check all pair on the serial without changing.
    """
    if "://" in text and "?" in text:
        from deltachrome.reference.openqualia import parse_label

        label = parse_label(text)
        if label is not None:
            text = label.target_id
    return re.sub(r"[^A-Z0-9]+", "", text.upper())


def _to_uint8_grey(pixels: np.ndarray, scale: float) -> np.ndarray:
    if pixels.ndim == 3 and pixels.shape[-1] == 1 and pixels.dtype == np.uint8:
        return np.ascontiguousarray(pixels[..., 0])
    gray = to_luma(np.ascontiguousarray(pixels))
    return np.clip(gray * (255.0 / scale), 0, 255).astype(np.uint8)


# -- decoders ---------------------------------------------------------------


def _decode_opencv(gray: np.ndarray) -> list[DetectedCode]:
    """QR only. Built in, so it always runs."""
    found: list[DetectedCode] = []
    detector = cv2.QRCodeDetector()
    try:
        data, points, _ = detector.detectAndDecode(gray)
    except cv2.error:
        return found
    if data:
        found.append(
            DetectedCode(text=data, symbology="QRCode", decoder="opencv", points=points)
        )
    return found


#: Standard (Industrial) 2 of 5 digit patterns: five bars per digit, exactly two
#: wide, with positional weights 1-2-4-7-0 (a sum of 11 encodes zero). Spaces are
#: all narrow separators and carry nothing.
_C25_WEIGHTS = (1, 2, 4, 7, 0)


def _decode_code25(gray: np.ndarray, min_bars: int = 15) -> list[DetectedCode]:
    """Decode Standard (Industrial) 2 of 5 from a bar-width scan.

    This symbology is old, carries no check digit and is not self-checking, so the
    general-purpose decoders have largely dropped it -- zxing-cpp and OpenCV both
    read Interleaved 2 of 5 but not this. LaserSoft's ISO 12641-2 targets use it,
    which is the whole reason their barcode would not read.

    Several scanlines are decoded independently and only a value that at least two
    of them agree on is returned. A single scanline through a dusty or scratched
    barcode is not evidence.
    """
    if gray.ndim != 2 or gray.shape[0] < 12:
        return []
    votes: dict[str, int] = {}
    height = gray.shape[0]
    for frac in (0.3, 0.4, 0.5, 0.6, 0.7):
        y = int(height * frac)
        band = gray[max(0, y - 4) : y + 4]
        if band.size == 0:
            continue
        row = band.mean(axis=0).astype(np.float64)
        if row.max() - row.min() < 40:
            continue
        dark = row < (row.max() + row.min()) / 2.0

        runs, current, length, start = [], bool(dark[0]), 0, 0
        for value in dark:
            if bool(value) == current:
                length += 1
            else:
                runs.append((current, start, length))
                start += length
                current, length = bool(value), 1
        runs.append((current, start, length))

        bars = [w for is_bar, _, w in runs if is_bar and w > 3]
        if len(bars) < min_bars:
            continue
        widths = np.array(bars, dtype=np.float64)
        narrow = float(np.median(widths[widths < np.median(widths) * 1.6]))
        if not np.isfinite(narrow) or narrow <= 0:
            continue
        symbols = "".join("w" if w >= narrow * 1.6 else "n" for w in widths)

        # Try each offset: the leading bar may be a start element or a stray mark.
        for offset in (0, 1, 2, 3):
            body = symbols[offset:]
            usable = len(body) - (len(body) % 5)
            if usable < 10:
                continue
            digits = []
            for i in range(0, usable, 5):
                group = body[i : i + 5]
                wide = [j for j, c in enumerate(group) if c == "w"]
                if len(wide) != 2:
                    digits = []
                    break
                total = sum(_C25_WEIGHTS[j] for j in wide)
                digits.append("0" if total == 11 else str(total))
            if digits:
                votes[("".join(digits))] = votes.get("".join(digits), 0) + 1
                break

    for text, count in sorted(votes.items(), key=lambda kv: -kv[1]):
        if count >= 2:
            return [DetectedCode(text=text, symbology="Code25", decoder="deltachrome")]
    return []


def _decode_zxing(gray: np.ndarray) -> list[DetectedCode]:
    """QR, Data Matrix, Code 39/128, EAN/UPC and more, when zxing-cpp is installed."""
    try:
        import zxingcpp
    except ImportError:
        return []
    found: list[DetectedCode] = []
    for result in zxingcpp.read_barcodes(gray):
        if result.text:
            found.append(
                DetectedCode(
                    text=result.text,
                    symbology=getattr(result.format, "name", str(result.format)),
                    decoder="zxing-cpp",
                    points=_zxing_points(result),
                )
            )
    return found


def _zxing_points(result) -> np.ndarray | None:
    """The four corners zxing reports, if this build reports them."""
    position = getattr(result, "position", None)
    corners = [getattr(position, name, None) for name in
               ("top_left", "top_right", "bottom_right", "bottom_left")]
    if position is None or any(c is None for c in corners):
        return None
    try:
        return np.array([[[float(c.x), float(c.y)] for c in corners]],
                        dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return None


def _placed(codes: list[DetectedCode], *, scale: float = 1.0,
            offset: tuple[int, int] = (0, 0)) -> list[DetectedCode]:
    """Move a window's coordinates into the capture's own.

    A decoder is handed a downsampled image or a tile, and answers in whatever
    it was given. Every caller of :attr:`DetectedCode.bounds` wants the capture,
    so the conversion happens once, here, rather than at each reader.
    """
    for code in codes:
        if code.points is None:
            continue
        code.points = np.asarray(code.points, dtype=np.float64) * scale + np.array(
            [float(offset[0]), float(offset[1])]
        )
    return codes


def available_decoders() -> tuple[list[str], list[str]]:
    """(installed, missing) decoder names, for reporting."""
    installed = ["opencv"]
    missing: list[str] = []
    try:
        import zxingcpp  # noqa: F401

        installed.append("zxing-cpp")
    except ImportError:
        missing.append("zxing-cpp")
    return installed, missing


def _decode_all(gray: np.ndarray, *, code25: bool = False) -> list[DetectedCode]:
    codes = _decode_opencv(gray) + _decode_zxing(gray)
    if code25:
        codes += _decode_code25(gray)
    keep: dict[str, DetectedCode] = {}
    for code in codes:
        if not _is_plausible(code.text):
            continue
        keep.setdefault(code.normalized, code)
    return list(keep.values())


# -- scanning ---------------------------------------------------------------


def scan_codes(
    image: LoadedImage,
    *,
    quick_max_dimension: int = 2600,
    tile: int = 1600,
    overlap: int = 400,
    max_tiles: int = 200,
    tiled: bool = True,
) -> CodeScanResult:
    """Find identification codes in a capture.

    A cheap downsampled pass runs first; tiling the full-resolution image only
    happens if that finds nothing, because a code a few hundred pixels across is
    unreadable once a 100 megapixel master has been scaled to fit a screen.
    """
    installed, missing = available_decoders()
    result = CodeScanResult(decoders_used=installed, decoders_unavailable=missing)

    stride = max(1, int(np.ceil(max(image.width, image.height) / quick_max_dimension)))
    quick = _to_uint8_grey(image.pixels[::stride, ::stride, :], image.scale)
    result.tiles_scanned = 1
    for code in _placed(_decode_all(quick), scale=float(stride)):
        code.source = f"whole image (1/{stride})"
        result.codes.append(code)
    if result.codes or not tiled:
        return result

    seen: set[str] = set()
    step = max(tile - overlap, tile // 2)
    for y in range(0, image.height, step):
        for x in range(0, image.width, step):
            if result.tiles_scanned >= max_tiles:
                return result
            window = image.pixels[y : y + tile, x : x + tile, :]
            if window.shape[0] < 64 or window.shape[1] < 64:
                continue
            result.tiles_scanned += 1
            found = _placed(_decode_all(_to_uint8_grey(window, image.scale)),
                            offset=(x, y))
            for code in found:
                if code.normalized in seen:
                    continue
                seen.add(code.normalized)
                code.source = f"tile at ({x}, {y})"
                result.codes.append(code)
    if result.codes:
        return result
    return _scan_wide(image, result)


#: The 1-D pass's window, and how far it moves. Wide because a **square tile is
#: the wrong shape for a barcode**: the LaserSoft ISO 12641-2's is 1820 px across
#: on its own master, so no 1600 px tile ever holds all of it, and what a
#: fragment decodes to is a confident wrong answer -- ``021``, ``30021``,
#: ``930021`` from three different crops of one barcode whose value is 1930021.
#: Short because the useful thing about a scanline through a barcode is that it
#: crosses nothing else, and a band the full height of a chart crosses its
#: patches.
WIDE_WINDOW = (3600, 420)
WIDE_STEP = (1200, 210)

#: Fewest digits a 2-of-5 read is believed at. The symbology has no check digit
#: and no guard pattern this parser relies on, so a window holding part of a
#: barcode decodes cleanly to part of a number. Five is above every fragment
#: measured here and below any serial worth printing.
MIN_1D_DIGITS = 5


def _scan_wide(image: LoadedImage, result: CodeScanResult) -> CodeScanResult:
    """A last pass in the shape a one-dimensional code actually has.

    Runs only when everything else has found nothing, and enables Standard 2 of 5
    -- which the general decoders dropped and which the tiled pass above must not
    use, because over arbitrary image content a symbology with no check digit
    eventually reads a plausible number out of noise. Two things make it safe
    enough to run here: a value must be agreed by at least two independent
    scanlines within one window, and it must be at least :data:`MIN_1D_DIGITS`
    long. Measured across every sample capture in this repository it reads the
    LaserSoft's barcode and returns nothing at all on the other seven.
    """
    width, height = WIDE_WINDOW
    xstep, ystep = WIDE_STEP
    votes: dict[str, DetectedCode] = {}
    for y in range(0, max(1, image.height - height // 2), ystep):
        for x in range(0, max(1, image.width - width // 2), xstep):
            window = image.pixels[y : y + height, x : x + width, :]
            if window.shape[0] < 40 or window.shape[1] < 200:
                continue
            result.tiles_scanned += 1
            gray = _to_uint8_grey(window, image.scale)
            for code in _decode_code25(gray):
                if len(code.text) < MIN_1D_DIGITS or not _PLAUSIBLE.match(code.text):
                    continue
                code.source = f"wide window at ({x}, {y})"
                votes.setdefault(code.normalized, code)
    result.codes.extend(_without_fragments(list(votes.values())))
    return result


def _without_fragments(codes: list[DetectedCode]) -> list[DetectedCode]:
    """Drop a 2-of-5 read that is part of a longer 2-of-5 read from the same scan.

    A window that cuts a barcode decodes the part it holds, and with no check
    digit and no guard pattern the part is a valid number: the LaserSoft 120 reads
    ``173003`` from one window and ``1730030`` from the next. Pairing was never
    misled by it, the fragment matching no file, but everything that lists what a
    chart prints was -- the placement screen said the chart printed both, and
    *Choose from bank* bound both, writing the fragment into the manifest as a
    lasting alias. A fragment is by definition contained in the whole read, so
    this needs no threshold and cannot remove a whole read. A fragment read where
    the whole never was survives; telling that apart needs the quiet zone either
    side, which is deferred.
    """
    two_of_five = [c.normalized for c in codes if c.symbology == "Code25"]
    return [
        code for code in codes
        if code.symbology != "Code25"
        or not any(code.normalized != other and code.normalized in other
                   for other in two_of_five)
    ]


def scan_regions(
    image: LoadedImage,
    regions: Iterable[tuple[int, int, int, int]],
    *,
    code25: bool = True,
) -> CodeScanResult:
    """Scan specific (x0, y0, x1, y1) regions at full resolution.

    Standard 2 of 5 is enabled here by default but not in the general scan: it has
    no check digit and no start/stop guard the parser can rely on, so run over
    arbitrary image content it would eventually read a plausible number out of
    noise. Pointed at a region the caller believes holds a barcode, it is useful.
    """
    installed, missing = available_decoders()
    result = CodeScanResult(decoders_used=installed, decoders_unavailable=missing)
    seen: set[str] = set()
    for x0, y0, x1, y1 in regions:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.width, x1), min(image.height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        result.tiles_scanned += 1
        window = _to_uint8_grey(image.pixels[y0:y1, x0:x1, :], image.scale)
        for code in _decode_all(window, code25=code25):
            if code.normalized in seen:
                continue
            seen.add(code.normalized)
            code.source = f"region ({x0}, {y0})-({x1}, {y1})"
            result.codes.append(code)
    result.codes = _without_fragments(result.codes)
    return result
