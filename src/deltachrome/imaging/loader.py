"""Image loading with mandatory ICC profile extraction.

Deltachrome refuses images with no embedded ICC profile. This is a deliberate
hard failure rather than a prompt-with-default: assuming sRGB for an Adobe RGB or
ECI RGB master produces Delta E values that are wrong by many units while looking
entirely reasonable, and a QC tool that can be confidently wrong is worse than one
that stops.

**Pixels are kept in their native integer dtype**, with a scale factor recorded
alongside. Converting a whole master to float64 up front is ruinous at the sizes
this domain works at: an 11600 x 8700 capture is 2.4 GB as float64 and 300 MB as
the uint8 it actually is. Sampling normalizes only the small window it needs, so
memory tracks the file rather than a multiple of it.

**A capture carries one or three colour records, and the count travels with it.**
Grayscale is not RGB with the chroma at zero -- it is a file with one record where
an RGB file has three, and the single plane is kept rather than replicated. That is
not thrift: white balance is a Delta E with lightness excluded, so on three
identical records it would return 0.00 on every patch of a chart the system rendered
badly, and report it as a pass. The count reaches the profile evaluator, which
refuses to let the three record-dependent metrics run on a one-record file. Pixels
and profile are also required to *agree*: a grayscale TIFF tagged Adobe RGB is
refused rather than resolved either way.

**JPEG 2000 is a first-class input.** JP2 is a preservation master format across
the sector, and it stores its ICC profile in the JP2 container's ``colr`` box
rather than anywhere Pillow surfaces it -- so Deltachrome parses the box structure
itself. Without that, every JP2 master would look like an untagged file and be
refused.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from deltachrome.errors import ImageError, MissingICCProfileError
from deltachrome.imaging.icc import ICCProfile, parse_icc

_TIFF_ICC_TAG = 34675
_TIFF_SUFFIXES = {".tif", ".tiff"}
#: TIFF ResolutionUnit values, as inches per unit. 1 means "no unit" --
#: the numbers are an aspect ratio, not a resolution, so there is nothing
#: to compare a measured rate against.
_RESOLUTION_UNIT_TO_INCH = {2: 1.0, 3: 2.54}
_JP2_SUFFIXES = {".jp2", ".jpf", ".jpx", ".j2k", ".jpc"}

#: JP2 enumerated color spaces (``colr`` box, METH=1). These are explicit
#: declarations in the file, not assumptions, so honouring them is not guesswork --
#: but the report says the color space came from an enumeration rather than an
#: embedded profile, because the two are not equally specific.
_JP2_ENUM_CS = {16: "sRGB", 17: "grayscale", 18: "sYCC", 20: "e-sRGB", 24: "ROMM-RGB"}


@dataclass
class LoadedImage:
    """An image plus everything needed to interpret its numbers colorimetrically."""

    path: Path
    pixels: np.ndarray  # (H, W, 3) in the file's native dtype
    scale: float  # divide by this to reach [0, 1]
    #: ``None`` only for a capture loaded with ``profile_required=False``, whose
    #: numbers are device values with no colorimetric meaning yet.
    profile: ICCProfile | None
    bit_depth: int
    dtype: str  # the file's storage dtype, not the working one
    container: str = "tiff"
    notes: list[str] = field(default_factory=list)
    #: What the file's header claims its resolution is, in ppi, if it claims
    #: anything. Metadata, never a measurement -- see :func:`read_header_resolution`.
    header_ppi: tuple[float, float] | None = None

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    @property
    def channels(self) -> int:
        """Colour records the capture carries: 3 for RGB, 1 for grayscale.

        Not a display detail. Three metrics here compare the records against each
        other or read chroma out of them, and on a one-record capture there is
        nothing for them to read -- see
        :data:`deltachrome.metrics.base.CAPTURE_COLOR_CAPABILITIES`.
        """
        return int(self.pixels.shape[2])

    @property
    def is_grayscale(self) -> bool:
        return self.channels == 1

    @property
    def quantisation_step(self) -> float:
        """Smallest representable code-value difference, in normalized units."""
        return 1.0 / (2**self.bit_depth - 1)

    def window(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """A normalized float64 view of one region. Only this region is converted."""
        return self.pixels[y0:y1, x0:x1, :].astype(np.float64) / self.scale

    def luma_window(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """One region as a single plane, for the steps that only need structure.

        Edge location, ruler ticks and code detection care where a boundary is, not
        what colour it is, and each of them used ``cvtColor(..., RGB2GRAY)`` -- which
        needs three channels and raises on a capture that has one. The reduction
        lives here instead so a grayscale capture is the trivial case rather than a
        special one.
        """
        return to_luma(self.window(y0, y1, x0, x1))

    @property
    def rgb(self) -> np.ndarray:
        """The whole image normalized to float64.

        Convenient for small images and tests; avoid it on masters, where it
        allocates eight times the file's pixel data. Use :meth:`window` instead.
        """
        return self.pixels.astype(np.float64) / self.scale

    def summary(self) -> str:
        lines = [
            f"{self.path.name}: {self.width}x{self.height} ({self.megapixels:.1f} MP), "
            f"{self.bit_depth}-bit, "
            f"{'grayscale' if self.is_grayscale else f'{self.channels}-channel'}, "
            f"{self.container.upper()}",
            f"  ICC: {self.profile.description if self.profile else 'none (device values)'}",
        ]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def load_image(path: str | Path, *, profile_required: bool = True) -> LoadedImage:
    """Load an image and its embedded ICC profile, or fail loudly.

    ``profile_required=False`` is for building a profile, and for nothing else.
    The capture a camera profile is made from usually has none -- RawTherapee's
    *Save Reference Image* writes an untagged linear TIFF -- and what is wanted
    from it is device values, which a profile would only stand in front of. Such
    a capture comes back with ``profile=None``, and every colorimetric step
    refuses it by having nothing to convert with. QC never passes this.
    """
    path = Path(path)
    if not path.exists():
        raise ImageError(f"No such image: {path}")

    suffix = path.suffix.lower()
    reader = (
        _load_tiff if suffix in _TIFF_SUFFIXES
        else _load_jp2 if suffix in _JP2_SUFFIXES
        else _load_pillow
    )
    try:
        array, icc_bytes, bit_depth, container, notes = reader(path)
    except ImageError:
        raise
    except Exception as failure:
        # A decoder's own exception names its internals -- "corrupted strip cannot
        # be reshaped from (230444,) to (1, 16, 11608, 3)" for a TIFF cut short in
        # copying -- which is a fact about the library and not one an operator can
        # act on. The original stays chained for whoever debugs it.
        raise ImageError(
            f"{path.name} could not be read. The file may be damaged, incomplete, "
            "or in a form this reader does not support."
        ) from failure

    if not icc_bytes and not profile_required:
        array = _to_records(array, path)
        scale, bits = _scale_for(array, bit_depth)
        return LoadedImage(
            path=path,
            pixels=array,
            scale=scale,
            profile=None,
            bit_depth=bits,
            dtype=str(array.dtype),
            container=container,
            notes=notes,
            header_ppi=read_header_resolution(path),
        )

    if not icc_bytes:
        raise MissingICCProfileError(
            f"{path.name} has no embedded ICC profile.\n"
            "Deltachrome will not guess a color space -- assuming sRGB for an "
            "Adobe RGB, ProPhoto or ECI RGB master silently corrupts every Delta E "
            "in the report.\n"
            "Fix: re-export the capture with its profile embedded, or assign the "
            "correct profile in your CMS-aware editor first."
        )

    profile = parse_icc(icc_bytes)
    array = _to_records(array, path)
    _check_records_match_profile(array, profile, path)
    scale, bits = _scale_for(array, bit_depth)

    return LoadedImage(
        path=path,
        pixels=array,
        scale=scale,
        profile=profile,
        bit_depth=bits,
        dtype=str(array.dtype),
        container=container,
        notes=notes,
        header_ppi=read_header_resolution(path),
    )


def read_header_resolution(path: str | Path) -> tuple[float, float] | None:
    """The resolution the file *claims*, in pixels per inch, or None if it claims none.

    Deliberately never used to derive a measurement. A header resolution is
    metadata: it can be set to anything, defaulted by an export dialog, or left
    over from a previous step, and believing it would put a number in a report that
    nothing in the capture supports. It is read for exactly one purpose -- so that
    reproduction scale accuracy can compare it against the rate measured off the
    chart's own printed ruler and say when the two disagree.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in _TIFF_SUFFIXES:
            import tifffile

            with tifffile.TiffFile(path) as tif:
                tags = tif.pages[0].tags
                x = tags.get("XResolution")
                y = tags.get("YResolution")
                unit = tags.get("ResolutionUnit")
                if x is None or y is None:
                    return None
                per_unit = _RESOLUTION_UNIT_TO_INCH.get(
                    int(unit.value) if unit is not None else 2
                )
                if per_unit is None:
                    return None
                return (_ratio(x.value) * per_unit, _ratio(y.value) * per_unit)

        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as img:
            dpi = img.info.get("dpi")
        if not dpi:
            return None
        return (float(dpi[0]), float(dpi[1]))
    except Exception:
        # A malformed resolution tag is not a reason to refuse an otherwise good
        # capture; the metric that wants it reports its absence instead.
        return None


def _ratio(value: Any) -> float:
    """TIFF rationals arrive as (numerator, denominator) or as a float."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]) / float(value[1] or 1)
    return float(value)


def _scale_for(array: np.ndarray, declared_bits: int | None) -> tuple[float, int]:
    """The divisor that maps this dtype into [0, 1], and the real bit depth."""
    if array.dtype.kind == "f":
        # Float TIFFs are already scene- or display-referred in [0, 1] by
        # convention. Values outside that range are real information (highlights
        # above diffuse white), so nothing is clipped here -- sampling flags it.
        return 1.0, declared_bits or 32
    if array.dtype.kind in "ui":
        info = np.iinfo(array.dtype)
        bits = declared_bits or int(np.ceil(np.log2(info.max + 1)))
        return float(info.max), bits
    raise ImageError(f"Unsupported pixel dtype {array.dtype}")


def _check_records_match_profile(array: np.ndarray, profile: ICCProfile, path: Path) -> None:
    """Refuse a capture whose pixels and whose profile disagree about the space.

    The two arrive from different places in the file and nothing makes them agree.
    A grayscale TIFF tagged with an RGB working space, or an RGB one tagged Gray
    Gamma 2.2, is a mistake made upstream -- and it is exactly the kind that would
    otherwise surface as an array-shape error deep inside a metric, or worse, be
    papered over by replicating or averaging channels.
    """
    channels = int(array.shape[2])
    if channels == profile.channels:
        return
    raise ImageError(
        f"{path.name} has {channels} channel(s) but its embedded ICC profile "
        f"({profile.description}) describes a {profile.color_space.strip()} space "
        f"with {profile.channels}. Deltachrome will not guess which is right: "
        "assuming the pixels means measuring through the wrong transform, and "
        "assuming the profile means inventing or discarding color records.\n"
        "Fix: re-export the capture with the profile that matches its pixels."
    )


def _to_records(array: np.ndarray, path: Path) -> np.ndarray:
    """Normalize the decoded array to (H, W, C) with C of 1 or 3.

    A grayscale capture keeps its single plane rather than being replicated into
    three. Replication is cheap and it is the wrong answer: colour registration
    would read exactly zero displacement and white balance exactly zero chroma on a
    file that measured neither, and both would be reported as passes.
    """
    if array.ndim == 2:
        return array[:, :, None]
    if array.ndim != 3:
        raise ImageError(f"{path.name}: expected a 2D or 3D array, got shape {array.shape}")
    channels = array.shape[2]
    if channels in (1, 3):
        return array
    if channels == 2:
        # Grey plus alpha. Drop the alpha, as for RGBA.
        return array[..., :1]
    if channels == 4:
        # Drop alpha (or a 4th ink); >3 channel handling is a roadmap item and
        # deliberately not faked here.
        return array[..., :3]
    raise ImageError(
        f"{path.name} has {channels} channels. Multi-channel (>3) capture support "
        "is on the roadmap and is not silently approximated by taking the first three."
    )


#: Weights for reducing three colour records to one luminance-like signal.
#:
#: ISO 12233's, which is what :mod:`deltachrome.imaging.sfr` and
#: :mod:`deltachrome.imaging.registration` already use to decide an edge's
#: orientation. Structure-finding used ``cv2.cvtColor(..., RGB2GRAY)`` before this,
#: which is BT.601 -- nobody chose those weights, they were OpenCV's default, and
#: having two conventions in one engine for the same reduction is one more than is
#: useful. Nothing colorimetric is measured through either.
LUMA_WEIGHTS = np.array([0.213, 0.715, 0.072])


def to_luma(block: np.ndarray) -> np.ndarray:
    """A (H, W, C) block reduced to one plane, C of 1 or 3.

    :data:`LUMA_WEIGHTS` for three records; the plane itself for one. Used where
    only *where* a boundary is matters -- edge search windows, ruler ticks, code
    detection -- and never on the path to a reported colour value.
    """
    block = np.asarray(block)
    if block.ndim == 2:
        return block.astype(np.float64)
    if block.shape[-1] == 1:
        return block[..., 0].astype(np.float64)
    return block[..., :3].astype(np.float64) @ LUMA_WEIGHTS


def to_display_records(block: np.ndarray) -> np.ndarray:
    """A (H, W, C) block widened to three channels, for drawing only.

    Every preview path here hands pixels to OpenCV or Qt, which want three. Doing
    the widening at the point of drawing keeps it out of the measurement path,
    where a replicated grey would be a claim the file does not make.
    """
    block = np.asarray(block)
    if block.ndim == 2:
        block = block[:, :, None]
    if block.shape[-1] == 1:
        return np.repeat(block, 3, axis=-1)
    return block[..., :3]


def _load_tiff(path: Path) -> tuple[np.ndarray, bytes | None, int | None, str, list[str]]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        array = page.asarray()
        tag = page.tags.get(_TIFF_ICC_TAG)
        icc = bytes(tag.value) if tag is not None else None
    return array, icc, None, "tiff", []


def _load_pillow(path: Path) -> tuple[np.ndarray, bytes | None, int | None, str, list[str]]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as img:
        icc = img.info.get("icc_profile")
        fmt = (img.format or "").lower()
        if img.mode not in ("RGB", "RGBA", "L", "I;16", "F"):
            img = img.convert("RGB")
        array = np.asarray(img)
    return array, icc, None, fmt or "image", []


def _load_jp2(path: Path) -> tuple[np.ndarray, bytes | None, int | None, str, list[str]]:
    """Load a JPEG 2000 master, taking its ICC profile from the JP2 container.

    **Decoded through OpenCV rather than Pillow**, and that is a correctness fix
    before it is a speed one. Pillow's JPEG 2000 plugin has no 16-bit RGB mode: it
    decodes a 16-bit three-component codestream to 8 bits per channel and says
    nothing, which threw away most of the precision a preservation master was
    captured at. This used to be a refusal -- convert to a 16-bit TIFF and analyze
    that -- which is a real cost imposed on the operator for a decoder limitation.
    OpenCV binds the same OpenJPEG library and reads the file at its own depth.

    **The two decoders were checked against each other rather than assumed
    equivalent**, because a decoder swap that silently disagrees is the worst kind
    of change to make here. On the one 8-bit master both can read, OpenCV's output
    is bit-for-bit identical to Pillow's across 11600 x 8700 x 3, and it is three
    times faster (9.4 s against 28.5). On the 16-bit master Pillow cannot read,
    OpenCV agrees bit-for-bit with **glymur**, an independent OpenJPEG binding --
    which is what says the 16-bit path is right and not merely different, since
    there is no Pillow answer to compare it against.

    OpenCV warns that the codestream declares no colour space and it is assuming
    sRGB. It is not applying anything: the identity against Pillow is over pixels,
    and neither decoder transforms. Colour management here comes from the ICC
    profile in the JP2 container, which is read separately and never from the
    codestream's own colour specification.

    **Pillow is kept as the fallback** rather than removed. It handles shapes this
    path deliberately does not claim to have tested -- anything that is not one or
    three components -- and a decoder returning ``None`` on a file it cannot open,
    which is how ``imread`` reports failure, must not become an exception three
    frames later.
    """
    import cv2

    header = _read_head(path)
    icc, notes = _jp2_color_spec(header, path)
    bits = _jp2_bit_depth(header)

    array = _quietly(lambda: cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
    if array is None or array.ndim not in (2, 3) or (
        array.ndim == 3 and array.shape[2] != 3
    ):
        array = _load_jp2_with_pillow(path)
    elif array.ndim == 3:
        # OpenCV hands back BGR. Copied rather than left as a reversed view: every
        # window taken from it afterwards would otherwise be a strided read.
        array = np.ascontiguousarray(array[:, :, ::-1])

    if bits is not None and array.dtype == np.uint8 and bits > 8:
        # Still possible: a codestream neither decoder reads at its declared depth.
        # Silently measuring through that would throw away the precision this tool
        # exists to protect, so it stays a refusal rather than a warning.
        raise ImageError(
            f"{path.name} is a {bits}-bit JPEG 2000, but the decoder returned 8-bit "
            "data. Measuring it would discard the file's precision.\n"
            "Fix: convert the master to a 16-bit TIFF with a CMS-aware tool "
            "(preserving the ICC profile) and analyze that."
        )

    return array, icc, bits, "jp2", notes


def _quietly(read):
    """Run an OpenCV read with its own logging turned down to errors.

    Every JPEG 2000 master here makes OpenCV print ``Image has unknown or
    unspecified color space, SRGB is assumed`` to stderr. It is true of the
    codestream and it is not a fact about this run: colour management comes from
    the ICC profile in the JP2 container, which is read separately. A line of
    library chatter on every decode is noise in a terminal and in a log, and worse
    in a desktop application that has no terminal to put it in.

    Turned down around the call and restored afterwards rather than set once at
    import: the log level is process-wide, and a library that silences a host
    application's diagnostics as a side effect of being imported is a bad citizen.
    Errors still come through, and a failed read is reported by ``imread``
    returning ``None`` rather than by anything printed.
    """
    from cv2.utils import logging as cv_logging

    previous = cv_logging.getLogLevel()
    cv_logging.setLogLevel(cv_logging.LOG_LEVEL_ERROR)
    try:
        return read()
    finally:
        cv_logging.setLogLevel(previous)


def _load_jp2_with_pillow(path: Path) -> np.ndarray:
    """The fallback decode, for a file OpenCV declines or returns an odd shape for."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as img:
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        return np.asarray(img)


def _read_head(path: Path, limit: int = 1 << 20) -> bytes:
    with path.open("rb") as handle:
        return handle.read(limit)


def _jp2_bit_depth(data: bytes) -> int | None:
    """Bits per component, from the JP2 header's ``ihdr`` box."""
    box = _find_box(data, b"ihdr", inside=b"jp2h")
    if box is None:
        return None
    offset, _length = box
    # ihdr layout after the 8-byte box header: HEIGHT(4) WIDTH(4) NC(2) BPC(1) ...
    # NC is two bytes, so BPC is at +18. Reading it at +16 picks up the high byte
    # of the channel count and reports every image as 1-bit.
    bpc = data[offset + 18]
    if bpc == 255:  # components differ in depth; the codestream is authoritative
        return None
    return (bpc & 0x7F) + 1


def _jp2_color_spec(data: bytes, path: Path) -> tuple[bytes | None, list[str]]:
    """Extract the color specification from the JP2 ``colr`` box.

    METH=2 carries a full embedded ICC profile. METH=1 carries an enumerated
    color space -- a definite declaration rather than an assumption, so it is
    honoured, but the report notes that it came from an enumeration.
    """
    box = _find_box(data, b"colr", inside=b"jp2h")
    if box is None:
        return None, []
    offset, length = box

    method = data[offset + 8]
    if method == 2:
        return data[offset + 11 : offset + length], []

    if method == 1:
        enum = struct.unpack_from(">I", data, offset + 11)[0]
        name = _JP2_ENUM_CS.get(enum, f"enumerated space {enum}")
        if enum != 16:
            raise ImageError(
                f"{path.name} declares the JP2 enumerated color space {name!r} and "
                "carries no embedded ICC profile. Only the sRGB enumeration is "
                "specific enough to measure through -- 'grayscale' names no tone "
                "curve, so nothing here can say what its code values mean.\n"
                "Fix: re-export with the capture's ICC profile embedded (a "
                "grayscale profile is fine; Deltachrome reads those)."
            )
        from PIL import ImageCms

        return (
            ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes(),
            [
                "Color space came from the JP2 enumerated field (sRGB), not an "
                "embedded ICC profile. That is an explicit declaration rather than a "
                "guess, but it carries no per-device calibration -- an embedded "
                "profile is preferable."
            ],
        )

    return None, []


def _find_box(data: bytes, want: bytes, inside: bytes | None = None) -> tuple[int, int] | None:
    """Locate a JP2 box, optionally within a named superbox. Returns (offset, length)."""

    def walk(start: int, end: int, target: bytes) -> tuple[int, int] | None:
        pos = start
        while pos + 8 <= end:
            length = struct.unpack_from(">I", data, pos)[0]
            btype = data[pos + 4 : pos + 8]
            header = 8
            if length == 1:
                if pos + 16 > end:
                    return None
                length = struct.unpack_from(">Q", data, pos + 8)[0]
                header = 16
            elif length == 0:
                length = end - pos
            if length < header:
                return None
            if btype == target:
                return pos, min(length, end - pos)
            pos += length
        return None

    if inside is None:
        return walk(0, len(data), want)

    outer = walk(0, len(data), inside)
    if outer is None:
        return None
    start, length = outer
    return walk(start + 8, start + length, want)
