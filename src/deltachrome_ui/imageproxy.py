"""Showing a master too large to hand to Qt whole.

The sample captures in this repository run from 46 MB to 577 MB, and a 577 MB
16-bit TIFF is around 1.7 gigapixels of ``QImage`` if you ask Qt to hold it as
8-bit RGB. Qt will not, and should not, be asked.

So the canvas is driven by two things:

* an **overview** -- the whole frame decimated to a few thousand pixels wide,
  built by striding the native integer array *before* any float conversion, the
  same trick :func:`deltachrome.imaging.sampling.render_overlay` uses. Cost is a
  preview's worth of memory rather than a multiple of the master's.
* **detail tiles** -- the visible rectangle only, read at whatever stride the
  current zoom actually resolves. Panning around a master at 1:1 costs one
  screenful at a time.

Scene coordinates are full-resolution image pixels throughout, at every zoom and
in both paths. That is the whole reason the geometry stays simple: what a corner
handle sits on *is* the number :func:`deltachrome.pipeline.analyze` wants for
``corners``, with no scaling step anywhere to get backwards.

**The preview is color-managed now**. Device RGB goes through the
embedded ICC profile to PCS XYZ, the same transform and the same full float
precision the measurement path uses, then to sRGB by the matrix
:mod:`~deltachrome_ui.swatches` already draws a patch swatch with. A linear-light
master used to look dark on the canvas while being measured correctly, which
looked like a rendering fault on the one screen where a placement is checked by
eye. What this still is not is a rendering to *the viewer's own monitor* -- sRGB
is assumed as the display's own space, the same assumption every untagged image
on the web already makes -- and no number anywhere in a report comes from it. A
profile the transform cannot evaluate falls back to the raw code-value divide
this always did, silently, because a preview that is momentarily less accurate
is a smaller failure than a canvas with nothing on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage

from deltachrome.imaging.loader import LoadedImage, to_display_records

#: Widest the overview is allowed to be. Enough to place corners roughly and to
#: see the whole chart at once; detail tiles handle everything finer.
DEFAULT_OVERVIEW_WIDTH = 2400

#: Most pixels a single detail tile may read. A cap on the read, not on the
#: region: asking for more area than this simply returns it at a coarser stride,
#: which is what a zoomed-out view wanted anyway.
DEFAULT_TILE_BUDGET = 4_000_000


#: How wide a frame-map thumbnail is asked for. The map draws the capture at
#: five hundred to nine hundred pixels in a normal window, so this is generous
#: and still an order of magnitude below the master -- which is the whole point,
#: because what it costs is set by how few resolution levels have to be decoded.
THUMBNAIL_WIDTH = 1200


def read_thumbnail(path, max_width: int = THUMBNAIL_WIDTH) -> QImage | None:
    """A small picture of a capture, without decoding the capture.

    For the frame map, which draws a few hundred pixels wide and needs no
    precision at all -- the outlines it draws come from the record, and the
    pixels only make the charts recognisable.

    **Why this is not ``load_image`` at a stride.** Decoding
    ``QA62_2.jp2`` whole is 11.2 s and 303 MB resident, to fill a rectangle
    480 px across. A JPEG 2000 codestream is natively multi-resolution, so the
    small version is *already in the file* and costs 0.59 s and 5 MB to read.
    An uncompressed TIFF has no such structure and needs none: it reads in
    0.2--0.5 s whatever its size, being a disk read.

    Two things were measured and are the reason this is not simpler. OpenCV's
    ``IMREAD_REDUCED_*`` flags buy **nothing** on JPEG 2000 -- 11.2 s at /2, /4
    and /8 alike -- because that path decodes the whole codestream and then
    downsamples. And Pillow's documented ``draft`` is a **no-op** for its
    JPEG2000 plugin: 31.9 s at every size asked for. The only thing that reads
    the resolution levels is the plugin's own ``reduce`` attribute.

    **Not color-managed**, unlike the overview and the tiles above it now are --
    this reads a codestream's own embedded reduced-resolution level directly
    rather than the array `to_display_rgb` is given, so it never reaches that
    path at all. A frame-map thumbnail is drawn at a few hundred pixels to make
    a chart *recognisable*, not to be checked by eye the way a placement is, so
    the cost of a full ICC evaluation buys nothing here that is worth the
    latency of asking for it.

    Returns ``None`` rather than raising when the file cannot be read at all --
    a missing or unreadable master leaves the map drawing its outlines, which is
    what it does when there are no pixels, and is not a failure of the screen.
    """
    path = Path(path)
    try:
        if path.suffix.lower() in (".jp2", ".j2k", ".jpf", ".jpx", ".j2c"):
            return _thumbnail_from_codestream(path, max_width)
        return _thumbnail_reduced(path, max_width)
    except Exception:  # noqa: BLE001 -- a thumbnail is never worth an error
        return None


def _thumbnail_from_codestream(path: Path, max_width: int) -> QImage | None:
    """A JPEG 2000's own reduced resolution level, decoded and nothing else.

    ``Image.open`` reads the header only, so the size is known before anything
    is decoded and the level can be chosen from it.
    """
    from PIL import Image

    # A preservation master is a decompression bomb by Pillow's default
    # reckoning -- 100 megapixels against its 89 million limit -- and the
    # engine's own loader disarms the check for exactly this reason. Without
    # it a 4x5 master raises on `open`, before a single pixel is read.
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        width = max(1, image.size[0])
        # The largest reduction that still leaves at least ``max_width``.
        level = 0
        while width >> (level + 1) >= max_width:
            level += 1
        # Assigning over ``Image.reduce`` -- which is a *method* on modern
        # Pillow -- is what the JPEG2000 plugin reads. Shadowing it on the
        # instance is the only way in; there is a test pinning that it still
        # works, because nothing about this API announces itself.
        image.reduce = level
        image.load()
        return _to_qimage(image.convert("RGB"))


def _thumbnail_reduced(path: Path, max_width: int) -> QImage | None:
    """Everything else, through OpenCV's reduced read.

    ``IMREAD_REDUCED_COLOR_*`` also brings a 16-bit master down to 8 bits,
    which is what a thumbnail wants and what the full path would otherwise
    have to do itself.
    """
    import cv2
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # see _thumbnail_from_codestream
    with Image.open(path) as probe:
        width = max(1, probe.size[0])
    flags = ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4),
             (2, cv2.IMREAD_REDUCED_COLOR_2), (1, cv2.IMREAD_COLOR))
    for divisor, flag in flags:
        if width // divisor >= max_width or divisor == 1:
            block = cv2.imread(str(path), flag)
            break
    if block is None:
        return None
    return _to_qimage_from_bgr(block)


def _to_qimage(image) -> QImage:
    """A Pillow RGB image as a QImage owning its own bytes."""
    return to_qimage(np.asarray(image, dtype=np.uint8))


def _to_qimage_from_bgr(block: np.ndarray) -> QImage:
    return to_qimage(np.ascontiguousarray(block[:, :, ::-1]))


#: sRGB primaries from D50-adapted XYZ (Bradford). A second copy of
#: :data:`deltachrome_ui.swatches._XYZ_D50_TO_SRGB` rather than a shared import:
#: that one converts a single swatch to a ``QColor`` and this one an array of
#: pixels to 8-bit RGB, different enough in shape that sharing the function
#: bought nothing -- but the matrix itself must stay identical, so a change to
#: either has to be made to both.
_XYZ_D50_TO_SRGB = np.array([
    [3.1338561, -1.6168667, -0.4906146],
    [-0.9787684, 1.9161415, 0.0334540],
    [0.0719453, -0.2289914, 1.4052427],
])


def _encode_srgb(linear: np.ndarray) -> np.ndarray:
    """The sRGB transfer function, applied to a whole array at once."""
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.maximum(linear, 0.0), 1 / 2.4) - 0.055,
    )


def to_display_rgb(block: np.ndarray, scale: float, profile=None, *,
                   linear: bool = False) -> np.ndarray:
    """Native-dtype pixels to contiguous 8-bit RGB, for display.

    **Color-managed when ``profile`` is given.** Device values go through the
    embedded ICC profile to PCS XYZ -- the same :meth:`ICCProfile.to_xyz` the
    measurement path uses, vectorized over the whole block rather than a
    handful of patch means -- then to sRGB by the same matrix
    :func:`deltachrome_ui.swatches.lab_to_display_rgb` draws a patch swatch
    with. This is still not a rendering to *the viewer's own monitor*: sRGB is
    assumed as the display's own space, which is the ordinary assumption every
    untagged image on the web already makes, and no number anywhere in a report
    comes from it.

    **Without a profile**, the fallback this always was: divide by the
    encoding's own maximum and truncate to 8 bits, no gamma assumption. A
    linear-light master looks dark under this path and correct under the one
    above -- which is exactly the case this distinction exists for.

    ``linear`` says the untagged values are linear in light, and puts them through
    the sRGB curve the managed path ends with. Only the profile screen asks, and
    only once it has read a capture's neutral ramp as linear: nothing else knows,
    and brightening a gamma-encoded capture would wash it out.

    A profile the transform cannot evaluate falls back to the same path rather
    than breaking the screen a corner is being dragged on; a preview that is
    momentarily less accurate is a smaller failure than one that is gone.
    """
    if profile is not None:
        try:
            return _to_display_rgb_managed(block, scale, profile)
        except Exception:  # noqa: BLE001 -- a preview is never worth an error
            pass
    return _to_display_rgb_raw(block, scale, linear=linear)


def _to_display_rgb_managed(block: np.ndarray, scale: float, profile) -> np.ndarray:
    # Not widened to three channels first -- unlike the raw path below, a
    # grayscale profile's ``to_xyz`` wants exactly the one channel it declares,
    # per-patch measurement's own rule (see ``ICCProfile.to_xyz``'s docstring:
    # grayscale is a different transform, not an RGB one fed a repeated grey).
    # Its output is XYZ either way, three channels regardless of the input.
    arr = np.asarray(block, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    normalized = arr / float(scale)
    if normalized.shape[-1] != profile.channels:
        # A file whose pixel channel count disagrees with its own profile is
        # refused by the loader already, so this should not happen in
        # practice; raising here is what sends it to the raw fallback above
        # rather than to a transform that would misread the data's shape.
        raise ValueError(
            f"{normalized.shape[-1]} pixel channel(s) does not match the "
            f"profile's {profile.channels}"
        )
    xyz = profile.to_xyz(normalized)
    linear = xyz @ _XYZ_D50_TO_SRGB.T
    encoded = np.clip(_encode_srgb(np.clip(linear, 0.0, None)), 0.0, 1.0)
    return np.ascontiguousarray((encoded * 255.0 + 0.5).astype(np.uint8))


def _to_display_rgb_raw(block: np.ndarray, scale: float, *,
                        linear: bool = False) -> np.ndarray:
    """No ICC transform: divide by the encoding's own maximum and truncate to
    8 bits, through the sRGB curve where the values are ``linear``."""
    # A grayscale master is widened to three here and nowhere earlier: the canvas
    # and the loupe both want RGB, and replicating in the measurement path would
    # have the engine assert three colour records the file does not carry.
    block = to_display_records(block)
    if block.dtype == np.uint8 and not linear:
        out = block if scale == 255.0 else None
        if out is not None:
            return np.ascontiguousarray(out)
    arr = block.astype(np.float32) / float(scale)
    np.clip(arr, 0.0, 1.0, out=arr)
    if linear:
        arr = _encode_srgb(arr)
    return np.ascontiguousarray((arr * 255.0 + 0.5).astype(np.uint8))


def to_qimage(rgb: np.ndarray) -> QImage:
    """Wrap contiguous 8-bit RGB as a QImage that owns its pixels.

    The ``.copy()`` is not waste. ``QImage`` over a numpy buffer does not keep the
    array alive, and the crash when the array is collected arrives later and
    somewhere else.
    """
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return image.copy()


@dataclass(frozen=True)
class Tile:
    """A region of the master, read at some stride, with where it came from."""

    image: QImage
    #: Region in full-resolution image pixels: (x0, y0, x1, y1).
    region: tuple[int, int, int, int]
    #: Pixels of the master per pixel of :attr:`image`. 1 is a true 1:1 read.
    stride: int

    @property
    def is_full_resolution(self) -> bool:
        return self.stride == 1


class ImageProxy:
    """Display-side view of one :class:`~deltachrome.imaging.loader.LoadedImage`."""

    def __init__(
        self,
        image: LoadedImage,
        *,
        overview_width: int = DEFAULT_OVERVIEW_WIDTH,
        tile_budget: int = DEFAULT_TILE_BUDGET,
        linear_display: bool = False,
    ) -> None:
        self.image = image
        #: Draw untagged values as linear light; see :func:`to_display_rgb`.
        self.linear_display = bool(linear_display) and image.profile is None
        self.tile_budget = int(tile_budget)
        self.overview_stride = max(1, int(np.ceil(image.width / max(1, overview_width))))
        self._overview_rgb = to_display_rgb(
            image.pixels[:: self.overview_stride, :: self.overview_stride, :],
            image.scale, image.profile, linear=self.linear_display,
        )
        self.overview = to_qimage(self._overview_rgb)

    # -- geometry ----------------------------------------------------------

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        """A point held inside the frame.

        Corners are allowed to sit exactly on the last pixel, since a chart
        photographed to the edge of the frame has its corner there.
        """
        return (
            float(min(max(x, 0.0), self.width - 1)),
            float(min(max(y, 0.0), self.height - 1)),
        )

    # -- reads -------------------------------------------------------------

    def stride_for(self, region: tuple[int, int, int, int], target_px: int) -> int:
        """Coarsest stride that still delivers ``target_px`` across the region.

        Bounded below by 1 -- there is no more detail than the file has -- and
        above by the tile budget, so a request for the whole frame at 1:1 quietly
        becomes a request for the whole frame at a stride that fits.
        """
        x0, y0, x1, y1 = region
        span = max(1, x1 - x0)
        by_zoom = max(1, int(np.floor(span / max(1, target_px))))
        area = max(1, (x1 - x0) * (y1 - y0))
        by_budget = max(1, int(np.ceil(np.sqrt(area / max(1, self.tile_budget)))))
        return max(by_zoom, by_budget)

    def tile(self, region: tuple[int, int, int, int], *, stride: int = 1) -> Tile:
        """Read one region at a given stride.

        The region is clipped to the frame first, so a view scrolled past the
        edge of the image returns the part that exists rather than failing.
        """
        x0, y0, x1, y1 = region
        # The origin is held one pixel inside the far edge so that a region
        # entirely off-frame still yields a real pixel. Otherwise the read is
        # empty, the QImage is null, and a null pixmap set on a graphics item
        # fails silently -- a blank canvas with no error anywhere.
        x0 = int(min(max(x0, 0), max(0, self.width - 1)))
        y0 = int(min(max(y0, 0), max(0, self.height - 1)))
        x1 = int(min(max(x1, x0 + 1), self.width))
        y1 = int(min(max(y1, y0 + 1), self.height))
        stride = max(1, int(stride))
        block = self.image.pixels[y0:y1:stride, x0:x1:stride, :]
        return Tile(
            image=to_qimage(to_display_rgb(block, self.image.scale, self.image.profile,
                                           linear=self.linear_display)),
            region=(x0, y0, x1, y1),
            stride=stride,
        )

    def loupe(self, x: float, y: float, half_width: int = 60) -> Tile:
        """A 1:1 read centerd on a point, for placing a corner exactly.

        Never decimated: the loupe exists precisely to show the pixels the
        overview cannot, so it reads at stride 1 regardless of the tile budget.
        A small window is cheap even on the largest master here.
        """
        half = max(1, int(half_width))
        return self.tile(
            (int(round(x)) - half, int(round(y)) - half,
             int(round(x)) + half + 1, int(round(y)) + half + 1),
            stride=1,
        )
