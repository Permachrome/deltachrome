"""Patch ROI sampling.

Each patch is sampled from an inset region of its bounding box -- 55% of the patch
by default -- because the outer margin is where printing registration error, patch
edge bleed, and the shadow of any surface texture live. The inset is a per-target
and per-profile setting rather than a constant, since a chart with larger patches
can afford a bigger fraction and a dense chart cannot.

The mean of the ROI is the measurement. Median and standard deviation are recorded
alongside it: a patch with a large mean/median divergence has a speck of dust or a
specular hit on it, and that is information the operator needs, not something to
quietly average away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from deltachrome.errors import GeometryError
from deltachrome.imaging.geometry import ChartPlacement
from deltachrome.imaging.loader import LoadedImage, to_display_records
from deltachrome.targets.schema import Patch, TargetDefinition
from deltachrome.text import plural

DEFAULT_ROI_INSET = 0.55
MIN_USEFUL_PIXELS = 100
#: Fraction of clipped pixels in an ROI above which the sample is untrustworthy.
CLIPPING_FLAG_FRACTION = 0.001
#: Mean-vs-median divergence, in normalized units, that suggests a contaminated ROI.
CONTAMINATION_FLAG = 0.005


#: How bright the brightest *other* channel must be, in normalized units, before a
#: channel held at the encoding floor is read as a GAMUT problem rather than as
#: underexposure. Both look identical in ``clipped_low``, which fires when any one
#: channel of any pixel is at the floor -- and they have opposite remedies, so a
#: report that cannot tell them apart sends the operator to the wrong control.
#:
#: The test is on the BRIGHTEST other channel, not on all of them, and that matters:
#: a saturated red has green at zero AND blue low, so requiring every other channel
#: to be bright would read it as underexposed. What separates the two cases is
#: whether anything in the patch is bright at all. A dark patch with everything near
#: the floor is underexposed: open up. A patch with one channel at zero and another
#: at 0.48 is a colour more saturated than the working space can encode: shoot in a
#: wider space.
#:
#: The LaserSoft ISO 12641-2 Advanced target is the case that made this worth
#: separating. 127 of its 864 patches flag ``clipped_low`` on an eciRGB v2 capture
#: and not one of them is underexposed: they are its saturated cyans, magentas and
#: yellows, with means as high as 0.60 and a channel pinned at exactly zero.
GAMUT_CLIP_BRIGHT_CHANNEL = 0.25

#: ICC classes whose encoding is a device's own response rather than a working
#: space. The pinned-channel pattern means something different there, and is
#: flagged ``channel_floor`` instead of ``gamut_clipped``: nothing is being
#: encoded INTO a space, so there is no wider one to move to. The channel is at
#: the floor because the sensor received almost no light in it -- measured on a
#: DT Nexus Profiling capture in a Phase One camera profile, whose eleven such
#: patches read within 2.6-4.2 dE2000 of their unit's measurement with chroma
#: intact, where the same chart in Adobe RGB loses a third of it.
DEVICE_ENCODING_CLASSES = frozenset({"scnr", "cmra"})


def _pinned_channels(pixels, mean, low) -> list[str]:
    """Which channels sit at the encoding floor while the patch is not simply dark.

    Returns the channel letters, or an empty list when the pattern does not hold: a
    grayscale capture, which has one record and so nothing to be out of gamut
    against; a patch dark in every channel, which is underexposure; or a patch where
    every channel is at the floor, which is the same thing.
    """
    if pixels.shape[1] < 3:
        return []
    at_floor = (pixels <= low).mean(axis=0)
    pinned = [
        name for index, name in enumerate("RGB")
        if at_floor[index] > CLIPPING_FLAG_FRACTION
    ]
    if not pinned or len(pinned) == pixels.shape[1]:
        return []
    others = [float(mean[i]) for i in range(pixels.shape[1]) if "RGB"[i] not in pinned]
    return pinned if max(others) >= GAMUT_CLIP_BRIGHT_CHANNEL else []


@dataclass
class PatchSample:
    """One patch measured off the image. Raw values, before any comparison."""

    patch_id: str
    row: int
    col: int
    rgb_mean: np.ndarray
    rgb_median: np.ndarray
    rgb_std: np.ndarray
    lab_std: np.ndarray | None  # per-channel Lab std within the ROI
    pixel_count: int
    clipped_high_fraction: float
    clipped_low_fraction: float
    polygon_px: np.ndarray
    flags: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return "empty_roi" not in self.flags

    @property
    def L_std(self) -> float:
        """Standard deviation of L* inside the ROI -- FADGI's noise measure."""
        return float("nan") if self.lab_std is None else float(self.lab_std[0])

    @property
    def contamination(self) -> float:
        return float(np.max(np.abs(self.rgb_mean - self.rgb_median)))


@dataclass
class SamplingResult:
    samples: dict[str, PatchSample]
    inset: float
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, patch_id: str) -> PatchSample:
        return self.samples[patch_id]

    def flagged(self) -> dict[str, list[str]]:
        return {pid: s.flags for pid, s in self.samples.items() if s.flags}


def sample_patches(
    image: LoadedImage,
    placement: ChartPlacement,
    target: TargetDefinition,
    *,
    inset: float = DEFAULT_ROI_INSET,
    patches: list[Patch] | None = None,
    measure_noise: bool = True,
) -> SamplingResult:
    """Sample every patch of ``target`` from ``image`` using ``placement``."""
    if not target.grid_regions:
        raise GeometryError(f"Target {target.target_id!r} has no patch geometry to sample.")
    if not 0.0 < inset <= 1.0:
        raise ValueError(f"ROI inset must be in (0, 1]; got {inset}")

    todo = patches if patches is not None else target.patches
    samples: dict[str, PatchSample] = {}
    warnings: list[str] = []

    step = image.quantisation_step
    high = 1.0 - step / 2.0
    low = step / 2.0

    small_rois = 0
    for patch in todo:
        # Geometry comes from the patch's own region: row and column only mean
        # something within one lattice, and a target may hold several.
        corners_rel = np.asarray(target.patch_corners_relative(patch, inset=inset))
        polygon = placement.to_pixel(corners_rel)
        sample = _sample_polygon(image, polygon, patch, high, low, measure_noise=measure_noise)
        if sample.pixel_count and sample.pixel_count < MIN_USEFUL_PIXELS:
            small_rois += 1
        samples[patch.patch_id] = sample

    if small_rois:
        warnings.append(
            f"{plural(small_rois, 'patch', 'patches')} too small to measure reliably."
        )

    empty = [pid for pid, s in samples.items() if not s.usable]
    if empty:
        warnings.append(
            f"{len(empty)} patch ROI(s) landed entirely outside the image: "
            + ", ".join(empty[:10])
            + (" ..." if len(empty) > 10 else "")
        )

    return SamplingResult(samples=samples, inset=inset, warnings=warnings)


def _sample_polygon(
    image: LoadedImage,
    polygon: np.ndarray,
    patch: Patch,
    high: float,
    low: float,
    measure_noise: bool = True,
) -> PatchSample:
    height, width = image.height, image.width

    x0 = int(np.floor(polygon[:, 0].min()))
    x1 = int(np.ceil(polygon[:, 0].max())) + 1
    y0 = int(np.floor(polygon[:, 1].min()))
    y1 = int(np.ceil(polygon[:, 1].max())) + 1

    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x1, width), min(y1, height)

    # Sized to the capture's own record count, not to three. A grayscale patch
    # measured as three NaNs would join the reference table as a colour it never had.
    nan3 = np.full(image.channels, np.nan)
    if cx1 <= cx0 or cy1 <= cy0:
        return PatchSample(
            patch_id=patch.patch_id,
            row=patch.row,
            col=patch.col,
            rgb_mean=nan3,
            rgb_median=nan3.copy(),
            rgb_std=nan3.copy(),
            lab_std=None,
            pixel_count=0,
            clipped_high_fraction=float("nan"),
            clipped_low_fraction=float("nan"),
            polygon_px=polygon,
            flags=["empty_roi"],
        )

    mask = np.zeros((cy1 - cy0, cx1 - cx0), dtype=np.uint8)
    shifted = np.round(polygon - np.array([cx0, cy0])).astype(np.int32)
    cv2.fillPoly(mask, [shifted], 1)

    # Only this patch's own window is converted to float. Normalizing a whole
    # master up front would cost eight times its pixel data in memory.
    window = image.window(cy0, cy1, cx0, cx1)
    pixels = window[mask.astype(bool)]

    flags: list[str] = []
    if pixels.size == 0:
        return PatchSample(
            patch_id=patch.patch_id,
            row=patch.row,
            col=patch.col,
            rgb_mean=nan3,
            rgb_median=nan3.copy(),
            rgb_std=nan3.copy(),
            lab_std=None,
            pixel_count=0,
            clipped_high_fraction=float("nan"),
            clipped_low_fraction=float("nan"),
            polygon_px=polygon,
            flags=["empty_roi"],
        )

    if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
        flags.append("roi_clipped_to_image")

    mean = pixels.mean(axis=0)
    median = np.median(pixels, axis=0)
    std = pixels.std(axis=0)

    lab_std = None
    if measure_noise:
        # FADGI states noise as the standard deviation of L* within a flat patch, so
        # it has to be computed on color-managed pixels, not on device values --
        # a fixed device-value spread means very different L* noise at the two ends
        # of the tone scale.
        from deltachrome.imaging.color import rgb_to_lab

        lab_std = rgb_to_lab(pixels, image.profile).std(axis=0)

    n = pixels.shape[0]
    clipped_high = float(np.any(pixels >= high, axis=1).sum()) / n
    clipped_low = float(np.any(pixels <= low, axis=1).sum()) / n

    if clipped_high > CLIPPING_FLAG_FRACTION:
        flags.append(f"clipped_high({clipped_high:.1%})")
    if clipped_low > CLIPPING_FLAG_FRACTION:
        flags.append(f"clipped_low({clipped_low:.1%})")
        pinned = _pinned_channels(pixels, mean, low)
        if pinned:
            # An untagged capture is device values too: there is no space it
            # was encoded into, so nothing is out of gamut.
            kind = (
                "channel_floor"
                if image.profile is None
                or getattr(image.profile, "profile_class", "") in DEVICE_ENCODING_CLASSES
                else "gamut_clipped"
            )
            flags.append(f"{kind}({''.join(pinned)})")
    if float(np.max(np.abs(mean - median))) > CONTAMINATION_FLAG:
        flags.append("mean_median_divergence")

    return PatchSample(
        patch_id=patch.patch_id,
        row=patch.row,
        col=patch.col,
        rgb_mean=mean,
        rgb_median=median,
        rgb_std=std,
        lab_std=lab_std,
        pixel_count=int(n),
        clipped_high_fraction=clipped_high,
        clipped_low_fraction=clipped_low,
        polygon_px=polygon,
        flags=flags,
    )


def render_overlay(
    image: LoadedImage,
    placement: ChartPlacement,
    result: SamplingResult,
    *,
    max_width: int = 1600,
) -> np.ndarray:
    """Draw sampled ROIs over the image, for visually verifying placement.

    With provisional geometry this is not a nicety -- it is the only way to confirm
    the ROIs actually sit on the patches before believing a single Delta E.

    The image is downsampled *before* any float conversion, so an overlay of a 100
    megapixel master costs a preview's worth of memory rather than the master's.
    """
    stride = max(1, int(np.ceil(image.width / max_width)))
    preview = to_display_records(image.pixels[::stride, ::stride, :])
    if preview.dtype.kind == "f":
        bgr = (np.clip(preview[..., ::-1], 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        bgr = (preview[..., ::-1].astype(np.float32) * (255.0 / image.scale)).astype(np.uint8)
    bgr = np.ascontiguousarray(bgr)

    def scaled(points: np.ndarray) -> np.ndarray:
        return np.round(np.asarray(points, dtype=np.float64) / stride).astype(np.int32)

    thickness = max(1, bgr.shape[1] // 600)
    cv2.polylines(bgr, [scaled(placement.corners_px)], True, (0, 255, 255), thickness)
    for sample in result.samples.values():
        if not sample.usable:
            continue
        color = (0, 0, 255) if sample.flags else (0, 255, 0)
        cv2.polylines(bgr, [scaled(sample.polygon_px)], True, color, max(1, thickness // 2))
    return bgr
