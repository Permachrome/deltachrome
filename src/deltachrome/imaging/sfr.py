"""Slanted-edge spatial frequency response, after ISO 12233.

The measurement in outline: an edge tilted a few degrees from the pixel grid is
sampled at a slightly different sub-pixel phase on every row. Collecting those rows
against the edge's own line yields an edge profile sampled far more finely than the
sensor pitch. Differentiating it gives the line spread function, and its Fourier
transform is the spatial frequency response.

Each stage below carries a parameter, and they are all exposed rather than baked in
-- different standards and different tools make different choices, and a QC tool
that hard-codes one cannot reproduce another's numbers:

===========================  ==========================================================
``edge_fit``                 ``linear`` (ISO) or ``quadratic``, for lens distortion
``supersample_bin_width``    ESF bin size in pixels; ISO's 4x oversampling is 0.25
``window``                   ``hann`` (ISO 12233 ed.4), ``hamming`` (ed.3), ``tukey``,
                             ``none``
``derivative``               ``central`` [-0.5,0,0.5] (ISO) or ``forward`` [-1,1]
``derivative_correction``    undo the difference filter's own frequency response
``centroid_threshold``       fraction of edge height used to locate each row's edge
``detrend``                  level the two plateaux before differentiating; **off**
===========================  ==========================================================

**Nothing may modify the profile between the ESF and the transform.** The DFT sees
the windowed LSF and divides by its own DC value, so anything subtracted from the
profile first moves that divisor -- and the divisor scales the entire curve. Two
steps here used to do exactly that, and both are gone; see :func:`_finish`.

**The image must be linearized first.** The method assumes the edge profile is
linear in luminance; a gamma-encoded profile is compressed at one end and stretched
at the other, which distorts the LSF and biases the MTF. The distortion reverses
with edge polarity, so it does not even bias consistently. Measured on a real
capture in eciRGB v2, linearizing through the profile's own transform cut the
edge-to-edge spread of MTF at 0.25 cycles/pixel by 34%, and shrank the gap between
dark-to-light and light-to-dark edges -- which the optics cannot actually cause --
by 70%. Imatest linearize for the same reason.

Two further details are easy to get wrong and both bias the answer upward, which is
the dangerous direction for a conformance tool:

* **The differencing filter is not an ideal derivative.** A central difference has
  response ``sin(2*pi*f*d)`` where the ideal is ``2*pi*f``, so it rolls off with
  frequency. Leaving it uncorrected under-reports high-frequency MTF; ISO corrects
  for it, and so does this.
* **Windowing before differentiating** would tilt the ESF plateaux and inject a
  spurious low-frequency component. The window belongs on the LSF.

**The window is Hann, following ISO 12233 edition 4.** Editions up to 2017 specify
Hamming; the 4th substitutes a Tukey window and settles on alpha = 1.0, which is
Hann. Burns et al. give the reason: Hann is zero at its endpoints, and "for ROIs
with left-to-right non-uniformity, suppressing the end values could reduce any
low-frequency discontinuity that may be introduced". Measured here that is worth a
little -- it takes the worst maximum modulation on an ISA 4x5 from 1.0037 to exactly
1.0000 and tightens the edge-to-edge spread of MTF50 -- while moving MTF50 itself by
0.06 points and leaving the closed-form Gaussian check unchanged. ``hamming`` remains
available for reproducing a figure computed under the earlier edition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import cv2
import numpy as np

EdgeFit = Literal["linear", "quadratic"]
Binning = Literal["distance", "iso_interleave"]
WindowKind = Literal["hamming", "hann", "tukey", "none"]
DerivativeKind = Literal["central", "forward"]

#: ISO 12233 oversamples the edge profile fourfold.
DEFAULT_BIN_WIDTH = 0.25
#: An edge too near the pixel grid repeats sub-pixel phases instead of filling them;
#: too far and the projection smears. ISO recommends a few degrees.
MIN_EDGE_ANGLE_DEG = 1.0
MAX_EDGE_ANGLE_DEG = 15.0
#: Sub-pixel phase coverage, in pixels of edge drift across the ROI, below which the
#: oversampled profile has gaps. See :func:`phase_coverage`.
MIN_PHASE_COVERAGE_PX = 1.0
#: Lowest frequency included when looking for the curve's peak. **Zero by default**,
#: which includes DC. That matches how the criterion is written: FADGI's sharpening
#: limits (< 1.15 / 1.10 / 1.05 / 1.02) all sit just above 1.0, which only makes
#: sense if an unsharpened system reports exactly 1.0 -- and it does, because the
#: curve is normalized to 1.0 at DC. GoldenThread reports it the same way.
#:
#: Raising it excludes the lowest transform bins, which carry the analysis window's
#: leakage as well as the system. That is available as a diagnostic, but it changes
#: the scale the thresholds were written against, so it is not the default.
DEFAULT_PEAK_SEARCH_MIN_FREQUENCY = 0.0
#: Frequency below which a peak is more likely leakage than sharpening. Unsharp
#: masking peaks near 0.09 cycles/pixel; a peak in the first couple of transform
#: bins is a prompt to check the spatial overshoot instead.
LEAKAGE_SUSPECT_FREQUENCY = 0.05
#: Distance from the edge, in pixels, within which a halo is looked for. A real
#: overshoot sits beside the edge; noise is everywhere, so searching the whole
#: profile measures the noisiest bin rather than the halo.
OVERSHOOT_SEARCH_PX = 12.0


class SFRError(ValueError):
    """The edge could not be analyzed."""


def phase_coverage(rows: int, angle_deg: float) -> float:
    """Pixels of edge drift across ``rows`` rows -- the ROI's sub-pixel phase coverage.

    This is the quantity that decides whether an ROI is long enough, and it is why
    "how many rows" has no answer on its own. The whole method depends on each row
    catching the edge at a different sub-pixel phase; the edge drifts by
    ``rows * tan(angle)`` pixels down the ROI, so that product is how much of the
    pixel's width gets sampled. Below one pixel the phases do not close and the
    oversampled profile has gaps no bin width can fill.

    It also means the two dimensions trade off: a steeper edge needs fewer rows for
    the same coverage, and a nearly-aligned edge may need more rows than the feature
    physically has.
    """
    return abs(rows * math.tan(math.radians(angle_deg)))


@dataclass
class SFRResult:
    """One edge's spatial frequency response and the quantities derived from it."""

    frequency: np.ndarray          # cycles per pixel; Nyquist is 0.5
    mtf: np.ndarray                # normalized to 1.0 at DC
    esf: np.ndarray
    lsf: np.ndarray
    edge_angle_deg: float
    bin_width: float
    rows_used: int
    phase_coverage_px: float = 0.0
    settings: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # -- readings off the curve --------------------------------------------

    def mtf_at(self, frequency: float) -> float:
        """MTF at a frequency in cycles per pixel."""
        return float(np.interp(frequency, self.frequency, self.mtf))

    def frequency_at(self, level: float) -> float | None:
        """First frequency where the MTF falls to ``level``, interpolated.

        The *first* crossing, deliberately: a sharpened response can cross a level
        more than once, and the later crossings describe the overshoot rather than
        the resolution limit.
        """
        mtf = self.mtf
        below = np.nonzero(mtf < level)[0]
        if below.size == 0:
            return None
        i = int(below[0])
        if i == 0:
            return float(self.frequency[0])
        f0, f1 = self.frequency[i - 1], self.frequency[i]
        m0, m1 = mtf[i - 1], mtf[i]
        if m0 == m1:
            return float(f1)
        return float(f0 + (m0 - level) * (f1 - f0) / (m0 - m1))

    @property
    def nyquist(self) -> float:
        return 0.5

    def sampling_efficiency(self, level: float = 0.10) -> float | None:
        """Frequency at which the MTF reaches ``level``, as a fraction of Nyquist.

        FADGI's SFR10: an efficiency of 0.9 means the system still resolves at 90%
        of the theoretical limit of its own sampling grid.
        """
        f = self.frequency_at(level)
        return None if f is None else f / self.nyquist

    def response_at_nyquist(self) -> float:
        return self.mtf_at(self.nyquist)

    def sfr50(self) -> float | None:
        """MTF50 as a fraction of the half-sampling (Nyquist) frequency."""
        f = self.frequency_at(0.50)
        return None if f is None else f / self.nyquist

    def peak_modulation(
        self, min_frequency: float = DEFAULT_PEAK_SEARCH_MIN_FREQUENCY
    ) -> float:
        """Highest MTF value between ``min_frequency`` and Nyquist.

        Above 1.0 means the response is being boosted somewhere -- sharpening. It
        shows as a halo at edges and cannot be undone later, which is why FADGI caps
        it rather than treating more contrast as better.

        **The lowest frequencies are excluded on purpose.** An edge profile a few
        hundred bins long resolves frequencies in steps of well under 0.01 cycles per
        pixel, and its first few FFT bins are dominated by the analysis window's
        spectral leakage rather than by the system. Searching from DC found every one
        of twenty edges on a real capture "peaking" between 0 and 0.016 cycles per
        pixel -- bin 2 of the transform -- and reported 1.12 where the honest figure
        is 1.03. Real sharpening peaks in the mid frequencies, and is still caught.
        """
        mask = (self.frequency >= min_frequency) & (self.frequency <= self.nyquist)
        if not mask.any():
            mask = self.frequency <= self.nyquist
        return float(self.mtf[mask].max()) if mask.any() else float(self.mtf.max())

    def overshoot(self, search_px: float = OVERSHOOT_SEARCH_PX) -> float:
        """Spatial overshoot at the edge, as a percentage of the edge height.

        The direct measurement of haloing: how far the edge profile shoots past its
        own plateau before settling. Unsharp masking leaves a bright rim on the light
        side and a dark one on the dark side, and this is exactly that rim.

        Worth having alongside the MTF peak because the two fail differently. The MTF
        peak is a Fourier quantity and the lowest transform bins carry the analysis
        window's leakage, so a peak there can be an artifact. Overshoot is spatial and
        leakage cannot produce it -- if the profile does not overshoot, there is no
        halo, whatever the transform says.

        Measured on a lightly smoothed profile and only within ``search_px`` of the
        edge. A halo sits beside the edge; noise is everywhere, so searching the whole
        profile on a real capture returns the noisiest bin rather than the overshoot.

        Calibrated against synthetic unsharp masking: a clean edge gives 0%, a mild
        sharpen about 11%, a strong one about 27%. Imatest treat roughly 25% as the
        limit of acceptability.
        """
        n = max(4, len(self.esf) // 8)
        left = float(np.median(self.esf[:n]))
        right = float(np.median(self.esf[-n:]))
        height = abs(right - left)
        if height < 1e-9:
            return 0.0

        # Light smoothing: a halo is several pixels wide, single-bin noise is not.
        span = max(3, int(round(0.75 / self.bin_width)) | 1)
        kernel = np.ones(span) / span
        smoothed = np.convolve(self.esf, kernel, mode="same")

        distance = (np.arange(len(self.esf)) - len(self.esf) / 2.0) * self.bin_width
        near = np.abs(distance) <= search_px
        near[:span] = near[-span:] = False
        if not near.any():
            return 0.0

        high, low = max(left, right), min(left, right)
        above = float(smoothed[near].max()) - high
        below = low - float(smoothed[near].min())
        return 100.0 * max(above, below, 0.0) / height

    def peak_frequency(
        self, min_frequency: float = DEFAULT_PEAK_SEARCH_MIN_FREQUENCY
    ) -> float | None:
        """Where the peak sits -- the quantity that separates a halo from leakage.

        Unsharp masking peaks near 0.09 cycles/pixel. A peak in the first couple of
        transform bins, with no matching spatial overshoot, is the analysis window
        rather than the system.
        """
        mask = (self.frequency >= min_frequency) & (self.frequency <= self.nyquist)
        if not mask.any():
            return None
        return float(self.frequency[mask][int(np.argmax(self.mtf[mask]))])

    def summary(self) -> str:
        efficiency = self.sampling_efficiency()
        fifty = self.sfr50()
        lsf = np.abs(self.lsf)
        above = np.nonzero(lsf >= lsf.max() / 2)[0]
        fwhm = (above.max() - above.min() + 1) * self.bin_width if above.size else float("nan")
        return (
            f"edge {self.edge_angle_deg:+.2f} deg, {self.rows_used} rows, "
            f"phase coverage {self.phase_coverage_px:.2f} px, bin {self.bin_width}\n"
            f"  SFR10 sampling efficiency: "
            f"{'n/a' if efficiency is None else f'{efficiency:.3f}'}\n"
            f"  MTF at Nyquist:            {self.response_at_nyquist():.3f}\n"
            f"  SFR50 / half-sampling:     {'n/a' if fifty is None else f'{fifty:.3f}'}\n"
            f"  peak modulation:           {self.peak_modulation():.3f}"
            f"  (at {self.peak_frequency() or 0:.3f} c/px)\n"
            f"  spatial overshoot:         {self.overshoot():.2f}%\n"
            f"  LSF full width half max:   {fwhm:.2f} px"
        )


# -- the pipeline -----------------------------------------------------------


def _as_luminance(
    roi: np.ndarray, channel: str | int, linearize: Callable | None = None
) -> np.ndarray:
    """Reduce the ROI to the one-dimensional signal the edge analysis works on.

    ``linearize`` maps encoded values to a linear signal -- normally the capture's
    own ICC transform. Without it the profile is measured in whatever encoding the
    file happens to use, which distorts the LSF.

    **Linearize first, then select the channel.** Two different linearizers are
    admissible here and they return different shapes. The luminance transform
    returns one plane, already reduced, and there is no channel left to choose.
    A *per-channel* linearizer -- :meth:`ICCProfile.channel_linearizer` -- returns
    three, and a single record is then taken from it. Selecting the channel first
    and linearizing after would push an encoded value through a curve built for
    the linearized one; selecting from a luminance plane is not possible at all.
    An earlier version of this function returned the linearized array immediately
    and silently ignored ``channel``, so asking for the red record politely
    handed back luminance.
    """
    data = roi
    if linearize is not None and roi.ndim == 3:
        data = np.asarray(linearize(roi), dtype=np.float64)
    if data.ndim == 2:
        return data.astype(np.float64)
    if data.shape[-1] == 1:
        # A grayscale capture has one record and it is the one every analysis runs
        # on. Asking for "r" or "luminance" on it is answered by the same plane
        # rather than refused: there is no other record it could have meant.
        return data[..., 0].astype(np.float64)
    if isinstance(channel, int):
        return data[..., channel].astype(np.float64)
    if channel in ("r", "g", "b"):
        return data[..., "rgb".index(channel)].astype(np.float64)
    # ISO 12233 weights for a luminance-like combination of the color record.
    weights = np.array([0.213, 0.715, 0.072])
    return data[..., :3].astype(np.float64) @ weights


#: The next three functions are the part of ISO 12233 that locates an edge, before
#: any of the frequency analysis. They are public because a second measurement uses
#: exactly these steps and nothing after them: color-channel registration is the
#: same edge location, run once per color record and differenced. See
#: :mod:`deltachrome.imaging.registration`.


def orient_edge(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return the ROI with the edge running roughly top-to-bottom.

    A horizontal edge is transposed rather than handled separately: one code path
    means one set of behaviors to verify.
    """
    dx = float(np.abs(np.diff(image, axis=1)).mean())
    dy = float(np.abs(np.diff(image, axis=0)).mean())
    return (image, False) if dx >= dy else (image.T, True)


def row_edge_positions(
    image: np.ndarray, threshold: float, smooth: float = 1.0, window: int = 12
) -> np.ndarray:
    """Sub-pixel edge location for each row, by centroid of the row's derivative.

    The centroid is taken over a window around the derivative's peak, not the whole
    row. Over the whole row, any noise clearing the threshold anywhere on either
    plateau drags the centroid toward the row's middle -- on a real 8-bit capture
    that scattered positions by tens of pixels, which smears the projected profile
    into a ramp and produces an MTF that is entirely fictitious.
    """
    if smooth > 0:
        blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), smooth, smooth)
    else:
        blurred = image.astype(np.float32)
    derivative = np.abs(np.diff(blurred, axis=1))
    positions = np.full(image.shape[0], np.nan)
    x = np.arange(derivative.shape[1]) + 0.5
    for r in range(derivative.shape[0]):
        row = derivative[r]
        peak_index = int(np.argmax(row))
        peak = float(row[peak_index])
        if peak <= 0:
            continue
        lo = max(peak_index - window, 0)
        hi = min(peak_index + window + 1, len(row))
        segment = row[lo:hi]
        weights = np.where(segment >= threshold * peak, segment, 0.0)
        total = weights.sum()
        if total > 0:
            positions[r] = float((x[lo:hi] * weights).sum() / total)
    return positions


def fit_edge_line(
    positions: np.ndarray, fit: EdgeFit, reject_sigma: float = 2.5, passes: int = 2
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit the edge line, rejecting rows whose located edge does not lie on it.

    A handful of rows can be spoiled by a dust speck or a defect. Left in, they pull
    the fit and blur the projected profile; the fit itself identifies them, so they
    are dropped and the fit repeated.
    """
    rows = np.arange(len(positions), dtype=np.float64)
    good = np.isfinite(positions)
    if good.sum() < 4:
        raise SFRError(
            f"Only {int(good.sum())} of {len(positions)} rows produced an edge "
            "position; the ROI may not contain a single clean edge."
        )
    order = 1 if fit == "linear" else 2
    coefficients = np.polyfit(rows[good], positions[good], order)
    for _ in range(passes):
        residual = positions - np.polyval(coefficients, rows)
        spread = np.nanstd(residual[good])
        if not np.isfinite(spread) or spread == 0:
            break
        keep = good & (np.abs(residual) <= reject_sigma * spread)
        if keep.sum() < max(4, 0.5 * good.sum()):
            break
        good = keep
        coefficients = np.polyfit(rows[good], positions[good], order)
    return np.polyval(coefficients, rows), coefficients, int(good.sum())


def _window(n: int, kind: WindowKind) -> np.ndarray:
    if kind == "none":
        return np.ones(n)
    if kind == "hamming":
        return np.hamming(n)
    if kind == "hann":
        return np.hanning(n)
    if kind == "tukey":
        # Alpha 0.25 -- a narrow taper, not the standard's choice. ISO 12233 ed.4
        # settled on Tukey alpha = 1.0, which *is* a Hann window and is available
        # above under that name. Kept because a shallower taper is occasionally
        # wanted as a diagnostic; do not mistake it for "the ISO Tukey window".
        alpha = 0.25
        w = np.ones(n)
        taper = int(alpha * (n - 1) / 2)
        if taper > 0:
            ramp = 0.5 * (1 + np.cos(np.pi * (np.arange(taper) / taper - 1)))
            w[:taper] = ramp
            w[-taper:] = ramp[::-1]
        return w
    raise SFRError(f"Unknown window {kind!r}")


def compute_sfr(
    roi: np.ndarray,
    *,
    channel: str | int = "luminance",
    edge_fit: EdgeFit = "linear",
    supersample_bin_width: float = DEFAULT_BIN_WIDTH,
    window: WindowKind = "hann",
    derivative: DerivativeKind = "central",
    derivative_correction: bool = True,
    centroid_threshold: float = 0.10,
    detrend: bool = False,
    integer_phase_cycles: bool = True,
    linearize: Callable | None = None,
    binning: Binning = "distance",
) -> SFRResult:
    """Compute the SFR of a single slanted edge in ``roi``.

    ``roi`` is a 2D or (H, W, C) array containing exactly one edge, tilted a few
    degrees from the pixel grid.
    """
    if roi.ndim not in (2, 3):
        raise SFRError(f"ROI must be 2D or 3D, got shape {roi.shape}")
    if not 0.0 < supersample_bin_width <= 1.0:
        raise SFRError(f"supersample_bin_width must be in (0, 1]; got {supersample_bin_width}")

    image = _as_luminance(roi, channel, linearize)
    image, transposed = orient_edge(image)
    if min(image.shape) < 8:
        raise SFRError(f"ROI is too small to analyze: {image.shape}")

    warnings: list[str] = []

    positions = row_edge_positions(image, centroid_threshold)
    fitted, coefficients, rows_used = fit_edge_line(positions, edge_fit)
    residual = float(np.nanstd(positions - fitted))
    if residual > 1.5:
        warnings.append(
            "Slanted edge poorly located."
        )

    slope = float(coefficients[-2]) if len(coefficients) >= 2 else 0.0
    angle = math.degrees(math.atan(slope))

    if integer_phase_cycles and abs(slope) > 1e-6:
        # Use a whole number of pixel-shift cycles. With a fractional cycle the
        # leftover rows revisit phases the rest already covered, so those bins get
        # more samples than their neighbors and the profile is subtly weighted.
        rows_per_cycle = 1.0 / abs(slope)
        usable = int(rows_per_cycle * math.floor(image.shape[0] / rows_per_cycle))
        if 8 <= usable < image.shape[0]:
            image = image[:usable]
            positions = positions[:usable]
            fitted = fitted[:usable]

    coverage = phase_coverage(image.shape[0], angle)
    if coverage < MIN_PHASE_COVERAGE_PX:
        warnings.append(
            "Slanted edge is too shallow for reliable analysis."
        )
    if abs(angle) < MIN_EDGE_ANGLE_DEG:
        warnings.append(
            "Slanted edge is too shallow for reliable analysis."
        )
    elif abs(angle) > MAX_EDGE_ANGLE_DEG:
        warnings.append(
            "Target is too tilted for reliable edge analysis."
        )

    def settings_for(bin_width: float) -> dict[str, Any]:
        return {
            "channel": channel, "edge_fit": edge_fit,
            "supersample_bin_width": bin_width, "window": window,
            "derivative": derivative,
            "derivative_correction": derivative_correction,
            "centroid_threshold": centroid_threshold, "detrend": detrend,
            "integer_phase_cycles": integer_phase_cycles,
            "linearized": linearize is not None, "binning": binning,
            "transposed": transposed,
            "edge_fit_residual_px": round(residual, 4),
            "rows_after_cropping": int(image.shape[0]),
        }

    if binning == "iso_interleave":
        esf = _iso_interleave_esf(image, fitted, supersample_bin_width)
        supersample_bin_width = 1.0 / round(1.0 / supersample_bin_width)
        return _finish(
            esf, supersample_bin_width, angle, transposed, rows_used, coverage,
            window, derivative, derivative_correction, detrend, warnings,
            settings_for(supersample_bin_width),
        )

    # Project every pixel onto the edge normal and bin. This is the step that turns
    # many coarsely sampled rows into one finely sampled profile.
    height, width = image.shape
    columns = np.arange(width)[None, :]
    distance = (columns - fitted[:, None]).ravel()
    values = image.ravel()

    span = min(abs(distance.min()), abs(distance.max()))
    if span < 3:
        raise SFRError("The fitted edge sits too close to the ROI border to profile.")
    edges = np.arange(-span, span + supersample_bin_width, supersample_bin_width)
    index = np.digitize(distance, edges) - 1
    valid = (index >= 0) & (index < len(edges) - 1)
    counts = np.bincount(index[valid], minlength=len(edges) - 1)
    sums = np.bincount(index[valid], weights=values[valid], minlength=len(edges) - 1)

    empty = counts == 0
    if empty.any():
        if empty.mean() > 0.25:
            raise SFRError(
                f"{empty.mean():.0%} of edge-profile bins are empty. The edge angle or "
                "ROI size cannot support this bin width."
            )
    esf = np.divide(sums, np.maximum(counts, 1), dtype=np.float64)
    if empty.any():
        filled = np.nonzero(~empty)[0]
        esf = np.interp(np.arange(len(esf)), filled, esf[filled])

    return _finish(
        esf, supersample_bin_width, angle, transposed, rows_used, coverage,
        window, derivative, derivative_correction, detrend, warnings,
        settings_for(supersample_bin_width),
    )


def _iso_interleave_esf(
    image: np.ndarray, fitted: np.ndarray, bin_width: float
) -> np.ndarray:
    """Build the oversampled profile the way ISO 12233 and Imatest do.

    Rather than placing each pixel at its own distance from the edge, each *scan
    line* is assigned to one of K phase bins by the fractional part of its edge
    position, the rows within a bin are averaged after aligning on the integer part,
    and the K bin-averages are interleaved.

    The two approaches quantise differently: this one folds together rows whose true
    phases differ by up to 1/K of a pixel, while distance binning carries each
    pixel's actual phase and quantises once. On a synthetic edge of known MTF they
    agree to 0.0002; on a real capture individual edges differ by up to 0.097, which
    is the same order as the spread across a field of edges.

    Note the interleave runs in decreasing phase order. Getting that backwards
    produces a profile that looks entirely plausible and reports MTF about 30% low.
    """
    k = max(2, int(round(1.0 / bin_width)))
    height, width = image.shape
    fractional = fitted - np.floor(fitted)
    which = np.clip((fractional * k).astype(int), 0, k - 1)
    offset = np.floor(fitted).astype(int)

    pad = width
    profiles: list[np.ndarray] = []
    for phase in range(k):
        rows = np.nonzero(which == phase)[0]
        if rows.size == 0:
            raise SFRError(
                f"Phase bin {phase} of {k} caught no scan lines, so the interleaved "
                "profile cannot be built. The edge tilt and ROI length do not cover "
                "every sub-pixel phase; distance binning tolerates this."
            )
        total = np.zeros(2 * pad)
        count = np.zeros(2 * pad)
        for row in rows:
            start = pad - int(offset[row])
            total[start : start + width] += image[row]
            count[start : start + width] += 1
        profile = np.full(2 * pad, np.nan)
        filled = count > 0
        profile[filled] = total[filled] / count[filled]
        profiles.append(profile)

    usable = np.nonzero(np.all(np.isfinite(np.vstack(profiles)), axis=0))[0]
    if usable.size < 8:
        raise SFRError("The interleaved phase bins do not overlap over enough pixels.")
    esf = np.empty(usable.size * k)
    for position, phase in enumerate(range(k - 1, -1, -1)):
        esf[position::k] = profiles[phase][usable]
    return esf


def _finish(
    esf, bin_width, angle, transposed, rows_used, coverage, window, derivative,
    derivative_correction, detrend, warnings, settings,
) -> SFRResult:
    """Differentiate, window, transform, correct. The only tail there is.

    Both binning paths end here, and that is deliberate rather than tidiness. This
    function used to be duplicated inline in :func:`compute_sfr`, and the copies
    drifted: a line that had no business being in either sat in both, and fixing one
    would have left the other reading 12% high with nothing to show it.
    """
    if detrend:
        # OFF BY DEFAULT, and not merely as a preference.
        #
        # This removes shading *within* the plateaux -- an illumination gradient
        # across the ROI -- without touching the step between them. The intent is
        # sound and ISO 12233 edition 4 carries a correction for the same thing. But
        # theirs is informative rather than normative, it is Koren & Koren's method,
        # which fits each side of the edge separately, and Burns et al. attach the
        # warning that its effect varies with ROI width. This is a cruder thing
        # wearing the same name: one slope common to both plateaux.
        #
        # What it costs, measured. On a synthetic edge with a gradient across the
        # ROI it improves the MTF error by about a tenth at 5-10% gradient and by
        # nothing at all at 20-40% -- against the 83% improvement Imatest report for
        # theirs. On real captures it is what keeps max modulation off 1.000: with it
        # off, all 36 edges of the UTT and all 20 of the ISA 4x5 report exactly
        # 1.000, matching iQ-Analyzer on 18 tiles and OpenDICE on 58 blocks. With it
        # on, the worst reads 1.012 and the maximum sits two bins above DC.
        #
        # A weak correction that moves the curve's own reference point is a bad
        # trade. Left available for anyone who wants ISO's informative correction and
        # accepts what it does; if it is ever wanted properly, implement the per-side
        # fit rather than turning this on.
        n = max(4, len(esf) // 10)
        x = np.arange(len(esf), dtype=np.float64)
        slope_left = float(np.polyfit(x[:n], esf[:n], 1)[0])
        slope_right = float(np.polyfit(x[-n:], esf[-n:], 1)[0])
        if slope_left * slope_right > 0:
            common = min(abs(slope_left), abs(slope_right)) * np.sign(slope_left)
            esf = esf - common * (x - x.mean())

    if derivative == "central":
        lsf = np.gradient(esf)
        filter_order = 2.0
    else:
        lsf = np.diff(esf, prepend=esf[0])
        filter_order = 1.0

    # THE PEAK IS LOOKED FOR IN THE INTERIOR, and that is a correctness fix rather
    # than tidiness. The oversampled ESF's outermost bins are fed by the fewest
    # pixels, so they are the noisiest, and the derivative at the very boundary is
    # one-sided on top of that. The transition itself is at the centre by
    # construction -- the ESF is binned about the fitted edge line -- so a maximum at
    # the array's end is never the edge.
    #
    # Found on a real capture: one edge of one QA-62 card whose last ESF bin dipped
    # 0.06 below its plateau, giving a derivative 1.59x the true peak. The window
    # centred on it, wrapped (see below), and the DC term came out half its proper
    # size -- so every point of the curve was inflated about twofold. It reported no
    # SFR50 at all, 0.987 modulation at Nyquist and a peak of 1.116, which failed
    # the sharpening criterion. Its three sibling edges, on the same square in the
    # same capture, read 68-71% SFR50.
    margin = max(4, len(lsf) // 20)
    peak = margin + int(np.argmax(np.abs(lsf[margin:len(lsf) - margin])))

    # NOTHING GOES HERE. The LSF is transformed as it comes off the derivative.
    #
    # There used to be a line subtracting the median of the LSF tails, on the
    # reasonable-sounding grounds that a plateau should differentiate to zero. It is
    # in no reference implementation, and it is worse than redundant. Subtracting a
    # constant c and then windowing adds -c*w(n) to the windowed LSF, whose transform
    # is the window's own spectrum -- a spike confined to the first two bins. So it
    # moves spectrum[0] and leaves the rest of the numerator alone, and since every
    # point is divided by spectrum[0], the whole curve is multiplied by a constant.
    # Measured per edge on real captures: 0.907 to 1.124, flat across frequency to
    # five decimal places, inflating the edge-to-edge spread of MTF50 by 70% on the
    # ISA 4x5, and lifting max modulation to 1.105 on a capture that iQ-Analyzer and
    # OpenDICE both read as 1.00.
    #
    # If a plateau really is sloping, correct the *ESF* -- see `detrend` above, and
    # read why that is off by default before turning it on.

    # Window the LSF, centered on the transition. Applying it before differentiating
    # would tilt the ESF plateaux and inject a spurious low-frequency term.
    # Shifted onto the transition with ZERO FILL, not np.roll. A roll is circular:
    # a window centred near one end of the array reappears at the other, where it
    # weights the far plateau instead of the transition. The window exists to
    # isolate the transition, so what falls off the end must fall off rather than
    # come back -- and on the edge that found this, the wrap flipped the sign of the
    # windowed LSF's sum, which is the quantity the whole curve is divided by.
    #
    # With the peak found in the interior above, this is a no-op on every edge in
    # this repository: the shift is a few bins and nothing wraps. It is the guard
    # that keeps a near-boundary peak from being a silent factor of two rather than
    # a visible truncation.
    w = _window(len(lsf), window)
    shift = int(peak) - len(lsf) // 2
    placed = np.zeros_like(w)
    if shift >= 0:
        placed[shift:] = w[:len(w) - shift]
    else:
        placed[:shift] = w[-shift:]
    lsf_windowed = lsf * placed

    spectrum = np.abs(np.fft.rfft(lsf_windowed))
    if spectrum[0] == 0:
        raise SFRError("The edge profile has no DC component; the ROI may be blank.")
    mtf = spectrum / spectrum[0]
    frequency = np.fft.rfftfreq(len(lsf_windowed), d=bin_width)

    if derivative_correction:
        argument = filter_order * frequency * bin_width
        correction = np.ones_like(frequency)
        nonzero = argument > 0
        correction[nonzero] = np.sinc(argument[nonzero])
        mtf = np.divide(mtf, np.where(np.abs(correction) < 0.05, np.nan, correction))
        mtf = np.nan_to_num(mtf, nan=0.0)

    keep = frequency <= 1.0
    return SFRResult(
        frequency=frequency[keep], mtf=mtf[keep], esf=esf, lsf=lsf,
        edge_angle_deg=angle if not transposed else -angle,
        bin_width=bin_width, rows_used=rows_used, phase_coverage_px=coverage,
        settings=settings, warnings=warnings,
    )
