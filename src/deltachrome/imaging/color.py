"""Colorimetric conversion and difference metrics.

Two rules govern this module:

1. Both sides of a comparison must share a white point and an observer. An image
   value converted to Lab against D50 and a reference value measured against D65
   are not comparable, and nothing downstream can detect that mistake. The
   conversion functions therefore take the white point explicitly.
2. Delta E formulae come from ``colour-science`` rather than being hand-rolled.
   CIEDE2000 in particular has well-known implementation traps (hue-angle
   averaging across the 0/360 boundary, the R_T rotation term); the test suite
   pins our results against Sharma et al.'s published verification data.
"""

from __future__ import annotations

from typing import Literal

import warnings

import numpy as np

with warnings.catch_warnings():
    # colour-science warns at import time when matplotlib is absent. Deltachrome's
    # engine never plots, so the warning is noise on every single run.
    warnings.simplefilter("ignore")
    from colour.difference import delta_E_CIE1976, delta_E_CIE1994, delta_E_CIE2000

from deltachrome.imaging.icc import CIE_D50, ICCProfile

DeltaEVariant = Literal[
    "delta_e_76", "delta_e_94", "delta_e_2000", "delta_e_ab", "delta_e_2000_ab"
]

#: Illuminant white points as XYZ, Y normalized to 1.0, 2 degree observer.
WHITE_POINTS: dict[str, tuple[float, float, float]] = {
    "D50": CIE_D50,
    "D55": (0.956797, 1.0, 0.921481),
    "D65": (0.950456, 1.0, 1.088754),
    "A": (1.098500, 1.0, 0.355800),
}

_EPSILON = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def xyz_to_lab(
    xyz: np.ndarray, white_point: tuple[float, float, float] = CIE_D50
) -> np.ndarray:
    """Convert XYZ to CIE L*a*b* against ``white_point``.

    Uses the CIE-recommended rational approximation below the linearity break
    (the 216/24389 form), not the older 0.008856/7.787 rounding, so results match
    modern instrument firmware.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[-1] != 3:
        raise ValueError(f"Expected trailing axis of 3, got shape {xyz.shape}")

    ratio = xyz / np.asarray(white_point, dtype=np.float64)
    f = np.where(
        ratio > _EPSILON,
        np.cbrt(np.clip(ratio, 0.0, None)),
        (_KAPPA * ratio + 16.0) / 116.0,
    )
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack(
        [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1
    )


def lab_to_xyz(
    lab: np.ndarray, white_point: tuple[float, float, float] = CIE_D50
) -> np.ndarray:
    """Inverse of :func:`xyz_to_lab`."""
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def inv(t: np.ndarray) -> np.ndarray:
        t3 = t**3
        return np.where(t3 > _EPSILON, t3, (116.0 * t - 16.0) / _KAPPA)

    xr = inv(fx)
    yr = np.where(L > _KAPPA * _EPSILON, ((L + 16.0) / 116.0) ** 3, L / _KAPPA)
    zr = inv(fz)
    return np.stack([xr, yr, zr], axis=-1) * np.asarray(white_point, dtype=np.float64)


def rgb_to_lab(
    rgb: np.ndarray,
    profile: ICCProfile,
    white_point: tuple[float, float, float] = CIE_D50,
) -> np.ndarray:
    """Device RGB in [0, 1] to Lab, through the image's own ICC profile.

    The profile's colorants are already adapted to the D50 PCS, so no chromatic
    adaptation happens here; the PCS XYZ is converted straight to Lab against
    ``white_point``, which defaults to CIE D50 to match spectrophotometer output.
    """
    return xyz_to_lab(profile.to_xyz(rgb), white_point=white_point)


def chromatic_adapt(
    xyz: np.ndarray,
    source_white: tuple[float, float, float],
    target_white: tuple[float, float, float],
    method: str = "Bradford",
) -> np.ndarray:
    """Adapt XYZ from one white point to another (Bradford by default).

    Only used when reference data was measured under an illuminant other than the
    one the image is evaluated against. Any use of this is reported, never silent:
    adaptation is a model, not a measurement.
    """
    from colour.adaptation import chromatic_adaptation_VonKries

    return chromatic_adaptation_VonKries(
        np.asarray(xyz, dtype=np.float64),
        np.asarray(source_white, dtype=np.float64),
        np.asarray(target_white, dtype=np.float64),
        transform=method,
    )


# -- difference metrics -----------------------------------------------------


def delta_e(
    sample: np.ndarray, reference: np.ndarray, variant: DeltaEVariant = "delta_e_2000"
) -> np.ndarray:
    """Color difference between two Lab arrays under the named variant."""
    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if variant == "delta_e_2000":
        return np.asarray(delta_E_CIE2000(sample, reference), dtype=np.float64)
    if variant == "delta_e_76":
        return np.asarray(delta_E_CIE1976(sample, reference), dtype=np.float64)
    if variant == "delta_e_94":
        return np.asarray(delta_E_CIE1994(sample, reference), dtype=np.float64)
    if variant == "delta_e_ab":
        return delta_e_ab(sample, reference)
    raise ValueError(f"Unknown Delta E variant {variant!r}")


def delta_e_ab(sample: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Chromaticity-only difference: Euclidean distance in the a*b* plane.

    This is FADGI's white balance measure. L* is excluded on purpose -- a gray
    patch reproduced too dark is a tone response problem, not a white balance one,
    and folding the two together hides which is which.
    """
    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    da = sample[..., 1] - reference[..., 1]
    db = sample[..., 2] - reference[..., 2]
    return np.sqrt(da**2 + db**2)


def delta_e_2000_ab(sample: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """CIEDE2000 with the lightness term removed -- Metamorfoze 2.0's white balance.

    Not the same quantity as :func:`delta_e_ab`, which is the plain Euclidean
    distance in the a*b* plane and is what FADGI asks for. This keeps CIEDE2000's
    chroma and hue weighting, and its rotation term, and drops only ``dL'``.

    Implemented by giving both colors the *reference's* lightness rather than by
    reimplementing the formula. That is exact rather than approximate: within
    CIEDE2000 the lightness enters the other terms only through ``S_L``, which
    multiplies the ``dL'`` term alone, so forcing ``dL' = 0`` leaves the chroma,
    hue and rotation terms computed exactly as they would have been. Reimplementing
    was the alternative, and CIEDE2000 has enough traps -- hue averaging across the
    0/360 boundary, the R_T rotation -- that not touching them is worth more than
    the directness.
    """
    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    sample, reference = np.broadcast_arrays(sample, reference)
    flattened_sample = np.stack(
        [reference[..., 0], sample[..., 1], sample[..., 2]], axis=-1
    )
    return delta_e(flattened_sample, reference, "delta_e_2000")


def delta_L(sample: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Signed lightness difference, sample minus reference."""
    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return sample[..., 0] - reference[..., 0]


def delta_L_2000(sample: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """The lightness term of CIEDE2000, |dL'| / (k_L * S_L).

    FADGI's tone response criterion is stated in "Delta L*2000" units. That is not
    plain |dL*|: CIEDE2000 divides the lightness difference by S_L, which is 1.0 at
    L*=50 and grows toward both ends of the scale, compressing differences in the
    deep shadows and bright highlights. At L* near 10, a 3.0 L* error scores about
    1.9. Computing this as plain |dL*| would therefore fail captures at the ends of
    the ramp that the standard passes.
    """
    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    L1 = sample[..., 0]
    L2 = reference[..., 0]
    L_bar = (L1 + L2) / 2.0
    S_L = 1.0 + (0.015 * (L_bar - 50.0) ** 2) / np.sqrt(20.0 + (L_bar - 50.0) ** 2)
    return np.abs(L1 - L2) / S_L


def chroma_ab(lab: np.ndarray) -> np.ndarray:
    """C*ab, the chroma of a Lab value. Used to identify achromatic patches."""
    lab = np.asarray(lab, dtype=np.float64)
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)


# -- optical density, and its uneasy relationship with L* --------------------

#: FADGI's transparency tables give a highlight aim of 96 +/- 2 L*. A density
#: ladder has no lightness of its own -- it describes a ratio of transmitted light,
#: not an appearance -- so turning one into aim L* values requires choosing what
#: lightness the target's *lightest* step should render at. Using the standard's own
#: highlight aim makes that choice the standard's rather than ours.
FADGI_TRANSPARENCY_HIGHLIGHT_L = 96.0


def density_to_luminance_factor(
    density: np.ndarray, *, reference_density: float = 0.0, reference_Y: float = 1.0
) -> np.ndarray:
    """Optical density to luminance factor, relative to a chosen anchor.

    ``D = -log10(T)``, so ``T = 10**-D``. The anchor says which density counts as
    the reference: film ladders quote absolute density including the base, so the
    lightest step of an ISA transmissive target is 0.05, not 0, and comparing it
    against a reference white means measuring from that step rather than from a
    hypothetical clear one.
    """
    density = np.asarray(density, dtype=np.float64)
    return reference_Y * 10.0 ** (-(density - reference_density))


def luminance_factor_to_density(
    Y: np.ndarray, *, reference_Y: float = 1.0
) -> np.ndarray:
    """The inverse. Y at or below zero has no density and comes back as infinity."""
    Y = np.asarray(Y, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return -np.log10(np.where(Y > 0.0, Y, np.nan) / reference_Y)


def lightness_from_luminance_factor(Y: np.ndarray) -> np.ndarray:
    """L* from a luminance factor, using the CIE piecewise function.

    Separate from :func:`xyz_to_lab` because a density ladder has only one
    dimension: there is no X or Z to convert, and inventing them would imply a
    chromaticity the density never carried.
    """
    Y = np.asarray(Y, dtype=np.float64)
    f = np.where(Y > _EPSILON, np.cbrt(np.maximum(Y, 0.0)), (_KAPPA * Y + 16.0) / 116.0)
    return 116.0 * f - 16.0


def density_to_lightness(
    density: np.ndarray,
    *,
    anchor_density: float | None = None,
    anchor_L: float = FADGI_TRANSPARENCY_HIGHLIGHT_L,
) -> np.ndarray:
    """Aim L* for a density ladder, anchored so ``anchor_density`` reads ``anchor_L``.

    **This is not a unit conversion.** Density describes the sample -- what fraction
    of the light it passes -- and is a property of the film in your hand. L*
    describes the appearance of a *reproduction*, and for a transmissive original
    there is no single correct one: it depends on how the system maps transmittance
    to output, which is the very thing a tone-response measurement is testing. So
    choosing an anchor is choosing a rendering intent, and the number this returns
    is an aim, not a measurement. See ``docs/understanding-your-results.md``.

    Two further limits worth knowing, both of which bite on real film ladders:

    * The relation is only exact for a **spectrally non-selective** sample. Density
      and Y weight the spectrum differently -- Y by the illuminant and the observer,
      visual density by the densitometer's own response -- and they agree only when
      the transmittance is flat enough that the weighting cannot matter.
    * Below ``Y = 216/24389`` (about ``D = 2.05`` from a white anchor) L* leaves its
      cube-root branch for a linear one, so equal density steps stop mapping to
      anything like equal L* steps. An ISA 4x5 ladder reaches D 2.25, so its last
      step is past that break.
    """
    density = np.asarray(density, dtype=np.float64)
    if anchor_density is None:
        anchor_density = float(np.min(density))
    Y = density_to_luminance_factor(
        density, reference_density=anchor_density, reference_Y=_Y_of_L(anchor_L)
    )
    return lightness_from_luminance_factor(Y)


def _Y_of_L(L: float) -> float:
    """Luminance factor of a given L*, the inverse of the CIE piecewise function."""
    f = (float(L) + 16.0) / 116.0
    return f**3 if f**3 > _EPSILON else (116.0 * f - 16.0) / _KAPPA


def lightness_per_density(L: np.ndarray) -> np.ndarray:
    """``|dL*/dD|`` at a given lightness -- how much L* one unit of density buys.

    The number that shows why the two scales are not interchangeable. On the
    cube-root branch ``L* = 116 * 10**(-D/3) - 16``, so ``dL*/dD = -(ln10/3)(L*+16)``:
    about 8.6 L* per 0.1 D near white, but only 2.8 near L* 20. A ladder with even
    density steps is not a ladder with even lightness steps, and a tolerance quoted
    in one says something quite different at each end when read in the other.
    """
    L = np.asarray(L, dtype=np.float64)
    return (np.log(10.0) / 3.0) * (L + 16.0)
