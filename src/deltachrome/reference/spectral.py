"""Turning a reference file's spectra into XYZ.

A reference file may carry Lab, XYZ, spectra, or several of those. Two of the
three were read and the third was stored and never used, which is a gap that
looks like nothing until it is the whole file: the LaserSoft ISO 12641-2's own
per-unit measurement is **864 patches of spectra and no colorimetry at all**, and
against it every patch read as unmeasured and the run stopped.

**Spectra are the better measurement, not a lesser one.** Lab in a reference file
is already an integration someone else performed, under an illuminant and observer
they chose and may not have stated; a spectrum is what the instrument saw, and it
can be integrated under whatever conditions the comparison actually needs. So the
integration here is not a fallback for a file missing its Lab -- it is the more
direct route, and where a file carries both, the Lab it states is still preferred
because a reader should not silently disagree with a number the file asserts.

The integration is the CIE's, and the only real decision in it is which illuminant
and observer to use. That is taken from the **file's own declaration** rather than
assumed, so the XYZ that comes out is described correctly by the metadata that came
with it -- which matters because the caller may then chromatically adapt it, and
adapting from an assumed illuminant would be worse than not adapting at all.
"""

from __future__ import annotations

import numpy as np

from deltachrome.reference.cie_data import ILLUMINANTS, OBSERVERS, SHAPE

#: Wavelengths the tables are on.
GRID = np.arange(SHAPE[0], SHAPE[1] + 1, SHAPE[2], dtype=np.float64)

#: Fewest wavelengths a spectrum must carry to be integrated. A handful of bands is
#: not a spectrum, and interpolating one onto a 5 nm grid would invent structure the
#: instrument never measured.
MIN_BANDS = 8

#: How a file may name its observer. CGATS writes ``OBSERVER_ANGLE`` and CxF an
#: observer element, and both are seen as an angle or as the year of the study.
OBSERVER_ALIASES = {2.0: "2", 1931.0: "2", 10.0: "10", 1964.0: "10"}

#: Reflectance factors above this are taken to be percentages rather than fractions.
#: Instruments write both, and a file mixing the two would be a different problem;
#: within one file the whole set is one convention or the other.
PERCENT_THRESHOLD = 3.0


def spectrum_to_xyz(
    spectrum: dict[float, float],
    *,
    illuminant: str = "D50",
    observer: str = "2",
) -> np.ndarray | None:
    """CIE XYZ for one measured spectrum, with Y = 1 for a perfect diffuser.

    Scaled to 1 rather than to 100 because that is what the rest of this engine
    uses; ``xyz_to_lab`` and the ICC path both work in that range.

    Returns ``None`` where the spectrum cannot be integrated -- too few bands, or an
    illuminant or observer that is not tabulated. That is a refusal rather than a
    guess for the usual reason: a value integrated under the wrong illuminant is
    wrong by several Delta E and looks entirely reasonable.
    """
    weights = _weights(illuminant, observer)
    if weights is None or len(spectrum) < MIN_BANDS:
        return None
    power, cmfs = weights

    wavelengths = np.asarray(sorted(spectrum), dtype=np.float64)
    values = np.asarray([spectrum[nm] for nm in wavelengths], dtype=np.float64)
    if float(np.max(values)) > PERCENT_THRESHOLD:
        values = values / 100.0

    # Resampled onto the tables' grid rather than the tables onto the file's,
    # because the tables are the thing with structure: an illuminant has emission
    # lines and the observer has curvature, and a reflectance spectrum between two
    # measured bands is the smooth part of the problem. Outside the measured range
    # the end value is held rather than extrapolated -- a reflectance is bounded and
    # a polynomial run past the data is not.
    resampled = np.interp(GRID, wavelengths, values)

    scale = float(np.sum(power * cmfs[:, 1]))
    if scale <= 0:
        return None
    return np.asarray(
        [float(np.sum(power * resampled * cmfs[:, axis])) / scale for axis in range(3)],
        dtype=np.float64,
    )


def _weights(illuminant: str, observer: str):
    """The illuminant's power and the observer's functions, or None if untabulated."""
    name = (illuminant or "D50").strip().upper()
    # Read as a number, not trimmed as text. ``"10".rstrip(".0")`` is ``"1"`` --
    # rstrip removes characters rather than a suffix -- which made the ten-degree
    # observer silently unavailable while the two-degree one worked by luck.
    angle = (observer or "2").strip() or "2"
    try:
        angle = OBSERVER_ALIASES.get(float(angle), angle)
    except ValueError:
        pass
    power = ILLUMINANTS.get(name)
    cmfs = OBSERVERS.get(angle)
    if power is None or cmfs is None:
        return None
    return np.asarray(power, np.float64), np.asarray(cmfs, np.float64)


#: The metadata key, and the run record's provenance key, saying that colorimetry
#: was integrated here from a file's spectra rather than stated by the file.
SPECTRA_INTEGRATED = "DELTACHROME_SPECTRA_INTEGRATED"


def fill_from_spectra(reference) -> int:
    """Give every spectra-only measurement in a set its XYZ, and say how many.

    Applied where a reference file is read, so that everything downstream -- role
    resolution, the reference Lab table, the frontend's own reads -- sees a file
    carrying colorimetry without any of them needing to know how it got there. The
    alternative, teaching each consumer to integrate, is how one file ends up
    integrated two different ways in one run.

    Integrated under the **file's own** declared illuminant and observer, so the
    values it produces are described correctly by the metadata that came with them.
    That matters because the caller may go on to chromatically adapt them, and
    adapting from an assumed illuminant is worse than not adapting at all.
    """
    from dataclasses import replace

    illuminant = reference.illuminant or "D50"
    observer = reference.observer or "2"
    filled = 0
    for name, measurement in list(reference.samples.items()):
        if measurement.lab is not None or measurement.xyz is not None:
            continue
        if not measurement.spectral:
            continue
        xyz = spectrum_to_xyz(
            measurement.spectral, illuminant=illuminant, observer=observer
        )
        if xyz is None:
            continue
        reference.samples[name] = replace(
            measurement, xyz=tuple(float(value) for value in xyz)
        )
        filled += 1

    if filled:
        # Provenance rather than a warning: the instrument's own software would
        # have integrated these differently had it been asked to, so the record
        # says it was done here and under what, and the run does not announce it.
        reference.metadata[SPECTRA_INTEGRATED] = (
            f"{filled} patches, {illuminant}, {observer} degree observer"
        )
    return filled
