"""Reading a reference file that carries spectra and no colorimetry.

The case that forced this is not a corner: the LaserSoft ISO 12641-2's own per-unit
measurement is **864 patches of spectra and nothing else**, which is what a
spectrophotometer actually records. Against it every patch read as unmeasured and
the run stopped, on an error message about patch ids that all matched perfectly.

The check these tests really rest on is the first one. A perfect diffuser must
integrate to the illuminant's own white point, by definition -- so if either the
tables or the integration were wrong, that identity would not hold, and it holds to
about two parts in a hundred thousand.
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.errors import MissingReferenceError
from deltachrome.imaging.color import WHITE_POINTS
from deltachrome.reference.cgats import MeasurementSet, PatchMeasurement
from deltachrome.reference.spectral import GRID, fill_from_spectra, spectrum_to_xyz
from deltachrome.targets.registry import load_target
from deltachrome.targets.roles import resolve_roles


def _flat(value: float = 1.0) -> dict[float, float]:
    return {float(nm): value for nm in GRID}


def test_a_perfect_diffuser_integrates_to_the_illuminants_white_point():
    """The identity that checks the tables and the integration at once.

    Nothing here is fitted or tuned, so there is no way for this to pass by
    accident: a reflectance of 1 at every wavelength integrated against an
    illuminant *is* that illuminant's white point.
    """
    for name, expected in WHITE_POINTS.items():
        xyz = spectrum_to_xyz(_flat(), illuminant=name)
        assert xyz is not None
        assert xyz == pytest.approx(np.asarray(expected), abs=3e-4), name


def test_the_generated_tables_are_the_published_ones():
    """The tables are generated rather than transcribed; this is why that is safe.

    ``tools/generate_cie_tables.py`` writes them from colour-science, and a stale or
    mis-regenerated table would be wrong by a few hundredths of a Delta E -- which
    is invisible in a report and still wrong.
    """
    colour = pytest.importorskip("colour")

    from deltachrome.reference.cie_data import ILLUMINANTS, OBSERVERS, SHAPE

    shape = colour.SpectralShape(*SHAPE)
    published = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape)
    assert np.asarray(OBSERVERS["2"]) == pytest.approx(
        np.asarray(published.values, float), abs=1e-8
    )
    d50 = colour.SDS_ILLUMINANTS["D50"].copy().extrapolate(shape).align(shape)
    assert np.asarray(ILLUMINANTS["D50"]) == pytest.approx(
        np.asarray(d50.values, float), abs=1e-6
    )


def test_percentages_and_fractions_read_alike():
    """Instruments write reflectance both ways and neither is wrong."""
    assert spectrum_to_xyz(_flat(0.4)) == pytest.approx(
        spectrum_to_xyz(_flat(40.0)), abs=1e-9
    )


def test_a_grey_is_neutral_and_scales():
    """Half the light, half the tristimulus -- reflectance is linear in XYZ."""
    white = spectrum_to_xyz(_flat(1.0))
    grey = spectrum_to_xyz(_flat(0.5))
    assert grey == pytest.approx(white * 0.5, abs=1e-9)


def test_a_handful_of_bands_is_not_a_spectrum():
    """Interpolating four bands onto a 5 nm grid invents structure nobody measured."""
    assert spectrum_to_xyz({400.0: 0.5, 500.0: 0.5, 600.0: 0.5, 700.0: 0.5}) is None


def test_an_untabulated_illuminant_is_refused_rather_than_assumed():
    """A value integrated under the wrong illuminant is wrong by several Delta E
    and looks entirely reasonable, which is the whole reason for refusing."""
    assert spectrum_to_xyz(_flat(), illuminant="F11") is None
    assert spectrum_to_xyz(_flat(), observer="47") is None


def test_the_observer_is_read_however_the_file_spells_it():
    for spelling in ("2", "2.0", "1931", ""):
        assert spectrum_to_xyz(_flat(), observer=spelling) is not None
    ten = spectrum_to_xyz(_flat(), observer="10")
    assert ten is not None and ten != pytest.approx(spectrum_to_xyz(_flat()), abs=1e-3)


# -- what the reader does with them --------------------------------------------


def _spectral_set(names, *, illuminant="D50", observer="2"):
    return MeasurementSet(
        samples={
            name: PatchMeasurement(sample_name=name, spectral=_flat(level))
            for name, level in names.items()
        },
        metadata={"ILLUMINATION_NAME": illuminant, "OBSERVER_ANGLE": observer},
        fields=["SAMPLE_NAME"],
    )


def test_spectra_become_xyz_when_the_file_is_read():
    reference = _spectral_set({f"A{i}": 0.2 + i * 0.05 for i in range(1, 10)})
    assert all(m.xyz is None for m in reference)
    assert fill_from_spectra(reference) == 9
    assert all(m.xyz is not None for m in reference)
    # Provenance, not an internal detail -- the instrument's own software would
    # have integrated it differently -- so the record carries it. It is not a
    # warning: nothing is wrong (2026-10-02 review).
    assert reference.provenance()["DELTACHROME_SPECTRA_INTEGRATED"].startswith("9 patches")
    assert not any("spectra" in w for w in reference.warnings)


def test_a_file_that_states_its_colorimetry_is_not_second_guessed():
    """Where a file asserts Lab, that is what is used. A reader silently disagreeing
    with a number the file states is worse than one that cannot read spectra."""
    stated = MeasurementSet(
        samples={"A1": PatchMeasurement(
            sample_name="A1", lab=(50.0, 1.0, -2.0), spectral=_flat(0.9)
        )},
        metadata={}, fields=["SAMPLE_NAME"],
    )
    assert fill_from_spectra(stated) == 0
    assert stated["A1"].lab == (50.0, 1.0, -2.0)
    assert stated["A1"].xyz is None


def test_roles_resolve_from_a_reference_carrying_only_xyz():
    """The second half of the same fault, and the wider one.

    Role resolution read ``lab`` and nothing else, so a file carrying XYZ -- or
    spectra, which become XYZ when the file is read -- had every patch counted as
    unmeasured whatever its sample names said.
    """
    target = load_target("colorchecker_sg_v1")
    reference = MeasurementSet(
        samples={
            patch.patch_id: PatchMeasurement(
                sample_name=patch.patch_id,
                xyz=(0.2 + 0.001 * index, 0.2 + 0.001 * index, 0.2 + 0.001 * index),
            )
            for index, patch in enumerate(target.patches)
        },
        metadata={}, fields=["SAMPLE_NAME"],
    )
    resolution = resolve_roles(target, reference)
    assert resolution.counts
    assert not resolution.unmeasured


def test_the_wrong_chart_and_an_unreadable_one_are_different_errors():
    """Reporting one as the other sent a reader hunting a chart mismatch that was
    not there, on a file naming all 864 patches correctly."""
    target = load_target("colorchecker_sg_v1")

    wrong_chart = MeasurementSet(
        samples={"ZZ99": PatchMeasurement(sample_name="ZZ99", lab=(50.0, 0.0, 0.0))},
        metadata={}, fields=["SAMPLE_NAME"],
    )
    with pytest.raises(MissingReferenceError, match="covers none of"):
        resolve_roles(target, wrong_chart)

    right_chart_no_colour = MeasurementSet(
        samples={
            patch.patch_id: PatchMeasurement(sample_name=patch.patch_id)
            for patch in target.patches
        },
        metadata={}, fields=["SAMPLE_NAME"],
    )
    with pytest.raises(MissingReferenceError, match="it is the right chart"):
        resolve_roles(target, right_chart_no_colour)
