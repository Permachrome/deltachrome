"""Colorimetry, pinned against published verification data.

CIEDE2000 is notoriously easy to implement almost-correctly: the hue-angle mean
across the 0/360 boundary and the R_T rotation term both have traps that produce
results which look reasonable on ordinary colors and are wrong on the specific
pairs that matter. Sharma, Wu & Dalal published a test set precisely to catch
those. Deltachrome delegates the formula to colour-science, so this suite pins the
integration -- argument order, units, array handling -- rather than re-deriving the
maths.
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.imaging.color import (
    chroma_ab,
    delta_L_2000,
    delta_e,
    delta_e_2000_ab,
    delta_e_ab,
    lab_to_xyz,
    xyz_to_lab,
)
from deltachrome.imaging.icc import CIE_D50

# Sharma, Wu & Dalal (2005), "The CIEDE2000 Color-Difference Formula:
# Implementation Notes, Supplementary Test Data, and Mathematical Observations".
# Lab1, Lab2, expected dE2000.
SHARMA = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0010), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0011), 7.2195),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0012), 7.2195),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0009, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0010, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0011, -2.4900), 4.7461),
    ((50.0000, 2.5000, 0.0000), (50.0000, 0.0000, -2.5000), 4.3065),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((50.0000, 2.5000, 0.0000), (61.0000, -5.0000, 29.0000), 22.8977),
    ((50.0000, 2.5000, 0.0000), (56.0000, -27.0000, -3.0000), 31.9030),
    ((50.0000, 2.5000, 0.0000), (58.0000, 24.0000, 15.0000), 19.4535),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.1736, 0.5854), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2972, 0.0000), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 1.8634, 0.5757), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2592, 0.3350), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
    ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


@pytest.mark.parametrize("lab1,lab2,expected", SHARMA)
def test_delta_e_2000_matches_sharma_verification_data(lab1, lab2, expected):
    value = delta_e(np.array(lab1), np.array(lab2), "delta_e_2000")
    assert float(value) == pytest.approx(expected, abs=1e-4)


def test_delta_e_2000_is_symmetric():
    for lab1, lab2, _ in SHARMA:
        forward = float(delta_e(np.array(lab1), np.array(lab2), "delta_e_2000"))
        backward = float(delta_e(np.array(lab2), np.array(lab1), "delta_e_2000"))
        assert forward == pytest.approx(backward, abs=1e-9)


def test_delta_e_2000_vectorises():
    a = np.array([lab1 for lab1, _, _ in SHARMA])
    b = np.array([lab2 for _, lab2, _ in SHARMA])
    values = delta_e(a, b, "delta_e_2000")
    assert values.shape == (len(SHARMA),)
    assert values == pytest.approx([e for _, _, e in SHARMA], abs=1e-4)


def test_lab_xyz_round_trip():
    lab = np.array([[96.0, -0.9, 3.2], [53.2, 80.1, 67.2], [10.0, 0.5, -0.4], [0.0, 0.0, 0.0]])
    assert xyz_to_lab(lab_to_xyz(lab)) == pytest.approx(lab, abs=1e-10)


def test_lab_uses_the_cie_rational_form_below_the_break():
    """Very dark values are where the 0.008856/7.787 rounding and the exact
    216/24389 form visibly disagree, and where a tone ramp's shadow end lives."""
    xyz = np.array([0.001, 0.001, 0.001]) * np.array(CIE_D50)
    lab = xyz_to_lab(xyz)
    assert float(lab[0]) == pytest.approx(24389.0 / 27.0 * 0.001, abs=1e-9)


def test_delta_e_ab_ignores_lightness():
    """FADGI's white balance measure must not react to a tone error."""
    a = np.array([50.0, 1.0, -2.0])
    b = np.array([90.0, 1.0, -2.0])
    assert float(delta_e_ab(a, b)) == pytest.approx(0.0)

    c = np.array([50.0, 4.0, 2.0])
    assert float(delta_e_ab(a, c)) == pytest.approx(5.0)


def test_delta_e_2000_ab_ignores_lightness_too():
    """Metamorfoze 2.0 states white balance as CIEDE2000 without the luminance
    term, so a pure tone error must not register as a color cast."""
    a = np.array([50.0, 1.0, -2.0])
    b = np.array([90.0, 1.0, -2.0])
    assert float(delta_e_2000_ab(a, b)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "reference,sample,expected",
    [
        # Every row is exactly 5.0 Euclidean units apart in a*b*.
        ([50.0, 1.0, -2.0], [50.0, 4.0, 2.0], 5.310),      # near-neutral
        ([50.0, 60.0, 0.0], [50.0, 64.0, 3.0], 1.697),     # saturated red
        ([50.0, 0.0, -60.0], [50.0, 3.0, -64.0], 1.028),   # saturated blue
    ],
)
def test_delta_e_2000_ab_is_not_delta_e_ab(reference, sample, expected):
    """Both get called "white balance in a*b*", and they are not interchangeable.

    Every case here is the same 5.0 Euclidean units, and CIEDE2000 reads them as
    5.31, 1.70 and 1.03 -- a factor of five apart -- because its chroma weighting
    grows with chroma and the Euclidean form has no weighting at all.

    The direction also reverses. Near the neutral axis, which is where a white
    balance is actually measured, CIEDE2000 reads *higher*: so scoring Metamorfoze's
    requirement with FADGI's Euclidean measure is not a harmless substitution in the
    safe direction, it is a permissive one.
    """
    reference, sample = np.array(reference), np.array(sample)
    assert float(delta_e_ab(reference, sample)) == pytest.approx(5.0)
    assert float(delta_e_2000_ab(sample, reference)) == pytest.approx(expected, abs=0.001)


def test_delta_e_2000_ab_matches_full_delta_e_2000_when_lightness_agrees():
    """The check that the chroma, hue and rotation terms are untouched -- only the
    dL' term is removed."""
    reference = np.array([60.0, -12.0, 30.0])
    sample = np.array([60.0, -6.0, 24.0])
    assert float(delta_e_2000_ab(sample, reference)) == pytest.approx(
        float(delta_e(sample, reference, "delta_e_2000")), abs=1e-12
    )


def test_delta_e_2000_ab_broadcasts_one_reference_over_many_samples():
    """How the flat field uses it: every spot against one spot."""
    samples = np.array([[95.0, 0.0, -2.0], [95.0, 0.0, 0.0], [95.0, 0.0, 2.0]])
    reference = np.array([95.0, 0.0, -2.0])
    values = delta_e_2000_ab(samples, reference)
    assert values.shape == (3,)
    assert float(values[0]) == pytest.approx(0.0, abs=1e-9)
    assert values[2] > values[1] > values[0]


def test_delta_l_2000_is_not_plain_delta_l():
    """At L*=50 the S_L weight is 1.0; toward the ends it grows, compressing the
    reported difference. Treating dL*2000 as |dL*| would fail captures at the
    ends of the ramp that the standard passes."""
    mid = float(delta_L_2000(np.array([51.5, 0, 0]), np.array([48.5, 0, 0])))
    assert mid == pytest.approx(3.0, abs=0.01)

    shadow = float(delta_L_2000(np.array([13.0, 0, 0]), np.array([10.0, 0, 0])))
    assert shadow < 2.0
    highlight = float(delta_L_2000(np.array([96.0, 0, 0]), np.array([93.0, 0, 0])))
    assert highlight < 2.0


def test_delta_l_2000_is_unsigned():
    up = float(delta_L_2000(np.array([53.0, 0, 0]), np.array([50.0, 0, 0])))
    down = float(delta_L_2000(np.array([50.0, 0, 0]), np.array([53.0, 0, 0])))
    assert up == pytest.approx(down)


def test_chroma_ab():
    assert float(chroma_ab(np.array([50.0, 3.0, 4.0]))) == pytest.approx(5.0)
