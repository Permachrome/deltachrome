"""Optical density, and the conversion to L* that is not really a conversion.

Transmissive targets are specified in density; every metric here works in L*. The
two describe the same physical quantity through different transfer functions, and
the places they part company are exactly the places a naive conversion goes wrong:
the CIE linearity break, which real film ladders reach; the spectral weighting,
which only agrees for a neutral; and the anchor, which has no arithmetic answer at
all. See the guide's "Density and L*" section.
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.imaging.color import (
    FADGI_TRANSPARENCY_HIGHLIGHT_L,
    _EPSILON,
    density_to_lightness,
    density_to_luminance_factor,
    lightness_from_luminance_factor,
    lightness_per_density,
    luminance_factor_to_density,
    xyz_to_lab,
)

# -- the parts that are ordinary arithmetic ---------------------------------


@pytest.mark.parametrize("density", [0.0, 0.05, 0.5, 1.0, 2.25, 3.5])
def test_density_and_luminance_factor_round_trip(density):
    Y = density_to_luminance_factor(np.asarray(density))
    assert float(luminance_factor_to_density(Y)) == pytest.approx(density, abs=1e-9)


def test_density_is_a_ratio_so_the_anchor_shifts_the_whole_ladder():
    ladder = np.array([0.05, 0.5, 1.0])
    absolute = density_to_luminance_factor(ladder)
    relative = density_to_luminance_factor(ladder, reference_density=0.05)
    assert np.allclose(relative / absolute, 10.0**0.05)
    assert float(relative[0]) == pytest.approx(1.0)


def test_lightness_from_luminance_factor_matches_the_lab_conversion():
    """The one-dimensional path must agree with the three-dimensional one, or a
    density-derived aim and a measured Lab would not be on the same scale."""
    for Y in (1.0, 0.9, 0.5, 0.18, 0.05, 0.008, 0.001):
        lab = xyz_to_lab(np.array([Y * 0.9642, Y, Y * 0.8251]))
        assert float(lightness_from_luminance_factor(np.asarray(Y))) == pytest.approx(
            float(lab[0]), abs=1e-6
        )


# -- the parts where the two scales part company ----------------------------


def test_the_closed_form_holds_above_the_break_and_fails_below_it():
    """``L* = 116 * 10**(-D/3) - 16`` is only the cube-root branch. Film ladders run
    past it -- the ISA 4x5 reaches D 2.25 -- so using the closed form everywhere
    would quietly misplace the darkest steps."""
    break_density = -np.log10(_EPSILON)          # about 2.053
    above = np.array([0.0, 0.5, 1.0, 2.0])
    closed_form = 116.0 * 10.0 ** (-above / 3.0) - 16.0
    assert np.allclose(density_to_lightness(above, anchor_L=100.0), closed_form, atol=1e-6)

    below = np.array([2.2, 2.5, 3.0])
    assert np.all(below > break_density)
    naive = 116.0 * 10.0 ** (-below / 3.0) - 16.0
    actual = density_to_lightness(below, anchor_L=100.0)
    assert np.all(actual > naive)
    assert float(np.abs(actual - naive).max()) > 0.1


def test_equal_density_steps_are_not_equal_lightness_steps():
    """The reason a tolerance in one says something different at each end of a ladder
    when read in the other."""
    ladder = np.arange(0.0, 1.6, 0.1)
    L = density_to_lightness(ladder, anchor_L=100.0)
    steps = -np.diff(L)
    # 8.6 L* for the first 0.1 D, 2.9 for the last: same density step, three times
    # the lightness.
    assert float(steps[0]) == pytest.approx(8.57, abs=0.05)
    assert float(steps[-1]) == pytest.approx(2.93, abs=0.05)
    assert steps[0] > 2.9 * steps[-1]
    # and the analytic slope agrees with the ladder's own differences
    midpoints = (L[:-1] + L[1:]) / 2.0
    assert np.allclose(lightness_per_density(midpoints) * 0.1, steps, rtol=0.02)


def test_the_slope_is_far_steeper_in_the_highlights():
    assert float(lightness_per_density(np.asarray(96.0))) == pytest.approx(85.96, abs=0.1)
    assert float(lightness_per_density(np.asarray(20.0))) == pytest.approx(27.63, abs=0.1)


# -- the anchor, which is a choice and not arithmetic -----------------------


def test_the_anchor_sets_what_the_lightest_step_renders_at():
    ladder = np.array([0.05, 0.5, 1.0, 2.25])
    at_96 = density_to_lightness(ladder)
    assert float(at_96[0]) == pytest.approx(FADGI_TRANSPARENCY_HIGHLIGHT_L)
    at_100 = density_to_lightness(ladder, anchor_L=100.0)
    assert float(at_100[0]) == pytest.approx(100.0)
    # Every other step moves too: the anchor is not a cosmetic offset.
    assert np.all(at_100[1:] > at_96[1:])


def test_the_anchor_defaults_to_the_ladders_own_minimum_density():
    """Film ladders quote density including the base, so the lightest step is 0.05,
    not 0. Anchoring at a hypothetical clear step would darken the whole ladder."""
    ladder = np.array([0.05, 0.5, 1.0])
    assert float(density_to_lightness(ladder)[0]) == pytest.approx(
        FADGI_TRANSPARENCY_HIGHLIGHT_L
    )
    explicit = density_to_lightness(ladder, anchor_density=0.0)
    assert float(explicit[0]) < FADGI_TRANSPARENCY_HIGHLIGHT_L


def test_the_isa_ladder_lands_where_the_generator_says_it_does():
    """Pins the shipped nominal file to the code that derives it."""
    from tools.generate_density_ladder_nominal import LADDERS

    isa = next(l for l in LADDERS if l.output.startswith("isa_transmissive_4x5"))
    L = density_to_lightness(np.asarray(isa.densities))
    assert float(L[0]) == pytest.approx(96.0)
    assert float(L[-1]) == pytest.approx(5.130, abs=0.001)
    assert np.all(np.diff(L) < 0)


def test_every_generated_ladder_file_matches_the_generator():
    from tools.generate_density_ladder_nominal import DEFINITIONS, LADDERS, build

    for ladder in LADDERS:
        assert (DEFINITIONS / ladder.output).read_text(encoding="utf-8") == build(ladder)


def test_each_ladder_names_its_steps_per_block_and_not_per_file():
    """OpenDICE numbers both copies of a ladder straight through, which says nothing
    about which copy a step is in. The definitions address the two blocks separately,
    so the ids have to carry the block -- and every one has to be unique, or the
    reference reader silently keeps one of a colliding pair."""
    from tools.generate_density_ladder_nominal import LADDERS, build

    for ladder in LADDERS:
        body = build(ladder).split("BEGIN_DATA\n")[1].split("END_DATA")[0]
        names = [line.split("\t")[0] for line in body.strip().splitlines()]
        assert len(names) == len(ladder.densities) * len(ladder.blocks)
        assert len(set(names)) == len(names)
        for block in ladder.blocks:
            assert f"{block}1" in names


def test_the_microform_polarities_do_not_share_a_ladder():
    """A positive and a negative of one chart are different specimens. Their ladders
    run opposite ways, so swapping the files inverts the ramp while still sampling
    every patch -- a failure that produces a full report and no error."""
    from tools.generate_density_ladder_nominal import LADDERS

    by_output = {l.output: l for l in LADDERS}
    positive = by_output["microform_35mm_positive_v1_nominal.txt"].densities
    negative = by_output["microform_35mm_negative_v1_nominal.txt"].densities
    assert np.all(np.diff(positive) < 0)
    assert np.all(np.diff(negative) > 0)
    assert positive != negative
