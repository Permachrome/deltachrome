"""What the reference numbers are, in three states rather than two.

A delta-E against a measurement of *this* chart is the capture system's error. A
delta-E against a printed aim for the model also contains whatever this specimen's
print differs from the aim by. Between those sits a third thing that is neither:
the mean of many individually measured units, which is not a measurement of the
chart in front of the camera and is also not a guess.

The middle state is only worth having because it can be *earned*. A mean is
centred when its held-out error is smaller than the spread of the units it was
built from -- when not having your own chart's measurement costs you less than the
difference between two real charts of the model. The Golden Thread Object Level
Target is the case that proves it matters: its printed table sits 0.853 dE2000
from an average unit where two units sit 0.610 apart, and the 40-unit mean that
replaced it sits 0.457. Same model, same file format, opposite answers.
"""

from __future__ import annotations

import pytest

from deltachrome.profiles.evaluator import advisory_line
from deltachrome.reference import read_reference
from deltachrome.reference.cgats import (
    MANUFACTURER_NOMINAL,
    PER_UNIT_MEASUREMENT,
    POPULATION_MEAN,
    parse_cgats,
)

HEADER = """CGATS.17

ORIGINATOR\t"test"
{keywords}
NUMBER_OF_FIELDS\t4
BEGIN_DATA_FORMAT
SAMPLE_NAME\tLAB_L\tLAB_A\tLAB_B
END_DATA_FORMAT

NUMBER_OF_SETS\t1
BEGIN_DATA
1\t50.0\t0.0\t0.0
END_DATA
"""


def build(**fields: object) -> object:
    lines = []
    for key, value in fields.items():
        lines.append(f'KEYWORD\t"{key}"')
        lines.append(f"{key}\t{value}")
    return parse_cgats(HEADER.format(keywords="\n".join(lines)))


# -- the trap ------------------------------------------------------------------


def test_a_basis_this_version_does_not_know_is_not_a_conformance_claim():
    """The single most dangerous line in this module, and it used to run the other
    way: ``MANUFACTURER_NOMINAL if declared == MANUFACTURER_NOMINAL else
    PER_UNIT_MEASUREMENT``. Every unrecognised value became a per-unit measurement
    and earned a full claim, so adding any new basis here would silently have
    promoted the files carrying it. The claim is granted for one value, not
    withheld for a list."""
    unknown = build(DELTACHROME_REFERENCE_BASIS='"spectrophotometer_live"')
    assert unknown.basis == MANUFACTURER_NOMINAL
    assert unknown.is_nominal is True


def test_an_ordinary_file_declaring_nothing_is_still_a_measurement():
    """A file straight off a spectrophotometer declares no basis and is one. The
    default may not be applied to make something look better than it is, but it
    also may not make the ordinary case look worse."""
    plain = build()
    assert plain.basis == PER_UNIT_MEASUREMENT
    assert plain.is_nominal is False


# -- earned, not asserted ------------------------------------------------------


def test_a_population_mean_must_state_that_it_is_centred():
    """Declaring the basis is not enough; a file has to show its working."""
    bare = build(DELTACHROME_REFERENCE_BASIS='"population_mean"')
    assert bare.basis == MANUFACTURER_NOMINAL, "no figures, no promotion"


def test_a_mean_further_from_a_unit_than_two_units_are_from_each_other_is_nominal():
    """The Object Level Target's printed table, in numbers: 0.853 against a 0.610
    spread. Further from an average chart than two charts are from each other makes
    it a worse predictor of the chart in front of you than a randomly chosen other
    unit would be, which is not something to dignify with its own tier."""
    offset = build(
        DELTACHROME_REFERENCE_BASIS='"population_mean"',
        DELTACHROME_NOMINAL_HELD_OUT_DE=0.853,
        DELTACHROME_NOMINAL_POPULATION_SPREAD_DE=0.610,
    )
    assert offset.population_is_centred is False
    assert offset.basis == MANUFACTURER_NOMINAL


def test_a_centred_mean_is_promoted_and_still_earns_no_conformance_claim():
    centred = build(
        DELTACHROME_REFERENCE_BASIS='"population_mean"',
        DELTACHROME_NOMINAL_HELD_OUT_DE=0.457,
        DELTACHROME_NOMINAL_POPULATION_SPREAD_DE=0.610,
        DELTACHROME_NOMINAL_POPULATION_UNITS=40,
    )
    assert centred.basis == POPULATION_MEAN
    assert centred.population_is_centred is True
    assert centred.population_units == 40
    # It is still not a measurement of the chart in front of the camera.
    assert centred.is_nominal is True


@pytest.mark.parametrize("held,spread", [(0.61, 0.61), (0.7, 0.61)])
def test_the_comparison_is_strict(held, spread):
    """Equal is not inside. A mean exactly as far out as the spread buys nothing."""
    assert build(
        DELTACHROME_REFERENCE_BASIS='"population_mean"',
        DELTACHROME_NOMINAL_HELD_OUT_DE=held,
        DELTACHROME_NOMINAL_POPULATION_SPREAD_DE=spread,
    ).basis == MANUFACTURER_NOMINAL


def test_unreadable_figures_do_not_promote():
    assert build(
        DELTACHROME_REFERENCE_BASIS='"population_mean"',
        DELTACHROME_NOMINAL_HELD_OUT_DE='"about half"',
        DELTACHROME_NOMINAL_POPULATION_SPREAD_DE=0.610,
    ).basis == MANUFACTURER_NOMINAL


# -- what the shipped files claim ----------------------------------------------


@pytest.mark.parametrize("name,units", [
    ("isa_object_level_1x_v1_nominal.txt", 40),
    ("isa_golden_thread_dlt_v1_nominal.txt", 50),
])
def test_both_golden_thread_nominals_are_centred_population_means(name, units):
    """Both were replaced by means of published per-unit measurements, and both
    were measured to sit inside their own populations. The figures are declared
    beside the basis so the promotion is checkable rather than asserted."""
    reference = read_reference("src/deltachrome/targets/definitions/" + name)
    assert reference.basis == POPULATION_MEAN
    assert reference.population_units == units
    assert reference.held_out_de < reference.population_spread_de


def test_the_half_x_printed_table_is_nominal_and_says_so():
    """No per-unit measurements of this model are published, so its printed batch
    table is the only aim data that exists for it. That is honestly nominal."""
    reference = read_reference(
        "src/deltachrome/targets/definitions/isa_object_level_05x_v1_nominal.txt"
    )
    assert reference.basis == MANUFACTURER_NOMINAL


# -- what the reader is told ---------------------------------------------------


def test_a_population_mean_says_something_different_from_a_printed_aim():
    population = advisory_line(reference_basis=POPULATION_MEAN, local_criteria=False)
    nominal = advisory_line(reference_basis=MANUFACTURER_NOMINAL, local_criteria=False)
    assert population and nominal and population != nominal
    assert "measured units" in population
    assert population.startswith("Advisory:") and nominal.startswith("Advisory:")


def test_a_measurement_of_this_chart_is_not_advised_against():
    assert advisory_line(reference_basis=PER_UNIT_MEASUREMENT, local_criteria=False) is None


def test_a_record_from_before_the_basis_was_stored_says_only_what_it_knows():
    """``None`` is not a caveat by itself -- the record simply does not say. What
    such a record cannot do is claim conformance on that silence, which the
    library handles separately."""
    assert advisory_line(reference_basis=None, local_criteria=False) is None
    assert advisory_line(reference_basis=None, local_criteria=True) is not None
