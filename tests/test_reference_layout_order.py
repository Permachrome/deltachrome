"""A reference file that numbers its patches and names none of them.

BabelColor PatchTool, among others, writes a measurement as ``SAMPLE_ID`` 1 to N
in the order the instrument read it, and says how long a row of that order was in
``LGOROWLENGTH``. The Wolf Faust IT8.7/2 file that prompted this was read down the
columns -- A1 to L1, then A2 -- and says 12. Read across the rows instead, the same
file judges the same capture at 39.0 dE2000 instead of 1.86, with every number
plausible-looking on its own.

What is tested is that the order is taken only where it can be checked, and that
the check is the chart's own declared ramp rather than anything in a capture.
"""

from __future__ import annotations

import pytest

from deltachrome.errors import ReferenceError
from deltachrome.reference.cgats import MeasurementSet, PatchMeasurement
from deltachrome.reference.sample_ids import reconcile_sample_ids
from deltachrome.targets.registry import load_target

ROWS, COLUMNS = 12, 22
LETTERS = "ABCDEFGHIJKL"


def _lab(patch_id: str) -> tuple[float, float, float]:
    """A plausible IT8 colour: column 16 a neutral ramp, everything else chromatic."""
    row, col = LETTERS.index(patch_id[0]), int(patch_id[1:])
    if col == 16:
        return (90.0 - 7.0 * row, 0.5, 1.0)
    return (30.0 + 2.0 * row, 20.0 + col, -15.0 + row)


def _numbered(order, *, row_length: int | None) -> MeasurementSet:
    """The colour array numbered 1..264 in ``order``, as a strip reader writes it."""
    samples = {
        str(k + 1): PatchMeasurement(sample_name=str(k + 1), lab=_lab(patch_id))
        for k, patch_id in enumerate(order)
    }
    metadata = {} if row_length is None else {"LGOROWLENGTH": str(row_length)}
    return MeasurementSet(samples=samples, metadata=metadata,
                          fields=["SAMPLE_ID", "LAB_L", "LAB_A", "LAB_B"])


DOWN_COLUMNS = [f"{LETTERS[r]}{c}" for c in range(1, COLUMNS + 1) for r in range(ROWS)]
ALONG_ROWS = [f"{LETTERS[r]}{c}" for r in range(ROWS) for c in range(1, COLUMNS + 1)]


@pytest.fixture(scope="module")
def it8():
    return load_target("it8_7_2_reflective_v1")


def test_a_file_read_down_the_columns_joins_every_patch(it8):
    reconciled, note = reconcile_sample_ids(_numbered(DOWN_COLUMNS, row_length=12), it8)

    assert len(reconciled.samples) == ROWS * COLUMNS
    for patch_id in ("A1", "L1", "A2", "F16", "L22"):
        assert reconciled.samples[patch_id].lab == _lab(patch_id)
    assert note and "down each column" in note and "LGOROWLENGTH" in note


def test_a_file_read_along_the_rows_joins_every_patch(it8):
    reconciled, note = reconcile_sample_ids(_numbered(ALONG_ROWS, row_length=22), it8)

    assert reconciled.samples["B1"].lab == _lab("B1")
    assert note and "along each row" in note


def test_a_row_length_that_contradicts_the_ramp_is_refused(it8):
    """Counted along the rows and labelled as if down the columns: the declared
    neutral ramp then lands on chromatic patches, and the file is refused rather
    than joined wrongly."""
    with pytest.raises(ReferenceError, match="neutral ramp"):
        reconcile_sample_ids(_numbered(ALONG_ROWS, row_length=12), it8)


def test_a_count_begun_at_the_other_end_is_refused(it8):
    """LGOROWLENGTH says which way the rows ran and not where the count began. A
    count from L22 back puts the ramp in reverse, and that is what catches it."""
    with pytest.raises(ReferenceError, match="neutral ramp"):
        reconcile_sample_ids(_numbered(DOWN_COLUMNS[::-1], row_length=12), it8)


def test_without_a_row_length_nothing_is_guessed(it8):
    with pytest.raises(ReferenceError, match="LGOROWLENGTH"):
        reconcile_sample_ids(_numbered(DOWN_COLUMNS, row_length=None), it8)


def test_a_square_array_cannot_say_which_way_it_was_read():
    """On a square lattice a row length equal to both dimensions settles nothing.
    The QA-62's OEC field is 6 x 6."""
    from deltachrome.targets.registry import available_targets

    square = []
    for target_id in available_targets():
        target = load_target(target_id)
        for region in target.grid_regions:
            if region.grid.rows == region.grid.columns and region.grid.rows > 1:
                square.append((target, region))
    if not square:
        pytest.skip("no shipped target has a square patch array")
    target, region = square[0]
    count = region.grid.rows * region.grid.columns
    if sum(r.grid.rows * r.grid.columns == count for r in target.grid_regions) != 1:
        pytest.skip("that target has two arrays of the same size")
    reference = MeasurementSet(
        samples={str(k): PatchMeasurement(sample_name=str(k), lab=(50.0, 0.0, 0.0))
                 for k in range(1, count + 1)},
        metadata={"LGOROWLENGTH": str(region.grid.rows)}, fields=["SAMPLE_ID"],
    )
    if any(p.patch_id in reference.samples for p in target.patches):
        pytest.skip("this target's own ids are numbers")
    with pytest.raises(ReferenceError, match="which way"):
        reconcile_sample_ids(reference, target)


def test_a_chart_that_prints_numbers_is_not_renumbered():
    """The Golden Thread prints 1 to 30. A file naming them 1 to 30 joins as
    written, and the layout reading never runs."""
    target = load_target("isa_golden_thread_dlt_v1")
    reference = MeasurementSet(
        samples={p.patch_id: PatchMeasurement(sample_name=p.patch_id, lab=(50.0, 0.0, 0.0))
                 for p in target.patches},
        metadata={"LGOROWLENGTH": "6"}, fields=["SAMPLE_ID"],
    )
    reconciled, note = reconcile_sample_ids(reference, target)

    assert note is None
    assert reconciled is reference


def test_a_named_file_is_untouched(it8):
    reference = MeasurementSet(
        samples={p: PatchMeasurement(sample_name=p, lab=_lab(p)) for p in DOWN_COLUMNS},
        metadata={"LGOROWLENGTH": "12"}, fields=["SAMPLE_NAME"],
    )
    reconciled, note = reconcile_sample_ids(reference, it8)

    assert note is None
    assert reconciled is reference
