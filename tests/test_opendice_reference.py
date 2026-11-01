"""OpenDICE target profiles read as reference files.

The format is a tab-separated table with a one-line header and nothing else, and
FADGI's website hands it out, so an operator coming from OpenDICE arrives holding
it. What is tested is that it is recognised by content, that its patch names reach
the chart's own, and that the two places it is ambiguous -- a name used twice, and
a ladder numbered straight through two blocks -- are settled only by a declaration
in the target, never by position alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deltachrome.errors import CGATSParseError, DuplicateSampleError
from deltachrome.imaging.color import density_to_lightness
from deltachrome.reference import read_reference, sniff_format
from deltachrome.reference.opendice import parse_opendice
from deltachrome.reference.sample_ids import reconcile_sample_ids
from deltachrome.targets.registry import load_target

DLT_HEAD = "\tL\ta \tb\tD\nPatch1\t38.07\t11.74\t13.51\t\nPatch10\t95.93\t-1.03\t2.84\t0.04\n"


@pytest.mark.parametrize(
    "header", ["\tL\ta \tb\tD", "\tL*\ta*\tb*\tD", "\tL\ta\tb\tD", "\tD"]
)
def test_every_header_in_circulation_is_recognised(header):
    assert sniff_format(header + "\nPatch1\t1\t2\t3\t0.1\n") == "opendice"


def test_cgats_and_cxf_are_not_mistaken_for_it():
    assert sniff_format("CGATS.17\nBEGIN_DATA_FORMAT\n") == "cgats"
    assert sniff_format("SAMPLE_ID\tL\ta\tb\n1\t2\t3\t4\n") is None


def test_patch_prefix_is_opendices_and_is_removed():
    reference = parse_opendice(DLT_HEAD)
    assert list(reference.samples) == ["1", "10"]
    assert reference["10"].lab == (95.93, -1.03, 2.84)
    assert reference["10"].raw["D"] == 0.04
    assert "D" not in reference["1"].raw


def test_stated_lightness_wins_over_density():
    # The density is kept as density. OpenDICE converts it back to L* and judges
    # against that; the file's own L* is the better number, and is the one used.
    reference = parse_opendice("\tL\ta\tb\tD\n7\t87.78\t0\t0\t0.10\n")
    assert reference["7"].lab[0] == 87.78


def test_density_only_ladder_takes_lightness_from_the_shipped_convention():
    reference = parse_opendice("\tD\nPatch1\t0.05\nPatch2\t1.15\nPatch3\t2.25\n")
    expected = density_to_lightness([0.05, 1.15, 2.25])
    assert [reference[k].lab[0] for k in "123"] == pytest.approx(expected)
    assert reference["1"].lab[0] == pytest.approx(96.0)
    assert all(reference[k].lab[1:] == (0.0, 0.0) for k in "123")
    assert "L* 96" in reference.metadata["DELTACHROME_LIGHTNESS_FROM_DENSITY"]


def test_part_of_a_lab_value_is_refused():
    with pytest.raises(CGATSParseError):
        parse_opendice("\tL\ta\tb\tD\n1\t50\t\t2\t\n")


def test_a_repeated_name_is_kept_until_something_settles_it():
    reference = parse_opendice("\tL\ta\tb\tD\nB1\t53\t-45\t38\t\nB1\t95\t0\t0\t0.06\n")
    assert len(reference) == 2
    assert reference.row_names == ["B1", "B1"]


def test_an_unsettled_repeat_is_refused():
    reference = parse_opendice("\tL\ta\tb\tD\nB1\t53\t-45\t38\t\nB1\t95\t0\t0\t0.06\n")
    with pytest.raises(DuplicateSampleError):
        reconcile_sample_ids(reference, load_target("colorchecker_sg_v1"))


def _utt_profile() -> str:
    rows = [f"{c}{i}\t50\t10\t-10\t" for c in "ABCDEF" for i in range(1, 10)]
    rows += [f"{s}{i}\t{100 - 4.5 * i:.1f}\t0\t0\t0.1" for s in "LURB" for i in range(1, 21)]
    return "\tL\ta \tb\tD\n" + "\n".join(rows) + "\n"


def test_utt_profile_is_matched_by_its_declared_row_order():
    reference = parse_opendice(_utt_profile())
    joined, note = reconcile_sample_ids(reference, load_target("image_engineering_utt_a3_v1"))
    assert note and "OpenDICE" in note
    # The colour B1 and the bottom grey scale's B1 land on different patches.
    assert joined["B1"].lab == (50.0, 10.0, -10.0)
    assert joined["GS_D_1"].lab[0] == pytest.approx(95.5)
    assert joined["GS_L_20"].lab[0] == pytest.approx(10.0)
    assert len(joined) == 134 and not joined.row_names


def test_a_row_order_is_used_only_where_the_names_agree():
    # Swap two rows: positions alone would still "fit", the names no longer do.
    lines = _utt_profile().splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    reference = parse_opendice("\n".join(lines) + "\n")
    with pytest.raises(DuplicateSampleError):
        reconcile_sample_ids(reference, load_target("image_engineering_utt_a3_v1"))


@pytest.mark.parametrize(
    "target_id, first_block, second_block, steps",
    [
        ("isa_transmissive_4x5_v1", "L", "R", 12),
        ("isa_transmissive_35mm_v1", "L", "R", 12),
        ("isa_transmissive_120_v1", "T", "B", 12),
        ("microform_35mm_positive_v1", "UL", "LR", 10),
        ("microform_35mm_negative_v1", "UL", "LR", 10),
    ],
)
def test_film_ladders_numbered_straight_through(target_id, first_block, second_block, steps):
    densities = [0.05 + 0.1 * i for i in range(steps)] * 2
    text = "\tD\n" + "".join(f"Patch{k + 1}\t{d:.2f}\n" for k, d in enumerate(densities))
    joined, _ = reconcile_sample_ids(parse_opendice(text), load_target(target_id))
    assert joined[f"{first_block}1"].raw["D"] == pytest.approx(0.05)
    assert joined[f"{second_block}{steps}"].raw["D"] == pytest.approx(densities[-1])


def test_read_reference_dispatches_on_content(tmp_path: Path):
    path = tmp_path / "Profile_DICE.txt"
    path.write_text(DLT_HEAD, encoding="utf-8")
    reference = read_reference(path)
    assert reference.source_path == path
    assert reference.metadata["DELTACHROME_SOURCE_FORMAT"] == "OpenDICE target profile"
    with pytest.raises(ValueError):
        read_reference(path, table=2)
