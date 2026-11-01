"""LaserSoft's barcode encodes its printed serial rather than spelling it.

Measured bar by bar on all four LaserSoft charts here -- two ISO 12641-2 units, an
IT8.7/2 and an IT8.7/1 -- every symbol decodes cleanly as Standard 2 of 5, and every
one is the printed serial with the product letter turned into a digit, the decade
dropped and twenty added to the month. It was taken for a misread for as long as
only the ISO 12641-2 was looked at, where the one changed digit is the month's tens.

What is pinned here is the rule on those four, and the two limits on where it may
apply: forward from a bank file's serial only, and only to a code read as 2 of 5.
"""

from __future__ import annotations

import pytest

from deltachrome.reference.library import (
    ReferenceLibrary,
    lasersoft_barcode,
    lasersoft_serials,
)

#: Printed serial and what the bars were measured to carry, one per chart.
MEASURED = (
    ("T1910021", "1930021"),  # ISO 12641-2, 4x5, 2019:10
    ("T1710030", "1730030"),  # ISO 12641-2, 120, 2017:10
    ("R100213", "002213"),    # IT8.7/2, 2010:02 -- OpenDICE's sample
    ("E180106", "282106"),    # IT8.7/1, 2018:01 -- OpenDICE's sample
)


@pytest.mark.parametrize("printed, bars", MEASURED)
def test_the_rule_reproduces_every_measured_barcode(printed, bars):
    assert lasersoft_barcode(printed) == bars


@pytest.mark.parametrize("printed, bars", MEASURED)
def test_going_back_names_the_serial_among_its_decades(printed, bars):
    """The barcode keeps only the year's last digit, so it names two serials."""
    assert printed in lasersoft_serials(bars)
    assert len(lasersoft_serials(bars)) == 2


@pytest.mark.parametrize("text", ["CCSG21", "T1913021", "X1910021", "T19", "M000390"])
def test_anything_not_shaped_like_a_lasersoft_serial_implies_nothing(text):
    """Including a month of 13: the rule is a statement about dates."""
    assert lasersoft_barcode(text) is None


def _bank(tmp_path, serial: str, filename: str = "measurement.txt") -> ReferenceLibrary:
    directory = tmp_path / "bank"
    directory.mkdir()
    (directory / filename).write_text(
        "CGATS.17\n"
        f'TARGET_SERIAL\t"{serial}"\n'
        "NUMBER_OF_FIELDS\t4\nBEGIN_DATA_FORMAT\nSAMPLE_NAME\tLAB_L\tLAB_A\tLAB_B\n"
        "END_DATA_FORMAT\nNUMBER_OF_SETS\t1\nBEGIN_DATA\nA1\t50\t0\t0\nEND_DATA\n",
        encoding="utf-8",
    )
    return ReferenceLibrary(directory)


def test_a_2of5_barcode_pairs_with_the_file_stating_its_serial(tmp_path):
    bank = _bank(tmp_path, "T1710030")
    resolution = bank.resolve(["1730030"], standard_2of5=["1730030"])

    assert resolution.path is not None
    assert "T1710030" in resolution.reason


def test_the_encoding_is_never_applied_to_a_code_of_another_symbology(tmp_path):
    """A QR or Code 128 reading 1730030 says what it says."""
    bank = _bank(tmp_path, "T1710030")

    assert bank.resolve(["1730030"]).path is None


def test_a_derived_barcode_is_not_listed_as_something_the_file_states(tmp_path):
    """A bank listing shows identifiers as facts the file carries. The barcode is
    worked out, and appears nowhere a person reads what the file says."""
    entry = _bank(tmp_path, "T1710030").entries[0]

    assert "1730030" in entry.encoded
    assert "1730030" not in entry.identifiers


def test_a_file_named_after_its_serial_pairs_too(tmp_path):
    bank = _bank(tmp_path, "", filename="T1910021.txt")

    assert bank.resolve(["1930021"], standard_2of5=["1930021"]).path is not None


def test_an_unpaired_barcode_says_which_serial_to_look_for(tmp_path):
    bank = _bank(tmp_path, "T1910021")
    reason = bank.resolve(["1730030"], standard_2of5=["1730030"]).reason

    assert "T1710030" in reason


def test_a_fragment_of_a_whole_read_is_dropped():
    """The LaserSoft 120 reads 173003 from one window and 1730030 from the next.
    Only the whole read is a fact about the chart; the fragment, listed beside it,
    was shown as something the chart prints and bound into the bank's manifest."""
    from deltachrome.imaging.codes import DetectedCode, _without_fragments

    codes = [DetectedCode("173003", "Code25", "deltachrome"),
             DetectedCode("1730030", "Code25", "deltachrome")]

    assert [c.text for c in _without_fragments(codes)] == ["1730030"]


def test_only_2of5_reads_are_folded_together():
    """A QR whose text happens to sit inside a barcode's is a different code."""
    from deltachrome.imaging.codes import DetectedCode, _without_fragments

    codes = [DetectedCode("CCSG21", "QRCode", "zxing-cpp"),
             DetectedCode("173003", "QRCode", "zxing-cpp"),
             DetectedCode("1730030", "Code25", "deltachrome")]

    assert len(_without_fragments(codes)) == 3
