"""CGATS parsing, with emphasis on the failure modes that corrupt results silently."""

from __future__ import annotations

import pytest

from deltachrome.errors import CGATSParseError, DuplicateSampleError
from deltachrome.reference.cgats import parse_cgats

MINIMAL = """CGATS.17
ORIGINATOR "i1Profiler"
DESCRIPTOR "ColorChecker SG M1"
CREATED "2026-01-14"
INSTRUMENTATION "i1Pro 3 PLUS"
ILLUMINATION_NAME "D50"
OBSERVER_ANGLE "2"
DEVCALSTD "XRDI"
NUMBER_OF_FIELDS 4
BEGIN_DATA_FORMAT
SAMPLE_NAME LAB_L LAB_A LAB_B
END_DATA_FORMAT
NUMBER_OF_SETS 3
BEGIN_DATA
A1 96.12 -0.91 3.18
B1 45.20 12.30 -8.40
K8 30.05 5.10 -2.20
END_DATA
"""


def test_parses_samples_and_metadata():
    result = parse_cgats(MINIMAL)
    assert len(result) == 3
    assert result["A1"].lab == pytest.approx((96.12, -0.91, 3.18))
    assert result.illuminant == "D50"
    assert result.observer == "2"
    assert result.instrument == "i1Pro 3 PLUS"
    assert "CREATED" in result.provenance()


def test_measurement_mode_is_extracted():
    text = MINIMAL.replace(
        'DESCRIPTOR "ColorChecker SG M1"',
        'MEASUREMENT_SOURCE "Illumination=M1 ObserverAngle=2"',
    )
    assert parse_cgats(text).measurement_mode == "M1"


def test_field_order_is_read_from_the_file_not_assumed():
    """Column order must come from DATA_FORMAT. A parser with hardcoded columns
    would read this file's b* as its L* and report plausible nonsense."""
    text = MINIMAL.replace(
        "SAMPLE_NAME LAB_L LAB_A LAB_B", "LAB_B LAB_A SAMPLE_NAME LAB_L"
    ).replace("A1 96.12 -0.91 3.18", "3.18 -0.91 A1 96.12").replace(
        "B1 45.20 12.30 -8.40", "-8.40 12.30 B1 45.20"
    ).replace("K8 30.05 5.10 -2.20", "-2.20 5.10 K8 30.05")

    result = parse_cgats(text)
    assert result["A1"].lab == pytest.approx((96.12, -0.91, 3.18))


def test_duplicate_sample_names_are_refused_not_overwritten():
    """The real reference file for this project's own SG listed K8 twice, the
    second entry being a mislabelled K9. Last-write-wins would have silently
    poisoned every metric involving that patch."""
    text = MINIMAL.replace("NUMBER_OF_SETS 3", "NUMBER_OF_SETS 4").replace(
        "\nEND_DATA\n", "\nK8 41.88 -7.02 14.55\nEND_DATA\n"
    )
    with pytest.raises(DuplicateSampleError) as excinfo:
        parse_cgats(text)

    message = str(excinfo.value)
    assert "K8" in message
    # The message must name the conflicting fields, so the operator can work out
    # which entry is the mislabelled one.
    assert "LAB_L" in message
    assert "30.05" in message and "41.88" in message


def test_identical_duplicates_are_still_refused_but_described_as_identical():
    text = MINIMAL.replace("NUMBER_OF_SETS 3", "NUMBER_OF_SETS 4").replace(
        "\nEND_DATA\n", "\nK8 30.05 5.10 -2.20\nEND_DATA\n"
    )
    with pytest.raises(DuplicateSampleError) as excinfo:
        parse_cgats(text)
    assert "identical" in str(excinfo.value)


def test_xyz_only_file_is_accepted():
    text = MINIMAL.replace(
        "SAMPLE_NAME LAB_L LAB_A LAB_B", "SAMPLE_NAME XYZ_X XYZ_Y XYZ_Z"
    )
    result = parse_cgats(text)
    assert result["A1"].xyz is not None
    assert result["A1"].lab is None


def test_spectral_fields_are_captured():
    text = """CGATS.17
NUMBER_OF_FIELDS 4
BEGIN_DATA_FORMAT
SAMPLE_NAME SPECTRAL_NM_380 SPECTRAL_NM_390 SPECTRAL_NM_400
END_DATA_FORMAT
BEGIN_DATA
A1 0.0512 0.0634 0.0781
END_DATA
"""
    result = parse_cgats(text)
    assert result["A1"].spectral == {380.0: 0.0512, 390.0: 0.0634, 400.0: 0.0781}


def test_quoted_values_with_spaces_survive():
    result = parse_cgats(MINIMAL)
    assert result.metadata["DESCRIPTOR"] == "ColorChecker SG M1"


def test_comments_are_stripped():
    text = MINIMAL.replace("A1 96.12 -0.91 3.18", "A1 96.12 -0.91 3.18 # near-white")
    assert parse_cgats(text)["A1"].lab == pytest.approx((96.12, -0.91, 3.18))


def test_row_with_wrong_column_count_is_an_error():
    text = MINIMAL.replace("B1 45.20 12.30 -8.40", "B1 45.20 12.30")
    with pytest.raises(CGATSParseError, match="values but DATA_FORMAT declares"):
        parse_cgats(text)


def test_data_without_a_format_block_is_an_error():
    text = "CGATS.17\nBEGIN_DATA\nA1 1 2 3\nEND_DATA\n"
    with pytest.raises(CGATSParseError, match="without a preceding"):
        parse_cgats(text)


def test_file_with_no_colorimetry_is_an_error():
    text = """CGATS.17
BEGIN_DATA_FORMAT
SAMPLE_NAME RGB_R RGB_G RGB_B
END_DATA_FORMAT
BEGIN_DATA
A1 255 255 255
END_DATA
"""
    with pytest.raises(CGATSParseError, match="no colorimetric data"):
        parse_cgats(text)


def test_set_count_mismatch_warns_rather_than_failing():
    text = MINIMAL.replace("NUMBER_OF_SETS 3", "NUMBER_OF_SETS 140")
    result = parse_cgats(text)
    assert "The file says it holds 140 patches; 3 were found." in result.warnings
    assert len(result) == 3
