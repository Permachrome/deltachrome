"""Multi-table CGATS files.

CGATS.17 allows more than one data table per file. Almost every reference file has
one, so the ordinary case stays silent. The case that matters is a file holding the
*same chart measured under different conditions* -- M0 and M1 in one export --
where the choice of table is a choice of measurement condition and every Delta E in
the report is against it.

So: default to M1, the D50 condition everything here is judged under, else to the
first; say so in one line when the default chose between conditions; and let the
operator name the one they mean.
"""

from __future__ import annotations

import pytest

from deltachrome.errors import CGATSParseError
from deltachrome.reference import read_reference
from deltachrome.reference.cgats import parse_cgats

HEADER = 'CGATS.17\nORIGINATOR\t"Deltachrome test"\nNUMBER_OF_TABLES\t{n}\n'


def table(rows, **keywords):
    lines = [f'{k}\t"{v}"' for k, v in keywords.items()]
    lines += [
        "NUMBER_OF_FIELDS\t4",
        "BEGIN_DATA_FORMAT",
        "SAMPLE_NAME\tLAB_L\tLAB_A\tLAB_B",
        "END_DATA_FORMAT",
        f"NUMBER_OF_SETS\t{len(rows)}",
        "BEGIN_DATA",
        *[f"{name}\t{L}\t0.0\t0.0" for name, L in rows],
        "END_DATA",
    ]
    return "\n".join(lines) + "\n"


def two_modes():
    return (
        HEADER.format(n=2)
        + table([("A1", 50.0), ("A2", 60.0)],
                MEASUREMENT_SOURCE="Illumination=M0", DESCRIPTOR="no filter")
        + table([("A1", 51.5), ("A2", 61.5)],
                MEASUREMENT_SOURCE="Illumination=M1", DESCRIPTOR="D50")
    )


# -- the ordinary case stays silent ----------------------------------------


def test_a_single_table_file_says_nothing_about_tables():
    text = HEADER.format(n=1) + table([("A1", 50.0), ("A2", 60.0)])
    result = parse_cgats(text)
    assert result.warnings == []
    assert result.table_index == 1
    assert len(result.tables_available) == 1


def test_a_header_claiming_more_tables_than_the_file_holds_is_reported():
    """The header is metadata and can be wrong; the data found is what was used."""
    text = HEADER.format(n=3) + table([("A1", 50.0)])
    result = parse_cgats(text)
    assert any("declares NUMBER_OF_TABLES 3" in w for w in result.warnings)
    assert len(result) == 1


# -- and the choice is announced -------------------------------------------


def test_the_default_is_m1_and_it_is_said_out_loud():
    """M1 is the D50 condition every comparison here is made under; an export of
    M0 and M1 lists M0 first, so "the first table" chose the wrong one."""
    result = parse_cgats(two_modes())
    assert result.table_index == 2
    assert result["A1"].lab[0] == pytest.approx(51.5)
    assert result.warnings == [
        "Reference file holds several measurements (M0, M1); M1 used."
    ]


def test_without_an_m1_table_the_default_is_the_first():
    text = (
        HEADER.format(n=2)
        + table([("A1", 50.0)], MEASUREMENT_SOURCE="Illumination=M0")
        + table([("A1", 52.0)], MEASUREMENT_SOURCE="Illumination=M2")
    )
    result = parse_cgats(text)
    assert result.table_index == 1
    assert result.warnings == [
        "Reference file holds several measurements (M0, M2); M0 used."
    ]


def test_tables_of_the_same_condition_say_nothing():
    """A formatting detail: the numbers are the same measurement either way."""
    text = (
        HEADER.format(n=2)
        + table([("A1", 50.0)], MEASUREMENT_SOURCE="Illumination=M1")
        + table([("A1", 50.0)], MEASUREMENT_SOURCE="Illumination=M1")
    )
    result = parse_cgats(text)
    assert result.table_index == 1
    assert result.warnings == []


def test_every_table_is_described_so_a_caller_can_offer_the_choice():
    """A library cannot prompt. What it can do is hand back what the alternatives
    are, so a CLI flag or a future UI does not have to re-parse to discover them."""
    available = parse_cgats(two_modes()).tables_available
    assert [t["table"] for t in available] == [1, 2]
    assert [t["measurement_mode"] for t in available] == ["M0", "M1"]
    assert [t["descriptor"] for t in available] == ["no filter", "D50"]
    assert all(t["rows"] == 2 for t in available)


# -- naming one ------------------------------------------------------------


def test_a_table_can_be_chosen_by_number_and_a_named_table_is_not_announced():
    result = parse_cgats(two_modes(), table=1)
    assert result.table_index == 1
    assert result["A1"].lab[0] == pytest.approx(50.0)
    assert result.warnings == []


def test_a_table_can_be_chosen_by_measurement_condition():
    result = parse_cgats(two_modes(), table="M1")
    assert result.table_index == 2
    assert result.measurement_mode == "M1"


def test_a_request_that_cannot_be_honoured_is_refused_not_defaulted():
    """Falling back to the first would give the operator a different measurement
    than the one they named, with the report showing the number they asked for."""
    with pytest.raises(CGATSParseError, match="file holds 2"):
        parse_cgats(two_modes(), table=5)
    with pytest.raises(CGATSParseError, match="measured under M2"):
        parse_cgats(two_modes(), table="M2")


def test_an_ambiguous_condition_request_is_refused():
    text = (
        HEADER.format(n=2)
        + table([("A1", 50.0)], MEASUREMENT_SOURCE="Illumination=M1")
        + table([("A1", 51.0)], MEASUREMENT_SOURCE="Illumination=M1")
    )
    with pytest.raises(CGATSParseError, match="name one by number"):
        parse_cgats(text, table="M1")


# -- provenance belongs to its own table -----------------------------------


def test_each_table_carries_its_own_header_and_the_file_header():
    """Accumulating keywords across the whole file gave the second table the first
    one's provenance as well as its own -- worst exactly where it matters most, when
    the header is what distinguishes the tables."""
    first = parse_cgats(two_modes(), table=1)
    second = parse_cgats(two_modes(), table=2)

    assert first.measurement_mode == "M0"
    assert second.measurement_mode == "M1"
    assert first.metadata["DESCRIPTOR"] == "no filter"
    assert second.metadata["DESCRIPTOR"] == "D50"
    # The file's own header reaches both.
    assert first.metadata["ORIGINATOR"] == second.metadata["ORIGINATOR"] == "Deltachrome test"


def test_repeated_keywords_still_join_within_one_table():
    """i1Profiler writes two WEIGHTING_FUNCTION lines, and keeping only the last
    would discard half the measurement's provenance."""
    text = HEADER.format(n=1) + (
        'WEIGHTING_FUNCTION\t"ILLUMINANT, D50"\n'
        'WEIGHTING_FUNCTION\t"OBSERVER, 2 degree"\n'
    ) + table([("A1", 50.0)])
    joined = parse_cgats(text).metadata["WEIGHTING_FUNCTION"]
    assert "ILLUMINANT" in joined and "OBSERVER" in joined


# -- and the other format --------------------------------------------------


def test_asking_a_cxf_file_for_a_table_is_an_error(tmp_path):
    """CxF holds one measurement set per file. Silently ignoring the request would
    let an operator believe they had selected something."""
    path = tmp_path / "ref.cxf"
    path.write_text("<?xml version='1.0'?><CxF3></CxF3>", encoding="utf-8")
    with pytest.raises(ValueError, match="Tables are a CGATS concept"):
        read_reference(path, table=2)


def test_the_run_record_says_which_table_was_used(tmp_path):
    from deltachrome.pipeline import analyze
    from fixtures import SyntheticChart

    chart = SyntheticChart()
    image, corners = chart.render_capture(tmp_path / "capture.tif")
    reference = chart.write_reference(tmp_path / "ref.txt")
    result = analyze(
        image, target="colorchecker_sg_v1", reference_path=reference,
        profile="fadgi_3star_prints_photographs_2023", corners=corners,
    )
    assert result.run.reference["table_index"] == 1
    assert len(result.run.reference["tables_available"]) == 1
