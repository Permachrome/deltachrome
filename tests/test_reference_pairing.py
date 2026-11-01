"""Reference formats, chart identification, and pairing a capture to a unit.

Reference measurements are per *physical unit*, and two charts of the same model
are indistinguishable in a photograph. Measuring against the wrong unit's file
produces numbers that look entirely reasonable and mean nothing -- so the chart's
own machine-readable identifier does the pairing, and a mismatch is surfaced rather
than assumed harmless.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from deltachrome.errors import CGATSParseError, DuplicateSampleError, MissingReferenceError
from deltachrome.imaging.codes import normalize_identifier, scan_codes, scan_regions
from deltachrome.imaging.loader import load_image
from deltachrome.reference import read_reference
from deltachrome.reference.cxf import parse_cxf
from deltachrome.reference.library import ReferenceLibrary, peek_identity
from fixtures import adobe_rgb_icc, write_tiff_with_profile

CXF = """<?xml version="1.0" encoding="UTF-8"?>
<CxF xmlns="http://colorexchangeformat.com/CxF3-core">
  <FileInformation>
    <Creator>Test Vendor</Creator>
    <Description>Test target, L* a* b* (light D50, viewing angle 2), M0</Description>
    <CreationDate>2026-01-05T10:00:00</CreationDate>
    <Tag Name="28178_TargetType" Value="ISO 12641-2 Candidate"/>
    <Tag Name="Serial" Value="T9990001"/>
  </FileInformation>
  <Resources>
    <ObjectCollection>
      <Object ObjectType="Target" Name="A1" Id="A1">
        <ColorValues>
          <ColorIEXYZ ColorSpecification="CS1"><X>3.201</X><Y>2.867</Y><Z>1.914</Z></ColorIEXYZ>
          <ColorCIELab ColorSpecification="CS1"><L>19.504</L><A>7.662</A><B>4.174</B></ColorCIELab>
          <ReflectanceSpectrum StartWL="380" ColorSpecification="CS1">0.0021 0.0066 0.0248</ReflectanceSpectrum>
        </ColorValues>
      </Object>
      <Object ObjectType="Target" Name="A2" Id="A2">
        <ColorValues>
          <ColorCIELab ColorSpecification="CS1"><L>50.0</L><A>1.0</A><B>-2.0</B></ColorCIELab>
        </ColorValues>
      </Object>
    </ObjectCollection>
  </Resources>
  <ColorSpecificationCollection>
    <ColorSpecification Id="CS1">
      <TristimulusSpec>
        <Illuminant>D50</Illuminant><Observer>2_Degree</Observer><Method>M0</Method>
      </TristimulusSpec>
      <MeasurementSpec>
        <MeasurementType>Colorimetric_Transmittance</MeasurementType>
        <Device><Manufacturer>Barbieri</Manufacturer><Model>Spectro 100xy</Model></Device>
      </MeasurementSpec>
    </ColorSpecification>
  </ColorSpecificationCollection>
</CxF>
"""


# -- CxF --------------------------------------------------------------------


def test_cxf_parses_colorimetry_and_spectra():
    result = parse_cxf(CXF)
    assert len(result) == 2
    assert result["A1"].lab == pytest.approx((19.504, 7.662, 4.174))
    assert result["A1"].xyz == pytest.approx((3.201, 2.867, 1.914))
    assert result["A1"].spectral == {380.0: 0.0021, 390.0: 0.0066, 400.0: 0.0248}


def test_cxf_conditions_map_onto_the_cgats_keywords_the_engine_reads():
    """Format has to stay an input concern; nothing downstream should care which
    kind of file a reference arrived in."""
    result = parse_cxf(CXF)
    assert result.illuminant == "D50"
    assert result.observer == "2"
    assert result.measurement_mode == "M0"
    assert result.instrument == "Barbieri Spectro 100xy"
    assert result.metadata["TARGET_SERIAL"] == "T9990001"


def test_cxf_transmittance_is_read_without_comment():
    """It used to be flagged on every transmittance file whichever guideline was
    chosen, which made it noise (2026-10-02 review). The mode is still read."""
    result = parse_cxf(CXF)
    assert not any("transmittance" in w for w in result.warnings)


def test_cxf_duplicate_names_are_refused():
    text = CXF.replace('Name="A2" Id="A2"', 'Name="A1" Id="A1dup"')
    with pytest.raises(DuplicateSampleError, match="A1"):
        parse_cxf(text)


def test_cxf_without_colorimetry_is_refused():
    text = CXF.replace("ColorCIELab", "Nonsense").replace("ColorIEXYZ", "Nonsense2")
    text = text.replace("ReflectanceSpectrum", "Nonsense3")
    with pytest.raises(CGATSParseError, match="none carry Lab, XYZ or spectral"):
        parse_cxf(text)


def test_non_cxf_xml_is_refused():
    with pytest.raises(CGATSParseError, match="expected <CxF>"):
        parse_cxf("<?xml version='1.0'?><NotCxF/>")


def test_read_reference_dispatches_on_content_not_extension(tmp_path):
    """Vendors ship CxF under .xml, and occasionally CGATS under .cxf."""
    misnamed = tmp_path / "reference.txt"
    misnamed.write_text(CXF, encoding="utf-8")
    assert len(read_reference(misnamed)) == 2

    cgats = tmp_path / "reference.cxf"
    cgats.write_text(
        "CGATS.17\nBEGIN_DATA_FORMAT\nSAMPLE_NAME LAB_L LAB_A LAB_B\nEND_DATA_FORMAT\n"
        "BEGIN_DATA\nA1 50 1 2\nEND_DATA\n",
        encoding="utf-8",
    )
    assert len(read_reference(cgats)) == 1


# -- identifiers ------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [("CCSG21", "CCSG21"), ("cc-sg-21", "CCSG21"), ("ccsg 21", "CCSG21"),
     ("T1910021", "T1910021"), ("t_1910021", "T1910021")],
)
def test_identifier_normalisation_survives_punctuation_and_case(text, expected):
    assert normalize_identifier(text) == expected


def test_peek_identity_reads_a_cxf_serial_without_a_full_parse(tmp_path):
    path = tmp_path / "unit.cxf"
    path.write_text(CXF, encoding="utf-8")
    entry = peek_identity(path)
    assert entry.format == "cxf"
    assert entry.serial == "T9990001"
    assert entry.target_type == "ISO 12641-2 Candidate"
    assert "T9990001" in entry.identifiers


def test_peek_identity_reads_a_cgats_serial_keyword(tmp_path):
    path = tmp_path / "chart.txt"
    path.write_text(
        'CGATS.17\nTARGET_SERIAL "CCSG21"\nDESCRIPTOR "unit 21"\n'
        "BEGIN_DATA_FORMAT\nSAMPLE_NAME LAB_L LAB_A LAB_B\nEND_DATA_FORMAT\n"
        "BEGIN_DATA\nA1 50 1 2\nEND_DATA\n",
        encoding="utf-8",
    )
    entry = peek_identity(path)
    assert entry.serial == "CCSG21"
    assert {"CCSG21", "CHART"} <= entry.identifiers


# -- the library ------------------------------------------------------------


@pytest.fixture
def library(tmp_path):
    (tmp_path / "T9990001.cxf").write_text(CXF, encoding="utf-8")
    (tmp_path / "unit_forty_three.txt").write_text(
        "CGATS.17\nDESCRIPTOR \"no serial here\"\n"
        "BEGIN_DATA_FORMAT\nSAMPLE_NAME LAB_L LAB_A LAB_B\nEND_DATA_FORMAT\n"
        "BEGIN_DATA\nA1 50 1 2\nEND_DATA\n",
        encoding="utf-8",
    )
    return ReferenceLibrary(tmp_path)


def test_library_indexes_by_embedded_serial_and_filename(library):
    assert library.find("T9990001")[0].path.name == "T9990001.cxf"
    assert library.find("unit-forty-three")[0].path.name == "unit_forty_three.txt"


def test_library_binding_gives_a_file_an_alias(library):
    assert library.find("CCSG43") == []
    library.bind("CCSG43", "unit_forty_three.txt")
    assert library.find("CCSG43")[0].path.name == "unit_forty_three.txt"

    manifest = json.loads((library.directory / "deltachrome-references.json").read_text())
    assert manifest["bindings"]["CCSG43"] == "unit_forty_three.txt"


def test_binding_a_missing_file_is_refused(library):
    with pytest.raises(MissingReferenceError, match="No such reference file"):
        library.bind("X", "not_here.txt")


def test_resolution_reports_when_nothing_matches(library):
    resolution = library.resolve(["CCSG99"])
    assert resolution.entry is None
    assert "no reference file" in resolution.reason
    assert resolution.candidates  # the operator is shown what is available


def test_resolution_refuses_to_choose_between_two_matches(library):
    """Guessing here would defeat the whole point of pairing by identifier."""
    library.bind("SHARED", "unit_forty_three.txt")
    resolution = library.resolve(["SHARED", "T9990001"])
    assert resolution.ambiguous is True
    assert resolution.entry is None
    assert "Refusing to choose" in resolution.reason


def test_resolution_with_no_codes_says_so(library):
    assert "No identification code" in library.resolve([]).reason


def test_a_library_pointed_at_a_file_is_refused(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(MissingReferenceError, match="not a directory"):
        ReferenceLibrary(path)


# -- reading codes off a capture -------------------------------------------


def _capture_with_qr(tmp_path, text, *, size=(1400, 1000), at=(1100, 760), module=6):
    qr = cv2.QRCodeEncoder.create().encode(text)
    qr = cv2.resize(qr, None, fx=module, fy=module, interpolation=cv2.INTER_NEAREST)
    canvas = np.full((size[1], size[0]), 140, np.uint8)
    x, y = at
    h, w = qr.shape
    canvas[y - 10 : y + h + 10, x - 10 : x + w + 10] = 255
    canvas[y : y + h, x : x + w] = qr
    rgb = np.repeat(canvas[:, :, None], 3, axis=2)
    return write_tiff_with_profile(tmp_path / "cap.tif", rgb, adobe_rgb_icc())


def test_a_code_is_read_off_a_capture(tmp_path):
    path = _capture_with_qr(tmp_path, "CCSG21")
    scan = scan_codes(load_image(path))
    assert "CCSG21" in scan.texts
    assert scan.codes[0].symbology == "QRCode"


def test_scanning_reports_which_decoders_were_available(tmp_path):
    path = _capture_with_qr(tmp_path, "CCSG21")
    scan = scan_codes(load_image(path))
    assert "opencv" in scan.decoders_used


def test_a_capture_with_no_code_reports_nothing_rather_than_failing(tmp_path):
    rgb = np.full((400, 600, 3), 128, np.uint8)
    path = write_tiff_with_profile(tmp_path / "plain.tif", rgb, adobe_rgb_icc())
    scan = scan_codes(load_image(path), tiled=False)
    assert scan.codes == []
    assert "No identification code decoded" in scan.describe()


def test_region_scanning_targets_a_known_area(tmp_path):
    path = _capture_with_qr(tmp_path, "T9990001")
    image = load_image(path)
    assert "T9990001" in scan_regions(image, [(1050, 700, 1400, 1000)]).texts
    assert scan_regions(image, [(0, 0, 300, 300)]).texts == []


def test_the_end_to_end_pairing_picks_the_right_unit(tmp_path, library):
    path = _capture_with_qr(tmp_path, "T9990001")
    scan = scan_codes(load_image(path))
    resolution = library.resolve(scan.texts)
    assert resolution.path.name == "T9990001.cxf"
    assert "matched" in resolution.reason


# -- the same file, spelled the other way --------------------------------------

#: The identical document with every ``Color`` spelled ``Colour``. This is not a
#: hypothetical: LaserSoft ship it, and the file above was written by Americanising
#: theirs by hand -- which is exactly why the reader looked correct for as long as
#: nobody read a real one.
CXF_BRITISH = (
    CXF.replace("ColorValues", "ColourValues")
    .replace("ColorIEXYZ", "ColourIEXYZ")
    .replace("ColorCIELab", "ColourCIELab")
    .replace("ColorSpecification", "ColourSpecification")
)


def test_a_british_spelled_cxf_reads_the_same_as_an_american_one():
    """ISO 17972-1 spells these ``Color``; a real vendor file spells them ``Colour``.

    Read strictly, none of those elements exists -- so a file carrying Lab *and* XYZ
    *and* spectra for 864 patches looked like a file carrying only spectra,
    ``ReflectanceSpectrum`` being the one element spelled the same either way. It
    went unnoticed because nothing failed: the spectra were integrated instead and
    agreed with the file's stated Lab to 0.066 Delta E2000 mean.
    """
    american = parse_cxf(CXF)
    british = parse_cxf(CXF_BRITISH)

    assert list(british.samples) == list(american.samples)
    assert british["A1"].lab == american["A1"].lab == (19.504, 7.662, 4.174)
    assert british["A1"].xyz == american["A1"].xyz == (3.201, 2.867, 1.914)
    assert british["A1"].spectral == american["A1"].spectral
    # The measurement conditions live under ColourSpecification too, so they were
    # invisible as well -- the report was showing no instrument and no illuminant
    # for a file that names both.
    assert british.metadata == american.metadata
    assert british.illuminant == "D50"
    assert british.metadata["INSTRUMENTATION"] == "Barbieri Spectro 100xy"


def test_a_files_own_lab_is_used_rather_than_its_spectra():
    """Where a file states Lab, that is the answer.

    Integrating its spectra instead would be a reader quietly disagreeing with a
    number the file asserts -- a small disagreement, invisible, and impossible for
    an operator to attribute. The spectra stay available for anything that wants
    them.
    """
    from deltachrome.reference.spectral import fill_from_spectra

    british = parse_cxf(CXF_BRITISH)
    assert fill_from_spectra(british) == 0
    assert british["A1"].lab == (19.504, 7.662, 4.174)
    assert not any("spectra and no colorimetry" in w for w in british.warnings)


def test_an_element_name_is_reported_as_the_file_spells_it():
    """The fold is for comparison, not for what the reader says back.

    Telling somebody their root element is ``<Colour...>`` when their file says
    something else -- or the reverse -- sends them looking for text that is not in
    their file.
    """
    from deltachrome.errors import CGATSParseError

    with pytest.raises(CGATSParseError, match="<ColourThing>"):
        parse_cxf('<ColourThing xmlns="http://colorexchangeformat.com/CxF3-core"/>')
