"""LUT-based ICC profiles (A2B0/A2B1/A2B2).

A device profile is LUT-based precisely because its response cannot be factored
into per-channel curves plus a linear matrix. These tests check that Deltachrome
evaluates the three tag encodings in the wild -- lut8Type, lut16Type and
lutAtoBType -- at float precision, and that the two encoding traps in the format
are handled the right way round.

The cross-validation trick: the fixture puts the tone curve in the input curves and
a *linear* function in the CLUT. Multilinear interpolation reproduces a linear
function exactly, so the LUT profile is analytically identical to the matrix
profile and any disagreement is encoding quantisation, not interpolation error.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from deltachrome.errors import UnsupportedICCProfileError
from deltachrome.imaging.color import xyz_to_lab
from deltachrome.imaging.icc import ICC_PCS_D50, parse_icc
from fixtures import (
    ADOBE_RGB_GAMMA,
    adobe_rgb_icc,
    adobe_rgb_matrix,
    build_icc,
    constant_lut_icc,
    encode_lab_legacy16,
    encode_lab_v4,
    linear_clut,
    lut_icc_from_matrix,
    mab_tag,
    mft_tag,
)

#: u1Fixed15 quantisation: 1 LSB is 1.99997/65535.
XYZ_QUANTISATION = 65535.0 / 32768.0 / 65535.0

SAMPLES = np.random.default_rng(20260818).uniform(0.0, 1.0, (500, 3))


# -- the pipeline evaluates correctly --------------------------------------


@pytest.mark.parametrize("tag_type", ["mft2", "mAB"])
def test_lut_profile_matches_the_analytic_matrix_profile(tag_type):
    lut = parse_icc(lut_icc_from_matrix(tag_type=tag_type))
    matrix = parse_icc(adobe_rgb_icc())

    assert lut.transform_kind == "lut"
    assert matrix.transform_kind == "matrix_trc"

    difference = np.abs(lut.to_xyz(SAMPLES) - matrix.to_xyz(SAMPLES)).max()
    assert difference < 2 * XYZ_QUANTISATION, (
        "the LUT path disagrees with the analytic path by more than the encoding's "
        "own quantisation step"
    )


@pytest.mark.parametrize("tag_type", ["mft2", "mAB"])
def test_lut_error_stays_far_below_a_meaningful_delta_e(tag_type):
    """The whole reason for evaluating LUTs in float rather than through an 8-bit
    CMS path: the error has to be negligible against a 3.5 dE budget."""
    lut = parse_icc(lut_icc_from_matrix(tag_type=tag_type))
    matrix = parse_icc(adobe_rgb_icc())

    from deltachrome.imaging.color import delta_e

    lab_lut = xyz_to_lab(lut.to_xyz(SAMPLES))
    lab_matrix = xyz_to_lab(matrix.to_xyz(SAMPLES))
    assert float(np.max(delta_e(lab_lut, lab_matrix))) < 0.05


def test_a_neutral_input_reads_only_the_cells_neutral_diagonal():
    """Tetrahedral, as Argyll and lcms read an RGB table.

    Every tetrahedron of a cell holds the cell's black and white corners, and an
    input with equal coordinates lies on the edge between them -- so it reads
    those two corners and nothing else, however coloured the other six are.
    Trilinear weighs all eight and reads a tint into a grey.
    """
    from deltachrome.imaging.icc import CLUT

    table = np.zeros((2, 2, 2, 3))
    table[1, 1, 1] = [1.0, 1.0, 1.0]
    # Every off-diagonal corner wildly coloured, all the same way.
    for corner in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]:
        table[corner] = [0.9, 0.0, 0.0]
    clut = CLUT(grid=(2, 2, 2), table=table)
    for t in (0.05, 0.3, 0.7):
        assert clut(np.array([t, t, t])) == pytest.approx([t, t, t], abs=1e-12)
    # And off the diagonal it is the tetrahedron the point is in: r > g > b walks
    # black -> red corner -> red+green corner -> white.
    point = np.array([0.6, 0.3, 0.1])
    expected = (0.4 * table[0, 0, 0] + 0.3 * table[1, 0, 0]
                + 0.2 * table[1, 1, 0] + 0.1 * table[1, 1, 1])
    assert clut(point) == pytest.approx(expected, abs=1e-12)


def test_clut_interpolation_is_exact_for_a_linear_table():
    """Interpolation reproduces a linear function exactly. If this drifts, the
    interpolator is wrong, not merely imprecise."""
    from deltachrome.imaging.icc import CLUT

    grid = 5
    axis = np.linspace(0.0, 1.0, grid)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    table = np.stack([r, 0.25 * r + 0.5 * g + 0.25 * b, b], axis=-1)
    clut = CLUT(grid=(grid, grid, grid), table=table)

    points = np.random.default_rng(7).uniform(0, 1, (200, 3))
    expected = np.stack(
        [points[:, 0], 0.25 * points[:, 0] + 0.5 * points[:, 1] + 0.25 * points[:, 2], points[:, 2]],
        axis=-1,
    )
    assert clut(points) == pytest.approx(expected, abs=1e-12)


def test_clut_hits_stored_values_exactly_at_grid_points():
    from deltachrome.imaging.icc import CLUT

    rng = np.random.default_rng(11)
    table = rng.uniform(0, 1, (4, 4, 4, 3))
    clut = CLUT(grid=(4, 4, 4), table=table)
    for i, j, k in [(0, 0, 0), (3, 3, 3), (1, 2, 3), (2, 0, 1)]:
        point = np.array([i, j, k], dtype=float) / 3.0
        assert clut(point) == pytest.approx(table[i, j, k], abs=1e-12)


def test_pipeline_shape_is_preserved():
    lut = parse_icc(lut_icc_from_matrix())
    assert lut.to_xyz(np.zeros(3)).shape == (3,)
    assert lut.to_xyz(np.zeros((5, 3))).shape == (5, 3)
    assert lut.to_xyz(np.zeros((4, 6, 3))).shape == (4, 6, 3)


# -- the PCS Lab encoding trap ---------------------------------------------

LAB_TARGET = np.array([96.0, -0.9, 3.2])


def decode_constant(tag_type: str, encoded: np.ndarray) -> np.ndarray:
    profile = parse_icc(constant_lut_icc(encoded, tag_type=tag_type))
    xyz = profile.to_xyz(np.array([0.5, 0.5, 0.5]))
    return xyz_to_lab(xyz, white_point=ICC_PCS_D50)


def test_lut16_uses_the_legacy_lab_encoding():
    """lut16Type encodes L* against 0xFF00."""
    assert decode_constant("mft2", encode_lab_legacy16(LAB_TARGET)) == pytest.approx(
        LAB_TARGET, abs=0.01
    )


def test_lut_a_to_b_uses_the_v4_lab_encoding():
    """lutAtoBType encodes L* against 0xFFFF."""
    assert decode_constant("mAB", encode_lab_v4(LAB_TARGET)) == pytest.approx(
        LAB_TARGET, abs=0.01
    )


def test_using_one_lab_decoder_for_both_would_be_a_real_error():
    """The two encodings differ by a factor of 1.0039 on L*. At the highlight end
    of a tonal ramp that is ~0.38 L*, enough on its own to flip a FADGI tone
    response verdict -- so this is a correctness trap, not a rounding detail."""
    mismatched = decode_constant("mft2", encode_lab_v4(LAB_TARGET))
    assert mismatched[0] - LAB_TARGET[0] == pytest.approx(0.375, abs=0.02)


def test_lut8_lab_round_trips_within_its_own_quantisation():
    """8-bit Lab has a 0.39 L* step; the value must land within that, not further."""
    decoded = decode_constant("mft1", encode_lab_v4(LAB_TARGET))
    assert decoded[0] == pytest.approx(LAB_TARGET[0], abs=0.4)


# -- which transform gets used ---------------------------------------------


def test_a_device_profile_prefers_its_lut_over_its_matrix_tags():
    """A scanner profile's matrix/TRC tags are a fallback for simple CMMs; the LUT
    is why the profile exists."""
    profile = parse_icc(
        lut_icc_from_matrix(profile_class=b"scnr", include_matrix_trc=True)
    )
    assert profile.transform_kind == "lut"
    assert "matrix/TRC" in profile.available_transforms


def test_a_display_profile_prefers_its_matrix_tags_over_a_lut():
    """For a working space the matrix is the exact analytic definition; a LUT there
    only re-samples it and would add interpolation error."""
    profile = parse_icc(
        lut_icc_from_matrix(profile_class=b"mntr", include_matrix_trc=True)
    )
    assert profile.transform_kind == "matrix_trc"


def test_the_choice_can_be_overridden_in_both_directions():
    blob = lut_icc_from_matrix(profile_class=b"scnr", include_matrix_trc=True)
    assert parse_icc(blob, prefer="matrix_trc").transform_kind == "matrix_trc"
    assert parse_icc(blob, prefer="lut").transform_kind == "lut"


def test_overriding_to_a_transform_the_profile_lacks_is_an_error():
    blob = lut_icc_from_matrix(include_matrix_trc=False)
    with pytest.raises(UnsupportedICCProfileError, match="no complete"):
        parse_icc(blob, prefer="matrix_trc")
    with pytest.raises(UnsupportedICCProfileError, match="no A2B tag"):
        parse_icc(adobe_rgb_icc(), prefer="lut")


def test_colorimetric_table_is_preferred_over_the_perceptual_one():
    """A2B0 is perceptual and carries gamut mapping. Measuring through it would
    report the profile's rendering intent as if it were the device's response."""
    matrix = adobe_rgb_matrix()
    perceptual = mft_tag(linear_clut(matrix * 0.5, grid=5))
    colorimetric = mft_tag(linear_clut(matrix, grid=9), input_curves=[
        np.linspace(0, 1, 1024) ** ADOBE_RGB_GAMMA for _ in range(3)
    ])
    blob = build_icc(
        [(b"A2B0", perceptual), (b"A2B1", colorimetric)], profile_class=b"scnr"
    )

    profile = parse_icc(blob)
    assert profile.pipeline.tag == "A2B1"
    reference = parse_icc(adobe_rgb_icc())
    assert np.abs(profile.to_xyz(SAMPLES) - reference.to_xyz(SAMPLES)).max() < 2 * XYZ_QUANTISATION


def test_falling_back_to_the_perceptual_table_is_reported():
    blob = lut_icc_from_matrix(tag=b"A2B0")
    profile = parse_icc(blob)
    assert profile.pipeline.tag == "A2B0"
    assert any("perceptual" in note for note in profile.notes)


# -- refusals that remain ---------------------------------------------------


def test_unknown_lut_tag_type_is_refused():
    blob = build_icc([(b"A2B1", b"mBA " + b"\x00" * 60)], profile_class=b"scnr")
    with pytest.raises(UnsupportedICCProfileError, match="unsupported LUT tag type"):
        parse_icc(blob)


def test_lut8_with_an_xyz_pcs_is_refused():
    """ICC.1:2010 defines no 8-bit PCSXYZ encoding, and it would be far too coarse
    to measure through even if it did."""
    blob = build_icc(
        [(b"A2B1", mft_tag(linear_clut(adobe_rgb_matrix(), grid=5), bits=8))],
        profile_class=b"scnr",
        pcs=b"XYZ ",
    )
    with pytest.raises(UnsupportedICCProfileError, match="no defined encoding"):
        parse_icc(blob)


def test_a_non_identity_matrix_on_device_input_is_refused():
    """That matrix applies only to PCSXYZ input. Applying it to device RGB anyway
    would silently distort every value."""
    payload = mft_tag(
        linear_clut(adobe_rgb_matrix(), grid=5), matrix=np.diag([1.0, 0.5, 1.0])
    )
    blob = build_icc([(b"A2B1", payload)], profile_class=b"scnr")
    with pytest.raises(UnsupportedICCProfileError, match="not the identity"):
        parse_icc(blob)


def test_a_lut_with_no_b_curves_is_refused():
    clut = linear_clut(adobe_rgb_matrix(), grid=5)
    payload = bytearray(mab_tag(clut))
    struct.pack_into(">I", payload, 12, 0)  # clear the B-curve offset
    blob = build_icc([(b"A2B1", bytes(payload))], profile_class=b"scnr")
    with pytest.raises(UnsupportedICCProfileError, match="B curves are required"):
        parse_icc(blob)


def test_a_profile_with_neither_structure_is_refused():
    blob = build_icc([], profile_class=b"mntr")
    with pytest.raises(UnsupportedICCProfileError, match="neither a complete matrix/TRC"):
        parse_icc(blob)


def test_summary_describes_the_lut_pipeline():
    profile = parse_icc(lut_icc_from_matrix())
    text = profile.summary()
    assert "transform: lut" in text
    assert "A2B1" in text and "CLUT 9x9x9" in text
