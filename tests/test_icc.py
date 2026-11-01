"""ICC parsing and the refusals that protect the measurement."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from PIL import ImageCms

from deltachrome.errors import MissingICCProfileError, UnsupportedICCProfileError
from deltachrome.imaging.icc import CIE_D50, ICC_PCS_D50, parse_icc
from deltachrome.imaging.loader import load_image
from fixtures import ADOBE_RGB_GAMMA, adobe_rgb_icc, adobe_rgb_matrix


def srgb_icc() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def test_parses_srgb_parametric_curves():
    profile = parse_icc(srgb_icc())
    assert profile.color_space == "RGB"
    assert profile.trc[0].kind == "parametric"
    # sRGB uses the type-3 piecewise form, not a pure power law.
    assert profile.trc[0].para_type == 3


def test_srgb_white_maps_to_the_pcs_white_point():
    profile = parse_icc(srgb_icc())
    xyz = profile.to_xyz(np.array([1.0, 1.0, 1.0]))
    assert xyz == pytest.approx(ICC_PCS_D50, abs=1e-4)


def test_parses_a_gamma_curve_profile():
    profile = parse_icc(adobe_rgb_icc())
    assert profile.trc[0].kind == "gamma"
    assert profile.trc[0].gamma == pytest.approx(ADOBE_RGB_GAMMA, abs=1e-4)
    assert profile.matrix == pytest.approx(adobe_rgb_matrix(), abs=2e-5)


def test_adobe_rgb_transform_is_exact_at_full_float_precision():
    """The reason this reader exists: an 8-bit Lab path would floor the error near
    0.4 dE. Round-tripping device RGB through the parsed profile must be exact."""
    profile = parse_icc(adobe_rgb_icc())
    device = np.array([[0.2, 0.5, 0.8], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    xyz = profile.to_xyz(device)
    linear = np.linalg.inv(profile.matrix) @ xyz.T
    recovered = np.clip(linear.T, 0, None) ** (1.0 / ADOBE_RGB_GAMMA)
    assert recovered == pytest.approx(device, abs=1e-9)


def test_tone_curve_applies_per_channel_independently():
    profile = parse_icc(adobe_rgb_icc())
    single = profile.to_xyz(np.array([0.3, 0.6, 0.9]))
    stacked = profile.to_xyz(np.array([[0.3, 0.6, 0.9], [0.3, 0.6, 0.9]]))
    assert stacked[0] == pytest.approx(single)


def test_cmyk_profile_is_refused():
    header = bytearray(128)
    struct.pack_into(">I", header, 0, 132)
    header[12:16] = b"prtr"
    header[16:20] = b"CMYK"
    header[20:24] = b"Lab "
    header[36:40] = b"acsp"
    blob = bytes(header) + struct.pack(">I", 0)
    with pytest.raises(UnsupportedICCProfileError, match="CMYK"):
        parse_icc(blob)


def test_truncated_blob_is_refused():
    with pytest.raises(UnsupportedICCProfileError, match="too short"):
        parse_icc(b"\x00" * 40)


def test_image_without_an_embedded_profile_is_refused(tmp_path):
    """The single most dangerous failure this tool could have is assuming sRGB for
    an Adobe RGB or ProPhoto master, so a missing profile is a hard stop."""
    import tifffile

    path = tmp_path / "no_profile.tif"
    tifffile.imwrite(path, np.zeros((16, 16, 3), dtype=np.uint16), photometric="rgb")

    with pytest.raises(MissingICCProfileError) as excinfo:
        load_image(path)
    assert "will not guess" in str(excinfo.value)


def test_cie_and_icc_d50_differ_only_slightly():
    """Documents the deliberate choice to evaluate against CIE D50 (what a
    spectrophotometer reports) rather than the ICC encoding of it."""
    assert np.max(np.abs(np.array(CIE_D50) - np.array(ICC_PCS_D50))) < 0.0004


def test_loaded_image_reports_the_files_storage_dtype_not_the_working_one(tmp_path):
    """The report shows this to the operator. Printing 'float64' -- the dtype
    Deltachrome normalizes to -- would tell them nothing about their file."""
    from fixtures import SyntheticChart

    path, _ = SyntheticChart().render_capture(tmp_path / "cap.tif")
    image = load_image(path)
    assert image.dtype == "uint16"
    assert image.bit_depth == 16
    assert image.rgb.dtype == np.float64
