"""JPEG 2000 loading.

JP2 is a preservation master format across cultural heritage digitisation, and it
stores its ICC profile in the JP2 container's ``colr`` box -- somewhere Pillow does
not surface. Without parsing that box, every JP2 master looks like an untagged file
and gets refused, which would make the tool useless for a large share of the
collections it is meant to serve.
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.errors import ImageError, MissingICCProfileError
from deltachrome.imaging.loader import load_image
from fixtures import adobe_rgb_icc, write_jp2

PIXELS = (np.random.default_rng(3).uniform(0.1, 0.9, (48, 64, 3)) * 255).astype(np.uint8)


def test_icc_profile_is_read_from_the_colr_box(tmp_path):
    path = write_jp2(tmp_path / "master.jp2", PIXELS, icc=adobe_rgb_icc())
    image = load_image(path)

    assert image.container == "jp2"
    assert image.profile.description.startswith("Synthetic Adobe RGB")
    assert image.width == 64 and image.height == 48


def test_bit_depth_comes_from_the_ihdr_box(tmp_path):
    """BPC sits at +18 in ihdr, after the two-byte channel count. Reading it at +16
    picks up the high byte of NC and reports every image as 1-bit -- which would
    also defeat the precision guard below."""
    path = write_jp2(tmp_path / "master.jp2", PIXELS, icc=adobe_rgb_icc())
    assert load_image(path).bit_depth == 8


def test_a_high_bit_depth_jp2_decoded_to_8_bits_is_refused(tmp_path):
    """Measuring a 16-bit master through an 8-bit decode would discard most of the
    precision it was captured at -- the exact loss this tool exists to prevent."""
    path = write_jp2(
        tmp_path / "master.jp2", PIXELS, icc=adobe_rgb_icc(), claim_bit_depth=16
    )
    with pytest.raises(ImageError, match="would discard the file's precision"):
        load_image(path)


def test_a_jp2_with_no_color_specification_is_refused(tmp_path):
    path = write_jp2(tmp_path / "bare.jp2", PIXELS, enum_cs=0)
    with pytest.raises((MissingICCProfileError, ImageError)):
        load_image(path)


def test_an_enumerated_srgb_space_is_honoured_but_flagged(tmp_path):
    """An enumerated color space is an explicit declaration, not a guess -- so it
    is used. It carries no per-device calibration though, and the report says so."""
    path = write_jp2(tmp_path / "srgb.jp2", PIXELS, enum_cs=16)
    image = load_image(path)

    assert image.profile.transform_kind == "matrix_trc"
    assert any("enumerated field" in note for note in image.notes)


def test_a_non_rgb_enumerated_space_is_refused(tmp_path):
    """sYCC is not three-channel RGB and must not be measured as if it were."""
    path = write_jp2(tmp_path / "sycc.jp2", PIXELS, enum_cs=18)
    with pytest.raises(ImageError, match="sYCC"):
        load_image(path)


def test_pixels_are_kept_in_their_native_dtype(tmp_path):
    """A 100 MP master is 300 MB as uint8 and 2.4 GB as float64. Converting up front
    would make real files unusable."""
    path = write_jp2(tmp_path / "master.jp2", PIXELS, icc=adobe_rgb_icc())
    image = load_image(path)

    assert image.pixels.dtype == np.uint8
    assert image.scale == 255.0
    assert image.dtype == "uint8"


def test_window_normalises_only_the_region_requested(tmp_path):
    path = write_jp2(tmp_path / "master.jp2", PIXELS, icc=adobe_rgb_icc())
    image = load_image(path)

    window = image.window(4, 12, 6, 20)
    assert window.shape == (8, 14, 3)
    assert window.dtype == np.float64
    assert 0.0 <= window.min() and window.max() <= 1.0
    assert window == pytest.approx(image.pixels[4:12, 6:20, :] / 255.0)


def test_window_matches_the_full_normalised_image(tmp_path):
    path = write_jp2(tmp_path / "master.jp2", PIXELS, icc=adobe_rgb_icc())
    image = load_image(path)
    assert image.window(0, image.height, 0, image.width) == pytest.approx(image.rgb)


def test_a_genuinely_16_bit_master_is_read_at_16_bits(tmp_path):
    """The refusal above is for a decoder that cannot; this is the ordinary case.

    Pillow has no 16-bit RGB JPEG 2000 mode -- it decodes such a codestream to 8
    bits per channel and says nothing -- so every 16-bit master used to be refused
    with an instruction to convert it to TIFF first. That is a real cost imposed on
    an operator for a decoder limitation. OpenCV binds the same OpenJPEG library and
    reads the file at its own depth, and it is the decoder the loader now uses.
    """
    pixels = (
        np.random.default_rng(11).uniform(0.1, 0.9, (48, 64, 3)) * 65535
    ).astype(np.uint16)
    path = write_jp2(tmp_path / "master.jp2", pixels, icc=adobe_rgb_icc())
    image = load_image(path)

    assert image.bit_depth == 16
    assert image.pixels.dtype == np.uint16
    assert np.array_equal(image.pixels, pixels), (
        "lossless in, lossless out -- a decoder that quietly rescaled would still "
        "have the right dtype"
    )


def test_the_decoded_pixels_are_rgb_and_not_the_decoders_own_order(tmp_path):
    """OpenCV hands back BGR. Getting this wrong swaps red and blue in every
    measurement while leaving neutrals, the placement and the geometry untouched --
    a chart would place perfectly and every Delta E would be wrong."""
    pixels = np.zeros((8, 8, 3), np.uint8)
    pixels[..., 0] = 200      # red
    pixels[..., 2] = 40       # blue
    path = write_jp2(tmp_path / "master.jp2", pixels, icc=adobe_rgb_icc())

    read = load_image(path).pixels
    assert read[0, 0, 0] > read[0, 0, 2], "red channel must still be the red channel"
    assert np.array_equal(read, pixels)


def test_a_contiguous_array_comes_back(tmp_path):
    """Reversing BGR gives a negatively strided view, and every window taken from it
    afterwards would be a strided read of a hundred-megapixel array."""
    path = write_jp2(tmp_path / "master.jp2", PIXELS, icc=adobe_rgb_icc())
    assert load_image(path).pixels.flags["C_CONTIGUOUS"]
