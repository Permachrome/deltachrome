"""The display proxy, and the invariant the placement canvas rests on.

Everything geometric in the frontend depends on one thing being true at every
zoom and in both read paths: **scene coordinates are full-resolution image
pixels**. A corner handle's position is therefore the number
``analyze(corners=...)` wants, with no conversion step to get backwards -- and a
conversion error there would produce a plausible wrong homography rather than a
crash, which is the failure mode this project treats as the dangerous one.

So the tests below are mostly about round-tripping a region through a strided
read and getting the same pixels back at the same coordinates.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from deltachrome.imaging.loader import LoadedImage  # noqa: E402
from deltachrome_ui.imageproxy import (  # noqa: E402
    ImageProxy,
    to_display_rgb,
    to_qimage,
)


def make_image(
    width=800, height=600, dtype=np.uint16, scale=65535.0, channels=3
) -> LoadedImage:
    """A gradient with a distinctive marker, so a misplaced read is visible."""
    ys, xs = np.mgrid[0:height, 0:width]
    pixels = np.zeros((height, width, channels), dtype=dtype)
    top = float(np.iinfo(dtype).max) if dtype != np.float32 else 1.0
    pixels[..., 0] = (xs / width * top).astype(dtype)
    if channels == 3:
        pixels[..., 1] = (ys / height * top).astype(dtype)
        pixels[..., 2] = top // 2 if dtype != np.float32 else 0.5
    return LoadedImage(
        path=__import__("pathlib").Path("synthetic.tif"),
        pixels=pixels,
        scale=scale,
        profile=None,  # type: ignore[arg-type]
        bit_depth=16,
        dtype=str(np.dtype(dtype)),
    )


# -- display conversion -------------------------------------------------


def test_sixteen_bit_maps_to_eight_by_the_encodings_own_maximum():
    block = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
    assert to_display_rgb(block, 65535.0).tolist() == [[[0, 128, 255]]]


def test_values_above_the_encoding_maximum_are_clipped_not_wrapped():
    block = np.array([[[70000, 65535, 0]]], dtype=np.uint32)
    assert to_display_rgb(block, 65535.0)[0, 0, 0] == 255


def test_a_grayscale_master_is_widened_for_display_and_only_for_display():
    """The engine keeps one colour record because replicating it would have three
    metrics report a comparison that was never made. The canvas needs three, so the
    widening happens here -- at the point of drawing, and nowhere earlier."""
    image = make_image(width=64, height=48, channels=1)
    assert image.channels == 1
    rgb = to_display_rgb(image.pixels, 65535.0)
    assert rgb.shape == (48, 64, 3)
    assert np.array_equal(rgb[..., 0], rgb[..., 2])

    proxy = ImageProxy(image, overview_width=32)
    assert proxy.overview.width() > 0
    tile = proxy.tile((0, 0, 16, 16))
    assert tile.image.width() == 16 and tile.image.height() == 16
    assert proxy.loupe(8, 8, half_width=4).is_full_resolution


def test_display_conversion_always_returns_contiguous_bytes():
    """QImage over a non-contiguous buffer reads whatever is between the rows."""
    strided = make_image().pixels[::3, ::4, :]
    assert to_display_rgb(strided, 65535.0).flags["C_CONTIGUOUS"]


# -- color management --


def test_a_linear_light_master_no_longer_reads_dark():
    """a linear-light profile's
    encoded 0.5 is *dark* divided raw by the encoding's own maximum, and *mid grey*
    once the gamma is actually applied."""
    from fixtures import adobe_rgb_matrix, build_matrix_icc

    from deltachrome.imaging.icc import parse_icc

    linear_profile = parse_icc(
        build_matrix_icc(adobe_rgb_matrix(), gamma=1.0, description="Linear [test]")
    )
    block = np.full((2, 2, 3), 32768, dtype=np.uint16)  # code value 0.5

    raw = to_display_rgb(block, 65535.0)
    managed = to_display_rgb(block, 65535.0, linear_profile)

    assert raw[0, 0, 0] == 128, "the raw path is the one that reads dark"
    assert managed[0, 0, 0] > raw[0, 0, 0] + 30, (
        "sRGB-encoding a linear 0.5 should read well above a raw-divided 0.5"
    )


def test_untagged_linear_values_are_drawn_through_the_srgb_curve():
    """Mid-grey reflectance, 18%, at 46% of the display rather than 18%."""
    block = np.array([[[11796, 11796, 11796]]], dtype=np.uint16)
    assert to_display_rgb(block, 65535.0)[0, 0, 0] == 46
    assert to_display_rgb(block, 65535.0, linear=True)[0, 0, 0] == 118
    eight = np.array([[[46, 46, 46]]], dtype=np.uint8)
    assert to_display_rgb(eight, 255.0, linear=True)[0, 0, 0] == 118


def test_only_an_untagged_capture_is_drawn_as_linear():
    image = make_image()
    proxy = ImageProxy(image, linear_display=True)
    assert proxy.linear_display
    plain = ImageProxy(image)
    region = (100, 100, 140, 140)
    assert proxy.tile(region).image.pixelColor(5, 5).red() \
        > plain.tile(region).image.pixelColor(5, 5).red()
    tagged = ImageProxy(image.__class__(**{**image.__dict__, "profile": object()}),
                        linear_display=True)
    assert not tagged.linear_display


def test_color_managed_and_raw_agree_closely_on_an_ordinary_profile():
    """Adobe RGB's own gamma is close enough to sRGB's that the two paths should
    not diverge wildly on ordinary content -- the fix is for the linear-light and
    wide-gamut cases, not a change that make every capture look different."""
    from fixtures import adobe_rgb_icc

    from deltachrome.imaging.icc import parse_icc

    profile = parse_icc(adobe_rgb_icc())
    block = np.full((2, 2, 3), 40000, dtype=np.uint16)

    raw = to_display_rgb(block, 65535.0)
    managed = to_display_rgb(block, 65535.0, profile)
    assert np.abs(raw.astype(int) - managed.astype(int)).max() < 15


def test_a_grayscale_master_is_color_managed_through_its_own_transform():
    """A grayscale profile's ``to_xyz`` wants one channel, not three -- feeding it
    a pre-widened block would raise inside ``ICCProfile.to_xyz`` and this would
    silently fall back to the raw path instead of actually managing the colour."""
    from fixtures import build_gray_icc

    from deltachrome.imaging.icc import parse_icc

    profile = parse_icc(build_gray_icc(gamma=2.2))
    image = make_image(width=8, height=8, channels=1)

    managed = to_display_rgb(image.pixels, 65535.0, profile)
    raw = to_display_rgb(image.pixels, 65535.0)
    assert managed.shape == raw.shape == (8, 8, 3)
    # Still neutral to a unit of 8-bit rounding -- a grayscale profile has no
    # colorants to introduce a cast; the same rounding noise
    # ``swatches.lab_to_display_rgb`` already carries for a chroma-zero Lab
    # triple through this same matrix.
    managed_i = managed.astype(int)
    assert np.abs(managed_i[..., 0] - managed_i[..., 1]).max() <= 1
    assert np.abs(managed_i[..., 1] - managed_i[..., 2]).max() <= 1
    assert not np.array_equal(managed, raw), "gamma 2.2 must move the reading"


def test_a_profile_that_cannot_evaluate_falls_back_silently():
    """A preview that is momentarily less accurate beats a canvas with nothing on
    it -- the whole reason this is a fallback rather than a raised error."""
    class BrokenProfile:
        channels = 3

        def to_xyz(self, rgb):
            raise RuntimeError("synthetic failure")

    block = np.full((2, 2, 3), 32768, dtype=np.uint16)
    assert np.array_equal(
        to_display_rgb(block, 65535.0, BrokenProfile()),
        to_display_rgb(block, 65535.0),
    )


def test_no_profile_is_the_original_raw_behaviour_unchanged():
    """The geometric round-trip tests elsewhere in this file build images with
    ``profile=None`` -- confirms that path is untouched by this."""
    block = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
    assert to_display_rgb(block, 65535.0, None).tolist() == \
        to_display_rgb(block, 65535.0).tolist()


def test_a_qimage_outlives_the_array_it_was_built_from():
    """QImage does not keep a numpy buffer alive, and the crash from that arrives
    later and somewhere else, so the wrapper copies."""
    import gc

    rgb = np.full((4, 5, 3), 200, dtype=np.uint8)
    image = to_qimage(rgb)
    del rgb
    gc.collect()
    assert image.width() == 5 and image.height() == 4
    assert image.pixelColor(0, 0).red() == 200


# -- geometry ------------------------------------------------------------


def test_the_overview_stride_covers_the_whole_frame():
    proxy = ImageProxy(make_image(1000, 800), overview_width=250)
    assert proxy.overview_stride == 4
    assert proxy.overview.width() == 250
    # Scene coordinates are image pixels, so the overview item is drawn scaled by
    # exactly this stride. Getting it wrong misplaces every corner.
    assert proxy.overview.width() * proxy.overview_stride >= proxy.width - 4


def test_a_narrow_image_is_never_upsampled():
    proxy = ImageProxy(make_image(100, 80), overview_width=2400)
    assert proxy.overview_stride == 1
    assert proxy.overview.width() == 100


def test_clamp_holds_a_point_on_the_last_pixel_not_past_it():
    """A chart photographed to the edge of the frame has its corner there."""
    proxy = ImageProxy(make_image(800, 600))
    assert proxy.clamp(-50.0, -50.0) == (0.0, 0.0)
    assert proxy.clamp(1e9, 1e9) == (799.0, 599.0)


# -- reads ---------------------------------------------------------------


def test_a_full_resolution_tile_returns_exactly_the_region_asked_for():
    proxy = ImageProxy(make_image(800, 600))
    tile = proxy.tile((100, 50, 300, 250), stride=1)
    assert tile.region == (100, 50, 300, 250)
    assert (tile.image.width(), tile.image.height()) == (200, 200)
    assert tile.is_full_resolution


def test_a_tile_reads_the_pixels_that_are_actually_there():
    """The round trip that the canvas's correctness rests on: a tile placed at
    ``region`` top-left and scaled by ``stride`` lands back on its own pixels."""
    image = make_image(800, 600)
    proxy = ImageProxy(image)
    x0, y0 = 321, 199
    tile = proxy.tile((x0, y0, x0 + 16, y0 + 16), stride=1)
    expected = to_display_rgb(image.pixels[y0 : y0 + 16, x0 : x0 + 16, :], image.scale)
    assert tile.image.pixelColor(0, 0).red() == int(expected[0, 0, 0])
    assert tile.image.pixelColor(15, 15).green() == int(expected[15, 15, 1])


def test_a_region_running_off_the_frame_returns_the_part_that_exists():
    """A view scrolled past the edge asks for pixels that are not there."""
    proxy = ImageProxy(make_image(800, 600))
    tile = proxy.tile((700, 500, 1200, 900), stride=1)
    assert tile.region == (700, 500, 800, 600)
    assert (tile.image.width(), tile.image.height()) == (100, 100)


def test_a_region_entirely_outside_the_frame_does_not_raise():
    proxy = ImageProxy(make_image(800, 600))
    assert proxy.tile((5000, 5000, 6000, 6000)).image.width() >= 1


def test_the_tile_budget_coarsens_a_large_request_rather_than_refusing_it():
    """Asking for the whole frame at 1:1 is what a zoomed-out view wanted anyway."""
    proxy = ImageProxy(make_image(4000, 3000), tile_budget=250_000)
    stride = proxy.stride_for((0, 0, 4000, 3000), target_px=4000)
    assert stride > 1
    assert (4000 // stride) * (3000 // stride) <= 250_000 * 1.1


def test_a_small_region_at_high_zoom_reads_at_full_resolution():
    proxy = ImageProxy(make_image(4000, 3000))
    assert proxy.stride_for((100, 100, 400, 400), target_px=1200) == 1


def test_the_loupe_ignores_the_tile_budget():
    """It exists precisely to show what the overview cannot, so it never decimates."""
    proxy = ImageProxy(make_image(4000, 3000), tile_budget=1)
    tile = proxy.loupe(2000.4, 1500.6, half_width=40)
    assert tile.stride == 1
    assert tile.is_full_resolution
    assert tile.image.width() == 81


def test_the_loupe_stays_inside_the_frame_at_a_corner():
    proxy = ImageProxy(make_image(800, 600))
    tile = proxy.loupe(2.0, 3.0, half_width=40)
    assert tile.region[0] == 0 and tile.region[1] == 0
    assert tile.stride == 1
