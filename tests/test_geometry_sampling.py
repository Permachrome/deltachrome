"""Homography placement and ROI sampling."""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.errors import GeometryError
from deltachrome.imaging.geometry import place_chart
from deltachrome.imaging.sampling import sample_patches
from deltachrome.targets.registry import load_target

SQUARE = [(100.0, 100.0), (1100.0, 100.0), (1100.0, 800.0), (100.0, 800.0)]


def test_relative_corners_map_to_the_placed_corners():
    placement = place_chart(SQUARE, (1200, 900))
    corners = placement.to_pixel(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))
    assert corners == pytest.approx(np.array(SQUARE), abs=1e-6)


def test_center_maps_to_the_center_for_a_rectangle():
    placement = place_chart(SQUARE, (1200, 900))
    assert placement.to_pixel(np.array([0.5, 0.5])) == pytest.approx([600.0, 450.0])


def test_perspective_placement_is_invertible():
    skewed = [(120.0, 90.0), (1080.0, 130.0), (1120.0, 810.0), (95.0, 770.0)]
    placement = place_chart(skewed, (1200, 900))
    for u, v in [(0.0, 0.0), (0.25, 0.75), (1.0, 1.0)]:
        px = placement.to_pixel(np.array([u, v]))
        assert np.all(np.isfinite(px))
    assert placement.to_pixel(np.array([[0.0, 0.0]]))[0] == pytest.approx(skewed[0], abs=1e-6)


def test_mis_ordered_corners_are_rejected():
    """Corners given as opposite pairs produce a homography that silently mirrors
    half the chart -- every patch would be sampled from the wrong place."""
    bowtie = [(100.0, 100.0), (1100.0, 700.0), (1000.0, 100.0), (100.0, 800.0)]
    with pytest.raises(GeometryError, match="convex quadrilateral"):
        place_chart(bowtie, (1200, 900))


def test_degenerate_corners_are_rejected():
    with pytest.raises(GeometryError, match="collinear or coincident"):
        place_chart([(10.0, 10.0)] * 4, (1200, 900))


def test_corners_outside_the_image_are_rejected():
    with pytest.raises(GeometryError, match="outside"):
        place_chart([(0.0, 0.0), (5000.0, 0.0), (5000.0, 800.0), (0.0, 800.0)], (1200, 900))


def test_effective_ppi_and_aspect_deviation():
    placement = place_chart(SQUARE, (1200, 900))
    ppi = placement.effective_ppi((292.1, 203.2))
    assert ppi == pytest.approx(87.2, abs=0.5)
    assert placement.aspect_deviation((292.1, 203.2)) < 0.02


def test_sampling_flags_clipping(tmp_path):
    from fixtures import SyntheticChart

    chart = SyntheticChart()
    path, corners = chart.render_capture(tmp_path / "cap.tif", gain=(3.0, 3.0, 3.0))
    from deltachrome.imaging.loader import load_image

    image = load_image(path)
    target = load_target("colorchecker_sg_v1")
    placement = place_chart(corners, (image.width, image.height))
    result = sample_patches(image, placement, target)

    flagged = [f for flags in result.flagged().values() for f in flags]
    assert any(f.startswith("clipped_high") for f in flagged)


def test_roi_inset_changes_the_sampled_pixel_count(tmp_path):
    from fixtures import SyntheticChart
    from deltachrome.imaging.loader import load_image

    chart = SyntheticChart()
    path, corners = chart.render_capture(tmp_path / "cap.tif")
    image = load_image(path)
    target = load_target("colorchecker_sg_v1")
    placement = place_chart(corners, (image.width, image.height))

    tight = sample_patches(image, placement, target, inset=0.3)
    loose = sample_patches(image, placement, target, inset=0.9)
    assert tight["A1"].pixel_count < loose["A1"].pixel_count


def test_invalid_inset_is_rejected(tmp_path):
    from fixtures import SyntheticChart
    from deltachrome.imaging.loader import load_image

    chart = SyntheticChart()
    path, corners = chart.render_capture(tmp_path / "cap.tif")
    image = load_image(path)
    target = load_target("colorchecker_sg_v1")
    placement = place_chart(corners, (image.width, image.height))
    with pytest.raises(ValueError, match="inset"):
        sample_patches(image, placement, target, inset=0.0)
