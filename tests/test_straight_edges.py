"""Edges declared one at a time, by their endpoints.

The engine's two older edge features both describe a *closed shape* and take their
edges from its sides: a rotated square gives four, a quarter-disc pair gives two.
Some charts do not build their edges that way. The ISA FADGI-ISO 19264 module
stacks two separate printed figures -- a band whose lower boundary slants, and a
block beside it whose side boundary slants -- so its horizontal and vertical edges
belong to different objects and share no centre, no size and no tilt.

Two things about that are worth testing rather than assuming:

* Describing the pair as one rotated square would invent a figure the chart does
  not print, and would emit two further "edges" that are really the borders of the
  white resolution field. Those borders are the highest-contrast boundaries in the
  whole module and they are **axis-aligned**, so they repeat one sub-pixel phase on
  every row and carry no information to oversample. Declaring edges individually is
  what makes it possible to take the two that slant and leave those alone.
* A lone edge has no shape to confirm a refinement against, so the refinement has
  to be able to decline. These tests pin both directions: that it corrects a
  placement that is out, and that it leaves one alone when what it found does not
  look like the edge that was declared.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deltachrome.imaging.edges import (
    STRAIGHT_EDGE_SEARCH_PX,
    _refine_straight_edge,
    extract_edges,
)
from deltachrome.imaging.geometry import place_chart
from deltachrome.imaging.loader import load_image
from deltachrome.imaging.sfr import compute_sfr
from deltachrome.targets.schema import target_from_dict
from fixtures import adobe_rgb_icc, write_tiff_with_profile

SIZE = 900


def stacked_figures_capture(tmp_path, tilt=5.0, sigma=0.7):
    """Two separate slanted figures, plus an axis-aligned rectangle beside them.

    A deliberately faithful miniature of the real module: a band across the top
    whose lower boundary slants, a block on the left whose right boundary slants,
    and a bright rectangle whose borders are the strongest transitions in the
    frame and are exactly axis-aligned. Anything that picks edges by contrast
    finds the rectangle; anything that picks them by slant finds the two figures.
    """
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    slope = np.tan(np.radians(tilt))

    image = np.full((SIZE, SIZE), 0.55)
    # Band across the top: its lower boundary descends to the right.
    image[yy < 200 + slope * (xx - SIZE / 2)] = 0.25
    # Block on the left, below the band: its right boundary leans.
    block = (yy > 320) & (yy < 780) & (xx > 120) & (xx < 380 + slope * (yy - 550))
    image[block] = 0.25
    # The bright rectangle: highest contrast anywhere, and square to the grid.
    image[330:820, 470:860] = 0.95

    image = cv2.GaussianBlur(image, (0, 0), sigma)
    data = (np.repeat(image[:, :, None], 3, axis=2) * 65535).astype(np.uint16)
    return write_tiff_with_profile(tmp_path / "stacked.tif", data, adobe_rgb_icc())


def straight_edge_target(edges):
    return target_from_dict({
        "target_id": "straight_probe",
        "display_name": "Straight edge probe",
        "capabilities": ["sfr"],
        "fiducials": {"method": "chart_corners", "corner_markers": [
            {"id": "TL", "relative_position": [0.0, 0.0]},
            {"id": "TR", "relative_position": [1.0, 0.0]},
            {"id": "BR", "relative_position": [1.0, 1.0]},
            {"id": "BL", "relative_position": [0.0, 1.0]}]},
        "regions": [{"region_id": "module", "kind": "slanted_edge",
                     "geometry": {"feature": "straight_edges", "straight_edges": edges}}],
        "patches": [],
    })


#: The two slanted boundaries of the fixture, in relative coordinates.
BAND_EDGE = {"id": "h", "orientation": "horizontal",
             "start": [0.15, 200 / SIZE - 0.35 * np.tan(np.radians(5.0))],
             "end": [0.85, 200 / SIZE + 0.35 * np.tan(np.radians(5.0))]}
BLOCK_EDGE = {"id": "v", "orientation": "vertical",
              "start": [380 / SIZE - (230 * np.tan(np.radians(5.0))) / SIZE, 320 / SIZE],
              "end": [380 / SIZE + (230 * np.tan(np.radians(5.0))) / SIZE, 780 / SIZE]}


def placed(path):
    image = load_image(path)
    placement = place_chart(
        [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
        (image.width, image.height),
    )
    return image, placement


# -- the feature -------------------------------------------------------------


def test_two_edges_are_cut_from_two_separate_figures(tmp_path):
    """Neither belongs to the other, and no shape connects them."""
    image, placement = placed(stacked_figures_capture(tmp_path))
    rois = extract_edges(
        image, placement, straight_edge_target([BAND_EDGE, BLOCK_EDGE]), roi_half_width=50
    )
    assert len(rois) == 2
    assert {r.orientation for r in rois} == {"horizontal", "vertical"}
    assert {r.edge_id for r in rois} == {"module_h", "module_v"}


def test_each_declared_edge_measures_the_slant_it_was_given(tmp_path):
    image, placement = placed(stacked_figures_capture(tmp_path, tilt=5.0))
    rois = extract_edges(
        image, placement, straight_edge_target([BAND_EDGE, BLOCK_EDGE]), roi_half_width=50
    )
    for roi in rois:
        result = compute_sfr(roi.pixels)
        assert abs(result.edge_angle_deg) == pytest.approx(5.0, abs=0.6)
        # A clean synthetic step has no overshoot, so the curve must not exceed 1.
        assert result.peak_modulation() == pytest.approx(1.0, abs=0.01)


def test_declaring_only_the_slanted_edges_leaves_the_rectangle_alone(tmp_path):
    """The point of declaring edges one at a time. The bright rectangle in the
    fixture is the strongest transition in the frame and is axis-aligned, so an
    ROI on it would have no sub-pixel phase to oversample at all."""
    image, placement = placed(stacked_figures_capture(tmp_path))
    rois = extract_edges(
        image, placement, straight_edge_target([BAND_EDGE, BLOCK_EDGE]), roi_half_width=50
    )
    for roi in rois:
        left, top, right, bottom = roi.bounds_px
        overlap_x = min(right, 860) - max(left, 470)
        overlap_y = min(bottom, 820) - max(top, 330)
        assert overlap_x <= 0 or overlap_y <= 0, f"{roi.edge_id} reaches the rectangle"


# -- refinement --------------------------------------------------------------


def test_refinement_pulls_a_displaced_declaration_back_onto_the_edge(tmp_path):
    """A definition describes a model; print registration and placement move the
    real edge by a few pixels on any given capture."""
    image, _ = placed(stacked_figures_capture(tmp_path))
    truth = np.array([[120.0, 320.0], [160.0, 780.0]])
    truth[:, 0] = [380 - 230 * np.tan(np.radians(5.0)), 380 + 230 * np.tan(np.radians(5.0))]

    displaced = truth + np.array([9.0, 0.0])
    a, b = _refine_straight_edge(image, displaced[0], displaced[1])
    moved_back = np.linalg.norm(np.array([a, b]) - truth, axis=1)
    assert moved_back.max() < 2.0
    # And it is a correction, not a coincidence: the declaration it started from
    # was further out than what it returned.
    assert np.linalg.norm(displaced - truth, axis=1).max() > moved_back.max()


def test_refinement_declines_when_there_is_nothing_to_snap_to(tmp_path):
    """A lone edge has no shape to confirm the find against, so the declaration is
    the only check available and a refinement that has wandered is worse than
    none. Pointed at flat field, it must hand back exactly what it was given."""
    image, _ = placed(stacked_figures_capture(tmp_path))
    # Well inside the uniform surround, far from every figure.
    a = np.array([40.0, 60.0])
    b = np.array([44.0, 300.0])
    out_a, out_b = _refine_straight_edge(image, a, b)
    assert np.allclose(out_a, a) and np.allclose(out_b, b)


def test_refinement_cannot_reach_a_neighbouring_edge(tmp_path):
    """The search radius is the guard. On the real chart the nearest other
    boundary is about 60 px from the one being refined, so a search wider than
    this would let one edge capture its neighbour."""
    image, _ = placed(stacked_figures_capture(tmp_path))
    truth_x = 380.0
    # Declared far enough out that the real edge is beyond the search radius.
    far = truth_x - STRAIGHT_EDGE_SEARCH_PX - 25
    a = np.array([far - 20.0, 320.0])
    b = np.array([far + 20.0, 780.0])
    out_a, out_b = _refine_straight_edge(image, a, b)
    assert abs(out_a[0] - truth_x) > STRAIGHT_EDGE_SEARCH_PX


def test_an_unrefined_run_uses_the_declaration_as_given(tmp_path):
    image, placement = placed(stacked_figures_capture(tmp_path))
    target = straight_edge_target([BLOCK_EDGE])
    refined = extract_edges(image, placement, target, roi_half_width=50, refine=True)
    exact = extract_edges(image, placement, target, roi_half_width=50, refine=False)
    assert len(refined) == len(exact) == 1
    # Both find the edge here, because the declaration is already right; what the
    # flag changes is whether the position was checked against the capture.
    assert refined[0].center_px == pytest.approx(exact[0].center_px, abs=3.0)
