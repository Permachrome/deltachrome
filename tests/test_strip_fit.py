"""How a row of patches is fitted, and why it is not fitted as a lattice.

The Golden Thread Object Level Target ships in two sizes and they are not the same
chart. The 1X is one row of thirty-four slots at one pitch, four of them the
resolution module; the 0.5X is two blocks of fifteen at pitches 1.4% apart with a
module spanning 4.37 of them. No integer slot count describes the second, which is
what took this finder off an index fit and onto the positions the definition
declares.

Everything here is either a pure function over positions or a small raster, because
the two real charts' captures are not in the repository and the failures being
guarded against are all arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.imaging import detect
from deltachrome.targets.registry import load_target


# -- the pitch ----------------------------------------------------------------


def test_the_pitch_is_the_common_divisor_of_the_gaps():
    """Patches butt, so every gap between two found ones is a whole number of
    pitches -- and the pitch is the largest spacing they all agree on."""
    found = np.array([0.0, 100.0, 200.0, 500.0, 600.0, 900.0])
    assert detect._strip_pitch(found, 34) == pytest.approx(100.0, abs=0.5)


def test_a_sub_multiple_is_not_preferred_to_the_pitch():
    """Half a pitch divides everything a whole pitch divides, and a quarter divides
    more still, so a search preferring the smallest candidate walks down to noise.
    It cost four of twenty Odonata captures a placement."""
    found = np.array([0.0, 100.0, 200.0, 400.0, 700.0])
    assert detect._strip_pitch(found, 34) == pytest.approx(100.0, abs=0.5)


def test_duplicates_from_the_threshold_sweep_do_not_become_the_pitch():
    """The blob search keeps the union over six saturation levels, so one patch
    arrives several times a pixel apart. Where those outnumber the patches -- half
    of these captures -- the median gap IS a duplicate, and filtering by a fraction
    of it returned a pitch of a fifth of a pixel."""
    found = np.array([0.0, 0.4, 1.1, 100.0, 100.3, 200.0, 200.2, 200.9, 300.0])
    assert detect._strip_pitch(found, 34) == pytest.approx(100.0, abs=1.0)


def test_a_gap_that_is_not_a_whole_pitch_does_not_have_to_agree():
    """The 0.5X's module spans 4.37 pitches, so a gap across it is not a multiple
    of anything. Enough gaps within the blocks agree, and that is the answer."""
    found = np.array([0.0, 100.0, 200.0, 637.0, 737.0, 837.0])
    assert detect._strip_pitch(found, 34) == pytest.approx(100.0, abs=1.0)


def test_too_little_to_measure_a_pitch_from_is_none_rather_than_a_guess():
    assert detect._strip_pitch(np.array([500.0]), 34) is None


# -- what the definition declares ---------------------------------------------


@pytest.mark.parametrize("target_id", ["isa_object_level_1x_v1", "isa_object_level_05x_v1"])
def test_both_object_level_targets_declare_thirty_patches_in_a_row(target_id):
    centres, outline = detect._declared_strip_centres(load_target(target_id))
    assert len(centres) == 30
    assert outline[0] == pytest.approx(0.0, abs=1e-6)
    assert outline[1] == pytest.approx(1.0, abs=1e-4)
    assert np.all(np.diff(centres) > 0), "centres come back in order along the strip"


def test_the_one_x_row_is_uniform_and_the_half_x_row_is_not():
    """The whole reason this fit is against declared positions rather than an index.

    Both charts print fifteen patches, a module, then fifteen more. On the 1X every
    patch sits on a thirty-four slot lattice; on the 0.5X the two blocks are at
    different pitches and the module between them is not a whole number of either.
    """
    one_x, _ = detect._declared_strip_centres(load_target("isa_object_level_1x_v1"))
    half_x, _ = detect._declared_strip_centres(load_target("isa_object_level_05x_v1"))

    def in_cells(centres):
        gaps = np.diff(centres)
        return gaps / np.median(gaps)

    assert in_cells(one_x)[14] == pytest.approx(5.0, abs=0.01), "1X: module is 4 slots"
    assert np.allclose(np.delete(in_cells(one_x), 14), 1.0, atol=0.01)

    # The 0.5X is irregular in two separate ways, and the module is only one of
    # them: its blocks sit at pitches 1.4% apart, and its patch 15 is cut short
    # where the left block ends, so even a neighbouring pair is not one cell.
    gaps = in_cells(half_x)
    assert not 4.99 < gaps[14] < 5.01, "the module is not a whole number of pitches"
    assert gaps[13] < 0.9, "patch 15 sits closer to 14 than a cell, being narrower"
    assert 1.0 < gaps[0] < 1.05


def test_the_two_blocks_are_declared_at_their_own_pitches():
    """1.4% apart, measured off the capture rather than assumed equal. Forcing one
    pitch on both puts patch 15 seventy pixels off centre on a 267 px patch."""
    target = load_target("isa_object_level_05x_v1")
    grids = {r.region_id: r.grid for r in target.grid_regions if r.grid is not None}
    assert set(grids) == {"patches_left", "patches_left_end", "patches_right"}
    left = grids["patches_left"].patch_spacing_relative[0]
    right = grids["patches_right"].patch_spacing_relative[0]
    assert left != right
    assert 1.005 < left / right < 1.030
    # The two blocks together fill the quad exactly: patch 1's outer left corner to
    # patch 30's outer right corner IS the fiducial quad.
    end = grids["patches_right"].grid_origin_relative[0] + grids["patches_right"].extent_relative[0]
    assert grids["patches_left"].grid_origin_relative[0] == pytest.approx(0.0, abs=1e-9)
    assert end == pytest.approx(1.0, abs=1e-4)
    # The left block no longer reaches the module: its last patch is declared apart.
    assert grids["patches_left"].columns == 14


# -- the fit ------------------------------------------------------------------


def _row(centres, scale, offset, keep):
    """Where the given declared patches would be found, at this scale.

    Each arrives three times a fraction of a pixel apart, because the blob search
    keeps the union over six saturation levels and that is what it hands the fit.
    """
    return np.array(
        [scale * centres[i] + offset + jitter for i in keep for jitter in (-0.4, 0.0, 0.5)],
        float,
    )


def test_the_fit_recovers_a_two_pitch_row_the_found_patches_only_sample():
    """Ten of thirty found, which is what the 0.5X gives: the search is for chroma
    and its neutral steps carry none."""
    target = load_target("isa_object_level_05x_v1")
    centres, outline = detect._declared_strip_centres(target)
    scale, offset = 1780.0, 224.0
    keep = [2, 3, 4, 5, 6, 7, 8, 21, 24, 29]
    along = _row(centres, scale, offset, keep)
    across = np.full(len(along), 90.0)
    cell = float(np.median(np.diff(centres)))
    pitch = scale * cell
    start = float(along.min()) - (2 + 0.5) * pitch

    fitted = detect._fit_declared_strip(
        along, across, centres, outline, [start], pitch / cell
    )
    assert fitted is not None
    line_along, _line_across, matched, residual = fitted
    assert matched == len(keep), "one patch found three times over is one patch"
    assert residual < 0.5
    assert float(line_along[0]) == pytest.approx(scale, rel=0.005)
    assert float(line_along[1]) == pytest.approx(offset, abs=4.0)


def test_the_fit_is_not_allowed_to_reopen_the_start_the_capture_established():
    """On a row at one pitch, shifting every found patch a slot along fits exactly
    as tightly. Re-matching against the fitted line and going round again looks
    like a refinement and lets the answer walk off the anchor -- 14 delta-E instead
    of 2.6, at a residual that still reads clean. So the match is made once."""
    import inspect

    source = inspect.getsource(detect._strip_assignment)
    assert "for _ in range" not in source, (
        "the assignment is matched once at the anchored start, never iterated"
    )


# -- where the row ends ---------------------------------------------------------


def _strip_raster(*, disc: bool) -> tuple[np.ndarray, np.ndarray]:
    """A card, a row of butted patches, and optionally a disc one cell beyond it.

    Both Golden Thread units print that disc, and it is the reason the walk that
    finds the row's end cannot simply ask whether a cell holds something other
    than the card: the disc does, and it sits exactly one cell out.
    """
    pitch, tall = 40, 60
    rgb = np.full((200, 800, 3), 0.5, np.float32)
    for slot in range(10):
        x = 300 + slot * pitch
        rgb[70:70 + tall, x:x + pitch] = (0.2 + 0.07 * slot, 0.3, 0.7 - 0.05 * slot)
    if disc:
        centre = (260, 100)
        ys, xs = np.mgrid[0:200, 0:800]
        rgb[((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2) < 16 ** 2] = 0.05
    blobs = np.array([[300 + slot * pitch + pitch / 2, 100.0, float(tall)]
                      for slot in range(10)], float)
    return rgb, blobs


def test_the_row_does_not_grow_into_a_disc_printed_beyond_it():
    """A solid disc one cell past the row departs from the card exactly as a patch
    does, so position alone cannot exclude it. What does is that the row's own
    patches butt together -- the line between two of them is patch on both sides --
    while anything beyond the row has card between it and the row."""
    rgb, blobs = _strip_raster(disc=True)
    reached = detect._strip_cells_beyond(rgb, float(blobs[0][0]), -40.0, blobs, 0, spare=3)
    assert reached == 0, "the disc is not the next patch, however much it looks like one"


def test_a_patch_the_chroma_search_missed_still_extends_the_row():
    """The whole point of asking the capture rather than the located patches: a
    neutral step carries no chroma and is walked straight past by the search, and
    it is still part of the row."""
    rgb, blobs = _strip_raster(disc=False)
    # Pretend the first two patches were never found.
    reached = detect._strip_cells_beyond(rgb, float(blobs[2][0]), -40.0, blobs[2:], 0, spare=3)
    assert reached == 2


# -- a patch the block cut short ------------------------------------------------


def test_the_half_x_declares_its_clipped_patch_apart_from_its_neighbours():
    """Patch 15 is not a fifteenth equal cell and declaring it as one costs a
    reading.

    Its neighbours are 266.9 px on the capture the geometry was measured from and
    it is 178: it runs from the boundary with patch 14 to the printed keyline that
    ends the block, and the block's width is not a whole number of pitches.
    Modelled uniformly, the sampling ROI overran the patch by about twenty-seven
    pixels onto the keyline and the shadow beside it -- the patch read 5.5 L* dark
    at 6.63 dE2000 with mean_median_divergence flagged, and it dragged the whole
    capture's noise figure to 10.6. Declared apart it reads 0.94 and the noise
    figure is 3.2.
    """
    target = load_target("isa_object_level_05x_v1")
    grids = {r.region_id: r.grid for r in target.grid_regions if r.grid is not None}
    assert set(grids) == {"patches_left", "patches_left_end", "patches_right"}

    run, clipped = grids["patches_left"], grids["patches_left_end"]
    assert run.columns == 14 and clipped.columns == 1
    assert clipped.patch_size_relative[0] < 0.75 * run.patch_spacing_relative[0], (
        "the clipped patch is materially narrower, which is the whole point"
    )

    # The three tile the quad in order, without overlapping.
    spans = [
        (g.grid_origin_relative[0], g.grid_origin_relative[0] + g.extent_relative[0])
        for g in (run, clipped, grids["patches_right"])
    ]
    assert spans[0][0] == pytest.approx(0.0, abs=1e-9)
    assert spans[2][1] == pytest.approx(1.0, abs=1e-4)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start >= end - 1e-9, "the blocks must not overlap"


def test_every_patch_the_half_x_declares_is_sampled_inside_itself():
    """The property the split exists to hold, checked on all thirty rather than on
    the one that was wrong: an ROI is the middle ``inset`` of a patch, so it lies
    inside that patch only if the patch is declared at its real width."""
    from deltachrome.imaging.sampling import DEFAULT_ROI_INSET

    target = load_target("isa_object_level_05x_v1")
    for patch in target.patches:
        grid = target.grid_for(patch)
        width = grid.patch_size_relative[0]
        corners = grid.corners_relative(patch.row, patch.col, inset=DEFAULT_ROI_INSET)
        us = [u for u, _v in corners]
        centre = grid.center_relative(patch.row, patch.col)[0]
        assert min(us) >= centre - width / 2 - 1e-9, patch.patch_id
        assert max(us) <= centre + width / 2 + 1e-9, patch.patch_id
