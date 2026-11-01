"""The three film targets built from OpenDICE's remaining published samples.

The ISA transmissive target in its 120 roll-film cut, and one microform resolution
chart in both polarities. What they have in common is that their aim data is a
**density ladder and nothing else** -- no Lab at all, because none of these charts
prints a colour patch -- so the reference values reaching a metric are derived
rather than published, and the derivation is a policy recorded in
``tools/generate_density_ladder_nominal.py``.

The traps each carries, and why each is worth a test:

* **A ladder is repeated and the copies are numbered straight through.** OpenDICE
  writes ``Patch1``-``Patch24`` for two twelve-step wedges and says nothing about
  which wedge is which. The definitions address the two blocks separately, so the
  ids have to carry the block, and a mismatch between generator and definition
  samples every patch correctly while comparing each against the wrong aim.
* **A positive and a negative of one chart are not one target.** They share every
  dimension of geometry and share no aim values: the ladders run opposite ways.
  Swapping the reference files produces a full report with an inverted ramp.
* **The microform ladder is an L.** Ten steps around a corner, declared as ten
  occupied cells of a 5 x 6 lattice. Nothing else here has holes in a grid on
  purpose.
* **Tick polarity is a property of the specimen.** The microform positive prints
  light ticks on a dark ground and its negative prints the opposite, through the
  same declared bands. Reading a light rule as a dark one does not fail -- it locks
  a "pitch" onto the noise between the ticks.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from deltachrome.imaging.color import density_to_lightness
from deltachrome.metrics.base import CAPTURE_COLOR_CAPABILITIES
from deltachrome.profiles.schema import available_profiles, load_profile
from deltachrome.reference import read_reference
from deltachrome.targets.registry import available_targets, load_target

ISA_120 = "isa_transmissive_120_v1"
MICROFORM_POSITIVE = "microform_35mm_positive_v1"
MICROFORM_NEGATIVE = "microform_35mm_negative_v1"

NEW_TARGETS = (ISA_120, MICROFORM_POSITIVE, MICROFORM_NEGATIVE)
MICROFORM_PAIR = (MICROFORM_POSITIVE, MICROFORM_NEGATIVE)


def nominal(target_id: str):
    return read_reference(load_target(target_id).nominal_reference_path)


# -- they exist and are well formed ----------------------------------------


@pytest.mark.parametrize("target_id", NEW_TARGETS)
def test_the_target_is_installed_and_loads(target_id):
    assert target_id in available_targets()
    load_target(target_id)


@pytest.mark.parametrize("target_id", NEW_TARGETS)
def test_every_patch_has_a_reference_value(target_id):
    """The join is by patch id and nothing checks it at load time. A block prefix
    that disagrees with the generator's leaves the patch data empty and the run
    refuses -- but only if every id is expected to match, which is what this pins."""
    reference = nominal(target_id)
    missing = [p.patch_id for p in load_target(target_id).patches
               if reference.get(p.patch_id) is None]
    assert missing == []


@pytest.mark.parametrize("target_id", NEW_TARGETS)
def test_the_reference_is_declared_nominal_so_a_run_is_indicative(target_id):
    """None of these charts is sold measured, and a run against a batch aim cannot
    be a conformance claim for the specimen."""
    from deltachrome.reference.cgats import MANUFACTURER_NOMINAL

    assert nominal(target_id).basis == MANUFACTURER_NOMINAL


# -- the ISA 120 ------------------------------------------------------------


def test_the_120_is_two_twelve_step_wedges_at_two_field_positions():
    target = load_target(ISA_120)
    assert len(target.patches) == 24
    assert {p.region for p in target.patches} == {"wedge_top", "wedge_bottom"}
    for region_id in ("wedge_top", "wedge_bottom"):
        cells = [p for p in target.patches if p.region == region_id]
        assert len(cells) == 12
        assert len({(p.row, p.col) for p in cells}) == 12


def test_the_120s_wedges_are_the_same_printed_block_twice():
    """The two blocks were fitted independently off the capture. Agreeing on pitch is
    the check that they are one design printed twice rather than two measurements of
    two things -- and the check that neither fit locked onto a neighbour."""
    grids = {r.region_id: r.grid for r in load_target(ISA_120).grid_regions}
    top, bottom = grids["wedge_top"], grids["wedge_bottom"]
    assert top.patch_spacing_relative[0] == pytest.approx(
        bottom.patch_spacing_relative[0], abs=0.0005
    )
    assert top.patch_spacing_relative[1] == pytest.approx(
        bottom.patch_spacing_relative[1], abs=0.0005
    )


def test_the_120s_ramp_runs_opposite_ways_in_the_two_wedges():
    """A property of the print, not a convention chosen here: the top block reads
    darkest-first and the bottom lightest-first. Numbering both in reading order
    would compare a highlight against a shadow aim at one field position."""
    target = load_target(ISA_120)
    by_id = {p.patch_id: p for p in target.patches}
    assert (by_id["T1"].row, by_id["T1"].col) == (2, 3)
    assert (by_id["B1"].row, by_id["B1"].col) == (0, 0)
    assert (by_id["T12"].row, by_id["T12"].col) == (0, 0)
    assert (by_id["B12"].row, by_id["B12"].col) == (2, 3)


def test_the_120_carries_four_rules_which_is_what_distortion_needs():
    """Geometric distortion is the *variation* of scale across the field, so it needs
    at least four rules -- and two per axis, or there is nothing to compare an axis
    against. Neither other ISA transmissive target reaches that."""
    from deltachrome.metrics.distortion import MIN_RULES

    target = load_target(ISA_120)
    scales = [r for r in target.regions if r.kind == "scale"]
    assert len(scales) >= MIN_RULES
    axes = [r.geometry["axis"] for r in scales]
    assert axes.count("u") >= 2 and axes.count("v") >= 2
    assert "geometric_distortion" in target.capabilities


def test_the_120s_metric_rules_contradict_their_printed_label_on_purpose():
    """The chart prints '1 division = 0.1 mm' and the definition reads it as 1 mm.
    That is a deliberate departure from what the chart says, so it has to be
    recorded on the region rather than only in a commit message."""
    for region in load_target(ISA_120).regions:
        if region.kind == "scale" and region.geometry.get("unit") == "mm":
            assert region.geometry["division_mm"] == 1.0
            assert "0.1 mm" in region.geometry["printed_label"]
            assert "CONTRADICTS" in region.geometry["measured_note"]


def test_the_120_has_five_quarter_disc_pairs_like_its_siblings():
    target = load_target(ISA_120)
    modules = [r for r in target.regions if r.kind == "slanted_edge"]
    assert len(modules) == 5
    assert {r.geometry["position"] for r in modules} == {"TL", "TR", "CENTER", "BL", "BR"}
    for module in modules:
        assert len(module.geometry["discs"]) == 2


def test_the_120s_disc_pairs_are_symmetric_about_their_own_position():
    """Each pair is two quarter discs meeting apex to apex, so the pair's midpoint is
    the fiducial. If the two declared centres were not symmetric about it, the
    midpoint an operator places would not be the origin the geometry assumes."""
    for module in load_target(ISA_120).regions:
        if module.kind != "slanted_edge":
            continue
        a, b = (np.asarray(d["center"]) for d in module.geometry["discs"])
        position = {"TL": (0, 0), "TR": (1, 0), "CENTER": (0.5, 0.5),
                    "BL": (0, 1), "BR": (1, 1)}[module.geometry["position"]]
        assert (a + b) / 2 == pytest.approx(position, abs=0.001)


# -- the microform pair -----------------------------------------------------


@pytest.mark.parametrize("target_id", MICROFORM_PAIR)
def test_the_microform_ladder_is_ten_cells_of_a_thirty_cell_lattice(target_id):
    """An L is not a rectangle. Declaring it as a lattice with holes needs no new
    geometry and says exactly which cells exist."""
    target = load_target(target_id)
    assert len(target.patches) == 20
    for region_id in ("ladder_upper_left", "ladder_lower_right"):
        cells = [p for p in target.patches if p.region == region_id]
        assert len(cells) == 10
        grid = next(r.grid for r in target.grid_regions if r.region_id == region_id)
        assert (grid.rows, grid.columns) == (5, 6)
        # Six along one arm and five down the other, sharing a corner.
        rows = {p.row for p in cells}
        cols = {p.col for p in cells}
        assert len(rows) == 5 and len(cols) == 6


@pytest.mark.parametrize("target_id", MICROFORM_PAIR)
def test_the_two_microform_ladders_occupy_mirrored_corners(target_id):
    """The lower-right ladder is the upper-left one rotated 180 degrees about the
    chart's centre. Getting one of them mirrored the wrong way still samples ten
    patches and still produces a monotone ramp."""
    target = load_target(target_id)
    upper = {(p.row, p.col) for p in target.patches if p.region == "ladder_upper_left"}
    lower = {(p.row, p.col) for p in target.patches if p.region == "ladder_lower_right"}
    assert upper == {(0, c) for c in range(6)} | {(r, 0) for r in range(5)}
    assert lower == {(4, c) for c in range(6)} | {(r, 5) for r in range(5)}


def test_the_two_polarities_share_their_geometry_exactly():
    """One chart, two specimens. Any difference between the definitions other than
    the reference file is a transcription error, and both were written from one
    measurement of the positive."""
    def geometry(target_id):
        data = json.loads(
            (load_target(target_id).source_path).read_text(encoding="utf-8")
        )
        return [
            {k: v for k, v in region.items() if k != "notes"}
            for region in data["regions"]
        ], data["fiducials"]["corner_markers"], data["fiducial_extent_mm"]

    assert geometry(MICROFORM_POSITIVE) == geometry(MICROFORM_NEGATIVE)


def test_the_two_polarities_do_not_share_a_reference_file():
    """The whole reason they are two definitions. Their ladders run opposite ways,
    so running one against the other's file inverts the ramp -- while sampling every
    patch, reporting every metric, and looking entirely normal."""
    positive, negative = (nominal(t) for t in MICROFORM_PAIR)
    assert (load_target(MICROFORM_POSITIVE).nominal_reference_path
            != load_target(MICROFORM_NEGATIVE).nominal_reference_path)

    def ladder(reference):
        return [float(reference[f"UL{i}"].lab[0]) for i in range(1, 11)]

    assert np.all(np.diff(ladder(positive)) > 0)   # UL1 densest, so darkest
    assert np.all(np.diff(ladder(negative)) < 0)   # UL1 clearest, so lightest


def test_the_microform_edges_are_declared_one_at_a_time():
    """A rotated square is refined by thresholding the figure out of its surround and
    taking the minimum-area rectangle. Here the figure is two triangles with the
    chart's own diagonal rule crossing the window, which produced peak modulations of
    21 and 60 and lost two of the four edges. Four straight edges is both the working
    description and the truthful one."""
    for target_id in MICROFORM_PAIR:
        region = next(r for r in load_target(target_id).regions
                      if r.kind == "slanted_edge")
        assert region.geometry["feature"] == "straight_edges"
        edges = region.geometry["straight_edges"]
        assert len(edges) == 4
        assert {e["id"] for e in edges} == {"top", "right", "bottom", "left"}


def test_the_microform_figure_is_a_rectangle_and_not_a_square():
    """Its sides differ by 6%, which is the printed figure rather than a
    rectification error -- the chart's own rules put the two axes at the same scale.
    Recorded because a half_size with two equal entries would look more plausible and
    would be wrong."""
    region = next(r for r in load_target(MICROFORM_POSITIVE).regions
                  if r.kind == "slanted_edge")
    edges = {e["id"]: (np.asarray(e["start"]), np.asarray(e["end"]))
             for e in region.geometry["straight_edges"]}
    across = np.linalg.norm(edges["top"][1] - edges["top"][0])
    down = np.linalg.norm(edges["right"][1] - edges["right"][0])
    assert across / down == pytest.approx(1.062, abs=0.005)


def test_the_microform_edges_are_slanted_by_about_five_degrees():
    """A slanted-edge ROI needs the edge a few degrees off the pixel grid, or every
    row samples the same sub-pixel phase and there is nothing to oversample."""
    region = next(r for r in load_target(MICROFORM_POSITIVE).regions
                  if r.kind == "slanted_edge")
    for edge in region.geometry["straight_edges"]:
        start, end = np.asarray(edge["start"]), np.asarray(edge["end"])
        angle = np.degrees(np.arctan2(*(end - start)[::-1])) % 90.0
        assert 2.0 < min(angle, 90.0 - angle) < 12.0


@pytest.mark.parametrize("target_id", MICROFORM_PAIR)
def test_the_microform_charts_claim_no_colour_capability(target_id):
    """No colour patches, and no manufacturer identified to have claimed the neutrals
    are spectrally neutral -- unlike the ISA transmissive targets, whose maker does.
    So these disclaim colour accuracy and white balance as facts about the chart,
    which is a different statement from the capture-side gate."""
    target = load_target(target_id)
    assert {"color_accuracy", "white_balance"} <= target.not_capable_of
    assert "color_registration" in target.capabilities


# -- the derived aims -------------------------------------------------------


@pytest.mark.parametrize("target_id", NEW_TARGETS)
def test_every_aim_is_neutral_and_the_ladder_is_monotone(target_id):
    """Density has no chromatic dimension, so a* and b* are a restated claim rather
    than data -- and the ladder has to be monotone or the OECF curve is meaningless."""
    reference = nominal(target_id)
    target = load_target(target_id)
    ramp = target.ramp()
    labs = np.array([reference[p.patch_id].lab for p in ramp], dtype=float)
    assert np.allclose(labs[:, 1:], 0.0)
    assert np.all(np.diff(labs[:, 0]) <= 0)


@pytest.mark.parametrize("target_id", NEW_TARGETS)
def test_the_lightest_step_lands_on_fadgis_transparency_highlight_aim(target_id):
    """A density ladder carries no lightness of its own, so something has to say what
    the clear step should render at. Using the standard's number makes it the
    standard's choice -- and moving it moves every step."""
    from deltachrome.imaging.color import FADGI_TRANSPARENCY_HIGHLIGHT_L

    labs = nominal(target_id)
    lightest = max(float(labs[p.patch_id].lab[0]) for p in load_target(target_id).patches)
    assert lightest == pytest.approx(FADGI_TRANSPARENCY_HIGHLIGHT_L, abs=0.001)


def test_the_120_carries_the_isa_film_wedge_not_its_own_published_ladder():
    """The swap this test once existed to make deliberate, made. OpenDICE's
    Profile_Negative120.txt (D 0.06 to 1.83) leaves 9.2 L* rms on its own sample;
    the 4x5's wedge leaves 2.2, and every ISA 35mm capture agrees. One wedge was
    adopted for all three ISA film targets, and the file has to say it was chosen partly by fit."""
    reference = nominal(ISA_120)
    densities = [float(reference[f"T{i}"].raw["D"]) for i in range(1, 13)]
    assert densities == [0.05, 0.12, 0.19, 0.33, 0.48, 0.72, 0.88, 1.15,
                         1.43, 1.71, 2.05, 2.25]
    lightness = density_to_lightness(np.asarray(densities))
    assert float(lightness[0]) == pytest.approx(96.0)
    assert "CHOSEN PARTLY BY FIT" in reference.nominal_caveat


# -- coverage ---------------------------------------------------------------


@pytest.mark.parametrize("target_id", NEW_TARGETS)
def test_no_capability_is_declared_that_no_metric_can_read(target_id):
    """A capability name is the join between a definition and a profile's metrics. An
    unrecognised one does not disable one metric -- it silently disables every metric
    that would have matched, and the run then passes on what is left."""
    target = load_target(target_id)
    required = {
        spec.requires_capability
        for profile_id in available_profiles()
        for spec in load_profile(profile_id).metrics
        if spec.requires_capability
    }
    assert target.capabilities <= required, (
        "A capability no shipped profile gates on changes nothing either way, and "
        "reads as a guarantee the engine will not honour -- the case in point was "
        "'noise', retired from the vocabulary rather than left declared and inert. "
        "See test_every_capability_is_gated_by_some_profile for the same check "
        "across every installed target."
    )


@pytest.mark.parametrize("target_id", NEW_TARGETS)
def test_a_grayscale_capture_would_lose_only_the_colour_metrics(target_id):
    """These are the targets most likely to meet a single-record master, and the
    point of the capture-side gate is that it costs them nothing else."""
    target = load_target(target_id)
    survives = target.capabilities - CAPTURE_COLOR_CAPABILITIES
    assert {"tonal_response", "sfr", "reproduction_scale"} <= survives
