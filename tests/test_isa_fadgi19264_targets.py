"""The ISA FADGI - ISO 19264 chart, in both the versions OpenDICE ships samples for.

Don Williams' device-level chart: 168 circular patches, a sixteen-step neutral
ramp, five resolution modules, a Siemens star and two printed rules. It is the
largest target modelled here and the one whose layout is least like what it looks
like, so most of this file is about the difference between the two.

* **It is one 14 x 21 lattice, not sixteen rows.** The chart prints row numbers 1
  to 16 and they are not evenly spaced -- the grey wedge, the printed rules and
  the serial box each take whole row slots. Read as sixteen rows the geometry
  cannot be expressed at all; read as twenty-one slots of which five are empty it
  fits to a tenth of a percent.
* **The two versions share that layout exactly and share almost no colours.** Only
  30 of 168 patches have any close match in the other version's reference file,
  and the worst pair differs by 74 dE2000. A capture of one placed against the
  other's definition samples every patch correctly.
* **The patch ids are numbers the commercial chart does not print.** It labels its
  grid A-N by 1-16; the reference data is keyed 1 to 168, so that is what the ids
  have to be, and the printed label is recorded per patch instead.
* **The declared rules are the inch combs, not the millimetre ones.** Read with
  this engine the fine 1 mm comb comes out 0.7-0.9% coarser than both the chart's
  own 10 mm major ticks and its inch comb, which agree with each other and with
  the capture's declared rate.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from deltachrome.imaging.color import delta_e
from deltachrome.reference import read_reference
from deltachrome.targets.registry import load_target
from deltachrome.targets.roles import (
    DEFAULT_ACHROMATIC_CHROMA_MAX,
    DEFAULT_UNIFORMITY_TOLERANCE,
    MIN_UNIFORMITY_CLUSTER,
)

EARLY = "isa_fadgi_19264_v1"
COMMERCIAL = "isa_fadgi_19264_commercial_v1"
BOTH = (EARLY, COMMERCIAL)

#: Printed row number for each lattice row that carries patches.
PRINTED_ROW = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 8: 7, 9: 8, 10: 9, 11: 10,
               12: 11, 13: 12, 14: 13, 15: 14, 17: 15, 20: 16}
#: The ramp, in the order the densities put it: down column E and column J.
RAMP = ["59", "60", "61", "62", "71", "72", "81", "82", "91", "92",
        "101", "102", "103", "104", "105", "106"]


def raw(target_id: str) -> dict:
    return json.loads(load_target(target_id).source_path.read_text(encoding="utf-8"))


def nominal(target_id: str):
    return read_reference(load_target(target_id).nominal_reference_path)


# -- the lattice ------------------------------------------------------------


@pytest.mark.parametrize("target_id", BOTH)
def test_the_patches_sit_on_a_lattice_of_twenty_one_rows(target_id):
    """Not sixteen. The printed row numbers skip whole slots of a regular
    lattice, and the empty slots are where the chart's other features sit."""
    grid = load_target(target_id).patch_grid
    assert (grid.rows, grid.columns) == (21, 14)
    used = {p.row for p in load_target(target_id).patches}
    assert used == set(PRINTED_ROW)
    assert sorted(set(range(21)) - used) == [6, 7, 16, 18, 19]


@pytest.mark.parametrize("target_id", BOTH)
def test_the_occupancy_is_the_pattern_the_capture_shows(target_id):
    """168 of 294 cells, and which ones is not decoration: it fixes the reading
    order, and the reading order is what joins a patch to its reference value."""
    per_row: dict[int, int] = {}
    for patch in load_target(target_id).patches:
        per_row[patch.row] = per_row.get(patch.row, 0) + 1
    assert sum(per_row.values()) == 168
    assert [per_row[r] for r in sorted(per_row)] == [
        8, 8, 8, 8, 8, 14, 12, 10, 10, 10, 14, 14, 14, 14, 8, 8
    ]


@pytest.mark.parametrize("target_id", BOTH)
def test_ids_run_one_to_one_sixty_eight_in_reading_order(target_id):
    """OpenDICE's numbering, and what the early chart prints beside its patches."""
    patches = load_target(target_id).patches
    assert [p.patch_id for p in patches] == [str(n) for n in range(1, 169)]
    ordered = sorted(patches, key=lambda p: (p.row, p.col))
    assert [p.patch_id for p in ordered] == [p.patch_id for p in patches]


def test_the_commercial_version_records_the_grid_label_it_prints():
    """Its ids are numbers the chart does not print anywhere, so the label a
    reader can actually see has to be recorded somewhere. The early chart prints
    the numbers themselves and needs no such note."""
    letters = "ABCDEFGHIJKLMN"
    for entry in raw(COMMERCIAL)["patches"]:
        row, col = entry["grid_position"]["row"], entry["grid_position"]["col"]
        assert entry["notes"].endswith(f"{letters[col]}{PRINTED_ROW[row]}")
    assert all("notes" not in e for e in raw(EARLY)["patches"])


# -- the ramp ---------------------------------------------------------------


@pytest.mark.parametrize("target_id", BOTH)
def test_the_ramp_is_sixteen_steps_down_two_columns(target_id):
    target = load_target(target_id)
    assert [p.patch_id for p in target.ramp()] == RAMP

    samples = nominal(target_id).samples
    lightness = [samples[p.patch_id].lab[0] for p in target.ramp()]
    # Descending, but not strictly: the two darkest steps are D 2.48 and D 2.57,
    # and the published lightness for them is 2.1 and 2.2 -- an inversion of a
    # tenth of an L* at the very bottom of the ladder, which is measurement noise
    # on a near-black patch rather than a step out of order. Asserting strict
    # descent would fail on that and would be asserting something untrue of the
    # data, so the tolerance is stated instead of the check being dropped.
    assert all(a - b > -0.5 for a, b in zip(lightness, lightness[1:]))
    assert lightness[0] - lightness[-1] > 90.0
    # Column E and column J of the printed grid, four rows of the lattice apart
    # from each other -- so the ladder runs down the chart in two columns rather
    # than along a row, and reading order happens to agree with it here.
    assert {p.col for p in target.ramp()} == {4, 5, 6, 7, 8, 9}


def test_both_versions_print_the_same_neutral_ramp():
    """The one part of the chart the redesign left alone, and the check that the
    two really are the same model rather than two unrelated charts."""
    early = {p: nominal(EARLY).samples[p].raw["D"] for p in RAMP}
    commercial = {p: nominal(COMMERCIAL).samples[p].raw["D"] for p in RAMP}
    assert early == commercial


# -- what separates the two versions ----------------------------------------


def test_the_two_versions_reference_files_are_not_interchangeable():
    """They share a lattice, an id scheme and a neutral ramp, and their colour
    patches were re-chosen. Everything that makes them look like one target is
    what makes swapping their reference files invisible."""
    early, commercial = nominal(EARLY).samples, nominal(COMMERCIAL).samples
    assert set(early) == set(commercial)

    ids = [str(n) for n in range(1, 169)]
    difference = delta_e(
        np.array([early[p].lab for p in ids]),
        np.array([commercial[p].lab for p in ids]),
        "delta_e_2000",
    )
    # Not a batch-to-batch drift: a redesign. Two thirds of the array moves more
    # than 5 dE2000 or is left alone entirely, and the worst pair is a different
    # colour altogether.
    assert float(np.max(difference)) > 60.0
    assert int((difference > 10).sum()) > 30


def test_the_two_versions_grid_geometry_is_measured_separately():
    """The modules sit differently against the array on the two charts -- 10.4
    patch pitches between the corner fields against 11.1 -- so neither
    definition's numbers transfer to the other, and both were measured."""
    early = load_target(EARLY).patch_grid
    commercial = load_target(COMMERCIAL).patch_grid
    assert early.patch_spacing_relative != commercial.patch_spacing_relative
    assert early.grid_origin_relative != commercial.grid_origin_relative
    # Vertically the two agree, which is what says the difference is real and
    # horizontal rather than a mis-measurement of one chart.
    assert early.patch_spacing_relative[1] == pytest.approx(
        commercial.patch_spacing_relative[1], rel=0.01
    )
    assert early.patch_spacing_relative[0] != pytest.approx(
        commercial.patch_spacing_relative[0], rel=0.02
    )


# -- placement --------------------------------------------------------------


def test_the_commercial_chart_places_on_its_printed_pluses():
    """It prints four, one just inside each corner, and they are better than the
    corner modules on every count: a bigger quad (2607 x 4011 px against 2086 x
    3353), better conditioned because it is a rectangle at the corners rather than
    a set of points inside the array (1.63 px of chart displacement per pixel of
    fiducial error, against 1.98), and findable -- the existing crosshair finder
    reads all four with their areas agreeing to 0.0%."""
    target = load_target("isa_fadgi_19264_commercial_v1")
    assert target.fiducials.method == "printed_crosshairs"
    assert target.fiducials.corner_ids == ["TL", "TR", "BR", "BL"]
    # Outside the frame on every side, because the frame is still the corner
    # modules' dpi fields and the pluses are at the chart's own corners.
    us = [u for u, _ in target.fiducials.corner_positions]
    vs = [v for _, v in target.fiducials.corner_positions]
    assert min(us) < 0.0 and max(us) > 1.0
    assert min(vs) < 0.0 and max(vs) > 1.0


def test_the_early_chart_places_on_its_corner_squares_and_one_dot():
    """It prints no pluses, so it places on what it does print at its corners: three
    black squares, and at the bottom right the dot standing where a fourth square
    would be. Each placement point IS the declared marker of that name, so the
    placement and the refinement features cannot drift apart. The two charts share
    a layout exactly and differ here, which is why each carries its own fiducial
    declaration rather than inheriting one."""
    target = load_target("isa_fadgi_19264_v1")
    assert target.fiducials.method == "corner_squares_and_dot"
    assert target.fiducials.corner_ids == ["TL", "TR", "BR", "BL"]
    centres = {r.region_id: tuple(r.geometry["center"]) for r in target.regions
               if r.kind == "marker" and "center" in r.geometry}
    assert [tuple(p) for p in target.fiducials.corner_positions] == [
        centres["corner_square_tl"], centres["corner_square_tr"],
        centres["registration_dot_8"], centres["corner_square_bl"],
    ]


def test_the_early_charts_placement_marks_form_a_rectangle():
    """The dot stands where a fourth square would: level with the bottom-left square
    and below the top-right one. That alignment is how a person finds the right dot
    of seventy-five, so it is pinned rather than left to the definition."""
    (tl, tr, br, bl) = load_target("isa_fadgi_19264_v1").fiducials.corner_positions
    assert abs(br[1] - bl[1]) < 0.002 and abs(br[0] - tr[0]) < 0.002
    assert abs(tl[1] - tr[1]) < 0.002 and abs(tl[0] - bl[0]) < 0.005


@pytest.mark.parametrize("target_id", BOTH)
def test_both_charts_declare_their_three_corner_squares_as_markers(target_id):
    """THREE, not four: the bottom-right corner is taken by that corner's module.
    They are declared because they are the only features on either chart that snap
    cleanly -- 0.24 to 0.48 px -- and because three is two thirds of a refinement
    path where these charts previously had none at all."""
    target = load_target(target_id)
    squares = [r for r in target.regions if r.region_id.startswith("corner_square_")]
    assert len(squares) == 3
    assert {r.geometry["position"] for r in squares} == {"TL", "TR", "BL"}
    for region in squares:
        assert region.kind == "marker"
        assert region.geometry["feature"] == "corner_square"
        # Outside the frame, which is what "just inside the chart's corner" means
        # when the frame is the corner modules rather than the chart.
        u, v = region.geometry["center"]
        assert not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0)


@pytest.mark.parametrize("target_id", BOTH)
def test_the_array_extends_outside_the_fiducial_quad(target_id):
    """A property of the coordinate FRAME rather than of the placement: the frame's
    own corners are the four corner modules' dpi fields, which sit inside the patch
    array, so patch coordinates run past 0 and 1 on both charts. That stays true
    after the commercial chart moved its placement out to the printed pluses --
    which is the point of expressing fiducials as positions in the frame rather
    than as the frame's corners, and the reason that change re-measured nothing."""
    target = load_target(target_id)
    corners = [
        corner
        for patch in target.patches
        for corner in target.patch_corners_relative(patch, inset=0.55)
    ]
    us = [u for u, _ in corners]
    vs = [v for _, v in corners]
    assert min(us) < 0.0 and max(us) > 1.0
    assert min(vs) < 0.0 and max(vs) > 1.0
    # Still bounded: an ROI reaching much past the modules would be off the chart.
    assert min(us) > -0.2 and max(us) < 1.2
    assert min(vs) > -0.25 and max(vs) < 1.15


# -- the printed rules ------------------------------------------------------


@pytest.mark.parametrize("target_id", BOTH)
def test_six_rule_segments_are_declared_three_on_each_axis(target_id):
    """Three segments per rule, because a comb absorbs bowing into its pitch and
    one band spanning the frame would measure the rate without ever showing how
    it varies -- and the variation is what geometric distortion is."""
    scales = [r for r in load_target(target_id).regions if r.kind == "scale"]
    assert len(scales) == 6
    axes = [r.geometry["axis"] for r in scales]
    assert axes.count("u") == 3 and axes.count("v") == 3


@pytest.mark.parametrize("target_id", BOTH)
def test_only_the_inch_comb_is_declared(target_id):
    """The millimetre comb on the same rule reads 0.7-0.9% coarser through this
    engine than the chart's own 10 mm major ticks and its inch comb, which agree
    with each other. Declaring it would put that systematic into reproduction
    scale and into geometric distortion, which pools every reading whatever unit
    it is in."""
    scales = [r for r in load_target(target_id).regions if r.kind == "scale"]
    assert {r.geometry["unit"] for r in scales} == {"in"}
    for region in scales:
        assert region.geometry["division_mm"] == pytest.approx(25.4 / 8)
        assert "not declared" in region.geometry["measured_note"].lower()


@pytest.mark.parametrize("target_id", BOTH)
def test_each_rule_segment_has_its_own_position_in_its_name(target_id):
    """The distortion metric groups readings by the second word of the region id,
    so segments that shared a name would pool into one position and the field
    spread would collapse to nothing."""
    scales = [r for r in load_target(target_id).regions if r.kind == "scale"]
    positions = {r.region_id.split("_")[1] for r in scales}
    assert positions == {"h1", "h2", "h3", "v1", "v2", "v3"}


# -- capabilities -----------------------------------------------------------


@pytest.mark.parametrize("target_id", BOTH)
def test_the_chart_can_measure_scale_and_distortion(target_id):
    caps = load_target(target_id).capabilities
    assert {"color_accuracy", "tonal_response", "white_balance",
            "reproduction_scale", "geometric_distortion"} <= caps


@pytest.mark.parametrize("target_id", BOTH)
def test_every_module_declares_two_edges_from_two_separate_figures(target_id):
    """Ten edges across five modules, declared one at a time. They are not the
    sides of a shape: each module stacks a band and a block, and the two slanted
    boundaries belong to different figures. The white resolution field's own
    borders are the strongest transitions in the module and are deliberately not
    among them -- they are axis-aligned, so there is no sub-pixel phase in them."""
    target = load_target(target_id)
    assert {"sfr", "color_registration"} <= target.capabilities
    modules = [r for r in target.regions if r.kind == "slanted_edge"]
    assert len(modules) == 5
    for region in modules:
        edges = region.geometry["straight_edges"]
        assert len(edges) == 2
        assert {e["orientation"] for e in edges} == {"horizontal", "vertical"}


@pytest.mark.parametrize("target_id", BOTH)
def test_every_declared_edge_actually_slants(target_id):
    """The property the whole measurement rests on. An edge aligned to the pixel
    grid repeats one sub-pixel phase on every row and cannot be oversampled, so a
    declaration that had drifted onto the resolution field's border would produce
    no reading at all -- and nothing else in the module is printed at an angle, so
    a slant in the declared range is strong evidence it is on the right boundary.
    """
    target = load_target(target_id)
    for region in (r for r in target.regions if r.kind == "slanted_edge"):
        for edge in region.geometry["straight_edges"]:
            start = np.array(edge["start"])
            end = np.array(edge["end"])
            direction = end - start
            # In relative coordinates the quad is not square, so the angle is
            # judged as a ratio of the run to the rise rather than in degrees.
            along, across = (
                (abs(direction[1]), abs(direction[0]))
                if edge["orientation"] == "vertical"
                else (abs(direction[0]), abs(direction[1]))
            )
            assert along > 0
            assert 0.01 < across / along < 0.25, f"{region.region_id} {edge['id']}"


@pytest.mark.parametrize("target_id", BOTH)
def test_lightness_uniformity_is_disclaimed_because_the_neutrals_are_a_ramp(target_id):
    """Uniformity needs repeated neutrals spread over the area. Every one of this
    chart's sixteen neutrals is a different step of one ladder."""
    assert "lightness_uniformity" in load_target(target_id).not_capable_of

    # Checked against the rule the role resolver actually applies: a uniformity
    # family is four or more neutrals within 1.5 dE2000 of each other. The
    # chart's largest such group is two.
    labs = np.array([
        s.lab for s in nominal(target_id).samples.values()
        if float(np.hypot(*s.lab[1:])) <= DEFAULT_ACHROMATIC_CHROMA_MAX
    ])
    largest = max(
        int((delta_e(np.repeat(lab[None, :], len(labs), axis=0), labs,
                     "delta_e_2000") <= DEFAULT_UNIFORMITY_TOLERANCE).sum())
        for lab in labs
    )
    assert largest < MIN_UNIFORMITY_CLUSTER


# -- the printed tone ladder and wedge ----------------------------------------


def test_only_the_commercial_chart_prints_a_stepped_tone_ladder():
    """The two versions differ here and the difference is easy to miss: the
    commercial release keylines eighteen steps above its wedge and labels each
    with an aim lightness, and the early chart prints the wedge alone."""
    commercial = {r.region_id for r in load_target(COMMERCIAL).regions}
    early = {r.region_id for r in load_target(EARLY).regions}
    assert "tone_ladder" in commercial and "tone_ladder" not in early
    assert "tone_wedge" in commercial and "tone_wedge" in early


def test_the_tone_ladder_is_a_marker_and_carries_the_reason_it_is_not_measured():
    """A deliberate refusal to measure, not an oversight. The ladder's only
    available reference is the numbers printed beside it, and those are DESIGN
    AIMS -- exact multiples of five, which no measured ladder lands on. The
    sixteen neutral circles carry batch-average L*a*b* instead, so declaring the
    ladder as patches would add a weaker measurement and put eighteen unusable
    patches into every run's warnings with no way for an operator to fix it."""
    ladder = next(r for r in load_target(COMMERCIAL).regions
                  if r.region_id == "tone_ladder")
    assert ladder.kind == "marker"
    assert ladder.grid is None, "a marker carries no patches"

    aims = ladder.geometry["printed_aim_lightness"]
    assert aims == list(range(95, 5, -5))
    assert len(aims) == ladder.geometry["steps"] == 18
    assert all(a % 5 == 0 for a in aims), "the giveaway that these are aims"
    assert "DESIGN AIMS" in ladder.notes
    assert "batch average" in ladder.notes


def test_no_ladder_patch_reaches_the_run():
    """The point of declaring it as a marker: the patch list is untouched, so no
    run gains a warning about eighteen patches it cannot judge."""
    for target_id in BOTH:
        target = load_target(target_id)
        assert len(target.patches) == 168
        assert not any(p.region in ("tone_ladder", "tone_wedge") for p in target.patches)


# -- refinement features ------------------------------------------------------


@pytest.mark.parametrize("target_id", BOTH)
def test_both_charts_declare_eight_registration_dots(target_id):
    """Before these, NEITHER chart could be refined at all: its five modules are
    declared as `straight_edges`, which the feature finder does not locate, so
    refinement found zero features and refused every time -- on charts where the
    difference between a readable printed rule and an unreadable one is about
    twenty pixels."""
    dots = [r for r in load_target(target_id).regions
            if r.region_id.startswith("registration_dot_")]
    assert len(dots) == 8
    for region in dots:
        assert region.kind == "marker"
        assert region.geometry["feature"] == "registration_dot"
        assert region.geometry["half_size"][0] == region.geometry["half_size"][1]


@pytest.mark.parametrize("target_id", BOTH)
def test_the_declared_dot_box_sits_inside_the_printed_dot(target_id):
    """Deliberate, and the one thing about this that looks like an error. The
    finder samples a feature's own level in an annulus at 0.78 to 1.13 of the
    declared half-size, so a box drawn ON a disc's edge samples half disc and half
    card, the two levels come out equal and it declines -- measured, all eight
    decline at 1.00 and at 0.85 of the radius, and all eight snap inside it."""
    for region in load_target(target_id).regions:
        if not region.region_id.startswith("registration_dot_"):
            continue
        half = region.geometry["half_size"][0]
        radius = region.geometry["printed_diameter_relative"] / 2
        assert half == pytest.approx(radius * 0.75, rel=1e-3)
        assert half < radius, "the box must sit inside the dot, not on its edge"


@pytest.mark.parametrize("target_id", BOTH)
def test_the_charts_now_carry_enough_features_for_a_homography(target_id):
    """Eight dots and three corner squares is eleven, and a homography is
    admissible only when the features over-determine its eight degrees of
    freedom. Getting there is the whole point of declaring the dots: three
    corner squares alone is one short of even a similarity."""
    from deltachrome.imaging.refine import FEATURE_KINDS, MODELS

    target = load_target(target_id)
    locatable = [
        r for r in target.regions
        if r.kind in FEATURE_KINDS
        and (r.geometry or {}).get("center") is not None
        and (r.geometry or {}).get("half_size") is not None
    ]
    assert len(locatable) == 11
    homography_dof = dict(MODELS)["homography"]
    assert len(locatable) >= homography_dof


@pytest.mark.parametrize("target_id", BOTH)
def test_the_dots_reason_records_why_the_resolution_fields_are_not_used(target_id):
    """The obvious fourth feature, and it does not work -- for two separate
    measured reasons. A later reader who reaches for it should meet them."""
    first = next(r for r in load_target(target_id).regions
                 if r.region_id == "registration_dot_1")
    assert "resolution fields" in first.notes.lower()
    assert "3:2" in first.notes and "isotropic" in first.notes.lower()
    # And the limit of what the dots buy, which is a property of their size.
    assert "WHERE IT STOPS" in first.notes
