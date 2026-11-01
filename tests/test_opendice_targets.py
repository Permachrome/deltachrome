"""The four patch-array targets built from OpenDICE's published samples.

Two Next Generation Targets and the two IT8.7 charts. What they have in common is
that each is a single regular lattice and nothing else -- no slanted edges, no
printed rules -- so almost everything that can go wrong here is a mapping error,
and a mapping error does not announce itself. It produces a complete run with
plausible numbers.

The traps each of these has, and why each is worth a test:

* **The two NGTs share a lattice exactly.** Same 13 x 10 array, same patch ids
  A0-M9, and different printed values -- their neutrals differ by nearly 4 L*. A
  capture of one placed against the other's definition samples every patch
  correctly and reports a chart-wide colour error belonging to neither chart.
* **The NGT's ramp is not in reading order.** Its twelve steps are printed as two
  rows running in opposite directions, so numbering them left to right interleaves
  the ladder and tone response compares a highlight against a shadow aim.
* **IT8 patches are butted.** Patch size equals spacing exactly, so the sampling
  inset is the only thing holding an ROI off the boundary -- there is no printed
  margin to fall back on.
* **A `not_capable_of` entry is documentation, not a gate.** The engine skips a
  metric on the strength of `capabilities` alone, so disclaiming something no
  profile gates on reads as a guarantee the engine will not honour. That is a
  mistake this file's last test would have caught, because it was made here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from deltachrome.imaging.color import delta_e
from deltachrome.profiles.schema import available_profiles, load_profile
from deltachrome.reference import read_reference
from deltachrome.targets.registry import available_targets, load_target

NGT = "avian_ngt_v1"
DT_NGT = "avian_dt_ngt_v2"
IT8_TRANSMISSIVE = "it8_7_1_transmissive_v1"
IT8_REFLECTIVE = "it8_7_2_reflective_v1"

NEW_TARGETS = (NGT, DT_NGT, IT8_TRANSMISSIVE, IT8_REFLECTIVE)
NGT_PAIR = (NGT, DT_NGT)
IT8_PAIR = (IT8_TRANSMISSIVE, IT8_REFLECTIVE)


def nominal(target_id: str):
    target = load_target(target_id)
    return read_reference(target.nominal_reference_path)


# -- what each chart is -----------------------------------------------------


@pytest.mark.parametrize("target_id", NGT_PAIR)
def test_the_ngts_are_thirteen_columns_by_ten_rows(target_id):
    target = load_target(target_id)
    assert len(target.patches) == 130
    assert set(target.patch_ids) == {f"{c}{r}" for c in "ABCDEFGHIJKLM" for r in range(10)}


@pytest.mark.parametrize("target_id", IT8_PAIR)
def test_the_it8s_are_twelve_rows_by_twenty_two_columns(target_id):
    target = load_target(target_id)
    array = {f"{r}{c}" for r in "ABCDEFGHIJKL" for c in range(1, 23)}
    assert array <= set(target.patch_ids)
    assert len(array) == 264


@pytest.mark.parametrize("target_id", IT8_PAIR)
def test_the_it8s_carry_the_grayscale_strip_below_the_colour_array(target_id):
    """Twenty-two butted steps with a Dmin block to their left and a Dmax block to
    their right. The step count was fitted rather than detected -- the ladder is
    nearly continuous in lightness and has no boundaries a gradient search can
    trust -- and the pitch it came out at is the colour array's own column pitch,
    so the strip is that lattice carried downward."""
    target = load_target(target_id)
    assert len(target.patches) == 288
    assert set(target.patch_ids) == (
        {f"{r}{c}" for r in "ABCDEFGHIJKL" for c in range(1, 23)}
        | {f"GS{i}" for i in range(24)}
    )
    ramp = next(r for r in target.regions if r.region_id == "grayscale")
    assert (ramp.grid.rows, ramp.grid.columns) == (1, 22)
    assert ramp.grid.patch_spacing_relative[0] == pytest.approx(1 / 22, abs=1e-6)
    assert ramp.grid.grid_origin_relative[0] == 0.0
    # The blocks at either end are their own regions because neither is exactly a
    # step wide -- a uniform 24-cell lattice fits both outer edges and drifts a
    # tenth of a cell in the middle.
    for region_id in ("grayscale_dmin", "grayscale_dmax"):
        block = next(r for r in target.regions if r.region_id == region_id)
        assert (block.grid.rows, block.grid.columns) == (1, 1)
        assert block.grid.patch_spacing_relative[0] != pytest.approx(1 / 22, abs=1e-4)


def test_only_the_dt_ngt_carries_printed_registration_marks():
    """The V2 adds four crosshairs the earlier chart does not have, and placement
    keys on them; the earlier NGT has to be placed by its patch array."""
    assert load_target(DT_NGT).fiducials.method == "printed_crosshairs"
    assert load_target(NGT).fiducials.method == "patch_array_corners"


@pytest.mark.parametrize("target_id", NEW_TARGETS)
def test_the_corners_go_round_the_chart_and_name_the_right_patches(target_id):
    """A bow-tie corner order raises nothing and maps every ROI to the frame's own
    average, so the schema refuses it on load. This pins the corners it accepts to
    the patches an operator is told to click."""
    target = load_target(target_id)
    assert target.fiducials.corner_ids == ["TL", "TR", "BR", "BL"]
    markers = json.loads(target.source_path.read_text())["fiducials"]["corner_markers"]
    named = [m["nearest_patch"] for m in markers]
    # The corner markers name the colour array's own corners, which on the IT8s
    # are no longer the first and last patch ids: the grayscale strip is declared
    # after the array and its GS23 sorts last.
    array = [p.patch_id for p in target.patches if p.region == "colour_array"] \
        or target.patch_ids
    assert named[0] == array[0] and named[2] == array[-1]


# -- the tonal ramp ---------------------------------------------------------


@pytest.mark.parametrize("target_id", NEW_TARGETS)
def test_the_ramp_is_ordered_by_density_and_therefore_by_lightness(target_id):
    """`ramp_position` has to run lightest to darkest, because tone response pairs
    each step against its aim by that number."""
    target = load_target(target_id)
    samples = nominal(target_id).samples
    lightness = [samples[p.patch_id].lab[0] for p in target.ramp()]
    assert len(lightness) == 12
    assert lightness == sorted(lightness, reverse=True)


def test_the_ngt_ramp_is_not_in_reading_order():
    """The trap this ordering exists to avoid. If the ladder happened to run in
    reading order, ordering by density would be an untested no-op and a definition
    that got it wrong would still pass the test above."""
    ramp = load_target(NGT).ramp()
    by_ladder = [p.patch_id for p in ramp]
    by_position = [p.patch_id for p in sorted(ramp, key=lambda p: (p.row, p.col))]
    assert by_ladder != by_position
    # Specifically: row 6 runs light to dark and row 2 dark to light, so the two
    # interleave rather than following one another.
    assert by_ladder[0] == "D6" and by_ladder[-1] == "I6"


@pytest.mark.parametrize("target_id", IT8_PAIR)
def test_the_it8_ramp_is_the_neutral_column_of_the_layout(target_id):
    """Column 16, rows A-L. Worth pinning rather than trusting: it is the only
    column whose chroma says it is neutral, and the next lowest is nearly three
    times as chromatic. Among columns 1-19, which are the layout's; 20-22 are each
    vendor's own and the fallback values no longer carry them."""
    target = load_target(target_id)
    assert [p.patch_id for p in target.ramp()] == [f"{r}16" for r in "ABCDEFGHIJKL"]

    samples = nominal(target_id).samples
    def mean_chroma(column: int) -> float:
        return float(np.mean([
            np.hypot(*samples[f"{row}{column}"].lab[1:]) for row in "ABCDEFGHIJKL"
        ]))

    others = [mean_chroma(c) for c in range(1, 20) if c != 16]
    assert mean_chroma(16) < min(others) / 2


# -- reference data ---------------------------------------------------------


@pytest.mark.parametrize("target_id", NEW_TARGETS)
def test_every_patch_has_a_reference_value_and_nothing_is_left_over(target_id):
    """A patch with no reference value is dropped from every colour metric, and a
    reference sample with no patch is measured by nothing. Both are silent.

    The IT8s' grayscale strip used to be the deliberate exception: they declare
    it, and OpenDICE's own profile carries no values for it at all. That gap is
    closed now -- both nominal files carry GS0-GS23 fetched directly from each
    chart's LaserSoft batch file, so there is nothing left over either way.
    The exception now is the IT8s' columns 20-22, which the standard leaves to each
    vendor and which no fallback can describe for anybody else's chart -- left out
    on purpose, and tested as exactly that set below. Nothing is ever left over the
    other way: a reference sample with no patch would be measured by nothing at all
    and say so to no one."""
    target = load_target(target_id)
    referenced = set(nominal(target_id).samples)
    declared = set(target.patch_ids)
    assert referenced <= declared, "a reference sample with no patch to measure it"
    vendor = ({f"{row}{col}" for row in "ABCDEFGHIJKL" for col in (20, 21, 22)}
              if target_id.startswith("it8_") else set())
    assert declared - referenced == vendor


@pytest.mark.parametrize("target_id", NEW_TARGETS)
def test_the_shipped_reference_is_declared_nominal(target_id):
    """It decides whether a run may carry a conformance claim, so it is not a
    detail: these are batch averages and a run against them is advisory."""
    assert nominal(target_id).metadata["DELTACHROME_REFERENCE_BASIS"] == "manufacturer_nominal"


def test_the_two_ngts_reference_files_are_not_interchangeable():
    """The reason each gets its own definition. Sharing a lattice is exactly what
    makes the confusion possible and exactly what makes it invisible."""
    ngt, dt = nominal(NGT).samples, nominal(DT_NGT).samples
    assert set(ngt) == set(dt), "the two charts do use the same patch ids"

    shared = sorted(ngt)
    difference = delta_e(
        np.array([ngt[p].lab for p in shared]),
        np.array([dt[p].lab for p in shared]),
        "delta_e_2000",
    )
    # 2.4 dE2000 on average, and the interesting part is that this is NOT enough to
    # fail a 3.5 threshold by itself. A swapped file would eat two thirds of the
    # budget and leave a plausible-looking result -- which is precisely why the
    # mistake has to be prevented by the definitions rather than caught by the
    # verdict. Individual patches go far past it: the worst here is over 30.
    assert float(np.mean(difference)) > 2.0
    assert float(np.max(difference)) > 20.0


def test_the_ngts_neutrals_are_printed_to_different_lightness():
    """The single clearest difference between the two charts, and the one that
    makes a swapped reference file show up in tone response as well as colour."""
    whites = ["A0", "A4", "A9", "G0", "G9", "M0", "M9"]
    ngt = np.mean([nominal(NGT).samples[p].lab[0] for p in whites])
    dt = np.mean([nominal(DT_NGT).samples[p].lab[0] for p in whites])
    assert dt - ngt > 3.0


# -- geometry ---------------------------------------------------------------


@pytest.mark.parametrize("target_id", IT8_PAIR)
def test_it8_patches_are_butted_so_the_inset_is_all_that_holds_the_roi_off(target_id):
    grid = next(r for r in load_target(target_id).regions
                if r.region_id == "colour_array").grid
    assert grid.patch_size_relative == grid.patch_spacing_relative

    # Neighbours at full size share an edge exactly; at the sampling inset they do
    # not. If patch size were ever measured as something smaller than the spacing
    # on this chart, that would be a mis-measurement rather than a margin.
    touching = grid.corners_relative(0, 0, inset=1.0)
    neighbour = grid.corners_relative(0, 1, inset=1.0)
    assert touching[1][0] == pytest.approx(neighbour[0][0], abs=1e-9)

    inset = grid.corners_relative(0, 0, inset=0.55)
    inset_neighbour = grid.corners_relative(0, 1, inset=0.55)
    assert inset[1][0] < inset_neighbour[0][0]


@pytest.mark.parametrize("target_id", NEW_TARGETS)
def test_no_sampling_roi_reaches_outside_the_placed_quad(target_id):
    """Every one of these targets is placed on its own array or on marks just
    outside it, so an ROI leaving the 0..1 box would be sampling the mount.

    The IT8 grayscale strip is the one thing that is genuinely outside it, because
    it is printed below the array the quad is placed on. It gets its own bounds
    rather than an exemption, so a strip that drifted would still be caught."""
    target = load_target(target_id)
    for patch in target.patches:
        strip = patch.patch_id.startswith("GS")
        for u, v in target.patch_corners_relative(patch, inset=0.55):
            assert -0.06 <= u <= 1.06, patch.patch_id
            if strip:
                assert 1.05 <= v <= 1.27, patch.patch_id
            else:
                assert -0.02 <= v <= 1.02, patch.patch_id


@pytest.mark.parametrize("target_id", NEW_TARGETS)
def test_the_geometry_is_measured_rather_than_provisional(target_id):
    target = load_target(target_id)
    assert not target.geometry_is_provisional
    raw = json.loads(target.source_path.read_text())
    grids = ([r["patch_grid"] for r in raw["regions"] if r.get("patch_grid")]
             if raw.get("regions") else [raw["patch_grid"]])
    measured = [g for g in grids if g.get("measured_from", {}).get("capture")]
    assert measured, "how it was measured has to be recorded"


# -- capability claims ------------------------------------------------------


@pytest.mark.parametrize("target_id", NEW_TARGETS)
def test_none_of_these_charts_claims_a_feature_it_does_not_print(target_id):
    """No slanted edges and no printed scale on any of the four, so SFR,
    registration and reproduction scale are disclaimed rather than left unsaid."""
    target = load_target(target_id)
    assert {"sfr", "color_registration", "reproduction_scale"} <= target.not_capable_of
    assert "color_accuracy" in target.capabilities


def test_only_the_ngts_can_measure_lightness_uniformity():
    """The IT8s do have a set of near-identical neutrals -- A13 to A19 -- and it is
    no use, because the five sit adjacent in one row. Uniformity needs neutrals
    spread over the chart's area, and measuring a six-centimetre run of print would
    report the chart's own consistency as the system's illumination."""
    for target_id in NGT_PAIR:
        assert "lightness_uniformity" in load_target(target_id).capabilities
    for target_id in IT8_PAIR:
        assert "lightness_uniformity" in load_target(target_id).not_capable_of


def test_a_disclaimed_capability_is_one_the_engine_actually_gates_on():
    """`not_capable_of` is documentation: the engine skips a metric on the strength
    of `capabilities` alone. So disclaiming a capability no profile gates on is a
    promise the engine will not keep -- the metric runs anyway and the definition
    says it cannot.

    Checked across every installed target, not just the new ones, because the rule
    is about the schema rather than about these four charts. `noise` is the live
    example: it is a known capability that no shipped profile requires (they all
    gate noise on `tonal_response`), so a target disclaiming it would be measured
    for noise regardless.
    """
    gated = {
        spec.requires_capability
        for profile_id in available_profiles()
        for spec in load_profile(profile_id).metrics
        if spec.requires_capability
    }
    for target_id in available_targets():
        target = load_target(target_id)
        ungated = sorted(target.not_capable_of - gated)
        assert not ungated, (
            f"{target_id} disclaims {ungated}, which no shipped profile gates on. "
            "The metric would run anyway; either a profile should require the "
            "capability, or the definition should qualify the reading in "
            "metric_defaults instead of disclaiming it."
        )


def test_every_capability_is_gated_by_some_profile():
    """The symmetric case of the test above, on the *positive* declarations. A capability nothing gates on is inert whichever way a target
    states it: ``noise`` was declared by nine targets and required by zero
    shipped profiles (they all gate the noise metric on ``tonal_response``
    instead), so the declaration changed nothing and was retired from the
    vocabulary. A material-specific caveat about what a reading means -- the
    IT8.7/1's film grain, still qualified in its own ``metric_defaults`` -- is
    the right home for that; a capability flag nothing reads is not.
    """
    gated = {
        spec.requires_capability
        for profile_id in available_profiles()
        for spec in load_profile(profile_id).metrics
        if spec.requires_capability
    }
    for target_id in available_targets():
        target = load_target(target_id)
        ungated = sorted(set(target.capabilities) - gated)
        assert not ungated, (
            f"{target_id} declares {ungated}, which no shipped profile requires. "
            "The declaration changes nothing; either a profile should require "
            "the capability, or it should not be in KNOWN_CAPABILITIES at all."
        )


@pytest.mark.parametrize("target_id", ["it8_7_1_transmissive_v1", "it8_7_2_reflective_v1"])
def test_the_it8_fallback_values_leave_the_vendor_columns_out(target_id):
    """IT8.7 fixes columns 1-19 and leaves A20-L22 to each manufacturer, so values
    for them are true of one vendor's chart and no other's. Against a Wolf Faust
    IT8.7/2's own measurement the LaserSoft values sit 3.8 dE2000 off in columns
    1-19 and 31.3 in 20-22. A fallback carrying them would fail every other
    vendor's chart on patches that were never meant to match; a chart's own file
    carries them, and they are judged then."""
    target = load_target(target_id)
    nominal = read_reference(target.nominal_reference_path)
    vendor = {f"{row}{col}" for row in "ABCDEFGHIJKL" for col in (20, 21, 22)}

    assert vendor <= {p.patch_id for p in target.patches}
    assert not vendor & set(nominal.samples)
    assert {f"{row}{col}" for row in "ABCDEFGHIJKL" for col in range(1, 20)} <= set(nominal.samples)
