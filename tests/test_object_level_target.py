"""The Golden Thread Object Level Target: a ruled strip, and what that costs.

A strip meant to lie beside the object rather than fill the frame. Thirty patches
in one row, a resolution module in the middle, an inch rule and a centimetre rule,
and — unusually — its own reference table printed beside the patches.

It is the only target here whose geometry was measured on **two different physical
units**, and the two are different versions of the product: the current one carries
a yellow border, a serial box and an "All values are batch averages" caption, and
the older one prints a **third fiducial disc at the centre** which the revamp
dropped. What makes one definition right for both is that the design underneath is
the same — the end discs agree on position to 0.0003 of the strip, and the two
printed reference tables differ by 0.99 ΔE2000, which is batch drift rather than a
redesign.

Two things here are unlike every other target and are what the tests are about:

* **The quad is 34 pitches long and one patch tall.** Nothing else in this project
  is that elongated, and it has a consequence: a strip laid down half a degree off
  square drifts a quarter of a patch height end to end, and features off the patch
  row run out of their declared bands. The remedy is to place along the strip's own
  axis, which clicking the real corners does.
* **One module means one field position**, and its block is shallow enough that the
  default ROI half-width takes in the chart's own printed value table.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from deltachrome.reference import read_reference
from deltachrome.targets.registry import load_target

TARGET = "isa_object_level_1x_v1"
#: Slots the resolution module occupies, of the strip's thirty-four.
MODULE_SLOTS = {15, 16, 17, 18}


def target():
    return load_target(TARGET)


def raw() -> dict:
    return json.loads(target().source_path.read_text(encoding="utf-8"))


def nominal():
    return read_reference(target().nominal_reference_path)


# -- the strip ---------------------------------------------------------------


def test_thirty_patches_on_a_single_row_of_thirty_four_slots():
    """The module takes four slots in the middle. Modelling the strip as thirty
    contiguous patches would put every patch after the module four places out."""
    grid = target().patch_grid
    assert (grid.rows, grid.columns) == (1, 34)

    patches = target().patches
    assert len(patches) == 30
    assert [p.patch_id for p in patches] == [str(n) for n in range(1, 31)]
    assert {p.col for p in patches} == set(range(34)) - MODULE_SLOTS
    assert {p.row for p in patches} == {0}


def test_the_patches_are_butted_so_the_inset_is_all_that_holds_the_roi_off():
    grid = target().patch_grid
    assert grid.patch_size_relative[0] == pytest.approx(grid.patch_spacing_relative[0])
    assert grid.patch_size_relative[0] == pytest.approx(1 / 34, abs=1e-6)
    # The quad is the patch row itself, so a patch fills its whole height.
    assert grid.patch_size_relative[1] == pytest.approx(1.0)


def test_the_ramp_is_twelve_steps_either_side_of_the_module():
    """Patches 10-15 and 16-21: the six before the module and the six after."""
    ramp = target().ramp()
    assert [p.patch_id for p in ramp] == [str(n) for n in range(10, 22)]
    lightness = [nominal().samples[p.patch_id].lab[0] for p in ramp]
    assert lightness == sorted(lightness, reverse=True)
    # They straddle the module rather than sitting on one side of it.
    assert min(p.col for p in ramp) < min(MODULE_SLOTS)
    assert max(p.col for p in ramp) > max(MODULE_SLOTS)


@pytest.mark.parametrize("patch_id", ["1", "30"])
def test_placement_keys_on_the_end_patches(patch_id):
    markers = raw()["fiducials"]["corner_markers"]
    assert target().fiducials.method == "patch_array_corners"
    assert patch_id in {m["nearest_patch"] for m in markers}


# -- the discs, and the two versions -----------------------------------------


def test_the_end_discs_are_recorded_but_not_used_for_placement():
    """They are the best-registered features on the target -- agreeing between two
    physical units to 0.0003 of the strip -- and there are only two of them, which
    cannot define a quad. So they are recorded for a future detector, not used."""
    marks = raw()["fiducials"]["printed_marks"]
    assert marks["used_for_placement"] is False
    discs = marks["discs_relative"]
    assert discs["left_end"][0] == pytest.approx(-0.0215, abs=0.002)
    assert discs["right_end"][0] == pytest.approx(1.0214, abs=0.002)


def test_the_older_version_prints_a_third_disc_at_the_centre():
    """ISA's revamp dropped it. Recorded because a definition that mentioned only
    two would misdescribe every older card, and because the centre disc is exactly
    what an automatic placement would want."""
    discs = raw()["fiducials"]["printed_marks"]["discs_relative"]
    centre = discs["centre_older_version_only"]
    assert centre[0] == pytest.approx(0.5, abs=0.002), "dead centre of the strip"
    # Not collinear with the end pair: it sits above the patch row, so the three
    # together fix rotation as well as position and scale.
    assert centre[1] < min(discs["left_end"][1], discs["right_end"][1]) - 0.5


def test_one_definition_covers_both_versions():
    notes = raw()["notes"]
    assert "TWO VERSIONS EXIST AND THIS DEFINITION SERVES BOTH" in notes
    measured = next(r for r in raw()["regions"] if r["region_id"] == "patches")
    assert len(measured["patch_grid"]["measured_from"]["pitch_px"]) == 2, "two units"


# -- the module --------------------------------------------------------------


def test_the_module_declares_two_edges_from_two_separate_figures():
    module = next(r for r in target().regions if r.kind == "slanted_edge")
    edges = module.geometry["straight_edges"]
    assert len(edges) == 2
    assert {e["orientation"] for e in edges} == {"horizontal", "vertical"}


def test_both_module_edges_narrow_their_own_roi():
    """The module packs a slanted band, a slanted block and a white resolution
    field into about a sixtieth of the strip, so the 50 px default reaches a
    neighbouring feature from either edge. Each one needed the override for its
    own reason and both landed on the same width.

    The horizontal edge: the block above it is 60 px deep and the chart's printed
    value table starts 25 px below it, and at 50 px the edge-line fit scattered
    17.7 px instead of 0.06 and peak modulation read 1.624.

    The vertical edge: its right-hand plateau is only about 48 px wide before the
    white 'dots per inch' field begins, so at 50 px that field lands INSIDE the
    ROI as a second edge. Overshoot read +620% on one specimen and +2626% on the
    other, and the second also warned that located edge positions scattered
    9.53 px about the fitted line. From 35 px down both specimens read exactly
    0.0000, while MTF50 moves by 0.003 across the whole sweep -- so the artefact
    was costing nothing except being reported as a real overshoot, which is the
    kind of wrong answer that looks like a finding."""
    module = next(r for r in target().regions if r.kind == "slanted_edge")
    edges = {e["id"]: e for e in module.geometry["straight_edges"]}
    assert edges["h"]["roi_half_width"] == 35
    assert edges["v"]["roi_half_width"] == 35


def test_the_horizontal_edge_is_declared_no_longer_than_the_block_it_bounds():
    """A declaration that overshoots its feature does not put the ROI off the
    feature -- `length_fraction` trims 45% of it away -- but it does put the ROI
    off the feature's MIDDLE, and it spends the margin that a differently printed
    specimen would need. The block ends at u 0.526 on one unit and 0.538 on the
    other; the declaration used to run to 0.5413."""
    module = next(r for r in target().regions if r.kind == "slanted_edge")
    edges = {e["id"]: e for e in module.geometry["straight_edges"]}
    start, end = edges["h"]["start"], edges["h"]["end"]
    assert start[0] >= 0.472 and end[0] <= 0.532
    # 0.55 of the declared span, centred, is what is actually measured.
    span = end[0] - start[0]
    used = (start[0] + 0.225 * span, end[0] - 0.225 * span)
    assert 0.474 < used[0] and used[1] < 0.526, "inside the block on both specimens"


def test_one_module_means_one_field_position():
    """Not a shortcoming to hide: it is what an object level target is."""
    modules = [r for r in target().regions if r.kind == "slanted_edge"]
    assert len(modules) == 1
    assert {"sfr", "color_registration"} <= target().capabilities


# -- the rules ---------------------------------------------------------------


def scales(pattern=None):
    """The chart's declared scales, optionally of one kind.

    Three now: the two printed rules, and the checkerboard above the metric one.
    They are all ``kind: "scale"`` because they are all the same measurement --
    pixels per millimetre -- and differ only in what the reader looks for.
    """
    found = [r for r in target().regions if r.kind == "scale"]
    if pattern is None:
        return found
    return [r for r in found if r.geometry.get("pattern", "ticks") == pattern]


def test_two_rules_in_two_units_cross_check_the_sampling_rate():
    """A scale misread by a factor still fits a beautiful straight line through its
    own ticks, so agreement between independently engraved scales is the only
    check available that each was read as the chart intends."""
    assert len(scales("ticks")) == 2
    assert {r.geometry["unit"] for r in scales("ticks")} == {"in", "cm"}


def test_the_checkerboard_is_declared_as_a_third_scale():
    """It marks the same centimetre divisions as the rule beneath it, with filled
    cells instead of thin marks -- and it is a far better feature to measure on:
    across three units it fits about thirty times more regularly than the ticks.
    On one capture here it is the only one of the two metric features that reads
    at all, its millimetre comb scattering 10% and its 10 mm majors being
    impossible to isolate at 600 ppi the way they are at 400."""
    blocks = scales("blocks")
    assert len(blocks) == 1
    checker = blocks[0].geometry
    assert checker["division_mm"] == 10.0, "one centimetre cells"
    assert checker["unit"] == "cm", "same unit family as the rule it sits above"
    # The cell size is inferred and the definition has to say so, because every
    # reading from this region scales with it.
    assert "INFERRED, NOT PUBLISHED" in checker["measured_note"]


def test_the_centimetre_rule_is_declared_on_its_ten_millimetre_majors():
    """The 1 mm comb reads correctly on either specimen alone and its band sits
    four to six pixels differently between them, so no single declared band serves
    both. The majors do."""
    cm = next(r for r in target().regions
              if r.kind == "scale" and r.geometry["unit"] == "cm")
    assert cm.geometry["division_mm"] == 10.0
    assert "1 mm comb" in cm.geometry["measured_note"]


# -- what it cannot do -------------------------------------------------------


def test_uniformity_and_distortion_are_disclaimed_with_reasons():
    """Both are judgements rather than absences, so both are stated. Uniformity
    needs repeated neutrals and the thirty patches include none; distortion needs
    scale at several places on both axes and the strip's rules run along one."""
    disclaimed = target().not_capable_of
    assert {"lightness_uniformity", "geometric_distortion"} <= disclaimed

    labs = np.array([s.lab for s in nominal().samples.values()])
    neutral = labs[np.hypot(labs[:, 1], labs[:, 2]) <= 4.0]
    lightness = np.sort(neutral[:, 0])
    assert min(np.diff(lightness)) > 2.0, "no two neutrals repeat"

    axes = {r.geometry["axis"] for r in target().regions if r.kind == "scale"}
    assert axes == {"u"}, "every rule runs along the strip"


def test_the_quad_extent_and_the_card_size_are_different_quantities():
    """physical_size_mm is ISA's published dimension for the whole card, 9.25:1.
    The quad the four placed corners bound is 34:1. Supplying the extent matters
    more here than on most targets, because the fallback when it is absent is the
    card, and a 9.25:1 expectation against a 34:1 placement makes the engine's
    aspect check -- which exists to catch corners clicked in the wrong order --
    meaningless."""
    extent = target().fiducial_extent_mm
    card = target().physical_size_mm
    assert card == (234.95, 25.4)
    assert extent == (214.86, 6.32)
    assert extent[0] / extent[1] == pytest.approx(34.0, abs=0.1)
    assert card[0] / card[1] == pytest.approx(9.25, abs=0.05)


def test_the_extent_follows_from_a_measured_disc_separation():
    """224 mm centre-to-centre between the end discs, measured on a physical
    specimen of the current version. Those discs sit 1.04252 quad widths apart on
    that version, so the quad is 214.86 mm and the patch pitch a thirty-fourth of
    it. Recomputed here so the arithmetic cannot drift from the note beside it."""
    measured_disc_separation_mm = 224.0
    v2_disc_separation_in_quad_widths = 1.02123 - (-0.02129)
    width = measured_disc_separation_mm / v2_disc_separation_in_quad_widths
    assert target().fiducial_extent_mm[0] == pytest.approx(width, abs=0.02)
    assert width / 34 == pytest.approx(6.32, abs=0.01), "patch pitch in mm"

    # Corroborated rather than circular: nothing from the captures fed into it, and
    # a block that long leaves 10 mm at each end of ISA's published card.
    card_length = target().physical_size_mm[0]
    assert (card_length - width) / 2 == pytest.approx(10.0, abs=0.2)


def test_the_printed_rules_remain_the_authority_for_sampling_rate():
    """The extent gives the geometry record an effective_ppi, and on a 34:1 quad
    that figure is weak: it averages the width- and height-derived rates equally,
    and the height rests on a hundred-pixel baseline. The rules are read per
    capture and need no assumption about which specimen this is."""
    notes = raw()["dimensional_reference"]["notes"]
    assert "printed rules remain the authority" in notes
    assert len(scales("ticks")) == 2
