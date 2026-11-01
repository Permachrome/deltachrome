"""Targets built from several separated groups.

A ColorChecker SG is one uniform lattice and needed nothing more than a single
grid. An ISA Golden Thread is not like that: four patch strips of differing length
at four positions, plus five slanted-edge modules, separated by printed text. Row
and column indices only mean something *within* one lattice, so each group carries
its own -- which is what the regions model exists for.
"""

from __future__ import annotations

import json

import pytest

from deltachrome.errors import TargetError
from deltachrome.targets.registry import load_target
from deltachrome.targets.schema import DEFAULT_REGION, Region, target_from_dict

SG = "colorchecker_sg_v1"
ISA = "isa_golden_thread_dlt_v1"
#: A target that still uses the top-level ``patch_grid`` shortcut. The SG used to be
#: the example here and no longer is: declaring its printed scale meant declaring
#: regions, and the two forms are mutually exclusive. The shortcut itself is
#: unchanged, and a definition written against it must keep working.
SHORTCUT = "avian_ngt_v1"


# -- the single-lattice case still reads simply ----------------------------


def test_a_target_declaring_one_patch_grid_becomes_one_region():
    """Existing definitions must keep working unchanged, and simple targets must
    stay simple to write."""
    target = load_target(SHORTCUT)
    assert [r.region_id for r in target.regions] == [DEFAULT_REGION]
    assert target.patch_grid is not None
    assert {p.region for p in target.patches} == {DEFAULT_REGION}


def test_patches_with_no_region_of_their_own_join_the_first_grid():
    """What makes adding a second region to a single-lattice target a small edit
    rather than a rewrite of its patch list: the SG declares a patch grid and a
    printed scale, and none of its 140 patches names a region."""
    target = load_target(SG)
    assert [r.region_id for r in target.regions] == ["patches", "ruler_mm"]
    assert {p.region for p in target.patches} == {"patches"}
    raw = json.loads(load_target(SG).source_path.read_text(encoding="utf-8"))
    assert not any("region" in p for p in raw["patches"])


def test_patch_grid_shortcut_is_none_for_a_multi_region_target():
    """There is no one grid, and code that assumed otherwise should be asking a
    patch for its own region instead of silently getting the first."""
    assert load_target(ISA).patch_grid is None
    # Four color/neutral strips plus five 2x2 sets of registration squares.
    assert len(load_target(ISA).grid_regions) == 9


def test_geometry_comes_from_the_patch_s_own_region():
    target = load_target(ISA)
    left = target["22"]        # top of the far-left color strip
    right = target["1"]        # top of the far-right color strip
    assert left.region != right.region
    assert target.center_relative(left)[0] < 0.4
    assert target.center_relative(right)[0] > 0.6


def test_row_indices_are_local_to_a_region():
    """Row 0 exists in every strip and means a different place in each."""
    target = load_target(ISA)
    firsts = [target["22"], target["16"], target["10"], target["1"]]
    assert all(p.row == 0 for p in firsts)
    xs = [target.center_relative(p)[0] for p in firsts]
    assert len(set(round(x, 3) for x in xs)) == 4


# -- validation -------------------------------------------------------------


def test_a_patch_in_an_undeclared_region_is_refused():
    raw = json.loads(load_target(SG).source_path.read_text(encoding="utf-8"))
    raw["patches"][0]["region"] = "nowhere"
    with pytest.raises(TargetError, match="undeclared region"):
        target_from_dict(raw)


def test_a_duplicated_region_id_is_refused():
    raw = json.loads(load_target(ISA).source_path.read_text(encoding="utf-8"))
    raw["regions"].append(dict(raw["regions"][0]))
    with pytest.raises(TargetError, match="region id twice"):
        target_from_dict(raw)


def test_a_grid_region_without_geometry_is_refused():
    with pytest.raises(TargetError, match="has no grid"):
        Region(region_id="x", kind="patch_grid", grid=None)


def test_an_unknown_region_kind_is_refused():
    with pytest.raises(TargetError, match="expected one of"):
        Region(region_id="x", kind="wishful_thinking")


def test_asking_a_non_grid_region_for_patch_geometry_is_an_error():
    target = load_target(ISA)
    edge = next(r for r in target.regions if r.kind == "slanted_edge")
    assert edge.grid is None
    assert not edge.holds_patches


# -- the ISA definition -----------------------------------------------------


def test_the_isa_target_has_four_strips_of_the_right_lengths():
    target = load_target(ISA)
    counts = {r.region_id: r.grid.rows for r in target.grid_regions}
    strips = {k: v for k, v in counts.items() if not k.startswith("uniformity_")}
    assert strips == {"color_right": 9, "highlight": 6, "gray": 6, "color_left": 9}
    # Each uniformity region is a 2x2: four squares at one module's corners.
    assert {k: v for k, v in counts.items() if k.startswith("uniformity_")} == {
        f"uniformity_{m}": 2 for m in ("tl", "tr", "center", "bl", "br")
    }
    assert len(target.patches) == 50
    printed = {p.patch_id for p in target.patches if p.region.startswith(("color", "gray", "highlight"))}
    assert printed == {str(i) for i in range(1, 31)}


def test_each_strip_is_a_single_column():
    for region in load_target(ISA).grid_regions:
        if region.region_id.startswith("uniformity_"):
            continue           # four squares at a module's corners, a 2x2
        assert region.grid.columns == 1


def test_the_neutral_strips_are_declared_as_the_ramp():
    ramp = load_target(ISA).ramp()
    assert [p.patch_id for p in ramp] == [str(i) for i in range(10, 22)]
    assert {p.region for p in ramp} == {"highlight", "gray"}


def test_the_slanted_edge_modules_are_declared_even_though_nothing_reads_them():
    """Silence would misrepresent the chart as lacking the feature."""
    target = load_target(ISA)
    edges = [r for r in target.regions if r.kind == "slanted_edge"]
    assert len(edges) == 5
    assert {r.geometry["position"] for r in edges} == {"TL", "TR", "BR", "BL", "CENTER"}
    assert target.supports("sfr")          # the chart can; the engine cannot yet


def test_the_isa_target_declares_its_rulers():
    """Twenty of them: a centimetre and an inch rule on all four sides of each of
    the five modules. The chart carries its own dimensional reference, so its
    sampling rate is measured rather than declared -- and the spread between the
    twenty is what its geometric distortion is read from."""
    target = load_target(ISA)
    scales = [r for r in target.regions if r.kind == "scale"]
    assert len(scales) == 20
    assert {r.geometry["unit"] for r in scales} == {"mm", "in"}
    assert {r.geometry["axis"] for r in scales} == {"u", "v"}
    assert all(r.geometry["bounds_relative"] for r in scales)
    assert target.dimensional_reference["available"] is True


def test_the_isa_definition_says_how_its_patch_ids_were_confirmed():
    """Published per-unit measurements now confirm the patch-id mapping; the
    definition says so, and still names the printed suffixes nobody has explained."""
    notes = load_target(ISA).notes
    assert "confirmed against published per-unit measurements" in notes
    assert "11(A)" in notes


def test_the_nominal_values_are_carried_as_nominal_not_as_a_measurement():
    """Whatever the nominal is derived from, it is not a measurement of this
    specimen, and the definition has to keep those two things apart rather than
    collapsing them.

    The wording this used to assert -- that these targets are "seldom bought
    measured" -- was true when written and is not any more: Digital Transitions
    publish per-unit measurements for 1,454 Device Level units under the
    OpenQualia label standard. So the test now pins the property that survives
    (this is nominal, not a specimen measurement) and, in place of the obsolete
    claim, that the definition points at the per-unit route which now exists.
    """
    from deltachrome.reference import read_reference

    target = load_target(ISA)
    nominal = target.nominal_reference
    assert nominal["basis"] == "manufacturer_nominal"
    assert "not a measurement of any individual chart" in nominal["notes"]
    assert "per-unit file is preferred" in nominal["notes"]

    path = target.nominal_reference_path
    assert path is not None and path.exists()
    reference = read_reference(path)
    assert reference.is_nominal
    assert len(reference) == 30
    # The twenty registration squares are deliberately absent: the chart prints no
    # values for them, and uniformity does not need any.
    assert len(target.patches) == 50
    present, missing = reference.coverage(target.patch_ids)
    assert len(present) == 30
    assert all(target[pid].role == "achromatic_border" for pid in missing)
    assert reference.illuminant == "D50" and reference.observer == "2"
    assert all(
        p.role in ("achromatic_ramp", "unassigned", "achromatic_border")
        for p in target.patches
    )


def test_the_nominal_values_are_internally_consistent():
    """A transcription error would show as a neutral out of order or a neutral that
    is not neutral. Both are cheap to check and neither can be checked later."""
    import numpy as np

    from deltachrome.reference import read_reference

    target = load_target(ISA)
    reference = read_reference(target.nominal_reference_path)
    ramp = [p for p in target.patches if p.role == "achromatic_ramp"]
    ramp.sort(key=lambda p: p.ramp_position)
    lightness = [reference[p.patch_id].lab[0] for p in ramp]
    assert lightness == sorted(lightness, reverse=True), lightness
    assert lightness[0] > 95 and lightness[-1] < 5
    chroma = [float(np.hypot(*reference[p.patch_id].lab[1:])) for p in ramp]
    assert max(chroma) < 2.0, chroma


def test_the_registration_squares_are_the_uniformity_sample():
    """Four per module across five modules: twenty points reaching every corner of
    the chart and its center, which is a better areal-coverage sample than either
    other target here offers."""
    target = load_target(ISA)
    assert target.supports("lightness_uniformity")
    squares = [p for p in target.patches if p.role == "achromatic_border"]
    assert len(squares) == 20
    assert len({p.region for p in squares}) == 5

    defaults = target.metric_defaults["lightness_uniformity"]
    assert defaults["sample_point_count"] == 20
    # The chart prints no values for them, and the definition has to say so rather
    # than leave a reader to wonder why no chart-corrected figure appears.
    assert "PRINTS NO VALUES" in defaults["notes"]
