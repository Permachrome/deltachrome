"""The Image Engineering UTT, and the things about it that are easy to get wrong.

Third kind of target definition, and the one that stretched the schema furthest:
twenty-eight regions rather than one lattice, four grey scales that are the same
printed bar in four orientations, and a uniformity set taken from the chart's
background rather than from patches printed to be measured.

Two of those are traps rather than features, and both are tested here because
neither announces itself in a result:

* Two of the four grey scales run **against** their axis, so a definition that
  numbers all four from the low-coordinate end reads two of them backwards. Tone
  response then compares L* 95 against an aim of 3 and fails enormously, which
  looks like a broken scanner rather than a broken definition.
* The four fiducial corners are matched positionally against the points an operator
  places, so listing them in the wrong order maps the chart onto a bow tie. That one
  produces no error at all -- see :func:`test_a_bow_tie_corner_order_is_refused`.
"""

from __future__ import annotations

import json

import pytest

from deltachrome.targets.registry import load_target
from deltachrome.targets.schema import TargetError, target_from_dict

UTT = "image_engineering_utt_a3_v1"
RAMPS = ("U", "D", "L", "R")


def raw() -> dict:
    return json.loads(load_target(UTT).source_path.read_text(encoding="utf-8"))


# -- what the chart is ------------------------------------------------------


def test_the_chart_carries_colour_tone_resolution_and_a_scale():
    """The point of the UTT: one sheet that answers a whole specification, which is
    why Metamorfoze and ISO 19264-1 are both written around it."""
    target = load_target(UTT)
    assert {
        "color_accuracy", "tonal_response", "white_balance", "lightness_uniformity",
        "sfr", "color_registration", "reproduction_scale", "geometric_distortion",
    } <= target.capabilities
    # It is a chart on a sheet, not a frame-filling field.
    assert "flat_field" in target.not_capable_of


def test_the_two_colour_bars_carry_the_same_27_colours_at_two_field_positions():
    ids = set(load_target(UTT).patch_ids)
    assert {f"{c}{r}" for c in "ABCDEF" for r in range(1, 10)} <= ids


def test_there_are_thirty_six_slanted_edges_at_nine_field_positions():
    regions = [r for r in load_target(UTT).regions if r.kind == "slanted_edge"]
    assert len(regions) == 9
    assert sum(r.geometry["edges"] for r in regions) == 36
    # Every square is tilted the same way and by an angle ISO 12233 contemplates.
    assert all(3.0 <= r.geometry["tilt_deg"] <= 7.0 for r in regions)


def test_the_modules_sit_on_a_three_by_three_lattice():
    """Nine positions, so the field is sampled at the corners, the edges and the
    centre -- which is what makes an edge-to-edge spread mean something."""
    centers = [
        tuple(r.geometry["center"]) for r in load_target(UTT).regions
        if r.kind == "slanted_edge"
    ]
    assert len({round(u, 2) for u, _ in centers}) == 3
    assert len({round(v, 2) for _, v in centers}) == 3


# -- the grey scales --------------------------------------------------------


def test_all_four_grey_scales_are_twenty_steps_of_the_tonal_ramp():
    target = load_target(UTT)
    for code in RAMPS:
        steps = [target[f"GS_{code}_{i}"] for i in range(1, 21)]
        assert all(p.role == "achromatic_ramp" for p in steps)
        assert [p.ramp_position for p in steps] == list(range(1, 21))
    assert len(target.by_role("achromatic_ramp")) == 80


@pytest.mark.parametrize("code,axis,descending", [
    ("U", 0, True),    # upper horizontal: step 1 at the right
    ("D", 0, False),   # lower horizontal: step 1 at the left
    ("L", 1, True),    # left vertical: step 1 at the bottom
    ("R", 1, False),   # right vertical: step 1 at the top
])
def test_two_of_the_four_ramps_run_against_their_axis(code, axis, descending):
    """The four scales are one bar printed four times with 180-degree rotational
    symmetry, so the light end of two of them sits at the *high* end of its
    coordinate. Numbering all four the same way reads two backwards, and tone
    response then measures L* 95 against an aim of L* 3.

    Confirmed against a vendor report on the same capture: read in this order, all
    four reproduce iQ-Analyzer's own per-step L* to about 0.1 in the highlights.
    """
    target = load_target(UTT)
    region = target.region_for(target[f"GS_{code}_1"])
    position = [
        region.grid.center_relative(target[f"GS_{code}_{i}"].row,
                                    target[f"GS_{code}_{i}"].col)[axis]
        for i in range(1, 21)
    ]
    assert (position[0] > position[-1]) is descending
    # and monotone either way, so the steps are in order rather than merely at the ends
    assert sorted(position, reverse=descending) == position


def test_the_ramps_are_the_same_bar_in_four_orientations():
    """A cross-check on the measured geometry rather than a property of the chart: if
    the fiducial quad's aspect were wrong, the horizontal pair and the vertical pair
    would come out different lengths."""
    target = load_target(UTT)
    lengths = []
    for code, axis in (("U", 0), ("D", 0), ("L", 1), ("R", 1)):
        grid = target.region_for(target[f"GS_{code}_1"]).grid
        pitch = grid.patch_spacing_relative[axis]
        # The vertical ramps are measured in v, whose unit is the quad's height.
        lengths.append(20 * pitch * (1.0 if axis == 0 else 1 / 1.4712))
    assert max(lengths) / min(lengths) < 1.01


# -- the uniformity set -----------------------------------------------------


def test_uniformity_comes_from_the_checkerboard_and_needs_no_reference():
    """The chart prints no values for its background squares, and uniformity needs
    none: it is the spread of a set of nominally identical patches about their own
    mean. Same arrangement as the Golden Thread's registration squares."""
    target = load_target(UTT)
    field = target.by_role("achromatic_border")
    assert len(field) == 25
    assert all(p.region == "checker_field" for p in field)

    nominal = target.nominal_reference_path.read_text(encoding="utf-8")
    assert not any(p.patch_id in nominal for p in field)


def test_the_uniformity_samples_reach_the_edges_and_the_corners():
    """A uniformity set clustered in the middle measures the middle. These are one
    per cell of a five-by-five division of the sheet."""
    target = load_target(UTT)
    grid = target.region_for(target.by_role("achromatic_border")[0]).grid
    centers = [
        grid.center_relative(p.row, p.col)
        for p in target.by_role("achromatic_border")
    ]
    us = [u for u, _ in centers]
    vs = [v for _, v in centers]
    assert min(us) < 0.1 and max(us) > 0.9
    assert min(vs) < 0.1 and max(vs) > 0.9


# -- the reference ----------------------------------------------------------


def test_nominal_values_cover_every_patch_that_is_compared_against_one():
    target = load_target(UTT)
    text = target.nominal_reference_path.read_text(encoding="utf-8")
    body = text[text.index("BEGIN_DATA\n") : text.index("END_DATA\n")]
    named = {line.split("\t")[0] for line in body.splitlines()[1:] if line.strip()}
    needed = {
        p.patch_id for p in target.patches if p.role != "achromatic_border"
    }
    assert needed <= named


def test_the_nominal_file_says_it_is_an_aim_and_not_a_measurement():
    """It costs a run its conformance claim, so the file has to carry the basis the
    engine reads and the sentence a person reads."""
    text = load_target(UTT).nominal_reference_path.read_text(encoding="utf-8")
    assert 'DELTACHROME_REFERENCE_BASIS\t"manufacturer_nominal"' in text
    assert "DELTACHROME_NOMINAL_CAVEAT" in text


# -- the traps --------------------------------------------------------------


def test_a_bow_tie_corner_order_is_refused():
    """Swapping two fiducial corners is a valid projective transform onto a bow tie.
    Every ROI still lands inside the image and every one reads about the frame's
    average, so the run comes back complete, plausible, and uniformly grey."""
    definition = raw()
    markers = definition["fiducials"]["corner_markers"]
    markers[2], markers[3] = markers[3], markers[2]
    with pytest.raises(TargetError, match="crosses the chart"):
        target_from_dict(definition)


def test_the_shipped_corner_order_goes_round_the_quad():
    assert [m["id"] for m in raw()["fiducials"]["corner_markers"]] == \
        ["TL", "TR", "BR", "BL"]


def test_distortion_is_claimed_only_with_the_caveat_that_it_is_narrower():
    """The chart's own distortion reference is its checkerboard, and a vendor tool
    reading that reports roughly ten times the number this engine reads off two
    printed rules. Both are right about different things; a definition that claimed
    the capability without saying so would look like it disagreed."""
    defaults = raw()["metric_defaults"]["geometric_distortion"]["notes"]
    assert "checkerboard" in defaults
    assert "narrower" in defaults


def test_the_rules_are_declared_in_segments_along_each_edge():
    """One band spanning a whole rule falls off the ticks -- the rule is not exactly
    parallel to the module lattice -- and a single comb absorbs bowing into its
    pitch. Segments fix both, and give distortion something to compare."""
    scales = [r for r in load_target(UTT).regions if r.kind == "scale"]
    assert len(scales) == 12
    assert {r.geometry["unit"] for r in scales} == {"mm", "in"}
    assert {r.geometry["axis"] for r in scales} == {"u", "v"}
    # Three positions per edge per unit, which _position() reads out of the id.
    assert len({r.region_id.split("_")[1] for r in scales}) == 6


# -- one ladder, four arms --------------------------------------------------


def test_the_four_arms_are_one_twenty_step_ladder_not_eighty_steps():
    """``ramp_position`` is the chart's structure; the measurement is its order.

    The UTT prints one twenty-step ladder along four arms of a star and declares
    position 1-20 four times over. Ranking the eighty patches individually made an
    eighty-step ladder out of it, ordered among four *exactly equal* nominal L*
    values by whatever the sort happened to yield -- so the OECF curve was drawn
    eighty points wide over twenty printed densities, and every run reported a
    disagreement with the declared order that did not exist.
    """
    import numpy as np

    from deltachrome.reference import read_reference
    from deltachrome.targets.roles import _order_ramp

    target = load_target(UTT)
    reference = read_reference(target.nominal_reference_path)
    labs = {
        p.patch_id: np.array(reference.get(p.patch_id).lab)
        for p in target.patches
        if reference.get(p.patch_id) is not None
        and reference.get(p.patch_id).lab is not None
    }

    ordered, note = _order_ramp(list(target.patches), labs)
    ramp = [p for p in ordered if p.role == "achromatic_ramp"]

    assert len(ramp) == 80
    assert sorted({p.ramp_position for p in ramp}) == list(range(1, 21))
    # The four arms of a step keep that step's number, whichever arm they are on.
    for step in range(1, 21):
        assert {p.patch_id for p in ramp if p.ramp_position == step} == {
            f"GS_{arm}_{step}" for arm in RAMPS
        }
    # And the declared order was right all along, so nothing is reported.
    assert note is None


def test_a_genuine_step_reversal_is_reported_and_names_only_what_moved():
    """The check earns its place -- but on the steps that moved, not the ladder.

    ``isa_fadgi_19264_v1`` declares its last two steps in the order 2.1 then
    2.2 L*, which is a real disagreement worth saying out loud. The note used to
    print both ladders in full to say it, which on this chart is 32 patch ids for
    a swap of two and on the UTT was 160 for a swap of none.
    """
    import numpy as np

    from deltachrome.reference import read_reference
    from deltachrome.targets.roles import _order_ramp

    target = load_target("isa_fadgi_19264_v1")
    reference = read_reference(target.nominal_reference_path)
    labs = {
        p.patch_id: np.array(reference.get(p.patch_id).lab)
        for p in target.patches
        if reference.get(p.patch_id) is not None
        and reference.get(p.patch_id).lab is not None
    }

    _, note = _order_ramp(list(target.patches), labs)

    assert note is not None
    assert "105" in note and "106" in note
    # Every other step held its place and none of them is named.
    unmoved = [
        p.patch_id
        for p in target.patches
        if p.role == "achromatic_ramp" and p.patch_id not in {"105", "106"}
    ]
    assert unmoved and not [pid for pid in unmoved if pid in note]
