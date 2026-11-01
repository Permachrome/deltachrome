"""The ISA 4x5 transmissive target, and the quarter-disc edge feature.

The DLT's slanted-edge feature is a rotated square: four sides, four usable edges.
This target's is a quarter disc, which has only two straight sides and an arc, and
telling them apart is not optional -- an ROI cut across the arc holds no single
transition and still returns a confident, entirely fictitious MTF.
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.imaging.edges import QUARTER_DISC_FILL, _refine_quarter_disc, extract_edges
from deltachrome.imaging.geometry import place_chart
from deltachrome.imaging.loader import load_image
from deltachrome.imaging.sfr import compute_sfr
from deltachrome.reference import read_reference
from deltachrome.targets.registry import load_target
from fixtures import adobe_rgb_icc, write_tiff_with_profile

TARGET = "isa_transmissive_4x5_v1"


# -- the definition ---------------------------------------------------------


def test_the_definition_describes_a_neutral_target_with_no_color_patches():
    """The chart carries no color patches at all, and FADGI's transparency category
    specifies no color accuracy metric either. Declaring the incapacity is what
    turns that into N/A rather than a silent omission."""
    target = load_target(TARGET)
    assert "color_accuracy" in target.not_capable_of
    assert not target.supports("color_accuracy")
    assert {"tonal_response", "white_balance", "sfr", "color_registration"} <= target.capabilities
    assert len(target.patches) == 24
    assert all(p.role == "achromatic_ramp" for p in target.patches)


def test_the_ramp_is_twelve_steps_measured_twice():
    """Twenty-four patches but twelve aims: the same ladder printed at two field
    positions. Reading it as a twenty-four step ramp would misdescribe the curve."""
    target = load_target(TARGET)
    reference = read_reference(target.nominal_reference_path)
    left = [reference[f"L{i}"].lab[0] for i in range(1, 13)]
    right = [reference[f"R{i}"].lab[0] for i in range(1, 13)]
    assert left == right
    assert left == sorted(left, reverse=True)
    assert len(set(left)) == 12


def test_the_nominal_file_is_labelled_nominal_and_says_why_it_is_derived():
    target = load_target(TARGET)
    reference = read_reference(target.nominal_reference_path)
    assert reference.is_nominal
    assert len(reference) == 24
    caveat = reference.nominal_caveat
    assert "DENSITY ONLY" in caveat
    assert "96" in caveat and "spectrally neutral" in caveat
    # The published quantity is carried through beside what was inferred from it.
    assert "D" in reference.fields


def test_every_patch_has_a_reference_value():
    target = load_target(TARGET)
    reference = read_reference(target.nominal_reference_path)
    present, missing = reference.coverage(target.patch_ids)
    assert not missing and len(present) == 24


def test_each_edge_region_is_a_pair_of_discs():
    """One region per pair, not per disc. The pair is what yields two edges per axis
    of opposite polarity, and the region id is what groups them."""
    target = load_target(TARGET)
    modules = [r for r in target.regions if r.kind == "slanted_edge"]
    assert len(modules) == 5
    for region in modules:
        assert len(region.geometry["discs"]) == 2
        assert region.geometry["feature"] == "quarter_disc_pair"


# -- the quarter-disc feature ----------------------------------------------


def quarter_disc_capture(path, *, side=400, tilt_deg=5.0, margin=160, dark=0.18, light=0.72):
    """A single dark quarter disc on a light ground, apex at the top-left corner
    of its bounding square, tilted like the real one."""
    size = side + 2 * margin
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    t = np.radians(tilt_deg)
    # coordinates in the square's own frame, origin at the apex
    ox, oy = margin, margin
    u = (xx - ox) * np.cos(t) + (yy - oy) * np.sin(t)
    v = -(xx - ox) * np.sin(t) + (yy - oy) * np.cos(t)
    inside = (u >= 0) & (v >= 0) & (u**2 + v**2 <= side**2)
    image = np.where(inside, dark, light)
    rgb = np.dstack([image] * 3)
    encoded = np.clip(rgb, 0, 1) ** (1 / 2.19921875)
    write_tiff_with_profile(path, np.round(encoded * 65535).astype(np.uint16), adobe_rgb_icc())
    return path, size, (ox, oy)


def test_a_quarter_disc_fills_pi_over_four_of_its_square():
    """The check that separates this feature from a solid square or a half. If the
    band did not exclude those, the wrong feature would be measured silently."""
    assert QUARTER_DISC_FILL[0] < np.pi / 4 < QUARTER_DISC_FILL[1]
    assert QUARTER_DISC_FILL[0] > 0.5 and QUARTER_DISC_FILL[1] < 1.0


def test_the_apex_is_located_and_only_the_straight_sides_are_used(tmp_path):
    path, size, (ox, oy) = quarter_disc_capture(tmp_path / "disc.tif")
    image = load_image(path)
    side = 400
    declared = np.array([[ox, oy], [ox + side, oy], [ox + side, oy + side], [ox, oy + side]],
                        dtype=np.float64)
    found = _refine_quarter_disc(image, declared)
    assert found is not None
    box, apex = found
    # The apex is the corner the disc's straight sides meet at -- the one nearest
    # the declared apex, which is where the synthetic disc was drawn.
    assert float(np.linalg.norm(box[apex] - np.array([ox, oy]))) < 25.0


def test_a_solid_square_is_not_mistaken_for_a_quarter_disc(tmp_path):
    """A quarter disc's bounding square holds a square-shaped component too. Only
    the fill ratio tells them apart, and confusing them would put two of the four
    ROIs nowhere near an edge."""
    size = 720
    image = np.full((size, size), 0.72)
    image[160:560, 160:560] = 0.18
    rgb = np.dstack([image] * 3)
    encoded = np.clip(rgb, 0, 1) ** (1 / 2.19921875)
    path = write_tiff_with_profile(
        tmp_path / "square.tif", np.round(encoded * 65535).astype(np.uint16), adobe_rgb_icc()
    )
    declared = np.array([[160, 160], [560, 160], [560, 560], [160, 560]], dtype=np.float64)
    assert _refine_quarter_disc(load_image(path), declared) is None


def test_both_straight_edges_of_a_synthetic_disc_measure(tmp_path):
    """End to end on a feature of known shape: two edges, both at the drawn tilt,
    both with a fit residual that says the ROI holds one clean transition."""
    from deltachrome.targets.schema import Region, TargetDefinition, Fiducials

    path, size, _ = quarter_disc_capture(tmp_path / "disc.tif", side=400, tilt_deg=5.0)
    image = load_image(path)
    placement = place_chart(
        [(0, 0), (size, 0), (size, size), (0, size)], image_size=(size, size)
    )
    target = TargetDefinition(
        target_id="synthetic", display_name="synthetic", manufacturer=None,
        target_type="test", physical_size_mm=None,
        capabilities=frozenset({"sfr"}), not_capable_of=frozenset(),
        fiducials=Fiducials(method="corners", corner_ids=[], corner_positions=[]),
        regions=[Region(
            region_id="module", kind="slanted_edge",
            geometry={"feature": "quarter_disc_pair", "discs": [
                {"center": [0.5, 0.5], "half_size": [200 / size, 200 / size],
                 "tilt_deg": 5.0},
            ]},
        )],
        patches=[],
    )
    edges = extract_edges(image, placement, target)
    assert len(edges) == 2
    assert {e.orientation for e in edges} == {"vertical", "horizontal"}
    for edge in edges:
        result = compute_sfr(edge.pixels, linearize=edge.linearize)
        assert abs(abs(result.edge_angle_deg) - 5.0) < 0.6
        assert result.settings["edge_fit_residual_px"] < 1.0


def test_each_edge_is_measured_in_every_color_record(tmp_path):
    """On the real extraction path, not a hand-built ROI: what makes this worth
    testing here is that `extract_edges` is where the per-channel linearizer is
    attached, and passing the luminance one instead returns luminance three times
    without complaining."""
    from deltachrome.pipeline import _channel_sfr
    from deltachrome.targets.schema import Region, TargetDefinition, Fiducials

    path, size, _ = quarter_disc_capture(tmp_path / "disc.tif", side=400, tilt_deg=5.0)
    image = load_image(path)
    placement = place_chart(
        [(0, 0), (size, 0), (size, size), (0, size)], image_size=(size, size)
    )
    target = TargetDefinition(
        target_id="synthetic", display_name="synthetic", manufacturer=None,
        target_type="test", physical_size_mm=None,
        capabilities=frozenset({"sfr"}), not_capable_of=frozenset(),
        fiducials=Fiducials(method="corners", corner_ids=[], corner_positions=[]),
        regions=[Region(
            region_id="module", kind="slanted_edge",
            geometry={"feature": "quarter_disc_pair", "discs": [
                {"center": [0.5, 0.5], "half_size": [200 / size, 200 / size],
                 "tilt_deg": 5.0},
            ]},
        )],
        patches=[],
    )
    for edge in extract_edges(image, placement, target):
        assert edge.linearize_channels is not None
        channels = _channel_sfr(edge)
        assert set(channels) == {"r", "g", "b"}
        # A neutral synthetic edge: the records should agree closely, which is
        # also the check that none of them came back as something else entirely.
        # Read at Nyquist rather than at SFR50, which this very sharp synthetic
        # edge never reaches inside the measured range.
        at_nyquist = [c.response_at_nyquist() for c in channels.values()]
        assert max(at_nyquist) - min(at_nyquist) < 0.02


def test_a_region_whose_feature_cannot_be_found_yields_no_edges(tmp_path):
    """Better nothing than an ROI placed on a guess: without the apex there is no way
    to tell a straight side from the arc."""
    from deltachrome.targets.schema import Region, TargetDefinition, Fiducials

    size = 600
    flat = np.full((size, size, 3), 0.5)
    path = write_tiff_with_profile(
        tmp_path / "flat.tif", np.round(flat * 65535).astype(np.uint16), adobe_rgb_icc()
    )
    image = load_image(path)
    placement = place_chart(
        [(0, 0), (size, 0), (size, size), (0, size)], image_size=(size, size)
    )
    target = TargetDefinition(
        target_id="synthetic", display_name="synthetic", manufacturer=None,
        target_type="test", physical_size_mm=None,
        capabilities=frozenset({"sfr"}), not_capable_of=frozenset(),
        fiducials=Fiducials(method="corners", corner_ids=[], corner_positions=[]),
        regions=[Region(
            region_id="module", kind="slanted_edge",
            geometry={"feature": "quarter_disc_pair", "discs": [
                {"center": [0.5, 0.5], "half_size": [0.3, 0.3], "tilt_deg": 5.0},
            ]},
        )],
        patches=[],
    )
    assert extract_edges(image, placement, target) == []


def test_the_square_feature_path_is_untouched(tmp_path):
    """The DLT still gets four edges from its rotated square."""
    from test_sfr import edge_target, rotated_square_capture

    path = rotated_square_capture(tmp_path)
    image = load_image(path)
    placement = place_chart(
        [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
        (image.width, image.height),
    )
    assert len(extract_edges(image, placement, edge_target())) == 4


# -- the 35mm, which the slide shares --------------------------------------

MM35 = "isa_transmissive_35mm_v1"


def test_the_35mm_carries_the_same_twelve_steps_twice():
    target = load_target(MM35)
    reference = read_reference(target.nominal_reference_path)
    left = [float(reference[f"L{i}"].raw["D"]) for i in range(1, 13)]
    right = [float(reference[f"R{i}"].raw["D"]) for i in range(1, 13)]
    assert left == right == [0.05, 0.12, 0.19, 0.33, 0.48, 0.72, 0.88, 1.15,
                             1.43, 1.71, 2.05, 2.25]
    assert len(target.patches) == 24


def test_the_right_hand_wedge_runs_in_reverse():
    """A property of the print, not a convention chosen here: the two blocks mirror
    each other, so the darkest step of the right block is top-left where the left
    block's lightest is."""
    target = load_target(MM35)
    left = {p.ramp_position: p for p in target.patches if p.region == "wedge_left"}
    right = {p.ramp_position: p for p in target.patches if p.region == "wedge_right"}
    lightest_left = left[1]
    lightest_right = right[2]          # positions interleave the two blocks
    assert (lightest_left.row, lightest_left.col) == (0, 0)
    assert (lightest_right.row, lightest_right.col) == (2, 3)


def test_the_35mm_carries_the_4x5_wedge_and_says_why():
    """Every 35mm capture tried, OpenDICE's own included, fits the 4x5's wedge best
    and the twelve L* values it replaced worst, so the wedge serves all three ISA
    film targets. The file must say what it replaced and that
    it was chosen partly by fit."""
    reference = read_reference(load_target(MM35).nominal_reference_path)
    four_by_five = read_reference(load_target("isa_transmissive_4x5_v1").nominal_reference_path)
    assert reference.is_nominal
    assert [reference[f"L{i}"].raw["D"] for i in range(1, 13)] == [
        four_by_five[f"L{i}"].raw["D"] for i in range(1, 13)
    ]
    source = reference.metadata["DELTACHROME_NOMINAL_SOURCE"]
    assert "replaces twelve recommended L* values" in source
    assert "CHOSEN PARTLY BY FIT" in reference.nominal_caveat


def test_the_35mm_metric_rules_are_read_against_their_own_printed_label():
    """The division used is 1 mm where the chart prints 0.1 mm, on the evidence that
    the label is a printing error. Contradicting a chart's own printing is exactly
    the kind of thing that must be written down where a reader will meet it, because
    the wrong division scales every rate by ten and still fits perfectly."""
    target = load_target(MM35)
    metric = [r for r in target.regions
              if r.kind == "scale" and r.geometry["unit"] == "mm"]
    assert len(metric) == 2
    for region in metric:
        rule = region.geometry
        assert rule["division_mm"] == 1.0
        assert rule["printed_label"] == "1 division = 0.1 mm"
        assert "CONTRADICTS THE CHART'S PRINTED LABEL" in rule["measured_note"]
        # What arbitrates it is the imperial engraving, which carries no metric
        # label and is therefore independent of the one under suspicion.
        assert "independent of it" in rule["measured_note"]


def test_the_35mm_metric_band_stops_clear_of_the_rules_own_baseline():
    """The one fact that makes this rule readable at all, and it is invisible in the
    numbers. The ticks stand up from a solid line; a band reaching down to include
    that line puts ink in every column, so the reader's low percentile down the band
    finds the baseline instead of the ticks and the comb collapses. Declared to the
    baseline this rule fitted seven ticks and was refused."""
    top = next(r for r in load_target(MM35).regions if r.region_id == "ruler_metric_top")
    assert "STOPS ABOVE THE RULE'S OWN BASELINE" in top.notes
    # The baseline sits at about v -0.175; the band must not reach it.
    assert top.geometry["bounds_relative"][3] < -0.176


def test_the_35mm_reads_two_rules_per_axis_in_two_units():
    """What took this target from a sampling rate to a distortion measurement. Two
    positions is what lets the field vary rather than only the scale; two units at
    each is what keeps a difference between engravings from reading as anisotropy."""
    target = load_target(MM35)
    scales = [r for r in target.regions if r.kind == "scale"]
    by_axis_unit = {}
    for region in scales:
        key = (region.geometry["axis"], region.geometry["unit"])
        by_axis_unit[key] = by_axis_unit.get(key, 0) + 1
    assert by_axis_unit == {("u", "mm"): 1, ("u", "in"): 1,
                            ("v", "mm"): 1, ("v", "in"): 1}
    assert "geometric_distortion" in target.capabilities
    # Unit and axis are not confounded -- both units appear on both axes -- so a
    # systematic difference between engravings can be told from a real anisotropy.
    assert target.dimensional_reference["available"] is True
    assert "Four printed rules" in target.dimensional_reference["summary"]


def test_the_35mm_has_its_own_fadgi_category():
    """Photographic Transparencies: 35mm up to 4x5, transcribed from section 3.9. It
    differs from the 4x5-and-larger table in the sampling rate alone, and by a factor
    of two -- so judging a 35mm original by the larger table would pass it at half
    the resolution its own category asks for."""
    from deltachrome.profiles.schema import load_profile

    for star in (1, 2, 3, 4):
        small = load_profile(f"fadgi_{star}star_photographic_transparencies_35mm_to_4x5_2023")
        large = load_profile(
            f"fadgi_{star}star_photographic_transparencies_4x5_and_larger_2023"
        )
        assert small.standard_reference["material_category"].startswith(
            "Photographic Transparencies: 35mm"
        )
        rate = lambda p: float(
            p.metric("sampling_frequency").pass_condition.split(">=")[1]
        )
        assert rate(small) == pytest.approx(2 * rate(large))


def test_both_transmissive_targets_share_a_feature_model():
    """Same quarter-disc pairs at the same five field positions, so everything built
    for one works on the other."""
    for target_id in ("isa_transmissive_4x5_v1", MM35):
        target = load_target(target_id)
        modules = [r for r in target.regions if r.kind == "slanted_edge"]
        assert len(modules) == 5
        assert all(len(r.geometry["discs"]) == 2 for r in modules)
        assert all(r.geometry["feature"] == "quarter_disc_pair" for r in modules)
