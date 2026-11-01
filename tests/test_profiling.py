"""Building a camera profile: untagged loading, the patches, the .ti3, colprof.

The end-to-end check has a known answer. The synthetic chart is rendered through
Adobe RGB, so its device values are Adobe RGB and a profile fitted to them must
be Adobe RGB again: read back through Deltachrome's own ICC reader, every patch
lands on its reference. That needs ``colprof``, which is never bundled, so it
runs only where ``DELTACHROME_COLPROF`` names one or it is on the PATH.
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.errors import MissingICCProfileError, ProfilingError
from deltachrome.imaging.loader import load_image
from deltachrome.profiling import (
    EXCLUDED_CHOSEN,
    EXCLUDED_FLAGGED,
    ColprofOptions,
    build_profile,
    find_colprof,
    format_ti3,
    gather_patches,
)
from deltachrome.reference import read_reference
from deltachrome.targets.registry import load_target
from fixtures import SyntheticChart

SG = "colorchecker_sg_v1"


@pytest.fixture
def chart(tmp_path):
    chart = SyntheticChart()
    reference = read_reference(chart.write_reference(tmp_path / "ref.txt"))
    image, corners = chart.render_capture(tmp_path / "capture.tif")
    return load_image(image, profile_required=False), reference, corners


def _colprof_or_skip():
    try:
        return find_colprof()
    except ProfilingError:
        pytest.skip("colprof is not installed here")


# -- loading ---------------------------------------------------------------


def test_an_untagged_capture_is_still_refused_by_default(tmp_path):
    import tifffile

    path = tmp_path / "bare.tif"
    tifffile.imwrite(path, np.zeros((40, 60, 3), dtype=np.uint16), photometric="rgb")
    with pytest.raises(MissingICCProfileError):
        load_image(path)


def test_an_untagged_capture_loads_as_device_values_when_asked(tmp_path):
    import tifffile

    path = tmp_path / "bare.tif"
    pixels = np.full((40, 60, 3), 32768, dtype=np.uint16)
    tifffile.imwrite(path, pixels, photometric="rgb")
    image = load_image(path, profile_required=False)
    assert image.profile is None
    assert image.bit_depth == 16
    assert np.allclose(image.window(0, 1, 0, 1), 32768 / 65535)
    assert "device values" in image.summary()


# -- the patches -----------------------------------------------------------


def test_every_patch_pairs_device_values_with_its_reference(chart):
    image, reference, corners = chart
    data = gather_patches(image, load_target(SG), reference, corners)
    assert len(data.used) == 140 and not data.excluded
    patch = data.used[0]
    assert patch.rgb.shape == (3,) and 0.0 <= patch.rgb.min() <= patch.rgb.max() <= 1.0
    # XYZ is relative to the ICC's own D50, so a perfect white would be Y = 1.
    assert 0.0 < patch.xyz[1] <= 1.0


def test_chosen_patches_are_excluded_and_kept_for_scoring(chart):
    image, reference, corners = chart
    data = gather_patches(image, load_target(SG), reference, corners, exclude=["B2", "C3"])
    assert len(data.used) == 138
    assert {pid: why for pid, (_, why) in data.excluded.items()} == {
        "B2": EXCLUDED_CHOSEN, "C3": EXCLUDED_CHOSEN,
    }


def test_a_patch_the_chart_does_not_print_is_refused(chart):
    image, reference, corners = chart
    with pytest.raises(ProfilingError, match="Z99"):
        gather_patches(image, load_target(SG), reference, corners, exclude=["Z99"])


def test_a_clipped_patch_is_excluded_as_flagged(tmp_path):
    chart = SyntheticChart()
    reference = read_reference(chart.write_reference(tmp_path / "ref.txt"))
    # Doubled exposure blows out every light patch.
    path, corners = chart.render_capture(tmp_path / "hot.tif", gain=(2.0, 2.0, 2.0))
    image = load_image(path, profile_required=False)
    data = gather_patches(image, load_target(SG), reference, corners)
    flagged = [pid for pid, (_, why) in data.excluded.items() if why == EXCLUDED_FLAGGED]
    assert flagged
    assert all(data.excluded[pid][0].flags for pid in flagged)


# -- the .ti3 --------------------------------------------------------------


def test_the_ti3_is_what_colprof_reads_for_an_input_device(chart):
    image, reference, corners = chart
    data = gather_patches(image, load_target(SG), reference, corners, exclude=["B2"])
    text = format_ti3(data, {"target": SG})
    lines = text.splitlines()
    assert lines[0] == "CTI3"
    assert 'DEVICE_CLASS "INPUT"' in lines
    assert 'COLOR_REP "XYZ_RGB"' in lines
    assert "SAMPLE_ID RGB_R RGB_G RGB_B XYZ_X XYZ_Y XYZ_Z" in lines
    assert f"NUMBER_OF_SETS {len(data.used)}" in lines
    # Provenance is declared before it is used, as CGATS requires.
    assert lines.index('KEYWORD "DELTACHROME_TARGET"') < lines.index(f'DELTACHROME_TARGET "{SG}"')
    assert 'DELTACHROME_EXCLUDED "B2"' in lines
    rows = lines[lines.index("BEGIN_DATA") + 1:lines.index("END_DATA")]
    assert len(rows) == len(data.used)
    values = np.array([[float(v) for v in row.split()[1:]] for row in rows])
    assert values[:, :3].max() <= 100.0
    first = data.used[0]
    assert np.allclose(values[0, :3], first.rgb * 100, atol=1e-5)
    assert np.allclose(values[0, 3:], first.xyz * 100, atol=1e-5)


# -- colprof's arguments ---------------------------------------------------


def test_the_scale_comes_before_absolute_white():
    # -u <scale> also switches on the automatic scaling, and the last -u wins.
    args = ColprofOptions(white="absolute", scale=1.02).arguments()
    assert args.index("-u") < args.index("-ua")
    assert args[args.index("-u") + 1] == "1.02"


def test_options_say_what_was_chosen():
    args = ColprofOptions(profile_type="xyz-table", quality="high", white="chart",
                          transmissive=True, restrict_primaries=True,
                          description="Studio").arguments()
    assert {"-ax", "-qh", "-Zt", "-R", "-DStudio"} <= set(args)
    assert not [a for a in args if a.startswith("-u")]
    args = ColprofOptions(profile_type="lab-table", smoothing=0.25).arguments()
    assert args[args.index("-r") + 1] == "0.25"


@pytest.mark.parametrize("kwargs", [
    {"profile_type": "display"},
    {"white": "perceptual"},
    {"quality": "ultra"},
    {"scale": 0.0},
    {"white": "chart", "scale": 1.1},
    # A matrix or shaper build is byte-identical at any -r, so it is not offered.
    {"profile_type": "shaper", "smoothing": 1.0},
    {"profile_type": "xyz-table", "smoothing": -1.0},
])
def test_options_colprof_cannot_express_are_refused(kwargs):
    with pytest.raises(ProfilingError):
        ColprofOptions(**kwargs)


def test_a_named_colprof_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(ProfilingError, match="not there"):
        find_colprof(tmp_path / "colprof")


def test_a_packaged_build_finds_the_colprof_it_carries(tmp_path, monkeypatch):
    """A fresh install needs nothing set: the bundle's own copy comes before the
    PATH, and a choice made in Settings or the environment still comes first."""
    import sys

    from deltachrome.profiling import colprof

    name = "colprof.exe" if sys.platform == "win32" else "colprof"
    bundled = tmp_path / colprof.BUNDLED_RELATIVE / name
    bundled.parent.mkdir(parents=True)
    bundled.write_text("")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.delenv(colprof.ENVIRONMENT_VARIABLE, raising=False)
    assert colprof.find_colprof() == bundled
    chosen = tmp_path / "chosen"
    chosen.write_text("")
    assert colprof.find_colprof(chosen) == chosen


def test_the_command_stops_before_decoding_when_colprof_is_missing(tmp_path):
    from deltachrome.cli import EXIT_ERROR, main

    code = main([
        "profile", str(tmp_path / "never-read.tif"), "--target", SG, "--detect",
        "--colprof", str(tmp_path / "colprof"), "--out", str(tmp_path / "out.icc"),
    ])
    assert code == EXIT_ERROR


# -- the computed scale ----------------------------------------------------


def test_the_white_scale_is_the_exposure_the_white_step_was_shot_at(tmp_path):
    from deltachrome.profiling import compute_scale, white_patch
    from fixtures import adobe_rgb_icc

    chart = SyntheticChart()
    reference = read_reference(chart.write_reference(tmp_path / "ref.txt"))
    # Shot at 90% exposure and read through the profile it was rendered with:
    # every Y comes back at 0.9 of its reference, so 0.9 is the scale that
    # puts the white step back where it belongs.
    path, corners = chart.render_capture(tmp_path / "dim.tif", gain=(0.9, 0.9, 0.9))
    image = load_image(path, profile_required=False)
    data = gather_patches(image, load_target(SG), reference, corners)
    scale, basis = compute_scale(adobe_rgb_icc(), data, "white")
    assert scale == pytest.approx(0.9, abs=0.002)
    assert basis == white_patch(data).patch_id
    ramp_scale, _ = compute_scale(adobe_rgb_icc(), data, "ramp")
    # A pure exposure error is the same factor at every step.
    assert ramp_scale == pytest.approx(0.9, abs=0.002)


def test_white_is_the_lightest_neutral_by_measurement_not_by_capture():
    from deltachrome.profiling import ProfilePatch, ProfilingSet, white_patch

    def patch(pid, L, a, b, recorded):
        return ProfilePatch(pid, np.full(3, recorded), np.zeros(3), np.array([L, a, b]))

    lit = patch("LIT", 94.0, 0.5, 0.5, 0.95)       # brightest in the image: the lights
    white = patch("WHITE", 96.0, -0.8, 2.5, 0.90)  # lightest neutral by measurement
    yellow = patch("YEL", 97.0, -5.0, 60.0, 0.97)  # lighter, and not neutral
    clipped = patch("CLIP", 98.0, 0.0, 0.0, 1.00)  # left out: clipped
    data = ProfilingSet("t", used=[lit, white, yellow],
                        excluded={"CLIP": (clipped, EXCLUDED_FLAGGED)})
    assert white_patch(data).patch_id == "WHITE"


def test_the_ramp_is_the_charts_own_lightest_first(chart):
    from deltachrome.profiling import ramp_patches

    image, reference, corners = chart
    data = gather_patches(image, load_target(SG), reference, corners)
    steps = ramp_patches(data)
    assert len(steps) == 12
    lightness = [float(patch.lab[0]) for patch in steps]
    assert lightness == sorted(lightness, reverse=True)


def test_a_scale_is_given_or_computed_not_both(chart):
    image, reference, corners = chart
    data = gather_patches(image, load_target(SG), reference, corners)
    with pytest.raises(ProfilingError, match="not both"):
        build_profile(data, ColprofOptions(scale=1.0), colprof=_colprof_or_skip(),
                      scale_for="white")


# -- held out --------------------------------------------------------------


def test_repeats_of_one_colour_share_a_fold(chart):
    from deltachrome.profiling.heldout import assign_folds, repeat_groups
    from fixtures import UNIFORMITY_IDS

    image, reference, corners = chart
    data = gather_patches(image, load_target(SG), reference, corners)
    ids = [patch.patch_id for patch in data.used]
    # The SG's border whites are one colour printed again and again: one group.
    whites = {ids.index(pid) for pid in UNIFORMITY_IDS if pid in ids}
    assert any(whites <= set(group) for group in repeat_groups(data.used))
    folds = assign_folds(data.used, 5)
    assert sorted(i for fold in folds for i in fold) == list(range(len(ids)))
    assert sum(1 for fold in folds if whites & set(fold)) == 1
    # Dealt so no fold is starved by the big group.
    assert min(map(len, folds)) >= len(ids) // 10


def test_the_folds_are_the_same_every_time(chart):
    from deltachrome.profiling.heldout import assign_folds

    image, reference, corners = chart
    data = gather_patches(image, load_target(SG), reference, corners)
    assert assign_folds(data.used, 5) == assign_folds(data.used, 5)


# -- the whole of it, where colprof exists ---------------------------------


def test_every_used_patch_is_scored_once_by_a_build_that_left_it_out(chart):
    colprof = _colprof_or_skip()
    image, reference, corners = chart
    data = gather_patches(image, load_target(SG), reference, corners)
    built = build_profile(data, ColprofOptions(profile_type="shaper"), colprof=colprof)
    assert built.held_out is not None
    assert set(built.held_out.per_patch) == {patch.patch_id for patch in data.used}
    # Predicting a patch is never easier than fitting it, on average.
    assert built.held_out.mean >= built.fit.mean
    skipped = build_profile(data, ColprofOptions(profile_type="shaper"),
                            colprof=colprof, folds=0)
    assert skipped.held_out is None


@pytest.mark.parametrize("aim", ["white", "ramp"])
def test_a_computed_scale_lands_where_it_aims(tmp_path, aim):
    from deltachrome.imaging.color import delta_L_2000
    from deltachrome.profiling import white_patch
    from deltachrome.profiling.scale import compute_scale

    colprof = _colprof_or_skip()
    chart = SyntheticChart()
    reference = read_reference(chart.write_reference(tmp_path / "ref.txt"))
    path, corners = chart.render_capture(tmp_path / "dim.tif", gain=(0.9, 0.9, 0.9))
    data = gather_patches(load_image(path, profile_required=False), load_target(SG),
                          reference, corners)
    built = build_profile(data, ColprofOptions(white="absolute"), colprof=colprof,
                          scale_for=aim)
    # Not 0.9: the fit maps the dim capture onto the reference itself, so the
    # scale corrects only what the fit leaves at the white -- which is small.
    assert built.scale == pytest.approx(1.0, abs=0.02)
    # Measured again on the rebuilt profile: nothing left to correct.
    again, _ = compute_scale(built.icc, data, aim)
    assert again == pytest.approx(1.0, abs=2e-4)
    white = white_patch(data)
    if aim == "white":
        from deltachrome.imaging.color import xyz_to_lab
        from deltachrome.imaging.icc import CIE_D50, parse_icc

        lab = xyz_to_lab(parse_icc(built.icc).to_xyz(white.rgb[None]), white_point=CIE_D50)
        assert float(delta_L_2000(lab, white.lab[None])[0]) < 0.01


def test_the_command_writes_a_profile(tmp_path):
    colprof = _colprof_or_skip()
    from deltachrome.cli import EXIT_OK, main

    chart = SyntheticChart()
    reference = chart.write_reference(tmp_path / "ref.txt")
    image, corners = chart.render_capture(tmp_path / "capture.tif")
    out = tmp_path / "camera.icc"
    code = main([
        "profile", str(image), "--target", SG,
        "--corners", " ".join(f"{x},{y}" for x, y in corners),
        "--reference", str(reference), "--no-verify-code",
        "--colprof", str(colprof), "--out", str(out),
    ])
    assert code == EXIT_OK
    assert out.read_bytes()[36:40] == b"acsp"


@pytest.mark.parametrize("profile_type", ["shaper", "gamma"])
def test_a_profile_of_an_adobe_rgb_render_is_adobe_rgb(chart, profile_type):
    colprof = _colprof_or_skip()
    image, reference, corners = chart
    data = gather_patches(image, load_target(SG), reference, corners, exclude=["B2"])
    built = build_profile(
        data, ColprofOptions(profile_type=profile_type, white="absolute"),
        colprof=colprof, keywords={"target": SG},
    )
    assert built.icc[36:40] == b"acsp"
    # Fitted through 16-bit quantisation and colprof's own smoothing: well under
    # any threshold anybody judges against, and far under a wrong answer.
    assert built.fit.mean < 0.5
    assert built.excluded is not None and built.excluded.count == 1
    assert built.excluded.max < 1.0
    # The .ti3 travels inside the profile, provenance and all.
    assert b"DELTACHROME_TARGET" in built.icc


# -- automatic -------------------------------------------------------------


def _trial(kind, mean, p90):
    from deltachrome.profiling.automatic import Trial

    return Trial(kind, "medium", None, mean, p90)


def test_a_more_elaborate_model_must_win_on_the_mean_and_the_90th_both():
    from deltachrome.profiling.automatic import pick

    # The LaserSoft's case: the table's mean is lowest and its tail is worst.
    trials = [_trial("matrix", 1.04, 1.55), _trial("shaper", 0.70, 1.18),
              _trial("xyz-table", 0.55, 1.58)]
    assert pick(trials).profile_type == "shaper"
    # The SG's: a better mean bought with a worse 90th keeps the matrix.
    assert pick([_trial("matrix", 1.29, 1.82),
                 _trial("single-shaper", 1.10, 1.99)]).profile_type == "matrix"


def test_a_win_inside_the_margin_goes_to_the_simpler_model():
    from deltachrome.profiling.automatic import TIE_FRACTION, pick

    better = 1.0 - TIE_FRACTION / 2
    trials = [_trial("matrix", 1.0, 2.0), _trial("shaper", better, 2.0 * better)]
    assert pick(trials).profile_type == "matrix"


def test_tables_are_tried_only_from_patches_enough_to_fill_one():
    from deltachrome.profiling import ProfilePatch, ProfilingSet
    from deltachrome.profiling.automatic import TABLE_MIN_PATCHES, candidates
    from deltachrome.profiling.colprof import TABLE_TYPES

    def of(n):
        patch = ProfilePatch("P", np.zeros(3), np.zeros(3), np.zeros(3))
        return ProfilingSet("t", used=[patch] * n, excluded={})

    few = {kind for kind, _ in candidates(of(TABLE_MIN_PATCHES - 1))}
    many = {kind for kind, _ in candidates(of(TABLE_MIN_PATCHES))}
    assert not few & set(TABLE_TYPES)
    assert set(TABLE_TYPES) <= many
    # Asked for by name, a table is tried however few the patches.
    assert candidates(of(140), types=["xyz-table"], qualities=["high"]) == [
        ("xyz-table", "high")]
    # A type chosen by hand with the quality left automatic tries every quality.
    assert {q for _, q in candidates(of(140), types=["shaper"])} == {"low", "medium", "high"}


def test_automatic_builds_the_candidate_it_chose(chart):
    from deltachrome.profiling import build_automatic

    colprof = _colprof_or_skip()
    image, reference, corners = chart
    data = gather_patches(image, load_target(SG), reference, corners)
    built = build_automatic(data, ColprofOptions(white="absolute"), colprof=colprof,
                            scale_for="white", types=["matrix", "shaper"],
                            qualities=["medium"])
    assert [(t.profile_type, t.quality) for t in built.trials] == [
        ("matrix", "medium"), ("shaper", "medium")]
    winner = next(t for t in built.trials if t.profile_type == built.options.profile_type)
    assert built.held_out.mean == pytest.approx(winner.held_out_mean)
    assert built.scale == winner.scale
    assert f"-a{'m' if winner.profile_type == 'matrix' else 's'}" in built.run.arguments


# -- repeats and encoding --------------------------------------------------


def _neutral(pid, L, device, a=0.0, b=0.0):
    from deltachrome.imaging.color import lab_to_xyz
    from deltachrome.imaging.icc import ICC_PCS_D50
    from deltachrome.profiling import ProfilePatch

    lab = np.array([L, a, b])
    return ProfilePatch(pid, np.full(3, device), lab_to_xyz(lab, white_point=ICC_PCS_D50), lab)


def test_a_colour_printed_again_is_a_repeat_and_a_fine_ramp_is_not():
    from deltachrome.profiling import repeat_families

    # Three readings of one white, a little apart, with nothing else near.
    whites = [_neutral(f"W{i}", 95.0 + 0.1 * i, 0.9) for i in range(3)]
    # A ramp stepping 0.6 L* at a time: each step nearer its neighbour than the
    # whites are to each other's spread allows, and every one a different colour.
    ramp = [_neutral(f"R{i}", 50.0 + 0.6 * i, 0.2) for i in range(6)]
    families = repeat_families(whites + ramp)
    assert families == [[0, 1, 2]]


def test_merged_repeats_are_one_reading_standing_for_every_member(chart):
    from deltachrome.profiling import merge_repeats, ramp_patches

    image, reference, corners = chart
    data = gather_patches(image, load_target(SG), reference, corners, exclude=["B2"])
    merged = merge_repeats(data)
    families = [patch for patch in merged.used if patch.members]
    assert families
    # Every patch is still accounted for, once.
    ids = [pid for patch in merged.used for pid in patch.ids]
    assert sorted(ids) == sorted(patch.patch_id for patch in data.used)
    first = families[0]
    group = [patch for patch in data.used if patch.patch_id in first.members]
    assert first.patch_id == first.members[0]
    assert np.allclose(first.rgb, np.mean([p.rgb for p in group], axis=0))
    assert np.allclose(first.xyz, np.mean([p.xyz for p in group], axis=0))
    # The ramp is found through a merged reading as through the patch itself,
    # and a left-out patch stays left out on its own.
    assert len(ramp_patches(merged)) == len(ramp_patches(data))
    assert merged.excluded == data.excluded


@pytest.mark.parametrize("gamma, label", [(1.0, "Linear"), (2.2, "Gamma 2.2"), (1.8, "Gamma 1.8")])
def test_the_encoding_is_read_off_the_neutrals(gamma, label):
    from deltachrome.imaging.color import lab_to_xyz
    from deltachrome.imaging.icc import ICC_PCS_D50
    from deltachrome.profiling import ProfilingSet
    from deltachrome.profiling.encoding import detect_encoding

    patches = []
    for i, L in enumerate([96, 80, 65, 50, 35, 20, 8]):
        y = float(lab_to_xyz(np.array([L, 0.0, 0.0]), white_point=ICC_PCS_D50)[1])
        patches.append(_neutral(f"N{i}", L, y ** (1.0 / gamma)))
    encoding = detect_encoding(ProfilingSet("t", used=patches, excluded={}))
    assert encoding.label == label
    assert encoding.exponent == pytest.approx(1.0 / gamma, rel=1e-6)


def test_no_encoding_is_claimed_from_too_few_neutrals():
    from deltachrome.profiling import ProfilingSet
    from deltachrome.profiling.encoding import detect_encoding

    patches = [_neutral("N0", 95, 0.9), _neutral("N1", 50, 0.2),
               _neutral("C", 50, 0.2, a=40.0)]
    assert detect_encoding(ProfilingSet("t", used=patches, excluded={})) is None


# -- the screen's engine half ----------------------------------------------


def test_a_selection_is_exactly_what_was_left_out(chart):
    from deltachrome.profiling import select_patches

    image, reference, corners = chart
    data = gather_patches(image, load_target(SG), reference, corners)
    chosen = select_patches(data, ["B2", "C3"])
    assert set(chosen.excluded) == {"B2", "C3"}
    assert all(why == EXCLUDED_CHOSEN for _, why in chosen.excluded.values())
    assert len(chosen.used) == len(data.all_patches) - 2
    # Nothing left out is nothing left out, flagged or not.
    assert not select_patches(data, []).excluded
    # Every patch carries the region it was sampled from, to be clicked on.
    assert set(data.regions) == {patch.patch_id for patch in data.all_patches}


def test_too_few_patches_left_is_refused(chart):
    from deltachrome.profiling import select_patches

    image, reference, corners = chart
    data = gather_patches(image, load_target(SG), reference, corners)
    keep = [patch.patch_id for patch in data.all_patches][:5]
    with pytest.raises(ProfilingError, match="at least"):
        select_patches(data, [p.patch_id for p in data.all_patches if p.patch_id not in keep])


def test_a_chart_with_nothing_to_profile_against_says_so():
    from deltachrome.profiling import profilable

    assert profilable(load_target(SG)) == (True, "")
    ok, why = profilable(load_target("applied_image_qa62_sfr_v1"))
    assert not ok and "reference" in why


def test_a_printed_aim_table_is_not_a_reference_for_a_profile():
    from types import SimpleNamespace

    from deltachrome.profiling import check_reference

    measured = SimpleNamespace(is_nominal=False, population_is_centred=False)
    assert check_reference(measured, "m.txt") == []
    centred = SimpleNamespace(is_nominal=True, population_is_centred=True)
    assert "mean of measured units" in check_reference(centred, "n.txt")[0]
    printed = SimpleNamespace(is_nominal=True, population_is_centred=False)
    with pytest.raises(ProfilingError, match="nominal"):
        check_reference(printed, "aims.txt")


def test_a_few_pixels_grazing_zero_do_not_leave_a_patch_out():
    from types import SimpleNamespace

    from deltachrome.profiling.patches import MARGINAL_LOW_FRACTION, _marginal_floor

    def sample(flags, fraction):
        return SimpleNamespace(flags=flags, clipped_low_fraction=fraction)

    # The LaserSoft master's dark patches: one channel grazing zero, nothing else.
    assert _marginal_floor(sample(["clipped_low(0.5%)"], 0.005))
    # More of the patch at the floor than that biases the mean, so it stays out.
    assert not _marginal_floor(sample(["clipped_low(5.0%)"], 0.05))
    assert MARGINAL_LOW_FRACTION < 0.05
    # A pinned channel beside a bright one is a colour the encoding lost.
    assert not _marginal_floor(sample(["clipped_low(0.2%)", "gamut_clipped(G)"], 0.002))
    assert not _marginal_floor(sample(["clipped_low(0.2%)", "mean_median_divergence"], 0.002))
    assert not _marginal_floor(sample([], 0.0))
