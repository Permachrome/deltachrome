"""Slanted-edge SFR, validated against an analytically known answer.

A step edge blurred by a Gaussian of width sigma has MTF exactly
``exp(-2 * pi^2 * sigma^2 * f^2)``. Point-sampling that analytic edge gives a test
image whose correct answer is known in closed form, so the whole chain -- edge
location, line fit, projection, binning, differentiation, windowing, transform and
the derivative-filter correction -- is checked against a number rather than against
itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from deltachrome.imaging.sfr import (
    MAX_EDGE_ANGLE_DEG,
    SFRError,
    compute_sfr,
)


def synthetic_edge(sigma, angle_deg=5.0, height=200, width=140, x0=70.0,
                   low=0.05, high=0.95, noise=0.0, seed=0):
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    edge = x0 + rows * np.tan(np.radians(angle_deg))
    profile = 0.5 * (1 + erf((cols - edge) / (sigma * np.sqrt(2))))
    image = low + (high - low) * profile
    if noise:
        image = image + np.random.default_rng(seed).normal(0, noise, image.shape)
    return image


def gaussian_mtf(sigma, f):
    return np.exp(-2 * np.pi**2 * sigma**2 * np.asarray(f) ** 2)


# -- the measurement is correct --------------------------------------------


@pytest.mark.parametrize("sigma", [0.4, 0.6, 0.9, 1.3])
def test_measured_mtf_matches_the_analytic_curve(sigma):
    result = compute_sfr(synthetic_edge(sigma))
    band = result.frequency <= 0.6
    error = np.abs(result.mtf[band] - gaussian_mtf(sigma, result.frequency[band]))
    assert error.max() < 0.02, f"max error {error.max():.4f}"


@pytest.mark.parametrize("sigma", [0.5, 0.8, 1.2])
def test_mtf50_matches_the_analytic_value(sigma):
    expected = np.sqrt(np.log(2) / (2 * np.pi**2 * sigma**2))
    measured = compute_sfr(synthetic_edge(sigma)).frequency_at(0.5)
    assert measured == pytest.approx(expected, rel=0.03)


def test_the_measurement_is_normalised_at_dc():
    result = compute_sfr(synthetic_edge(0.7))
    assert result.mtf[0] == pytest.approx(1.0)


@pytest.mark.parametrize("angle", [-8.0, -5.0, 3.0, 5.0, 8.0])
def test_the_edge_angle_is_recovered(angle):
    result = compute_sfr(synthetic_edge(0.7, angle_deg=angle))
    assert result.edge_angle_deg == pytest.approx(angle, abs=0.15)


def test_a_horizontal_edge_gives_the_same_answer_as_a_vertical_one():
    """Transposing rather than branching means one code path to verify."""
    vertical = compute_sfr(synthetic_edge(0.7))
    horizontal = compute_sfr(synthetic_edge(0.7).T)
    assert horizontal.mtf_at(0.25) == pytest.approx(vertical.mtf_at(0.25), abs=0.01)


def test_contrast_and_polarity_do_not_change_the_result():
    """MTF is normalized, so a darker edge or a reversed one measures the same."""
    base = compute_sfr(synthetic_edge(0.7))
    faint = compute_sfr(synthetic_edge(0.7, low=0.40, high=0.60))
    flipped = compute_sfr(synthetic_edge(0.7, low=0.95, high=0.05))
    for other in (faint, flipped):
        assert other.mtf_at(0.3) == pytest.approx(base.mtf_at(0.3), abs=0.02)


def test_noise_does_not_destroy_the_measurement():
    """Real captures are noisy; the per-row centroid must be windowed and the line
    fit robust, or scattered edge positions smear the profile into a ramp."""
    clean = compute_sfr(synthetic_edge(0.8))
    noisy = compute_sfr(synthetic_edge(0.8, noise=0.01, seed=3))
    assert noisy.mtf_at(0.3) == pytest.approx(clean.mtf_at(0.3), abs=0.05)
    assert noisy.settings["edge_fit_residual_px"] < 1.0


# -- the derivative-filter correction --------------------------------------


def test_omitting_the_derivative_correction_under_reports_high_frequencies():
    """A difference filter is not an ideal derivative: it rolls off with frequency.
    Leaving it in flatters the result, which is the dangerous direction."""
    sigma = 0.7
    corrected = compute_sfr(synthetic_edge(sigma), derivative_correction=True)
    raw = compute_sfr(synthetic_edge(sigma), derivative_correction=False)

    ideal = gaussian_mtf(sigma, 0.5)
    assert raw.mtf_at(0.5) < corrected.mtf_at(0.5)
    assert abs(corrected.mtf_at(0.5) - ideal) < abs(raw.mtf_at(0.5) - ideal)


# -- parameters -------------------------------------------------------------


@pytest.mark.parametrize("bin_width", [0.125, 0.25, 0.5])
def test_supersample_bin_width_is_a_parameter_and_all_choices_agree(bin_width):
    result = compute_sfr(synthetic_edge(0.8), supersample_bin_width=bin_width)
    assert result.bin_width == bin_width
    assert result.mtf_at(0.3) == pytest.approx(gaussian_mtf(0.8, 0.3), abs=0.03)


@pytest.mark.parametrize("window", ["hamming", "hann", "tukey", "none"])
def test_window_choice_is_a_parameter(window):
    result = compute_sfr(synthetic_edge(0.8), window=window)
    assert result.settings["window"] == window
    assert result.mtf_at(0.25) == pytest.approx(gaussian_mtf(0.8, 0.25), abs=0.05)


def test_the_default_window_is_hann_following_iso_12233_edition_4():
    """Editions to 2017 specify Hamming; the 4th substitutes a Tukey window and
    settles on alpha = 1.0, which is Hann. Burns et al. give the reason: zero
    endpoints, so a non-uniform ROI cannot introduce a low-frequency discontinuity."""
    from inspect import signature

    assert signature(compute_sfr).parameters["window"].default == "hann"
    assert compute_sfr(synthetic_edge(0.8)).settings["window"] == "hann"


def test_hann_is_zero_at_its_endpoints_and_hamming_is_not():
    """The property edition 4 selected it for, and the whole difference between them."""
    from deltachrome.imaging.sfr import _window

    assert _window(64, "hann")[0] == pytest.approx(0.0, abs=1e-12)
    assert _window(64, "hann")[-1] == pytest.approx(0.0, abs=1e-12)
    assert _window(64, "hamming")[0] == pytest.approx(0.08, abs=0.005)


def test_the_earlier_edition_window_is_still_reachable():
    """A figure computed under ISO 12233:2017 has to remain reproducible."""
    result = compute_sfr(synthetic_edge(0.8), window="hamming")
    assert result.settings["window"] == "hamming"
    assert result.mtf_at(0.25) == pytest.approx(gaussian_mtf(0.8, 0.25), abs=0.05)


def test_the_two_windows_agree_on_a_clean_edge():
    """Burns et al.: "The choice of window function was found to have little effect
    on the resulting e-SFR." They differ in what they do to a *noisy* or non-uniform
    ROI, not in what they measure."""
    hann = compute_sfr(synthetic_edge(0.8), window="hann")
    hamming = compute_sfr(synthetic_edge(0.8), window="hamming")
    assert hann.sfr50() == pytest.approx(hamming.sfr50(), rel=0.005)


@pytest.mark.parametrize("fit", ["linear", "quadratic"])
def test_edge_fit_is_a_parameter(fit):
    result = compute_sfr(synthetic_edge(0.8), edge_fit=fit)
    assert result.settings["edge_fit"] == fit
    assert result.mtf_at(0.3) == pytest.approx(gaussian_mtf(0.8, 0.3), abs=0.03)


@pytest.mark.parametrize("derivative", ["central", "forward"])
def test_derivative_kernel_is_a_parameter(derivative):
    result = compute_sfr(synthetic_edge(0.8), derivative=derivative)
    assert result.settings["derivative"] == derivative
    assert result.mtf_at(0.25) == pytest.approx(gaussian_mtf(0.8, 0.25), abs=0.05)


def test_every_stage_records_the_setting_it_used():
    """Two runs of the same capture under different settings are different
    measurements, so the settings travel with the result."""
    result = compute_sfr(synthetic_edge(0.8))
    for key in ("edge_fit", "supersample_bin_width", "window", "derivative",
                "derivative_correction", "centroid_threshold", "detrend"):
        assert key in result.settings


# -- refusals and warnings --------------------------------------------------


def test_an_edge_aligned_to_the_pixel_grid_is_warned_about():
    """Rows then repeat the same sub-pixel phases instead of filling them."""
    result = compute_sfr(synthetic_edge(0.8, angle_deg=0.2))
    assert "Slanted edge is too shallow for reliable analysis." in result.warnings


def test_an_over_tilted_edge_is_warned_about():
    result = compute_sfr(synthetic_edge(0.8, angle_deg=MAX_EDGE_ANGLE_DEG + 6))
    assert "Target is too tilted for reliable edge analysis." in result.warnings


def test_a_blank_roi_is_refused():
    with pytest.raises(SFRError):
        compute_sfr(np.full((80, 80), 0.5))


def test_a_tiny_roi_is_refused():
    with pytest.raises(SFRError, match="too small"):
        compute_sfr(synthetic_edge(0.5, height=6, width=6, x0=3))


def test_an_impossible_bin_width_is_refused():
    with pytest.raises(SFRError, match="supersample_bin_width"):
        compute_sfr(synthetic_edge(0.7), supersample_bin_width=0)


# -- reading the curve ------------------------------------------------------


def test_readings_are_expressed_against_nyquist():
    result = compute_sfr(synthetic_edge(0.55))
    assert result.nyquist == 0.5
    efficiency = result.sampling_efficiency(0.10)
    assert efficiency == pytest.approx(result.frequency_at(0.10) / 0.5)
    assert result.sfr50() == pytest.approx(result.frequency_at(0.50) / 0.5)


def test_an_unsharpened_edge_reads_exactly_one():
    """The criterion's limits all sit just above 1.0, which only makes sense if an
    unsharpened system reads exactly 1.0. It does, because the curve is normalized at
    DC and never rises above it without sharpening. GoldenThread reports it this way."""
    result = compute_sfr(synthetic_edge(0.8))
    assert result.peak_modulation() == pytest.approx(1.0, abs=1e-6)
    assert result.peak_frequency() == pytest.approx(0.0, abs=1e-9)


def test_the_peak_floor_is_available_as_a_diagnostic_but_is_not_the_default():
    """Raising the floor excludes the leakage-prone lowest bins, but it also changes
    the scale the thresholds were written against, so it is opt-in."""
    from deltachrome.imaging.sfr import (DEFAULT_PEAK_SEARCH_MIN_FREQUENCY,
                                         LEAKAGE_SUSPECT_FREQUENCY)

    assert DEFAULT_PEAK_SEARCH_MIN_FREQUENCY == 0.0
    result = compute_sfr(synthetic_edge(0.8))
    assert result.peak_modulation(LEAKAGE_SUSPECT_FREQUENCY) < result.peak_modulation()


def test_sharpening_shows_as_a_peak_above_one():
    """An overshoot at the edge is contrast the system did not capture; it shows as
    an MTF above 1 and cannot be undone later."""
    image = synthetic_edge(0.9)
    blurred = np.apply_along_axis(
        lambda r: np.convolve(r, np.ones(5) / 5, mode="same"), 1, image
    )
    sharpened = np.clip(image + 1.4 * (image - blurred), 0, 1)
    assert compute_sfr(sharpened).peak_modulation() > 1.10


def test_frequency_at_returns_the_first_crossing():
    """A sharpened response can cross a level more than once; the later crossings
    describe the overshoot, not the resolution limit."""
    result = compute_sfr(synthetic_edge(0.7))
    level = 0.5
    f = result.frequency_at(level)
    before = result.frequency < f
    assert (result.mtf[before] >= level - 1e-9).all()


def test_a_level_the_curve_never_reaches_returns_none():
    assert compute_sfr(synthetic_edge(0.8)).frequency_at(-1.0) is None


# -- locating edges on a target --------------------------------------------
#
# A slanted-edge region declares roughly where its feature sits; the exact edge is
# found in the capture. Declaring exact positions would bake one specimen's print
# registration into a definition meant to describe a model.


from deltachrome.imaging.edges import _square_corners, extract_edges
from deltachrome.imaging.geometry import place_chart
from deltachrome.imaging.loader import load_image
from deltachrome.targets.schema import target_from_dict
from fixtures import adobe_rgb_icc, write_tiff_with_profile


def edge_target(center=(0.5, 0.5), half=(0.25, 0.25), tilt=5.0):
    return target_from_dict({
        "target_id": "edge_probe",
        "display_name": "Edge probe",
        "capabilities": ["sfr"],
        "fiducials": {"method": "chart_corners", "corner_markers": [
            {"id": "TL", "relative_position": [0.0, 0.0]},
            {"id": "TR", "relative_position": [1.0, 0.0]},
            {"id": "BR", "relative_position": [1.0, 1.0]},
            {"id": "BL", "relative_position": [0.0, 1.0]}]},
        "regions": [{"region_id": "module", "kind": "slanted_edge",
                     "geometry": {"center": list(center), "half_size": list(half),
                                  "tilt_deg": tilt}}],
        "patches": [],
    })


def rotated_square_capture(tmp_path, size=900, half=220, tilt=5.0, sigma=0.7):
    """A dark square rotated by ``tilt`` on a light field, softened by a Gaussian."""
    import cv2

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    t = np.radians(tilt)
    cx = cy = size / 2
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    inside = (np.abs(u) < half) & (np.abs(v) < half)
    image = np.where(inside, 0.25, 0.75)
    image = cv2.GaussianBlur(image, (0, 0), sigma)
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    data = (rgb * 65535).astype(np.uint16)
    return write_tiff_with_profile(tmp_path / "edges.tif", data, adobe_rgb_icc())


def test_four_edges_are_cut_from_a_square(tmp_path):
    path = rotated_square_capture(tmp_path)
    image = load_image(path)
    target = edge_target()
    placement = place_chart(
        [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
        (image.width, image.height),
    )
    rois = extract_edges(image, placement, target, roi_half_width=50)
    assert len(rois) == 4
    assert {r.orientation for r in rois} == {"vertical", "horizontal"}
    assert all(r.region_id == "module" for r in rois)


def test_each_extracted_edge_measures_the_blur_it_was_given(tmp_path):
    sigma = 0.9
    path = rotated_square_capture(tmp_path, sigma=sigma)
    image = load_image(path)
    placement = place_chart(
        [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
        (image.width, image.height),
    )
    rois = extract_edges(image, placement, edge_target(), roi_half_width=50)
    for roi in rois:
        result = compute_sfr(roi.pixels)
        assert abs(result.edge_angle_deg) == pytest.approx(5.0, abs=0.5)
        # The rendered square is blurred, then encoded through a gamma TRC, so the
        # measured width is close to but not exactly the nominal sigma.
        assert result.mtf_at(0.25) == pytest.approx(gaussian_mtf(sigma, 0.25), abs=0.12)


def test_the_declared_position_only_has_to_be_approximate(tmp_path):
    """Refinement finds the real square, so a definition need not encode one
    specimen's exact print registration."""
    path = rotated_square_capture(tmp_path)
    image = load_image(path)
    placement = place_chart(
        [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
        (image.width, image.height),
    )
    exact = extract_edges(image, placement, edge_target(), roi_half_width=50)
    offset = extract_edges(
        image, placement, edge_target(center=(0.53, 0.47), half=(0.22, 0.28)),
        roi_half_width=50,
    )
    assert len(offset) == 4
    for a, b in zip(exact, offset):
        assert a.center_px[0] == pytest.approx(b.center_px[0], abs=25)
        assert a.center_px[1] == pytest.approx(b.center_px[1], abs=25)


def test_rotation_is_applied_before_scaling_into_an_anisotropic_frame():
    """The fiducial quad need not be square. Scaling before rotating shears the
    feature -- on the ISA target, whose quad aspect is about 1.75, a 4.8 degree tilt
    came out as edges tens of degrees apart and the measurement was meaningless."""
    square = _square_corners(np.array([0.5, 0.5]), np.array([0.1, 0.1]), 45.0)
    sides = [np.linalg.norm(square[(i + 1) % 4] - square[i]) for i in range(4)]
    assert max(sides) == pytest.approx(min(sides), rel=1e-9)

    stretched = _square_corners(np.array([0.5, 0.5]), np.array([0.1, 0.2]), 30.0)
    scaled_back = stretched * np.array([1.0, 0.5])
    sides = [np.linalg.norm(scaled_back[(i + 1) % 4] - scaled_back[i]) for i in range(4)]
    assert max(sides) == pytest.approx(min(sides), rel=1e-9)


def test_a_region_without_geometry_yields_no_edges(tmp_path):
    path = rotated_square_capture(tmp_path)
    image = load_image(path)
    target = target_from_dict({
        "target_id": "bare", "display_name": "bare", "capabilities": ["sfr"],
        "fiducials": {"method": "chart_corners", "corner_markers": [
            {"id": "TL", "relative_position": [0.0, 0.0]},
            {"id": "TR", "relative_position": [1.0, 0.0]},
            {"id": "BR", "relative_position": [1.0, 1.0]},
            {"id": "BL", "relative_position": [0.0, 1.0]}]},
        "regions": [{"region_id": "m", "kind": "slanted_edge"}],
        "patches": [],
    })
    placement = place_chart(
        [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
        (image.width, image.height),
    )
    assert extract_edges(image, placement, target) == []


def test_the_metric_reports_incomplete_when_no_edge_was_found():
    from deltachrome.metrics.base import MetricContext, MetricSpec
    from deltachrome.metrics.sfr import sfr_metric

    result = sfr_metric(
        MetricContext(target=edge_target(), data=[], extras={"edges": []}),
        MetricSpec(metric_id="sfr10_sampling_efficiency", pass_condition="value > 0.9"),
    )
    assert result.outcome == "incomplete"
    assert result.passed is None
    assert "No slanted edge" in result.reason


def test_the_worst_edge_is_chosen_in_the_direction_the_criterion_cares_about():
    """Higher is better for sampling efficiency, lower for response at Nyquist. A
    single 'worst' rule would flatter one of them."""
    from deltachrome.metrics.sfr import _worst

    values = np.array([0.5, 0.9, 0.7])
    assert _worst(values, "value > 0.90") == 0      # least efficient
    assert _worst(values, "value < 0.20") == 1      # most response at Nyquist
    assert _worst(values, "0.45 <= value <= 0.65") == 1   # furthest from the band center


# -- ROI sizing -------------------------------------------------------------
#
# "How many rows" has no answer on its own. The method needs each row to catch the
# edge at a different sub-pixel phase, and the edge drifts by rows*tan(angle) pixels
# down the ROI, so the requirement couples the two dimensions to the tilt.

from deltachrome.imaging.sfr import MIN_PHASE_COVERAGE_PX, phase_coverage


def test_phase_coverage_is_rows_times_the_tangent_of_the_tilt():
    assert phase_coverage(100, 5.0) == pytest.approx(100 * np.tan(np.radians(5.0)))
    # A steeper edge needs fewer rows for the same coverage.
    assert phase_coverage(100, 10.0) > phase_coverage(100, 5.0)


def test_too_little_phase_coverage_is_warned_about():
    """Below one pixel of drift the phases do not close and the oversampled profile
    has gaps that no bin width can fill."""
    shallow = synthetic_edge(0.7, angle_deg=0.3, height=60, width=140)
    result = compute_sfr(shallow)
    assert result.phase_coverage_px < MIN_PHASE_COVERAGE_PX
    assert "Slanted edge is too shallow for reliable analysis." in result.warnings


def test_a_long_enough_edge_has_ample_coverage():
    result = compute_sfr(synthetic_edge(0.7, angle_deg=5.0, height=200))
    assert result.phase_coverage_px > 10
    assert not any("phase" in w for w in result.warnings)


def test_rows_are_cropped_to_whole_phase_cycles():
    """With a fractional cycle the leftover rows revisit phases the rest already
    covered, so those bins get more samples than their neighbors."""
    result = compute_sfr(synthetic_edge(0.7, angle_deg=5.0, height=200))
    rows = result.settings["rows_after_cropping"]
    assert rows <= 200
    cycles = rows * np.tan(np.radians(5.0))
    assert cycles == pytest.approx(round(cycles), abs=0.05)


def test_integer_cycle_cropping_can_be_turned_off():
    result = compute_sfr(
        synthetic_edge(0.7, angle_deg=5.0, height=200), integer_phase_cycles=False
    )
    assert result.settings["rows_after_cropping"] == 200


def test_roi_length_uses_the_available_edge_not_the_bare_minimum(tmp_path):
    """Every extra row is another independent sample of the same transition, so
    length buys noise reduction. Sizing to the coverage floor would throw most of
    the edge away and leave the estimate noisy."""
    path = rotated_square_capture(tmp_path, size=900, half=260, tilt=5.0)
    image = load_image(path)
    placement = place_chart(
        [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
        (image.width, image.height),
    )
    rois = extract_edges(image, placement, edge_target(half=(0.29, 0.29)))
    for roi in rois:
        rows = max(roi.pixels.shape[:2])
        assert rows > 150, f"{roi.edge_id} only {rows} rows"


# -- reporting units --------------------------------------------------------


def test_sfr_metrics_report_in_the_units_fadgi_states():
    """FADGI gives sampling efficiency as a percentage ratio ("> 90%") and SFR50 as a
    "Percentage of Half Sampling Frequency"; response at Nyquist and sharpening are
    modulation values. The numbers should read the way the table does."""
    from deltachrome.metrics.sfr import SCALE, UNITS

    assert SCALE["sfr10_sampling_efficiency"] == 100.0
    assert SCALE["sfr50"] == 100.0
    assert SCALE["sfr_response_at_nyquist"] == 1.0
    assert SCALE["sharpening"] == 1.0
    assert "%" in UNITS["sfr10_sampling_efficiency"]
    assert "%" in UNITS["sfr50"]
    assert UNITS["sfr_response_at_nyquist"] == "modulation"


def test_profile_thresholds_are_in_the_same_units_as_the_readings():
    from deltachrome.profiles.schema import load_profile

    profile = load_profile("fadgi_3star_prints_photographs_2023")
    assert profile.metric("sfr10_sampling_efficiency").pass_condition == "value > 80"
    assert profile.metric("sfr50").pass_condition == "40 <= value <= 75"
    assert profile.metric("sfr_response_at_nyquist").pass_condition == "value < 0.3"
    assert profile.metric("sharpening").pass_condition == "value < 1.05"


def test_edge_aggregation_is_an_explicit_choice():
    """Neither FADGI nor ISO 19264 says how to reduce a field of edges, so the choice
    is the tool's and must be visible, not silent."""
    from deltachrome.errors import ProfileError
    from deltachrome.metrics.base import MetricContext, MetricSpec
    from deltachrome.metrics.sfr import sfr_metric

    class FakeEdge:
        def __init__(self, image, name):
            self.pixels, self.edge_id, self.region_id, self.notes = image, name, "m", []

    edges = [FakeEdge(synthetic_edge(s), f"e{i}") for i, s in enumerate((0.5, 0.8, 1.1))]
    ctx = MetricContext(target=edge_target(), data=[], extras={"edges": edges})

    def run(how):
        return sfr_metric(ctx, MetricSpec(
            metric_id="sfr50", algorithm={"edge_aggregation": how},
            pass_condition="20 <= value <= 95"))

    worst, mean, median = run("worst"), run("mean"), run("median")
    # For a two-sided band, "worst" is the edge furthest from the band's center, so
    # it is one of the extremes rather than the smallest.
    assert worst.value in (worst.metadata["min"], worst.metadata["max"])
    assert worst.value != pytest.approx(mean.value)
    assert median.value == pytest.approx(mean.value, rel=0.25)
    assert worst.metadata["edge_aggregation"] == "worst"

    with pytest.raises(ProfileError, match="edge_aggregation"):
        run("average-ish")


def test_the_worst_and_the_mean_are_both_reported_whichever_is_used():
    from deltachrome.metrics.base import MetricContext, MetricSpec
    from deltachrome.metrics.sfr import sfr_metric

    class FakeEdge:
        def __init__(self, image, name):
            self.pixels, self.edge_id, self.region_id, self.notes = image, name, "m", []

    edges = [FakeEdge(synthetic_edge(s), f"e{i}") for i, s in enumerate((0.5, 0.9))]
    result = sfr_metric(
        MetricContext(target=edge_target(), data=[], extras={"edges": edges}),
        MetricSpec(metric_id="sfr50", pass_condition="20 <= value <= 95"),
    )
    for key in ("min", "max", "mean", "spread", "worst_edge", "edges_measured"):
        assert key in result.metadata


# -- linearization ----------------------------------------------------------
#
# The method assumes the edge profile is linear in luminance. A gamma-encoded
# profile is compressed at one end and stretched at the other, which distorts the
# LSF -- and in opposite directions for edges of opposite polarity, so it does not
# even bias consistently. Imatest linearize for the same reason.


def encode(image, gamma=1.0 / 2.2):
    """Apply a display gamma, as a capture file would carry."""
    return np.clip(image, 0, 1) ** gamma


def decode(image, gamma=1.0 / 2.2):
    return np.clip(image, 0, 1) ** (1.0 / gamma)


def test_measuring_encoded_values_distorts_the_result():
    linear = synthetic_edge(0.8, low=0.05, high=0.95)
    truth = compute_sfr(linear).mtf_at(0.25)

    encoded = encode(linear)
    unlinearised = compute_sfr(encoded).mtf_at(0.25)
    linearized = compute_sfr(
        np.repeat(encoded[:, :, None], 3, axis=2),
        linearize=lambda rgb: decode(rgb[..., 1]),
    ).mtf_at(0.25)

    assert abs(linearized - truth) < abs(unlinearised - truth)


def test_the_encoding_bias_reverses_with_edge_polarity():
    """This is the tell: the optics cannot care which way an edge runs, so a
    polarity-dependent difference is measurement error."""
    rising = synthetic_edge(0.8, low=0.05, high=0.95)
    falling = synthetic_edge(0.8, low=0.95, high=0.05)

    enc = [compute_sfr(encode(e)).mtf_at(0.25) for e in (rising, falling)]
    lin = [
        compute_sfr(
            np.repeat(encode(e)[:, :, None], 3, axis=2),
            linearize=lambda rgb: decode(rgb[..., 1]),
        ).mtf_at(0.25)
        for e in (rising, falling)
    ]
    assert abs(lin[0] - lin[1]) < abs(enc[0] - enc[1])


def test_analysing_encoded_values_is_recorded_in_the_settings():
    """Unreachable from a run -- linearizing comes from the capture's required ICC
    profile -- so it is a setting in the record rather than a warning."""
    result = compute_sfr(np.repeat(synthetic_edge(0.8)[:, :, None], 3, axis=2))
    assert result.settings["linearized"] is False


def test_linearising_removes_the_warning():
    roi = np.repeat(encode(synthetic_edge(0.8))[:, :, None], 3, axis=2)
    result = compute_sfr(roi, linearize=lambda rgb: decode(rgb[..., 1]))
    assert result.settings["linearized"] is True
    assert not any("encoded values" in w for w in result.warnings)


def test_extracted_edges_carry_the_captures_own_transform(tmp_path):
    """Taken from the embedded ICC rather than assuming a gamma exponent, so it works
    for LUT profiles as well as matrix ones."""
    path = rotated_square_capture(tmp_path)
    image = load_image(path)
    placement = place_chart(
        [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
        (image.width, image.height),
    )
    rois = extract_edges(image, placement, edge_target())
    assert all(r.linearize is not None for r in rois)

    sample = np.array([[[0.5, 0.5, 0.5]]])
    luminance = rois[0].linearize(sample)
    assert luminance.shape == (1, 1)
    # A gamma-encoded mid gray is darker than 0.5 in linear luminance.
    assert 0.1 < float(luminance[0, 0]) < 0.35


def test_a_profile_can_turn_linearisation_off():
    from deltachrome.metrics.base import MetricContext, MetricSpec
    from deltachrome.metrics.sfr import sfr_metric

    class FakeEdge:
        def __init__(self, image):
            self.pixels = np.repeat(image[:, :, None], 3, axis=2)
            self.edge_id, self.region_id, self.notes = "e0", "m", []
            self.linearize = lambda rgb: decode(rgb[..., 1])

    ctx = MetricContext(
        target=edge_target(), data=[], extras={"edges": [FakeEdge(encode(synthetic_edge(0.8)))]}
    )
    on = sfr_metric(ctx, MetricSpec(metric_id="sfr50", pass_condition="20 <= value <= 95"))
    off = sfr_metric(
        ctx,
        MetricSpec(metric_id="sfr50", algorithm={"linearize": False},
                   pass_condition="20 <= value <= 95"),
    )
    assert on.metadata["linearized"] is True
    assert off.metadata["linearized"] is False
    assert on.value != pytest.approx(off.value)


# -- ISO interleave binning -------------------------------------------------


@pytest.mark.parametrize("sigma", [0.5, 0.8, 1.1])
def test_iso_interleave_binning_agrees_with_distance_binning(sigma):
    """ISO and Imatest bin whole scan lines into phase groups and interleave the
    group averages; Deltachrome bins each pixel by its continuous distance. For a
    uniform tilt they are the same measurement."""
    image = synthetic_edge(sigma)
    distance = compute_sfr(image).mtf_at(0.25)
    interleaved = compute_sfr(image, binning="iso_interleave").mtf_at(0.25)
    assert interleaved == pytest.approx(distance, abs=0.002)
    assert interleaved == pytest.approx(gaussian_mtf(sigma, 0.25), abs=0.02)


def test_the_interleave_order_matters():
    """Reversing it produces a profile that looks entirely plausible and reports MTF
    well off the truth. Only a known answer catches that -- which is why the ordering
    is asserted against the analytic curve rather than against itself."""
    sigma = 0.8
    image = synthetic_edge(sigma)
    correct = compute_sfr(image, binning="iso_interleave")
    truth = gaussian_mtf(sigma, 0.25)
    assert correct.mtf_at(0.25) == pytest.approx(truth, abs=0.02)

    # Rebuild the same profile with the phase groups interleaved the other way.
    reversed_esf = correct.esf.copy()
    k = 4
    for start in range(k):
        reversed_esf[start::k] = correct.esf[k - 1 - start :: k]
    from deltachrome.imaging.sfr import _finish

    spoiled = _finish(
        reversed_esf, correct.bin_width, correct.edge_angle_deg, False,
        correct.rows_used, correct.phase_coverage_px, "hamming", "central", True,
        True, [], {},
    )
    assert abs(spoiled.mtf_at(0.25) - truth) > 4 * abs(correct.mtf_at(0.25) - truth)


def test_binning_choice_is_recorded():
    for choice in ("distance", "iso_interleave"):
        assert compute_sfr(synthetic_edge(0.8), binning=choice).settings["binning"] == choice


def test_interleave_refuses_when_a_phase_bin_is_empty():
    """Distance binning tolerates uneven phase coverage; interleaving cannot."""
    with pytest.raises(SFRError, match="Phase bin"):
        compute_sfr(synthetic_edge(0.8, angle_deg=0.02, height=30), binning="iso_interleave")


# -- distinguishing real haloing from transform leakage --------------------


def test_a_clean_edge_has_no_spatial_overshoot():
    assert compute_sfr(synthetic_edge(0.9)).overshoot() == pytest.approx(0.0, abs=0.2)


def test_unsharp_masking_produces_overshoot_and_a_mid_frequency_peak():
    """The signature of real sharpening: the curve lifts above 1 *after* DC, and the
    edge profile physically overshoots its own plateau."""
    import cv2

    base = synthetic_edge(0.9)
    blurred = cv2.GaussianBlur(base, (0, 0), 2.0)
    sharpened = np.clip(base + 1.6 * (base - blurred), 0, 1)

    result = compute_sfr(sharpened)
    assert result.overshoot() > 2.0
    assert result.peak_modulation(0.0) > 1.10
    # Unsharp masking peaks well above the leakage-dominated bins.
    assert result.peak_frequency(0.0) > 0.05


def test_overshoot_is_immune_to_the_leakage_that_lifts_the_transform():
    """The two measures fail differently, which is why both are reported: a peak in
    the lowest transform bins with no matching overshoot is not a halo."""
    clean = compute_sfr(synthetic_edge(0.9))
    assert clean.overshoot() < 0.5
    assert clean.peak_modulation() <= 1.0


def test_the_sharpening_metric_reports_the_evidence_not_just_a_verdict():
    from deltachrome.metrics.base import MetricContext, MetricSpec
    from deltachrome.metrics.sfr import sfr_metric

    class FakeEdge:
        def __init__(self, image):
            self.pixels = np.repeat(image[:, :, None], 3, axis=2)
            self.edge_id, self.region_id, self.notes, self.linearize = "e0", "m", [], None

    ctx = MetricContext(
        target=edge_target(), data=[], extras={"edges": [FakeEdge(synthetic_edge(0.9))]}
    )
    result = sfr_metric(ctx, MetricSpec(metric_id="sharpening", pass_condition="value < 1.05"))
    for key in ("peak_over_all_frequencies", "peak_frequency_range_cyc_px",
                "spatial_overshoot_pct", "peak_search_min_frequency"):
        assert key in result.metadata


def test_overshoot_is_calibrated_against_known_sharpening():
    """A clean edge gives 0%, a mild sharpen about 10%, a strong one about 27% --
    and the ordering must hold, since it is what makes the number interpretable."""
    import cv2

    base = synthetic_edge(0.9, low=0.25, high=0.75)   # headroom, so nothing clips
    blurred = cv2.GaussianBlur(base, (0, 0), 2.0)

    clean = compute_sfr(base).overshoot()
    mild = compute_sfr(base + 0.7 * (base - blurred)).overshoot()
    strong = compute_sfr(base + 1.6 * (base - blurred)).overshoot()

    assert clean == pytest.approx(0.0, abs=0.5)
    assert 5.0 < mild < 16.0
    assert 20.0 < strong < 35.0
    assert clean < mild < strong


def test_overshoot_ignores_noise_away_from_the_edge():
    """A halo sits beside the edge; noise is everywhere. Searching the whole profile
    returns the noisiest bin instead of the overshoot -- on a real capture that
    inflated the figure from 0.35% to 2.33%."""
    rng = np.random.default_rng(11)
    base = synthetic_edge(0.9, low=0.25, high=0.75)
    noisy = base + rng.normal(0, 0.004, base.shape)

    result = compute_sfr(noisy)
    assert result.overshoot() < 3.0
    # Widening the search window lets distant noise in.
    assert result.overshoot(search_px=200.0) >= result.overshoot()


def test_clipping_would_hide_overshoot_so_the_calibration_uses_headroom():
    """Recorded because it caught a bad calibration: with the edge running 0.05-0.95,
    unsharp masking clips and overshoot saturates at the same value for every
    strength, which looks like the measure not working."""
    import cv2

    clipped_base = synthetic_edge(0.9, low=0.05, high=0.95)
    blurred = cv2.GaussianBlur(clipped_base, (0, 0), 2.0)
    mild = compute_sfr(np.clip(clipped_base + 0.7 * (clipped_base - blurred), 0, 1))
    strong = compute_sfr(np.clip(clipped_base + 1.6 * (clipped_base - blurred), 0, 1))
    assert mild.overshoot() == pytest.approx(strong.overshoot(), abs=0.5)



def test_the_default_aggregation_follows_opendice():
    """Neither standard specifies it. OpenDICE -- the open FADGI-aligned
    implementation -- displays "the RGB average sampling efficiency values of all the
    edge regions", so the default is the mean and a Deltachrome figure lines up with
    a GoldenThread report of the same capture without conversion."""
    from deltachrome.metrics.base import MetricContext, MetricSpec
    from deltachrome.metrics.sfr import DEFAULT_EDGE_AGGREGATION, sfr_metric

    assert DEFAULT_EDGE_AGGREGATION == "mean"

    class FakeEdge:
        def __init__(self, image, name):
            self.pixels, self.edge_id, self.region_id = image, name, "m"
            self.notes, self.linearize = [], None

    edges = [FakeEdge(synthetic_edge(s), f"e{i}") for i, s in enumerate((0.5, 0.8, 1.1))]
    ctx = MetricContext(target=edge_target(), data=[], extras={"edges": edges})
    result = sfr_metric(
        ctx, MetricSpec(metric_id="sfr50", pass_condition="20 <= value <= 95")
    )
    assert result.metadata["edge_aggregation"] == "mean"
    assert result.value == pytest.approx(result.metadata["mean"])


def test_the_worst_edge_is_still_reported_when_the_mean_is_judged():
    """A mean can pass a system with one badly soft corner; the worst and the spread
    are what show it, so they are always present."""
    from deltachrome.metrics.base import MetricContext, MetricSpec
    from deltachrome.metrics.sfr import sfr_metric

    class FakeEdge:
        def __init__(self, image, name):
            self.pixels, self.edge_id, self.region_id = image, name, "m"
            self.notes, self.linearize = [], None

    edges = [FakeEdge(synthetic_edge(s), f"e{i}") for i, s in enumerate((0.5, 0.5, 1.6))]
    result = sfr_metric(
        MetricContext(target=edge_target(), data=[], extras={"edges": edges}),
        MetricSpec(metric_id="sfr50", pass_condition="20 <= value <= 95"),
    )
    assert result.metadata["min"] < result.value
    assert result.metadata["spread"] > 0
    assert result.metadata["worst_edge"]


# -- per color record -------------------------------------------------------


def _channel_roi(sharpness=(1.6, 1.1, 2.4), size=(400, 90)):
    """One edge, deliberately sharpest in green and softest in blue."""
    import numpy as np

    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    edge = w / 2 + 0.09 * yy
    planes = [1.0 / (1.0 + np.exp(-(xx - edge) / s)) for s in sharpness]
    return np.clip(np.stack(planes, axis=-1), 0.0, 1.0)


def test_asking_for_a_channel_returns_that_channel_not_luminance():
    """It used to return luminance. `_as_luminance` applied the linearizer and
    returned immediately, so `channel` was accepted and silently ignored -- the
    politest possible way to answer a different question."""
    from deltachrome.imaging.sfr import compute_sfr

    roi = _channel_roi()
    fifties = {
        channel: compute_sfr(roi, channel=channel).sfr50()
        for channel in ("r", "g", "b")
    }
    assert fifties["g"] > fifties["r"] > fifties["b"]


def test_a_per_channel_lineariser_still_leaves_a_channel_to_choose():
    """Linearize first, then select. A luminance transform returns one plane and
    there is nothing left to pick; a per-channel one returns three."""
    import numpy as np

    from deltachrome.imaging.sfr import compute_sfr

    roi = np.clip(_channel_roi() ** (1 / 2.2), 0.0, 1.0)

    def per_channel(rgb):
        return np.asarray(rgb, dtype=np.float64) ** 2.2

    red = compute_sfr(roi, channel="r", linearize=per_channel).sfr50()
    blue = compute_sfr(roi, channel="b", linearize=per_channel).sfr50()
    assert red > blue


def test_a_luminance_lineariser_is_still_honoured_unchanged():
    """The reduced plane it returns is the signal, and no channel is taken from
    it -- which is what every existing caller relies on."""
    import numpy as np

    from deltachrome.imaging.sfr import compute_sfr

    roi = _channel_roi()

    def to_luminance(rgb):
        return np.asarray(rgb, dtype=np.float64)[..., 1]

    with_transform = compute_sfr(roi, linearize=to_luminance).sfr50()
    green_alone = compute_sfr(roi, channel="g").sfr50()
    assert with_transform == pytest.approx(green_alone)


# -- the LSF's peak, and the window placed on it -------------------------------


def _finish_with(esf):
    """Run the tail of the SFR path over a hand-built ESF.

    At this level rather than over pixels, deliberately. The condition is a property
    of the *oversampled profile* -- its outermost bin is fed by the fewest pixels and
    so carries the most noise -- and a synthetic ROI that reproduces it reliably is a
    harder thing to write than the profile itself. Building the profile states the
    case exactly and cannot pass by accident.
    """
    from deltachrome.imaging.sfr import _finish

    return _finish(
        np.asarray(esf, dtype=np.float64), 0.25, 5.0, False, 300, 12.0,
        "hann", "central", True, False, [], {},
    )


def _clean_esf(bins=500, blur=6.0):
    """A smooth step in the middle of the profile, as every real edge produces."""
    x = np.arange(bins, dtype=np.float64) - bins / 2.0
    return 0.08 + 0.22 / (1.0 + np.exp(x / blur))


# -- the LSF's peak, and the window placed on it -------------------------------


def test_the_lsf_peak_is_looked_for_in_the_interior():
    """The transition is at the centre by construction, so a maximum at the array's
    end is never the edge -- it is the sparsest bin's noise, differentiated.

    Found on a real capture: one edge of a QA-62 whose last ESF bin dipped 0.06 below
    its plateau, giving a derivative 1.59x the true peak. The window centred on it,
    wrapped, and the DC term the whole curve is divided by came out half its proper
    size -- so the curve was inflated about twofold. It reported no SFR50 at all,
    0.987 modulation at Nyquist and a peak of 1.116, where its three sibling edges on
    the same square read 68-71% SFR50.
    """
    clean = _clean_esf()
    healthy = _finish_with(clean)

    dipped = clean.copy()
    dipped[-1] -= 0.06
    guarded = _finish_with(dipped)

    # The dip really does out-shout the transition, or this proves nothing.
    lsf = np.gradient(dipped)
    assert abs(lsf[-1]) > abs(lsf[len(lsf) // 2]), (
        "the fixture must reproduce the condition: a boundary bin louder than the edge"
    )

    assert guarded.peak_modulation() == pytest.approx(1.0, abs=0.02), (
        "modulation above the DC value means the curve was divided by the wrong "
        "number, not that the system is sharpening"
    )
    assert guarded.sfr50() is not None, "a clean edge must still yield an SFR50"
    assert guarded.sfr50() == pytest.approx(healthy.sfr50(), rel=0.02), (
        "one noisy bin at the far end of the profile must not move the reading"
    )


def test_the_window_does_not_wrap_around_the_lsf():
    """np.roll is circular: a window centred near one end reappears at the other,
    where it weights the far plateau instead of the transition -- and on the edge
    that found this, the wrap flipped the sign of the windowed LSF's sum, which is
    the quantity the whole curve is normalised by.

    With the peak found in the interior this is a no-op on every edge here. It is the
    guard that keeps a near-boundary peak from being a silent factor of two rather
    than a visible truncation.
    """
    import inspect

    from deltachrome.imaging import sfr as sfr_module

    source = inspect.getsource(sfr_module._finish)
    assert "np.roll(w" not in source, (
        "the window must be shifted with zero fill, not rolled"
    )
    assert "placed[shift:]" in source and "placed[:shift]" in source


def test_a_normal_edge_is_untouched_by_either_guard():
    """The claim that matters: 156 of the 164 slanted edges across every committed
    placement in this repository produce a bit-identical curve, and the seven others
    differ by at most 0.00001 in modulation. The one that changed materially was
    broken -- an ISA 120 module reporting a peak modulation of 2.402 and an SFR50 of
    52.39%, which read 1.000 and 31.69% once the boundary bin stopped winning."""
    result = _finish_with(_clean_esf())

    assert result.peak_modulation() == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < result.sfr50() < 100.0
    assert np.asarray(result.mtf).max() <= 1.0 + 1e-6, "nothing rises above DC"
