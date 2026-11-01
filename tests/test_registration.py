"""Color-channel misregistration, validated against injected shifts.

A synthetic edge can be built with a displacement between color records that is
known exactly, so the estimator is checked against a number rather than against
itself. The cases that matter are the ones where a naive implementation reports a
displacement that is not there: records of different sharpness, records of
different exposure read through a gamma encoding, and the apparent size difference
that shows up on a square feature and flips sign with edge polarity.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from deltachrome.errors import ProfileError
from deltachrome.imaging.edges import EdgeROI
from deltachrome.imaging.icc import parse_icc
from deltachrome.imaging.registration import (
    CHANNEL_PAIRS,
    field_points,
    field_vectors,
    measure_edge_registration,
    measure_edges,
    pair_name,
)
from deltachrome.metrics.base import MetricContext, MetricSpec, get_metric

METRIC = "color_channel_misregistration"


def channel(sigma=0.7, shift=0.0, angle_deg=5.0, height=400, width=100, x0=50.0,
            low=0.05, high=0.95, noise=0.0, seed=0, polarity=+1):
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    edge = x0 + shift + rows * np.tan(np.radians(angle_deg))
    profile = 0.5 * (1 + erf((cols - edge) / (sigma * np.sqrt(2))))
    if polarity < 0:
        profile = 1.0 - profile
    image = low + (high - low) * profile
    if noise:
        image = image + np.random.default_rng(seed).normal(0, noise, image.shape)
    return image


def rgb_edge(*, shift_r=0.0, shift_g=0.0, shift_b=0.0, **kw):
    return np.stack(
        [channel(shift=shift_r, **kw),
         channel(shift=shift_g, **kw),
         channel(shift=shift_b, **kw)],
        axis=-1,
    )


def roi(pixels, edge_id="e0", region_id="module"):
    return EdgeROI(
        edge_id=edge_id, region_id=region_id, pixels=pixels,
        orientation="vertical", center_px=(0.0, 0.0), bounds_px=(0, 0, 0, 0),
    )


# -- the measurement is correct --------------------------------------------


@pytest.mark.parametrize("shift", [0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
def test_an_injected_shift_is_recovered(shift):
    measured = measure_edge_registration(rgb_edge(shift_b=shift))
    assert measured.shift("B", "G") == pytest.approx(shift, abs=0.002)
    assert measured.shift("G", "R") == pytest.approx(0.0, abs=0.002)


def test_the_displacement_axis_follows_the_edge_orientation():
    vertical = measure_edge_registration(rgb_edge(shift_b=0.3))
    horizontal = measure_edge_registration(
        np.transpose(rgb_edge(shift_b=0.3), (1, 0, 2))
    )
    assert vertical.axis == "x"
    assert horizontal.axis == "y"
    assert horizontal.shift("B", "G") == pytest.approx(vertical.shift("B", "G"), abs=0.002)


@pytest.mark.parametrize("sigma_b", [0.9, 1.2, 1.8])
def test_records_of_different_sharpness_do_not_read_as_displaced(sigma_b):
    """Blue is routinely the softest record. A blur difference is not a displacement,
    and an estimator that confused the two would report misregistration on every
    lens with longitudinal chromatic aberration."""
    pixels = np.stack(
        [channel(sigma=0.7), channel(sigma=0.7), channel(sigma=sigma_b)], axis=-1
    )
    measured = measure_edge_registration(pixels)
    assert abs(measured.shift("B", "G")) < 0.01


def test_polarity_alone_does_not_change_the_reading():
    for polarity in (+1, -1):
        measured = measure_edge_registration(rgb_edge(shift_b=0.2, polarity=polarity))
        assert measured.shift("B", "G") == pytest.approx(0.2, abs=0.002)


def test_the_standard_error_falls_with_row_count_and_rises_with_noise():
    quiet = measure_edge_registration(rgb_edge(noise=0.004, height=400))
    noisy = measure_edge_registration(rgb_edge(noise=0.02, height=400))
    short = measure_edge_registration(rgb_edge(noise=0.004, height=60))
    assert noisy.standard_error_px > quiet.standard_error_px
    assert short.standard_error_px > quiet.standard_error_px
    # And the reading itself stays near zero at a realistic noise level.
    assert abs(quiet.shift("B", "G")) < 0.02


# -- pairing separates displacement from apparent size ----------------------


def _square(translation, size_term, region="module"):
    """The two edges one axis of a square feature presents.

    ``translation`` moves both records' edges the same way -- a displacement.
    ``size_term`` moves them oppositely -- the blue square measuring bigger or
    smaller than the green one, which is not a displacement at all.
    """
    return [
        roi(rgb_edge(shift_b=translation + size_term, polarity=+1), "e0", region),
        roi(rgb_edge(shift_b=translation - size_term, polarity=-1), "e2", region),
    ]


def test_pairing_opposite_edges_separates_the_two_terms():
    measured, failures = measure_edges(_square(translation=0.12, size_term=0.20))
    assert not failures
    point = field_points(measured)[0]
    assert point.paired
    assert point.translation["G-B"] == pytest.approx(-0.12, abs=0.005)
    assert abs(point.antisymmetric["G-B"]) == pytest.approx(0.20, abs=0.005)


def test_leaving_the_edges_unpaired_folds_the_size_term_into_the_answer():
    """The reason pairing is the default: judged edge by edge, a size difference
    that is not a displacement is reported as one, at full strength."""
    edges = _square(translation=0.0, size_term=0.20)
    measured, _ = measure_edges(edges)
    paired = field_points(measured, pairing="opposite_edges")
    per_edge = field_points(measured, pairing="per_edge")
    assert max(p.spread_px for p in paired) < 0.01
    assert max(p.spread_px for p in per_edge) == pytest.approx(0.20, abs=0.005)


def test_a_lone_edge_is_noted_only_where_its_partner_was_printed():
    """A square printed two edges on this axis and one failed: the reading carries
    the size term at full strength, and the operator is told which module. A chart
    printing one edge per axis by design -- the ISA FADGI 19264 -- says nothing,
    because nothing failed."""
    measured, _ = measure_edges([roi(rgb_edge(shift_b=0.1))])
    region, axis = measured[0].region_id, measured[0].axis

    lost = field_points(measured, declared={(region, axis): 2})[0]
    assert not lost.paired
    assert lost.antisymmetric == {}
    assert lost.notes == [f"Edge not measured on {region}."]

    by_design = field_points(measured, declared={(region, axis): 1})[0]
    assert not by_design.paired
    assert by_design.notes == []


def test_the_displacement_vector_is_never_smaller_than_its_axes():
    edges = [
        roi(rgb_edge(shift_b=0.2), "e0", "module"),
        roi(rgb_edge(shift_b=0.2), "e2", "module"),
        roi(np.transpose(rgb_edge(shift_b=0.2), (1, 0, 2)), "e1", "module"),
        roi(np.transpose(rgb_edge(shift_b=0.2), (1, 0, 2)), "e3", "module"),
    ]
    measured, _ = measure_edges(edges)
    points = field_points(measured)
    vectors = field_vectors(points)
    assert set(vectors) == {"module"}
    magnitude = vectors["module"]["worst_magnitude_px"]
    assert magnitude == pytest.approx(0.2 * np.sqrt(2), abs=0.01)
    assert magnitude >= max(p.spread_px for p in points)


# -- linearization ----------------------------------------------------------


@pytest.mark.parametrize("icc", ["matrix_trc", "lut"])
def test_the_channel_linearizer_never_mixes_the_records(icc):
    """The invariant that makes a per-channel measurement possible at all. A 3x3 to
    XYZ would blend the records together and average a displacement between two of
    them into all three, so a linearization that mixes channels would quietly
    destroy the measurement it was meant to enable."""
    from fixtures import adobe_rgb_icc, lut_icc_from_matrix

    profile = parse_icc(adobe_rgb_icc() if icc == "matrix_trc" else lut_icc_from_matrix())
    linearize, description = profile.channel_linearizer()
    assert profile.transform_kind == icc
    assert description and description != "none"

    base = np.array([[[0.2, 0.4, 0.6]]])
    changed = np.array([[[0.2, 0.9, 0.6]]])
    out_base = linearize(base)
    out_changed = linearize(changed)
    assert out_base[..., 0] == pytest.approx(out_changed[..., 0])
    assert out_base[..., 2] == pytest.approx(out_changed[..., 2])
    assert out_base[..., 1] != pytest.approx(out_changed[..., 1])


def test_the_channel_linearizer_actually_linearises():
    """Both constructions must undo the encoding, not merely keep the channels
    separate. A ramp that is even in luminance is uneven once encoded; linearized
    again it has to come back even, because the edge analysis assumes a profile
    linear in luminance."""
    from fixtures import ADOBE_RGB_GAMMA, adobe_rgb_icc, lut_icc_from_matrix

    even_in_luminance = np.linspace(0.05, 0.9, 9)
    encoded = np.stack([even_in_luminance ** (1 / ADOBE_RGB_GAMMA)] * 3, axis=-1)
    for blob in (adobe_rgb_icc(), lut_icc_from_matrix()):
        linearize, _ = parse_icc(blob).channel_linearizer()
        recovered = linearize(encoded)[..., 1]
        recovered = recovered / recovered.max()
        assert recovered == pytest.approx(even_in_luminance / 0.9, abs=0.005)


def test_a_gamma_encoding_biases_the_reading_but_only_slightly():
    """Records of different exposure read through a gamma encoding pick up a
    polarity-dependent bias. It is real, which is why linearization is on by
    default, and it is small, which is why a profile that cannot supply a
    per-channel curve is still worth measuring."""
    worst = 0.0
    for polarity in (+1, -1):
        green = channel(polarity=polarity, low=0.05, high=0.95) ** (1 / 2.2)
        blue = channel(polarity=polarity, low=0.04, high=0.72) ** (1 / 2.2)
        pixels = np.stack([green, green, blue], axis=-1)
        worst = max(worst, abs(measure_edge_registration(pixels).shift("B", "G")))
    assert 0.0 < worst < 0.02


# -- the plugin -------------------------------------------------------------


def _spec(**algorithm):
    return MetricSpec(
        metric_id=METRIC, requires_capability="color_registration",
        aggregation="max", pass_condition="value < 0.33", units="pixels",
        algorithm=algorithm,
    )


def _run(edges, **algorithm):
    ctx = MetricContext(target=None, data=[], extras={"edges": edges})
    return get_metric(METRIC)(ctx, _spec(**algorithm))


def test_the_metric_reports_incomplete_without_edges():
    result = _run([])
    assert result.outcome == "incomplete"
    assert result.passed is None
    assert "no slanted edge" in (result.reason or "").lower()


def test_the_metric_measures_and_judges():
    result = _run(_square(translation=0.10, size_term=0.0))
    assert result.outcome == "evaluated"
    assert result.value == pytest.approx(0.10, abs=0.01)
    assert result.passed is True
    assert result.metadata["points_paired"] == 1
    assert result.metadata["threshold"] == 0.33


def test_the_metric_fails_a_capture_beyond_the_limit():
    result = _run(_square(translation=0.6, size_term=0.0))
    assert result.passed is False


def test_every_reading_is_reported_whichever_drives_the_verdict():
    edges = _square(0.1, 0.0, "a") + _square(0.4, 0.0, "b")
    worst = _run(edges)
    mean = _run(edges, registration_aggregation="mean")
    assert worst.value == pytest.approx(0.4, abs=0.01)
    assert mean.value == pytest.approx(0.25, abs=0.01)
    for result in (worst, mean):
        for key in ("min", "max", "mean", "median", "spread"):
            assert key in result.metadata


def test_the_aggregation_default_is_the_conservative_one():
    """Unlike sampling efficiency, misregistration is a local defect: one corner
    fringing badly is a fault whether or not the rest of the field is clean."""
    edges = _square(0.1, 0.0, "a") + _square(0.4, 0.0, "b")
    assert _run(edges).metadata["registration_aggregation"] == "worst"


def test_the_report_carries_the_settings_that_produced_it():
    result = _run(_square(0.1, 0.0))
    for key in ("pairing", "component", "registration_aggregation",
                "measurement_floor_px", "linearization", "channel_pairs"):
        assert key in result.metadata
    assert result.metadata["channel_pairs"] == [pair_name(p) for p in CHANNEL_PAIRS]


def test_a_reading_at_the_measurement_floor_is_recorded_not_announced():
    """The floor stays in the record. It no longer prints a note, since it never
    changes the verdict (2026-10-02 review)."""
    result = _run(_square(translation=0.0, size_term=0.0))
    assert result.metadata["measurement_floor_px"] >= 0
    assert not any("measurement floor" in note for note in result.notes)


def test_the_antisymmetric_term_is_named_rather_than_folded_in():
    """Kept out of the verdict and kept in the record.

    Its *figures* are a column in the values table now -- per position, axis and
    pair -- where they were three sentences ending "Worst here: G-B", which is a
    sorted column read aloud. What the note still has to carry is what the
    column cannot say about itself: why pairing happens at all.
    """
    result = _run(_square(translation=0.0, size_term=0.25))
    assert result.value < 0.02
    assert result.metadata["antisymmetric_term_px"]["G-B"] == pytest.approx(0.25, abs=0.01)
    # Neither the numbers nor the explanation of pairing are read aloud any more:
    # the figures are a column, and why pairing happens is documentation.
    assert not any("antisymmetric" in note for note in result.notes)


def test_unknown_reduction_choices_are_refused():
    for algorithm in ({"pairing": "nonsense"}, {"component": "nonsense"},
                      {"registration_aggregation": "nonsense"}):
        with pytest.raises(ProfileError):
            _run(_square(0.1, 0.0), **algorithm)


def test_no_note_claims_a_standard_specifies_the_reduction():
    """Neither FADGI nor ISO 19264-1 says how a field of edges is reduced. The
    sentence saying so went to documentation; what must never appear in its place
    is a note implying the opposite."""
    result = _run(_square(0.1, 0.0))
    assert not any("specif" in note for note in result.notes)


# -- what a run keeps and what a report shows -------------------------------


def test_registration_is_recorded_independently_of_any_metric(tmp_path):
    """Recorded by the pipeline rather than inside the metric, so a run under a
    profile that declares no registration criterion still carries the raw
    per-record edge locations and can be re-scored later."""
    from deltachrome.imaging.edges import extract_edges
    from deltachrome.imaging.geometry import place_chart
    from deltachrome.imaging.loader import load_image
    from deltachrome.results.edges import build_registration_records, registration_summary
    from test_sfr import edge_target, rotated_square_capture

    image = load_image(rotated_square_capture(tmp_path))
    placement = place_chart(
        [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
        (image.width, image.height),
    )
    measured, failures = measure_edges(extract_edges(image, placement, edge_target()))
    assert not failures and len(measured) == 4

    records = build_registration_records(measured)
    assert {r.edge_id for r in records} == {m.edge_id for m in measured}
    for record in records:
        assert set(record.channel_locations_px) == {"R", "G", "B"}
        assert record.polarity in ("rising", "falling")
        assert record.linearization != ""

    summary = registration_summary(measured)
    assert len(summary["positions"]) == 2
    assert all(p["paired"] for p in summary["positions"])
    # A synthetic capture has no misregistration to find, and none is invented.
    assert summary["worst_position_px"] < 0.02


def test_the_report_lays_the_measurement_out_the_way_opendice_does():
    """One column per channel pair in OpenDICE's order and sign, positions down the
    page, axes separated -- so this can be read against a GoldenThread screen
    column for column instead of being mentally converted."""
    from deltachrome.reporting.console import render

    class Run:
        run_id, created_utc = "x", "2026-01-01T00:00:00+00:00"
        image = {"filename": "c.tif", "width": 10, "height": 10, "bit_depth": 16}
        color_management = {"icc_profile": "p"}
        target = {"display_name": "t", "patch_count": 0}
        reference = {"path": None, "sample_count": 0, "provenance": {},
                     "measurement_mode": None}
        profile = {"display_name": "prof"}
        geometry, sampling = {}, {}
        overall, overall_explanation, warnings = "PASS", "fine", []
        patches: list = []
        edges: list = []
        edge_field_summary: dict = {}
        registration: list = []
        rulers: dict = {}
        registration_summary = {
            "positions": [
                {"region": "module_tl", "axis": "x", "edges": ["a", "b"], "paired": True,
                 "translation_px": {"G-B": 0.011, "G-R": -0.009, "B-R": -0.020},
                 "antisymmetric_px": {"G-B": 0.030, "G-R": -0.005, "B-R": -0.035},
                 "standard_error_px": 0.003},
                {"region": "module_tl", "axis": "y", "edges": ["c"], "paired": False,
                 "translation_px": {"G-B": -0.012, "G-R": 0.007, "B-R": 0.019},
                 "antisymmetric_px": {}, "standard_error_px": 0.004},
            ],
            "displacement_vectors": {
                "module_tl": {"center_px": [1.0, 2.0], "worst_magnitude_px": 0.028,
                              "pairs": {}},
            },
            "rotation_between_records_deg": {"R-G_mean_deg": -0.0005},
        }

    class Result:
        run = Run()

        class evaluation:
            results: list = []

    text = render(Result())
    assert "CHANNEL REGISTRATION" in text
    for pair in ("G-B", "G-R", "B-R"):
        assert pair in text
    assert "+0.011" in text and "-0.020" in text      # signed, not absolute
    assert "unpaired" in text
    assert "antisymmetric term" in text
    assert "worst displacement vector" in text
    assert "rotation between records" in text


# -- direction, not just magnitude -------------------------------------------


class FakePlacement:
    """A 1:1 placement over a square frame: pixels are relative × size."""

    def __init__(self, size=1000):
        self.image_size = (size, size)
        self.size = size

    def to_relative(self, points):
        return np.asarray(points, dtype=np.float64) / self.size


def _vectors(field):
    """{region: {"center_px": [...], "pairs": {pair: {dx, dy, magnitude_px}}}}"""
    out = {}
    for region, (center, (dx, dy)) in field.items():
        out[region] = {
            "center_px": list(center),
            "pairs": {"G-B": {"dx": dx, "dy": dy,
                              "magnitude_px": float(np.hypot(dx, dy))}},
            "worst_magnitude_px": float(np.hypot(dx, dy)),
        }
    return out


def _radial_field(strength=0.001):
    """A magnification difference: every arrow points out of the frame center,
    growing in proportion to distance from it."""
    field = {}
    for i, (x, y) in enumerate(
        [(200, 200), (800, 200), (800, 800), (200, 800), (500, 200)]
    ):
        ox, oy = x - 500, y - 500
        field[f"m{i}"] = ((x, y), (ox * strength, oy * strength))
    return _vectors(field)


def _uniform_field(dx=0.4, dy=0.15):
    """A bodily offset: the same displacement everywhere."""
    return _vectors({
        f"m{i}": ((x, y), (dx, dy))
        for i, (x, y) in enumerate(
            [(200, 200), (800, 200), (800, 800), (200, 800), (500, 200)]
        )
    })


def test_a_magnification_difference_reads_as_radial_and_growing():
    """It and a bodily offset produce the same magnitudes. Only the directions
    tell them apart, which is why direction is worth recording at all."""
    from deltachrome.metrics.registration import (
        _decomposition_note,
        _radial_decomposition,
    )

    decomposition = _radial_decomposition(_radial_field(), FakePlacement())
    reading = decomposition["by_pair"]["G-B"]
    assert reading["radial_growth_with_radius"] > 0.9
    assert reading["mean_abs_radial_px"] > reading["mean_abs_tangential_px"]
    assert "different magnification" in _decomposition_note(decomposition)
    assert "confirm against a second target" in _decomposition_note(decomposition)


def test_a_bodily_offset_reads_as_coherent_rather_than_radial():
    from deltachrome.metrics.registration import (
        _decomposition_note,
        _radial_decomposition,
    )

    decomposition = _radial_decomposition(_uniform_field(), FakePlacement())
    assert decomposition["by_pair"]["G-B"]["direction_coherence"] > 0.99
    assert "bodily offset" in _decomposition_note(decomposition)


def test_the_two_are_not_separated_by_magnitude_alone():
    """The point of the whole exercise, asserted directly: build the two fields
    so their magnitudes match, and check only the decomposition tells them
    apart."""
    from deltachrome.metrics.registration import _radial_decomposition

    radial = _radial_field(strength=0.001)
    size = np.hypot(*[radial["m0"]["pairs"]["G-B"][k] for k in ("dx", "dy")])
    uniform = _uniform_field(dx=size, dy=0.0)

    a = [v["worst_magnitude_px"] for v in radial.values()]
    b = [v["worst_magnitude_px"] for v in uniform.values()]
    assert max(a) == pytest.approx(max(b), rel=0.3)

    def coherence(field):
        return _radial_decomposition(
            field, FakePlacement())["by_pair"]["G-B"]["direction_coherence"]

    assert coherence(radial) < 0.5
    assert coherence(uniform) > 0.99


def test_the_decomposition_never_mixes_two_channel_pairs():
    """The bug this was written for, and the reason the fixtures above missed it.

    Every fixture here carries a single pair, so nothing exercised what happens
    when the *worst* pair differs from position to position -- which it does on
    a real capture. ``G-B`` and ``B-R`` do not share a sign convention: the same
    physical displacement points opposite ways in the two, so taking whichever
    is largest at each position pools two populations 180 degrees apart.

    Measured on the ISA Golden Thread run, where the worst pair is ``B-R`` at
    four modules and ``G-B`` at the fifth, the pooled coherence read 0.598 and
    the run reported that neither pattern was clear. Per pair the same capture
    reads 0.986 and 0.979 -- two clean bodily offsets.

    Built here as the minimal version of that: one pair uniform left, the other
    uniform right, with the second larger at exactly one position so that a
    worst-pair rule has to straddle them.
    """
    from deltachrome.metrics.registration import _radial_decomposition

    places = [(200, 200), (800, 200), (800, 800), (200, 800), (500, 200)]
    field = {}
    for i, (x, y) in enumerate(places):
        # "ab" always points +x; "ba" always points -x, and is the larger of the
        # two at the last position only.
        big = 0.5 if i == len(places) - 1 else 0.1
        field[f"m{i}"] = {
            "center_px": [x, y],
            "pairs": {
                "ab": {"dx": 0.3, "dy": 0.0, "magnitude_px": 0.3},
                "ba": {"dx": -big, "dy": 0.0, "magnitude_px": big},
            },
            "worst_magnitude_px": max(0.3, big),
        }

    by_pair = _radial_decomposition(field, FakePlacement())["by_pair"]
    # Each pair is internally perfectly coherent.
    assert by_pair["ab"]["direction_coherence"] > 0.99
    assert by_pair["ba"]["direction_coherence"] > 0.99

    # And pooling by worst magnitude -- what this used to do -- is not merely a
    # weaker reading of the same thing but a different and wrong one. Computed
    # here rather than asserted as a number, so the test says what it is against.
    angles = []
    for entry in field.values():
        worst = max(entry["pairs"], key=lambda k: entry["pairs"][k]["magnitude_px"])
        v = entry["pairs"][worst]
        angles.append(np.arctan2(v["dy"], v["dx"]))
    pooled = float(np.abs(np.exp(1j * np.asarray(angles)).mean()))
    assert pooled < 0.7
    assert min(by_pair[p]["direction_coherence"] for p in ("ab", "ba")) - pooled > 0.25


def test_the_note_names_the_pair_rather_than_the_field():
    """A coherence figure with no pair beside it cannot be acted on: the fault
    it points at belongs to two records, not to the capture as a whole."""
    from deltachrome.metrics.registration import (
        _decomposition_note,
        _radial_decomposition,
    )

    note = _decomposition_note(_radial_decomposition(_uniform_field(), FakePlacement()))
    assert "G-B" in note
    assert "bodily offset" in note


def test_too_few_positions_to_decompose_says_nothing_rather_than_guessing():
    from deltachrome.metrics.registration import _radial_decomposition

    two = _vectors({"m0": ((200, 200), (0.1, 0.1)), "m1": ((800, 800), (0.1, 0.1))})
    assert _radial_decomposition(two, FakePlacement()) == {}


def test_the_arrow_drawn_is_the_pair_the_number_came_from():
    """Three arrows per position would invite comparing a picture against a
    figure taken from only one of them."""
    from deltachrome.metrics.registration import _vector_field

    vectors = {"m0": {
        "center_px": [200.0, 800.0],
        "pairs": {
            "G-B": {"dx": 0.05, "dy": 0.0, "magnitude_px": 0.05},
            "G-R": {"dx": 0.40, "dy": 0.10, "magnitude_px": 0.412},
            "B-R": {"dx": 0.20, "dy": 0.0, "magnitude_px": 0.20},
        },
        "worst_magnitude_px": 0.412,
    }}
    reading = _vector_field(vectors, FakePlacement())[0]
    assert reading.source == "G-R"
    assert reading.vector_px == (0.40, 0.10)
    assert reading.position_relative == pytest.approx((0.2, 0.8))


def test_no_placement_means_no_arrows_rather_than_arrows_in_pixel_space():
    from deltachrome.metrics.registration import _vector_field

    assert _vector_field(_radial_field(), None) == []
