"""Per-edge records, kept whatever the verdict.

A pass/fail reduces a field of edges to one number, and that number is not what
tells you *why* a system underperforms. The differences between edges are the
diagnosis -- astigmatism shows as one orientation resolving better than the other,
field curvature as softness growing toward the corners, tilt as softness on one
side. None of that survives aggregation, so every edge is recorded in full.
"""

from __future__ import annotations

import json

import pytest

from deltachrome.results.edges import CURVE_STEP, build_edge_record, field_summary


class FakeEdge:
    def __init__(self, edge_id, region, orientation, center):
        self.edge_id, self.region_id = edge_id, region
        self.orientation, self.center_px = orientation, center
        self.bounds_px = (0, 0, 100, 100)


class FakeResult:
    def __init__(self, fifty=0.70, nyquist=0.25, peak=1.0, overshoot=0.0, angle=-4.6):
        self._fifty, self._nyq, self._peak, self._over = fifty, nyquist, peak, overshoot
        self.edge_angle_deg, self.phase_coverage_px, self.rows_used = angle, 33.0, 400
        self.warnings = []
        self.settings = {"edge_fit_residual_px": 0.07, "linearized": True,
                         "binning": "distance"}

    def sampling_efficiency(self, level=0.10): return 1.25
    def sfr50(self): return self._fifty
    def response_at_nyquist(self): return self._nyq
    def peak_modulation(self, *a): return self._peak
    def peak_frequency(self, *a): return 0.01
    def overshoot(self, *a): return self._over
    def mtf_at(self, f): return max(0.0, 1.0 - f)


def record(edge_id="m_e0", region="m", orientation="horizontal", center=(100.0, 50.0), **kw):
    return build_edge_record(FakeEdge(edge_id, region, orientation, center),
                             FakeResult(**kw), (0.5, 0.25))


# -- what a record holds ----------------------------------------------------


def test_a_record_carries_orientation_and_position():
    """Without these the per-edge values cannot diagnose anything -- an aberration is
    defined by how the measurement varies with direction and field position."""
    r = record()
    assert r.orientation == "horizontal"
    assert r.center_px == [100.0, 50.0]
    assert r.center_relative == [0.5, 0.25]
    assert r.region == "m"


def test_a_record_carries_measurement_quality_not_just_the_reading():
    """A soft edge and a badly measured edge are different problems."""
    r = record()
    assert r.edge_angle_deg == pytest.approx(-4.6)
    assert r.phase_coverage_px == pytest.approx(33.0)
    assert r.fit_residual_px == pytest.approx(0.07)
    assert r.rows_used == 400
    assert r.linearized is True
    assert r.binning == "distance"


def test_a_record_carries_every_reading_not_only_the_judged_one():
    r = record()
    for value in (r.sampling_efficiency_pct, r.sfr50_pct, r.mtf_at_nyquist,
                  r.peak_modulation, r.peak_frequency, r.overshoot_pct):
        assert value is not None


def test_the_curve_is_stored_so_the_field_can_be_replotted_without_the_capture():
    r = record()
    curve = r.mtf_curve
    assert len(curve["mtf"]) == len(curve["frequency_cyc_px"])
    assert curve["frequency_cyc_px"][1] == pytest.approx(CURVE_STEP)
    assert 40 < len(curve["mtf"]) < 80          # enough to plot, small enough to keep


def test_readings_are_stored_in_the_units_the_report_uses():
    r = record(fifty=0.70)
    assert r.sfr50_pct == pytest.approx(70.0)
    assert r.sampling_efficiency_pct == pytest.approx(125.0)


def test_records_serialise():
    from dataclasses import asdict

    json.dumps(asdict(record()))


# -- the field summary ------------------------------------------------------


def make_field(horizontal_fifty, vertical_fifty, regions=("tl", "tr", "bl", "br")):
    out = []
    for region in regions:
        for i, (orientation, fifty) in enumerate(
            (("horizontal", horizontal_fifty), ("vertical", vertical_fifty))
        ):
            out.append(record(f"{region}_e{i}", region, orientation, (0.0, 0.0), fifty=fifty))
    return out


def test_the_summary_splits_by_orientation():
    summary = field_summary(make_field(0.75, 0.65))
    assert summary["by_orientation"]["horizontal"]["sfr50_pct"]["mean"] == pytest.approx(75.0)
    assert summary["by_orientation"]["vertical"]["sfr50_pct"]["mean"] == pytest.approx(65.0)


def test_an_orientation_difference_is_reported_against_the_field_spread():
    """A difference smaller than the spread across the whole field is not evidence of
    anything, so the ratio is what makes the number readable."""
    summary = field_summary(make_field(0.75, 0.65))
    difference = summary["orientation_difference"]
    assert difference["sfr50_horizontal_minus_vertical_pct"] == pytest.approx(10.0)
    assert difference["as_fraction_of_field_spread"] > 1.0


def test_no_orientation_difference_when_both_read_the_same():
    summary = field_summary(make_field(0.70, 0.70))
    assert summary["orientation_difference"][
        "sfr50_horizontal_minus_vertical_pct"] == pytest.approx(0.0)


def test_the_summary_hedges_rather_than_diagnosing():
    """A horizontal/vertical difference is consistent with astigmatism and also with
    motion during capture. The tool points; it does not conclude."""
    note = field_summary(make_field(0.75, 0.65))["orientation_difference"]["note"]
    assert "astigmatism" in note
    assert "pointer rather than a diagnosis" in note


def test_the_summary_splits_by_field_position():
    records = [record(f"{r}_e0", r, "horizontal", (0.0, 0.0), fifty=f)
               for r, f in (("tl", 0.60), ("tr", 0.70), ("center", 0.80))]
    by_region = field_summary(records)["by_region"]
    assert by_region["tl"]["sfr50_pct"]["mean"] == pytest.approx(60.0)
    assert by_region["center"]["sfr50_pct"]["mean"] == pytest.approx(80.0)


def test_an_empty_field_summarises_to_nothing_rather_than_failing():
    assert field_summary([]) == {}


def test_a_summary_survives_edges_that_never_reached_the_level():
    records = [record("a", "m", "horizontal", (0.0, 0.0))]
    records[0].sfr50_pct = None
    summary = field_summary(records)
    assert summary["by_orientation"]["horizontal"]["sfr50_pct"] is None
    assert "orientation_difference" not in summary


# -- the run record ---------------------------------------------------------


def test_a_run_on_a_target_without_edges_records_none_rather_than_failing(tmp_path):
    """Most targets carry no slanted edges; the fields exist and are empty."""
    from deltachrome.pipeline import analyze
    from fixtures import SyntheticChart

    chart = SyntheticChart()
    reference = chart.write_reference(tmp_path / "ref.txt")
    image, corners = chart.render_capture(tmp_path / "cap.tif")
    result = analyze(
        image, target="colorchecker_sg_v1", reference_path=reference,
        profile="fadgi_3star_prints_photographs_2023", corners=corners,
    )
    data = result.run.to_dict()
    assert data["edges"] == []
    assert data["edge_field_summary"] == {}
    assert data["registration"] == []
    assert data["registration_summary"] == {}


def test_edges_are_measured_independently_of_any_metric(tmp_path):
    """Recorded by the pipeline, not inside an SFR metric, so the diagnostic data
    exists even under a profile that declares no SFR criteria at all."""
    from deltachrome.imaging.edges import extract_edges
    from deltachrome.imaging.geometry import place_chart
    from deltachrome.imaging.loader import load_image
    from deltachrome.pipeline import _measure_edges
    from test_sfr import edge_target, rotated_square_capture

    path = rotated_square_capture(tmp_path)
    image = load_image(path)
    placement = place_chart(
        [(0, 0), (image.width, 0), (image.width, image.height), (0, image.height)],
        (image.width, image.height),
    )
    edges = extract_edges(image, placement, edge_target())
    records = _measure_edges(edges, placement)

    assert len(records) == len(edges) == 4
    assert {r.orientation for r in records} == {"horizontal", "vertical"}
    # Positions come back in the target's own relative frame, so they are comparable
    # between captures at different scales.
    for r in records:
        assert 0.0 <= r.center_relative[0] <= 1.0
        assert 0.0 <= r.center_relative[1] <= 1.0
        assert r.mtf_curve["mtf"]


def test_the_report_lists_every_edge_and_the_field_pattern():
    from deltachrome.reporting.console import render

    class Run:
        run_id, created_utc = "x", "2026-01-01T00:00:00+00:00"
        image = {"filename": "c.tif", "width": 10, "height": 10, "bit_depth": 16}
        color_management = {"icc_profile": "p"}
        target = {"display_name": "t", "patch_count": 0}
        reference = {"path": None, "sample_count": 0, "provenance": {},
                     "measurement_mode": None}
        profile = {"display_name": "prof"}
        geometry = {}
        sampling = {}
        overall, overall_explanation, warnings = "PASS", "fine", []
        patches: list = []
        # Real EdgeRecords, not dictionaries. The report reads the live objects the
        # pipeline builds, and a stub that swapped in dicts hid a crash on every
        # capture that actually had edges.
        edges = [
            record("m_e0", "m", "horizontal", (0.1, 0.2), fifty=70.0),
            record("m_e1", "m", "vertical", (0.3, 0.4), fifty=60.0),
        ]
        edge_field_summary = field_summary(make_field(0.70, 0.60))
        registration: list = []
        registration_summary: dict = {}
        rulers: dict = {}

    class Result:
        run = Run()

        class evaluation:
            results: list = []

    text = render(Result())
    assert "EDGES (2 measured" in text
    assert "m_e0" in text and "m_e1" in text
    assert "FIELD PATTERN" in text
    assert "vertical - horizontal" in text
    assert "pointer rather than a diagnosis" in text

    # Named by the direction the response was read in, not by the way the edge
    # runs -- ISO 12233's own convention, and the one the report uses. A
    # near-horizontal edge reads the vertical resolution. One application
    # printing one word for opposite things in its terminal output and its
    # report is worse than either convention on its own.
    horizontal_edge = next(l for l in text.splitlines() if "m_e0" in l)
    vertical_edge = next(l for l in text.splitlines() if "m_e1" in l)
    assert "vertical" in horizontal_edge and "horizontal" not in horizontal_edge
    assert "horizontal" in vertical_edge and "vertical" not in vertical_edge


# -- the color records ------------------------------------------------------


def test_the_channel_curves_share_the_luminance_frequency_axis():
    """Three identical copies of one axis, in every edge of every run, is storage
    spent on nothing."""
    from deltachrome.results.edges import EdgeRecord

    record = EdgeRecord(
        edge_id="e0", region="r", orientation="horizontal", center_px=[0.0, 0.0],
        center_relative=[0.5, 0.5], bounds_px=[0, 0, 1, 1], edge_angle_deg=-5.0,
        phase_coverage_px=30.0, fit_residual_px=0.1, rows_used=50, linearized=True,
        binning="distance", sampling_efficiency_pct=90.0, sfr50_pct=70.0,
        mtf_at_nyquist=0.3, peak_modulation=1.0, peak_frequency=0.0,
        overshoot_pct=0.0,
        mtf_curve={"frequency_cyc_px": [0.0, 0.5], "mtf": [1.0, 0.3]},
        channels={
            "r": {"mtf": [1.0, 0.24], "sfr50_pct": 63.0, "mtf_at_nyquist": 0.24},
            "g": {"mtf": [1.0, 0.31], "sfr50_pct": 71.0, "mtf_at_nyquist": 0.31},
            "b": {"mtf": [1.0, 0.19], "sfr50_pct": 58.0, "mtf_at_nyquist": 0.19},
        },
    )
    for entry in record.channels.values():
        assert "frequency_cyc_px" not in entry
        assert len(entry["mtf"]) == len(record.mtf_curve["frequency_cyc_px"])
    assert record.channel_spread_pct == pytest.approx(13.0)


def test_one_measured_record_is_not_a_spread():
    from deltachrome.results.edges import EdgeRecord

    record = EdgeRecord(
        edge_id="e0", region="r", orientation="horizontal", center_px=[0.0, 0.0],
        center_relative=[0.5, 0.5], bounds_px=[0, 0, 1, 1], edge_angle_deg=-5.0,
        phase_coverage_px=30.0, fit_residual_px=0.1, rows_used=50, linearized=True,
        binning="distance", sampling_efficiency_pct=90.0, sfr50_pct=70.0,
        mtf_at_nyquist=0.3, peak_modulation=1.0, peak_frequency=0.0,
        overshoot_pct=0.0, mtf_curve={},
        channels={"g": {"mtf": [], "sfr50_pct": 71.0, "mtf_at_nyquist": 0.31}},
    )
    assert record.channel_spread_pct is None


def test_measuring_the_color_records_leaves_the_luminance_record_untouched():
    """The verdict rests on the luminance measurement. Per-channel curves are
    diagnostics recorded beside it, and if adding them could move any luminance
    field they would not be diagnostics -- they would be a second opinion."""
    from dataclasses import asdict

    edge, result = FakeEdge("m_e0", "m", "horizontal", (100.0, 50.0)), FakeResult()
    without = asdict(build_edge_record(edge, result, (0.5, 0.25)))
    with_channels = asdict(build_edge_record(
        edge, result, (0.5, 0.25), {"r": FakeResult(fifty=0.63), "g": FakeResult()}
    ))
    assert with_channels.pop("channels")
    assert without.pop("channels") == {}
    assert without == with_channels


def test_a_chromatic_spread_that_grows_to_the_corners_is_reported_as_such():
    """Lateral chromatic aberration is radial -- nil on the axis, worst at the
    corners. A record simply softer everywhere is a different fault, and a mean
    is the one summary shape that cannot tell them apart."""
    records = []
    for i, (x, y) in enumerate([(0.5, 0.5), (0.6, 0.6), (0.15, 0.15), (0.9, 0.9)]):
        r = record(edge_id=f"e{i}")
        r.center_relative = [x, y]
        radius = ((x - 0.5) ** 2 + (y - 0.5) ** 2) ** 0.5
        r.channels = {
            "r": {"mtf": [], "sfr50_pct": 70.0 - 40.0 * radius, "mtf_at_nyquist": 0.2},
            "g": {"mtf": [], "sfr50_pct": 70.0, "mtf_at_nyquist": 0.3},
        }
        records.append(r)

    chromatic = field_summary(records)["chromatic_difference"]
    assert chromatic["grows_with_field_radius"] > 0.99
    assert chromatic["worst_edge"] == "e3"


def test_a_record_softer_everywhere_does_not_read_as_radial():
    records = []
    for i, (x, y) in enumerate([(0.5, 0.5), (0.2, 0.2), (0.85, 0.15), (0.9, 0.9)]):
        r = record(edge_id=f"e{i}")
        r.center_relative = [x, y]
        r.channels = {
            "r": {"mtf": [], "sfr50_pct": 62.0, "mtf_at_nyquist": 0.2},
            "g": {"mtf": [], "sfr50_pct": 70.0, "mtf_at_nyquist": 0.3},
        }
        records.append(r)

    chromatic = field_summary(records)["chromatic_difference"]
    assert chromatic["sfr50_channel_spread_pct"]["mean"] == pytest.approx(8.0)
    # Every edge reads the same, so there is no correlation to report at all.
    assert chromatic["grows_with_field_radius"] is None


def test_an_edge_with_no_color_records_keeps_the_summary_from_forming():
    assert "chromatic_difference" not in field_summary([record()])
