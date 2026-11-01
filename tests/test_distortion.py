"""Geometric distortion, read from the scales a chart prints across its field.

The measurement is: magnification is supposed to be the same everywhere and in
every direction; how much is it not? A rule is a local measurement of scale at a
known place along a known axis, so a chart printing several of them answers that
directly.

The ISA Golden Thread prints twenty -- a centimetre and an inch rule on all four
sides of each of its five modules -- which is ten readings per axis at five field
positions, and each axis measured twice through independently engraved scales.
"""

from __future__ import annotations

import numpy as np
import pytest

from deltachrome.imaging.rulers import RulerReading
from deltachrome.metrics import get_metric
from deltachrome.metrics.base import MetricContext, MetricSpec
from deltachrome.targets.registry import load_target

METRIC = "geometric_distortion"
TARGET = "isa_golden_thread_dlt_v1"


def reading(region_id, axis, ppi, *, pitch=24.0, error=0.0, unit="mm"):
    """One rule, at a chosen rate. ``ppi`` is what the metric compares."""
    division_mm = 1.0
    pitch_px = ppi / 25.4 * division_mm
    return RulerReading(
        region_id=region_id, unit=unit, axis=axis, division_mm=division_mm,
        divisions_used=18, span_px=pitch_px * 17, pitch_px=pitch_px,
        residual_px=error,
    )


def run(readings, condition="value <= 2"):
    target = load_target(TARGET)
    ctx = MetricContext(target=target, data=[], extras={"rulers": readings})
    spec = MetricSpec(metric_id=METRIC, requires_capability="geometric_distortion",
                      pass_condition=condition, units="%")
    return get_metric(METRIC)(ctx, spec)


def field(h_rates, v_rates):
    positions = ["tl", "tr", "center", "bl", "br"]
    out = []
    for position, rate in zip(positions, h_rates):
        out.append(reading(f"ruler_{position}_top", "u", rate))
    for position, rate in zip(positions, v_rates):
        out.append(reading(f"ruler_{position}_left", "v", rate))
    return out


def both_units(h_rates, v_rates, *, offset=1.0):
    """The same field measured twice, once per unit, with the imperial scale printed
    ``offset`` times coarser than the metric one. Both units on both axes, which is
    what every chart here prints and what makes the offset separable."""
    out = field(h_rates, v_rates)
    for r in list(out):
        out.append(reading(r.region_id.replace("ruler_", "ruler_in_"), r.axis,
                           r.ppi * offset, unit="in"))
    return out


# -- the arithmetic ---------------------------------------------------------


def test_a_perfectly_even_field_reads_zero():
    result = run(field([600.0] * 5, [600.0] * 5))
    assert result.outcome == "evaluated"
    assert result.value == pytest.approx(0.0, abs=1e-9)
    assert result.passed is True


def test_a_known_scale_spread_comes_back_as_that_percentage():
    """606 to 618 about a mean of 612 is exactly 2%."""
    result = run(field([606.0, 609.0, 612.0, 615.0, 618.0], [612.0] * 5))
    assert result.metadata["components"]["field_spread"] == pytest.approx(
        (618 - 606) / np.mean([606, 609, 612, 615, 618, 612, 612, 612, 612, 612]) * 100,
        abs=0.001,
    )


def test_axis_anisotropy_is_measured_and_reported_separately():
    """Squares into rectangles: every position agrees with itself, but the two
    directions disagree. A field spread alone would show it as spread with no clue
    which way it ran."""
    result = run(field([600.0] * 5, [606.0] * 5))
    components = result.metadata["components"]
    assert components["axis_anisotropy"] == pytest.approx(6 / 603 * 100, abs=0.01)
    assert components["horizontal_mean_ppi"] == 600.0
    assert components["vertical_mean_ppi"] == 606.0


def test_one_sheared_position_governs_even_when_the_field_looks_even():
    """The failure a pooled figure hides: four positions perfect, one with its two
    axes disagreeing. Pooling puts that in with everything else; the per-position
    anisotropy names it."""
    readings = field([600.0] * 5, [600.0, 612.0, 600.0, 600.0, 600.0])
    result = run(readings)
    assert result.metadata["per_position"]["tr"]["anisotropy_percent"] == pytest.approx(
        12 / 606 * 100, abs=0.01
    )
    assert result.metadata["components"]["worst_local_anisotropy"] == pytest.approx(
        12 / 606 * 100, abs=0.01
    )
    assert result.value >= result.metadata["components"]["worst_local_anisotropy"]


def test_the_worst_component_governs():
    """One tolerance covers pincushion, barrel and shear, so the worst of them
    decides rather than an average of things that are not commensurable."""
    result = run(field([600.0] * 5, [600.0, 640.0, 600.0, 600.0, 600.0]))
    components = result.metadata["components"]
    # The metadata is rounded for reporting; the value is not.
    assert result.value == pytest.approx(max(
        components["field_spread"], components["worst_local_anisotropy"]
    ), rel=1e-4)
    assert result.aggregation == "worst component"


def test_a_field_outside_the_tolerance_fails():
    result = run(field([600.0, 600.0, 600.0, 600.0, 640.0], [600.0] * 5))
    assert result.value > 2.0
    assert result.passed is False


# -- what it will not pretend to know ---------------------------------------


def test_too_few_rules_is_incomplete_rather_than_a_number():
    """One rule measures the sampling rate and says nothing at all about
    distortion, which is a variation between rules."""
    result = run([reading("ruler_center_top", "u", 600.0)])
    assert result.outcome == "incomplete"
    assert result.value is None
    assert "variation" in result.reason
    assert "one rule measures the sampling rate" in result.reason.lower()


def test_the_measurement_floor_is_recorded_rather_than_announced():
    """The rules can only be read so repeatably, and a spread under that is the
    reading, not the lens."""
    readings = field([600.0, 600.05, 600.0, 600.0, 600.02], [600.0] * 5)
    for r in readings:
        object.__setattr__(r, "residual_px", 1.2)
    result = run(readings)
    # The floor stays in the record; it is no longer a note, because it never
    # changes the verdict and an operator cannot move it (2026-10-02 review).
    assert result.metadata["measurement_floor_percent"] > 0
    assert not any("upper bound" in note for note in result.notes)


# -- against the chart itself ----------------------------------------------


def test_the_golden_thread_declares_the_capability_and_the_rules_to_back_it():
    target = load_target(TARGET)
    assert target.supports("geometric_distortion")
    scales = [r for r in target.regions if r.kind == "scale"]
    assert len({r.geometry["axis"] for r in scales}) == 2
    assert len(scales) >= 4


def test_every_rule_is_kept_individually_in_the_result():
    """The per-position differences are the diagnostic; the aggregate is not."""
    result = run(field([600.0] * 5, [606.0] * 5))
    assert len(result.per_patch) == 10
    assert {p.role for p in result.per_patch} == {"scale_u", "scale_v"}


# -- a chart whose own scales disagree ---------------------------------------


def test_a_printing_offset_between_units_is_not_reported_as_distortion():
    """A chart's metric and imperial engravings
    disagreeing moves every rule of one family together -- it has no place and no
    direction, so it is not distortion. Pooling the raw rates put it there anyway:
    on a perfectly even field, an 0.8% offset between the two scales arrived as 0.8%
    of pincushion."""
    even = both_units([600.0] * 5, [600.0] * 5, offset=1.008)
    result = run(even)
    assert result.value == pytest.approx(0.0, abs=1e-6)
    assert result.metadata["components"]["unit_disagreement"] == pytest.approx(
        0.8, abs=0.01
    )
    assert result.metadata["unit_correction"]["mm"] > 1.0
    assert result.metadata["unit_correction"]["in"] < 1.0


def test_real_distortion_survives_the_correction():
    """The correction must not be able to flatten the measurement it sits inside. A
    2% field spread stays 2% whether or not the chart's two scales agree."""
    rates = [606.0, 609.0, 612.0, 615.0, 618.0]
    without = run(both_units(rates, [612.0] * 5)).value
    with_offset = run(both_units(rates, [612.0] * 5, offset=1.008)).value
    assert without == pytest.approx(with_offset, abs=0.02)
    assert without > 1.9


def test_the_correction_is_refused_when_unit_and_axis_are_confounded():
    """A chart printing millimetres only across and inches only down gives no way to
    tell 'the two engravings differ' from 'the two axes differ'. Normalising would
    erase a real anisotropy, so it is not done -- and the reason is said out loud."""
    readings = [reading(f"ruler_p{i}_top", "u", 600.0, unit="mm") for i in range(3)]
    readings += [reading(f"ruler_p{i}_left", "v", 604.8, unit="in") for i in range(3)]
    result = run(readings)
    assert result.metadata["unit_correction"] == {}
    assert result.value > 0.7
    assert any("NOT corrected" in note for note in result.notes)


def test_one_unit_alone_is_left_exactly_as_it_was():
    """Most charts print one unit, and nothing about this may move their numbers."""
    rates = [606.0, 609.0, 612.0, 615.0, 618.0]
    result = run(field(rates, [612.0] * 5))
    assert result.metadata["unit_correction"] == {}
    assert "unit_disagreement" not in result.metadata["components"]
    assert result.value == pytest.approx((618 - 606) / np.mean(rates + [612.0] * 5) * 100,
                                         abs=0.001)
