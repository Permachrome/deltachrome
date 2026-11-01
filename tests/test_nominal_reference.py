"""Running against the manufacturer's nominal values instead of a measurement.

The engine prefers a per-unit measurement and always will. But some targets are
effectively never available measured -- the ISA Golden Thread is seldom bought
measured and its patches are too small for a hand spectrophotometer -- so a hard
refusal does not protect anyone, it just puts the target out of reach. Where the
manufacturer publishes aim values the definition carries them and the run falls
back to them, which is what OpenDICE does with its preloaded sample profiles.

What must not happen is the substitution going unnoticed. Everything below exists
to make sure it cannot: the run warns, the reference is labeled, the record stores
the basis, and the verdict loses its conformance claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deltachrome.errors import MissingReferenceError
from deltachrome.pipeline import analyze
from deltachrome.reference import MANUFACTURER_NOMINAL, PER_UNIT_MEASUREMENT, read_reference
from deltachrome.targets.registry import load_target
from fixtures import SyntheticChart

DEFINITIONS = Path(__file__).resolve().parents[1] / "src/deltachrome/targets/definitions"
PROFILE = "fadgi_3star_prints_photographs_2023"


@pytest.fixture
def nominal_target(tmp_path):
    """A copy of the SG definition that ships nominal values, plus a capture.

    The nominal file is deliberately *not* the same numbers as the chart: real aim
    values differ from any individual specimen, and a fixture where they agreed
    exactly would not exercise the thing being guarded against.
    """
    chart = SyntheticChart()
    image, corners = chart.render_capture(tmp_path / "capture.tif")
    measured = chart.write_reference(tmp_path / "measured.txt")

    nominal = tmp_path / "sg_nominal.txt"
    text = measured.read_text(encoding="utf-8").replace(
        'ORIGINATOR "Deltachrome synthetic fixture"',
        'ORIGINATOR "Deltachrome synthetic fixture"\n'
        'KEYWORD "DELTACHROME_REFERENCE_BASIS"\n'
        f'DELTACHROME_REFERENCE_BASIS "{MANUFACTURER_NOMINAL}"\n'
        'KEYWORD "DELTACHROME_NOMINAL_CAVEAT"\n'
        'DELTACHROME_NOMINAL_CAVEAT "Aim values for the model, not this chart."',
    )
    nominal.write_text(text, encoding="utf-8")

    definition = json.loads((DEFINITIONS / "colorchecker_sg_v1.json").read_text("utf-8"))
    definition["target_id"] = "sg_with_nominal_v1"
    definition["reference_data"]["nominal_values"] = {
        "file": nominal.name,
        "basis": MANUFACTURER_NOMINAL,
        "notes": "Aim values for the model, not a measurement of any individual chart.",
    }
    path = tmp_path / "sg_with_nominal.json"
    path.write_text(json.dumps(definition), encoding="utf-8")
    return path, image, corners, measured


# -- the fallback happens, and is preferred away from -----------------------


def test_a_target_with_nominal_values_runs_without_a_reference_file(nominal_target):
    target, image, corners, _ = nominal_target
    result = analyze(image, target=target, profile=PROFILE, corners=corners)

    assert result.reference.basis == MANUFACTURER_NOMINAL
    assert result.evaluation.reference_basis == MANUFACTURER_NOMINAL
    assert len(result.run.patches) > 0


def test_a_supplied_measurement_wins_over_the_nominal_values(nominal_target):
    """The fallback is a fallback. Naming a file must always take precedence."""
    target, image, corners, measured = nominal_target
    result = analyze(
        image, target=target, reference_path=measured, profile=PROFILE, corners=corners
    )
    assert result.reference.basis == PER_UNIT_MEASUREMENT
    assert result.conformance_claim is True
    assert "nominal" not in " ".join(result.warnings).lower()


def test_the_fallback_can_be_refused(nominal_target):
    """And the refusal says the values exist and were declined, which is a different
    problem from a target that never had any."""
    target, image, corners, _ = nominal_target
    with pytest.raises(MissingReferenceError, match="nominal fallback was refused"):
        analyze(image, target=target, profile=PROFILE, corners=corners,
                allow_nominal_reference=False)


def test_a_target_without_nominal_values_still_refuses(tmp_path):
    """Unchanged for every target that has none: no reference, no run."""
    chart = SyntheticChart()
    image, corners = chart.render_capture(tmp_path / "capture.tif")
    with pytest.raises(MissingReferenceError, match="nothing to measure against"):
        analyze(image, target="colorchecker_sg_v1", profile=PROFILE, corners=corners)


# -- and it is impossible to miss -------------------------------------------


def test_the_run_says_so_once_in_its_caveat_and_not_again_as_a_warning(nominal_target):
    """The overview's caveat line is where this is said; a warning repeating it on
    the same report was cut in the 2026-10-02 review."""
    target, image, corners, _ = nominal_target
    result = analyze(image, target=target, profile=PROFILE, corners=corners)

    assert result.evaluation.advisory_line.startswith("Advisory:")
    assert not any("nominal" in w for w in result.warnings)


def test_a_nominal_run_is_not_a_conformance_claim(nominal_target):
    """The load-bearing consequence. A capture judged against aim values carries the
    chart's own print tolerance inside every difference, so neither a pass nor a
    failure settles conformance for the specimen."""
    target, image, corners, _ = nominal_target
    result = analyze(image, target=target, profile=PROFILE, corners=corners)

    assert result.conformance_claim is False
    assert result.evaluation.verdict_line.endswith("(ADVISORY)")
    assert "not a conformance result" in result.evaluation.overall_explanation
    assert result.run.conformance_claim is False


def test_a_measured_run_keeps_its_plain_verdict(nominal_target):
    target, image, corners, measured = nominal_target
    result = analyze(
        image, target=target, reference_path=measured, profile=PROFILE, corners=corners
    )
    assert result.evaluation.verdict_line == result.overall
    assert "ADVISORY" not in result.evaluation.verdict_line
    assert "conformance result" not in result.evaluation.overall_explanation


def test_the_run_record_stores_the_basis_beside_the_verdict(nominal_target):
    """A consumer reading ``overall`` alone must find the qualifier next to it, not
    three levels down in provenance."""
    target, image, corners, _ = nominal_target
    data = analyze(image, target=target, profile=PROFILE, corners=corners).run.to_dict()

    assert data["conformance_claim"] is False
    assert data["reference"]["basis"] == MANUFACTURER_NOMINAL
    assert data["reference"]["nominal_caveat"]


def test_the_report_says_so_on_the_verdict_line_and_in_the_header(nominal_target):
    from deltachrome.reporting.console import render

    target, image, corners, _ = nominal_target
    text = render(analyze(image, target=target, profile=PROFILE, corners=corners))

    assert "MANUFACTURER NOMINAL" in text
    assert "ADVISORY -- NOT A CONFORMANCE TEST" in text
    assert "ABOUT THE REFERENCE VALUES" in text


# -- the label itself -------------------------------------------------------


def test_an_ordinary_reference_file_is_a_per_unit_measurement(tmp_path):
    """The default is the honest one: a file from a spectrophotometer says nothing
    about basis, and it is a measurement. Nominal data has to declare itself."""
    path = SyntheticChart().write_reference(tmp_path / "ref.txt")
    reference = read_reference(path)
    assert reference.basis == PER_UNIT_MEASUREMENT
    assert reference.is_nominal is False
    assert reference.nominal_caveat is None


def test_the_isa_target_ships_nominal_values_and_the_sg_does_not():
    """The SG is bought measured or measured by its owner; the ISA is neither."""
    isa = load_target("isa_golden_thread_dlt_v1")
    assert isa.nominal_reference_path is not None
    assert read_reference(isa.nominal_reference_path).is_nominal

    for target_id in ("colorchecker_sg_v1", "lasersoft_iso12641_2_advanced_v1"):
        assert load_target(target_id).nominal_reference_path is None


def test_a_definition_naming_a_missing_nominal_file_is_not_silently_ignored(tmp_path):
    """A typo in the filename would otherwise turn into 'this target has no nominal
    values', which reads as a deliberate choice rather than a broken one."""
    definition = json.loads((DEFINITIONS / "colorchecker_sg_v1.json").read_text("utf-8"))
    definition["reference_data"]["nominal_values"] = {"file": "not_there.txt"}
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(definition), encoding="utf-8")

    chart = SyntheticChart()
    image, corners = chart.render_capture(tmp_path / "capture.tif")
    with pytest.raises(MissingReferenceError, match="not_there.txt"):
        analyze(image, target=path, profile=PROFILE, corners=corners)
