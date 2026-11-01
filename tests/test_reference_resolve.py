"""Pairing a capture with its own unit's reference file.

This policy used to live in ``cli.py``, keyed off an ``argparse.Namespace``, and
was covered by no test at all. Both facts mattered once a second front end needed
it: policy only one front end can reach is policy the two can disagree about, and
disagreeing about *which chart's file* a capture is measured against produces
numbers that look entirely reasonable and mean nothing.

The decode-count test at the bottom is the other half of the move. The old
arrangement scanned the capture for codes by loading it, then handed a *path* to
``analyze``, which loaded it again -- so every run with code verification on, the
default, decoded the master twice. On a JPEG 2000 preservation master that is
around 25 seconds of pure repetition.
"""

from __future__ import annotations

import pytest

from deltachrome.errors import MissingReferenceError
from deltachrome.imaging.loader import load_image
from deltachrome.reference.resolve import (
    DEFERRED,
    EXPLICIT,
    PAIRED,
    resolve_reference,
)
from deltachrome.targets.registry import load_target
from fixtures import SyntheticChart
from test_reference_pairing import _capture_with_qr

SG = "colorchecker_sg_v1"
ISA = "isa_golden_thread_dlt_v1"


@pytest.fixture
def sg():
    return load_target(SG)


@pytest.fixture
def isa():
    return load_target(ISA)


# -- nothing named -----------------------------------------------------------


def test_naming_nothing_defers_rather_than_deciding(sg):
    """The pipeline owns the fallback rule and tells three situations apart.
    Deciding it a second time here would be a copy free to drift."""
    choice = resolve_reference(target=sg)
    assert choice.path is None
    assert choice.basis == DEFERRED


def test_a_target_with_nominal_values_defers_without_comment(isa):
    """The run's own caveat says it was judged on manufacturer values; saying it
    here first as well printed the same fact twice."""
    choice = resolve_reference(target=isa)
    assert choice.path is None
    assert choice.basis == DEFERRED
    assert choice.notes == []


def test_a_target_without_nominal_values_promises_nothing(sg):
    """The SG ships none, so there is no fallback to pre-announce."""
    assert resolve_reference(target=sg).notes == []


def test_deferring_never_touches_the_image(sg):
    """No scan is needed, so no decode should happen -- passing a path that does
    not exist proves the file was never opened."""
    choice = resolve_reference(target=sg, image="/no/such/master.tif")
    assert choice.basis == DEFERRED


# -- an explicit file --------------------------------------------------------


def test_an_explicit_file_is_used_as_given(sg, tmp_path):
    reference = SyntheticChart().write_reference(tmp_path / "ref.txt")
    choice = resolve_reference(target=sg, explicit_path=reference, verify_code=False)
    assert choice.path == reference
    assert choice.basis == EXPLICIT


def test_verification_off_needs_no_capture(sg, tmp_path):
    reference = SyntheticChart().write_reference(tmp_path / "ref.txt")
    choice = resolve_reference(
        target=sg, explicit_path=reference, image=None, verify_code=False
    )
    assert choice.path == reference


def test_a_matching_code_raises_nothing(sg, tmp_path):
    """The placement screen states the code it read, including that none was, so a
    match needs no line of its own."""
    reference = SyntheticChart().write_reference(tmp_path / "CCSG21.txt")
    capture = _capture_with_qr(tmp_path, "CCSG21")
    choice = resolve_reference(
        target=sg, explicit_path=reference, image=load_image(capture)
    )
    assert not choice.warnings
    assert not choice.code_mismatch


def test_a_mismatched_code_is_a_warning_not_a_note(sg, tmp_path):
    reference = SyntheticChart().write_reference(tmp_path / "CCSG99.txt")
    capture = _capture_with_qr(tmp_path, "CCSG21")
    choice = resolve_reference(
        target=sg, explicit_path=reference, image=load_image(capture)
    )
    assert choice.code_mismatch
    assert choice.warnings == [
        "Chart's printed code CCSG21 doesn't match the reference file's CCSG99."
    ]
    # Still returned: the operator may know something the chart does not.
    assert choice.path is not None


def test_a_file_claiming_no_identity_cannot_mismatch(sg, tmp_path):
    """An i1Profiler export states no serial, and a descriptive filename claims
    no code: nothing in it can disagree with the chart, so a file measured on
    this very chart was being reported as not matching it."""
    reference = SyntheticChart().write_reference(tmp_path / "ColorChecker_SG_M1.txt")
    capture = _capture_with_qr(tmp_path, "CCSG21")
    choice = resolve_reference(
        target=sg, explicit_path=reference, image=load_image(capture)
    )
    assert not choice.code_mismatch
    assert not choice.warnings


def test_a_capture_with_no_code_at_all_is_not_a_mismatch(sg, tmp_path):
    """Most charts carry no code. Absence must not read as disagreement."""
    reference = SyntheticChart().write_reference(tmp_path / "ref.txt")
    plain, _ = SyntheticChart().render_capture(tmp_path / "plain.tif")
    choice = resolve_reference(
        target=sg, explicit_path=reference, image=load_image(plain)
    )
    assert not choice.warnings


# -- a library ---------------------------------------------------------------


def test_a_library_pairs_by_the_charts_printed_code(sg, tmp_path):
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    SyntheticChart().write_reference(library_dir / "CCSG21.txt")
    capture = _capture_with_qr(tmp_path, "CCSG21")
    choice = resolve_reference(
        target=sg, library_dir=library_dir, image=load_image(capture)
    )
    assert choice.basis == PAIRED
    assert choice.path.name == "CCSG21.txt"
    assert any("paired automatically" in note for note in choice.notes)


def test_a_library_that_cannot_pair_refuses(sg, tmp_path):
    """Guessing here is the one thing that must not happen."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    SyntheticChart().write_reference(library_dir / "CCSG99.txt")
    capture = _capture_with_qr(tmp_path, "CCSG21")
    with pytest.raises(MissingReferenceError):
        resolve_reference(target=sg, library_dir=library_dir, image=load_image(capture))


def test_pairing_without_a_capture_says_what_is_missing(sg, tmp_path):
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    with pytest.raises(ValueError, match="no image"):
        resolve_reference(target=sg, library_dir=library_dir, image=None)


# -- decoding the master once ------------------------------------------------


def count_decodes(monkeypatch) -> list[str]:
    """Count every decode, whichever binding of ``load_image`` reaches it."""
    import deltachrome.imaging.loader as loader_module
    import deltachrome.pipeline as pipeline_module
    import deltachrome.reference.resolve as resolve_module

    calls: list[str] = []
    real = loader_module.load_image

    def counting(path, *args, **kwargs):
        calls.append(str(path))
        return real(path, *args, **kwargs)

    for module in (loader_module, pipeline_module, resolve_module):
        monkeypatch.setattr(module, "load_image", counting, raising=False)
    return calls


def test_analyze_accepts_an_already_decoded_image(tmp_path, monkeypatch):
    chart = SyntheticChart()
    reference = chart.write_reference(tmp_path / "ref.txt")
    path, corners = chart.render_capture(tmp_path / "capture.tif")
    image = load_image(path)

    calls = count_decodes(monkeypatch)
    from deltachrome.pipeline import analyze

    result = analyze(
        image, target=SG, reference_path=reference,
        profile="fadgi_3star_documents_unbound_general_2023", corners=corners,
    )
    assert calls == [], "a decoded image was handed in and was decoded again"
    assert result.overall == "PASS"


def test_a_path_and_a_loaded_image_give_the_same_answer(tmp_path):
    chart = SyntheticChart()
    reference = chart.write_reference(tmp_path / "ref.txt")
    path, corners = chart.render_capture(tmp_path / "capture.tif")
    from deltachrome.pipeline import analyze

    kwargs = dict(
        target=SG, reference_path=reference,
        profile="fadgi_3star_documents_unbound_general_2023", corners=corners,
    )
    by_path = analyze(path, **kwargs)
    by_image = analyze(load_image(path), **kwargs)
    for a, b in zip(by_path.evaluation.results, by_image.evaluation.results):
        assert a.metric_id == b.metric_id
        assert a.value == pytest.approx(b.value, nan_ok=True)


def test_the_cli_decodes_the_master_exactly_once(tmp_path, monkeypatch, capsys):
    """The regression guard. Code verification is on by default, and it used to
    cost a second full decode of the master on every single run."""
    chart = SyntheticChart()
    reference = chart.write_reference(tmp_path / "ref.txt")
    path, corners = chart.render_capture(tmp_path / "capture.tif")

    calls = count_decodes(monkeypatch)
    from deltachrome.cli import main

    exit_code = main([
        "analyze", str(path), "--target", SG, "--reference", str(reference),
        "--profile", "fadgi_3star_documents_unbound_general_2023",
        "--corners", " ".join(f"{x},{y}" for x, y in corners),
    ])
    capsys.readouterr()
    assert exit_code == 0
    assert len(calls) == 1, f"decoded {len(calls)} times: {calls}"


@pytest.mark.parametrize("symbology, mismatch", [("Code25", False), ("QRCode", True)])
def test_a_lasersoft_barcode_agrees_with_the_serial_it_encodes(sg, tmp_path, symbology, mismatch):
    """LaserSoft's barcode carries 1930021 for a chart printed T1910021 -- the
    serial rewritten, not misread -- so a file for that unit is not a mismatch.
    Only for a code read as Standard 2 of 5, LaserSoft's symbology; the same
    digits in a QR code say only what they say."""
    from deltachrome.imaging.codes import CodeScanResult, DetectedCode

    reference = SyntheticChart().write_reference(tmp_path / "T1910021.txt")
    scan = CodeScanResult(codes=[DetectedCode("1930021", symbology, "deltachrome")])
    choice = resolve_reference(target=sg, explicit_path=reference, scan=scan)
    assert choice.code_mismatch is mismatch
