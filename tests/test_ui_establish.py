"""Reading a saved run's capture again, from its own record.

A run reopened from the library holds every number and no pixels, so the profile
picker that lets a fresh run be judged against another standard is a label
instead -- and until now the only way back to a live measurement was the
placement screen, reached through "Re-run Capture". What these tests pin is that
the record supplies everything a re-establishment needs on its own -- the
target, the placement, the reference and the run options -- and that the shell
ends up in exactly the state a fresh analysis would have left it in.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from PySide6.QtWidgets import QApplication  # noqa: E402

from deltachrome.pipeline import analyze  # noqa: E402

SG = "colorchecker_sg_v1"
FADGI3 = "fadgi_3star_documents_unbound_general_2023"


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    from fixtures import SyntheticChart

    tmp = tmp_path_factory.mktemp("establish")
    chart = SyntheticChart()
    reference = chart.write_reference(tmp / "ref.txt")
    image, corners = chart.render_capture(tmp / "capture.tif", skew=18)
    return analyze(
        image, target=SG, reference_path=reference, profile=FADGI3, corners=corners
    )


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    from deltachrome_ui.app import MainWindow

    monkeypatch.setenv("DELTACHROME_RUN_DIR", str(tmp_path / "runs"))
    window = MainWindow()
    try:
        yield window
    finally:
        window.close()


def saved_row(result, tmp_path, name="run.json"):
    """A saved run, exactly as the library would list it."""
    from deltachrome_ui.library import summarise

    path = result.run.save(tmp_path / name)
    return summarise(json.loads(path.read_text()), path)


# -- the worker itself ---------------------------------------------------------


def test_the_worker_reads_a_bare_run_record_again(result):
    """a live measurement, from a record,
    with nothing supplied but the record itself."""
    from deltachrome_ui.workers import RecordEstablisher

    outcome = []
    worker = RecordEstablisher(image=result.image, charts=[result.run.to_dict()])
    worker.finished.connect(outcome.append)
    worker.run()

    assert len(outcome) == 1
    results, failures = outcome[0]
    assert not failures
    assert len(results) == 1
    fresh = results[0]
    assert fresh.measurement is not None, "must be rescoreable, not another record"
    assert fresh.run.run_id != result.run.run_id, "a fresh read, not the same run"
    assert fresh.run.overall == result.run.overall
    assert fresh.run.profile["profile_id"] == result.run.profile["profile_id"]


def test_a_chart_with_no_placement_is_skipped_and_the_rest_still_run(result):
    """A capture record holds several charts, and one that cannot say enough of
    itself -- no corners, here -- must not cost the others their answer."""
    from deltachrome_ui.workers import RecordEstablisher

    bare = dict(result.run.to_dict())
    bare["geometry"] = {}
    outcome = []
    worker = RecordEstablisher(image=result.image, charts=[result.run.to_dict(), bare])
    worker.finished.connect(outcome.append)
    worker.run()

    results, failures = outcome[0]
    assert len(results) == 1
    assert len(failures) == 1


def test_every_chart_unreadable_is_reported_as_a_failure(result):
    from deltachrome_ui.workers import RecordEstablisher

    failed = []
    worker = RecordEstablisher(image=result.image, charts=[{"geometry": {}}])
    worker.failed.connect(failed.append)
    worker.run()

    assert len(failed) == 1


# -- what the shell does with it ------------------------------------------------


def test_can_rescore_becomes_true_once_established(window, result, tmp_path):
    row = saved_row(result, tmp_path)
    window._open_run(row)
    assert not window.result.can_rescore

    window._on_established(([result], []))

    assert window.result.can_rescore
    assert window.result.result.run.run_id == result.run.run_id


def test_it_is_a_fresh_result_afterwards_not_a_saved_one(window, result, tmp_path):
    """Save is what turns a fresh measurement into a library record, whether it
    arrived by placing a chart from scratch or by reading one again -- so this
    is the same shape a just-analyzed capture is in, not a saved one with its
    pixels back."""
    row = saved_row(result, tmp_path)
    window._open_run(row)
    window._on_established(([result], []))

    assert not window.result.saved
    assert window._open_record is None
    assert result.run.run_id in window._fresh_results


def test_the_reader_keeps_their_place(window, result, tmp_path):
    row = saved_row(result, tmp_path)
    window._open_run(row)
    before = window.result.current_page
    window._on_established(([result], []))
    assert window.result.current_page == before


def test_a_fresh_run_has_nothing_to_establish(window, result):
    """The reading is already in memory. The guard keeps the request from being
    made at all, rather than asking the engine to refuse it."""
    window._on_analysed(result)
    assert window.result.can_rescore

    window._establish_measurement()

    assert window.result.can_rescore
    assert window.result.result.run.run_id == result.run.run_id


def test_nothing_open_asks_for_nothing(window):
    window._establish_measurement()  # must not raise


def test_a_second_chart_is_asked_for_by_index_after_establishing(
    window, result, tmp_path
):
    """A capture record holds several charts and this reads all of them, not
    only the one on screen -- so stepping to another chart afterwards is still
    a read of a live measurement, not a fall back to record-only numbers."""
    from deltachrome.session import capture_record
    from deltachrome_ui.library import summarise

    other = analyze(
        result.image, target=SG,
        reference_path=result.reference.source_path, profile=FADGI3,
        corners=[(x, y) for x, y in result.placement.corners_px],
    )
    record = capture_record([result, other], labels=["chart 1", "chart 2"])
    directory = tmp_path / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    path = record.save(directory / f"{record.capture_id}.json")
    window._open_run(summarise(json.loads(path.read_text()), path))

    window._on_established(([result, other], []))

    assert window.result.chart_count == 2
    window._show_chart(1)
    assert window.result.result.run.run_id == other.run.run_id
    assert window.result.can_rescore, "the second chart must be live too"
