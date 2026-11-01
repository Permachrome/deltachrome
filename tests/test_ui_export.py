"""Exporting from the report, and the list of charts both Export and Save read.

The engine's export is pinned in ``test_export.py``. What is pinned here is the
shell's half: that the button is there, that it writes the capture on screen
rather than another one, and -- found while wiring it -- that a single chart
measured after an unsaved board no longer leaves the board in the list Save,
Export and re-judging all read.
"""

from __future__ import annotations

import csv
import json

import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from deltachrome.pipeline import analyze  # noqa: E402

SG = "colorchecker_sg_v1"
FADGI3 = "fadgi_3star_documents_unbound_general_2023"


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    from fixtures import SyntheticChart

    tmp = tmp_path_factory.mktemp("export_ui")
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


def _answer_dialog(monkeypatch, path, chosen="CSV (*.csv)", typed_filter=None):
    """Choose ``chosen`` from the format menu, then name ``path`` in the dialog.

    ``typed_filter`` is what the dialog hands back as its filter, which is
    deliberately ignored: the format was settled before the dialog opened.
    """
    from deltachrome_ui.app import MainWindow

    offered = []
    monkeypatch.setattr(MainWindow, "_ask_export_format",
                        lambda self, button=None: chosen)

    def dialog(_parent, _title, start, filters):
        offered.append((start, filters))
        return str(path), typed_filter or filters
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(dialog))
    return offered


def test_the_report_offers_export_on_fresh_and_saved_runs(qt_app, result):
    from deltachrome_ui.result_view import ResultView

    for saved in (False, True):
        view = ResultView()
        view.set_result(result, saved=saved)
        assert view.export_button is not None
        assert view.export_button.kind == "export"
        assert view.export_button.toolTip()


def test_a_fresh_run_exports_the_run_on_screen(window, result, tmp_path, monkeypatch):
    window._on_analysed(result)
    out = tmp_path / "out.csv"
    _answer_dialog(monkeypatch, out)
    window.result.exportRequested.emit()
    lines = list(csv.reader(out.open(encoding="utf-8-sig")))
    assert dict(line for line in lines if len(line) == 2)["Image"] == \
        result.run.image["filename"]
    assert len(lines) > 10


def test_the_dialogs_filter_supplies_a_missing_extension(window, result, tmp_path,
                                                         monkeypatch):
    window._on_analysed(result)
    _answer_dialog(monkeypatch, tmp_path / "report", chosen="XML (*.xml)")
    window.result.exportRequested.emit()
    assert (tmp_path / "report.xml").exists()


def test_a_saved_run_exports_its_record_as_saved(window, result, tmp_path, monkeypatch):
    from deltachrome.session import capture_record
    from deltachrome_ui.library import summarise

    path = capture_record([result]).save(tmp_path / "saved.json")
    window._open_run(summarise(json.loads(path.read_text()), path))
    out = tmp_path / "saved.txt"
    _answer_dialog(monkeypatch, out, chosen="Text (*.txt)")
    window.result.exportRequested.emit()
    text = out.read_text(encoding="utf-8")
    assert result.run.image["filename"] in text
    assert "Average Color Accuracy" in text


def test_a_lone_chart_after_an_unsaved_board_is_what_save_and_export_see(
        window, result):
    """The board stayed in the shell's list when a single chart was measured
    after it, so Save wrote the board, Export exported it and re-judging the
    chart re-judged the board's charts instead."""
    from types import SimpleNamespace

    board = {f"board-{n}": SimpleNamespace(run=SimpleNamespace(run_id=f"board-{n}"))
             for n in range(4)}
    window._fresh_results = dict(board)
    window._on_analysed(result)
    assert list(window._fresh_results.values()) == [result]


def test_a_lone_charts_rejudgement_replaces_it_in_the_list(window, result):
    """The other half of the fix above: once a lone chart is in the list, judging
    it again must replace it there, or Save writes the judgement it replaced."""
    from deltachrome.pipeline import rescore

    window._on_analysed(result)
    again = rescore(result, profile="fadgi_4star_documents_unbound_general_2023")
    window._on_rescored([again])
    assert list(window._fresh_results.values()) == [again]


def test_the_format_menu_decides_the_format_not_the_prefilled_extension():
    """The dialog opens on ``name.csv``; choosing Text must write text, though
    the platform may hand the name back still ending ``.csv``."""
    from pathlib import Path

    from deltachrome_ui.app import export_path

    assert export_path("/x/QA62_2.csv", "Text (*.txt)") == Path("/x/QA62_2.txt")
    assert export_path("/x/QA62_2.csv", "XML (*.xml)") == Path("/x/QA62_2.xml")
    assert export_path("/x/QA62_2.csv", "CSV (*.csv)") == Path("/x/QA62_2.csv")
    # A name with no extension, or a dot that is part of the name, gains one.
    assert export_path("/x/QA62_2", "Text (*.txt)") == Path("/x/QA62_2.txt")
    assert export_path("/x/run.v2", "XML (*.xml)") == Path("/x/run.v2.xml")


def test_the_dialog_offers_only_the_format_already_chosen(window, result, tmp_path,
                                                          monkeypatch):
    """The macOS panel saved as the prefilled type whatever its own menu said,
    and asked to replace ``name.csv`` with Text chosen. Offered one format, its
    name and its type cannot disagree."""
    window._on_analysed(result)
    offered = _answer_dialog(monkeypatch, tmp_path / "out.csv",
                             chosen="Text (*.txt)", typed_filter="CSV (*.csv)")
    window.result.exportRequested.emit()
    start, filters = offered[0]
    assert start.endswith(".txt") and filters == "Text (*.txt)"
    assert (tmp_path / "out.txt").exists()
    assert not (tmp_path / "out.csv").exists()


def test_dismissing_the_format_menu_exports_nothing(window, result, tmp_path,
                                                     monkeypatch):
    window._on_analysed(result)
    offered = _answer_dialog(monkeypatch, tmp_path / "out.csv", chosen=None)
    window.result.exportRequested.emit()
    assert offered == []
