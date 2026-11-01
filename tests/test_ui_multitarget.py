"""A frame holding several charts, once it has been measured.

The run used to land on the **session** -- the board's combined verdict -- and
every reader crossed it to reach a report. Nothing on that screen is reached by
pressing anything: it lists the captures, and clicking one opens the report the
run was made to produce. So the report comes first now and the combined verdict
is one press away in its bar, which is the same distance the session put the
reports at, turned round.

What is pinned here is mostly the two things that change with it.

**The way between charts has to be visible without being opened.** A drop-down
was right while the reader arrived through a list of the charts; as the only
route between them it has to say, unopened, whether there is another one.

**A capture is judged as one thing.** A capture record holds one combined verdict
against one profile, so re-judging a board from the report of card 3 has to
re-judge the board. It did not, and the judgement it replaced is what Save wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from deltachrome.pipeline import analyze  # noqa: E402
from deltachrome_ui.widgets import Stepper  # noqa: E402

SG = "colorchecker_sg_v1"
FADGI3 = "fadgi_3star_documents_unbound_general_2023"
FADGI4 = "fadgi_4star_documents_unbound_general_2023"


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def charts(tmp_path_factory):
    """Two runs off one frame.

    Two analyzes of the same synthetic capture, which is what a board of
    identical cards measures to: one decode, one placement each, two runs with
    ids of their own. What this exercises is the shell's handling of a list of
    them, and nothing in that cares whether the charts differ.
    """
    from fixtures import SyntheticChart

    tmp = tmp_path_factory.mktemp("multitarget")
    chart = SyntheticChart()
    reference = chart.write_reference(tmp / "ref.txt")
    image, corners = chart.render_capture(tmp / "capture.tif", skew=18)
    return [
        analyze(image, target=SG, reference_path=reference, profile=FADGI3,
                corners=corners)
        for _ in range(2)
    ]


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    from deltachrome_ui.app import MainWindow

    monkeypatch.setenv("DELTACHROME_RUN_DIR", str(tmp_path / "runs"))
    window = MainWindow()
    try:
        yield window
    finally:
        window.close()


def bar_buttons(view) -> list[str]:
    return [b.text() for b in view.header.findChildren(QPushButton) if b.text()]


def stepper(view) -> Stepper | None:
    found = view.header.findChildren(Stepper)
    return found[0] if found else None


def combined_button(view) -> QPushButton:
    """The way from a chart's report back up to the capture's own screen."""
    return next(
        b for b in view.header.findChildren(QPushButton)
        if b.text() == "All Targets"
    )


# -- where a multi-chart run lands --------------------------------------------


def test_a_board_lands_on_all_targets(window, charts):
    """And this has been round twice, which is why the reason is written down.

    It first landed here, was moved to the first chart's report because nothing
    on this screen was reached by *pressing* anything -- it listed the captures,
    so every reader crossed it -- and is back because the screen changed. It is
    a picture of the capture now, with every target outlined where it was
    measured and each one's report a click on its own outline. That answers the
    question a board is laid out to ask before any one card does, and the click
    lands on the target the reader meant rather than on whichever came first.

    The report underneath is still built, holding chart 0, because the stepper,
    Save and the frame map all read the list it was given."""
    window._on_charts_analysed((charts, []))

    assert window.stack.currentWidget() is window.session
    assert len(window.session.session.captures) == 2
    assert (window.result.chart_index, window.result.chart_count) == (0, 2)


def test_a_lone_chart_still_lands_on_its_report(window, charts):
    """There is nothing to combine, so there is no such screen to land on."""
    window._on_charts_analysed(([charts[0]], []))
    assert window.stack.currentWidget() is window.result
    assert window.result.chart_count == 1


def test_the_stepper_says_where_you_are_without_being_opened(window, charts):
    """The whole of how a reader moves between the charts of a board, so it has
    to say unopened whether there is another one. The ends are dead rather than
    wrapping, which is the same fact said by the arrows."""
    window._on_charts_analysed((charts, []))
    window.stack.setCurrentWidget(window.result)
    step = stepper(window.result)

    assert step is not None
    assert step.label.text() == "Target 1 of 2"
    assert not step.previous.isEnabled(), "there is nothing before the first"
    assert step.following.isEnabled()

    step.following.click()
    step = stepper(window.result)
    assert step.label.text() == "Target 2 of 2"
    assert step.previous.isEnabled()
    assert not step.following.isEnabled()
    assert window.stack.currentWidget() is window.result, "a read, not a screen"


def test_an_arrow_with_nowhere_to_go_is_visibly_spent(qt_app):
    """It was drawn in exactly the resting colour, which on a control whose whole
    job is to say whether there is another chart is no answer at all: both arrows
    looked live on the first of five and on the last."""
    def ink(button) -> set[int]:
        image = button.grab().toImage()
        return {
            image.pixel(x, y) & 0xFFFFFF
            for x in range(image.width()) for y in range(image.height())
        }

    first, middle = Stepper(0, 5), Stepper(2, 5)
    assert not first.previous.isEnabled()
    assert middle.previous.isEnabled()
    assert ink(first.previous) != ink(middle.previous), (
        "a spent arrow has to look different from one that works"
    )
    assert ink(first.following) == ink(middle.following), (
        "and one that works still looks like every other one"
    )


def test_the_label_does_not_move_the_buttons_beside_it(qt_app):
    """Everything to the right of it is the bar's actions. A reader stepping
    through twelve charts must not have the buttons shift under the pointer
    between chart 9 and chart 10."""
    widths = {Stepper(index, 12).label.width() for index in range(12)}
    assert len(widths) == 1


def test_a_single_chart_run_has_no_stepper_and_no_combined_verdict(window, charts):
    """Both controls are about a capture holding several. One chart is the
    ordinary case and must look exactly as it always did."""
    window._on_analysed(charts[0])

    assert stepper(window.result) is None
    assert "Combined Verdict" not in bar_buttons(window.result)


def test_a_board_that_measured_nothing_does_not_land_on_an_empty_report(
    window, charts, monkeypatch
):
    """Every chart in the frame refused. There is no report to land on, and the
    placement screen is where the corners that produced it are still on."""
    from PySide6.QtWidgets import QMessageBox

    said = []
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *args, **kwargs: said.append(args[2])),
    )
    window._on_charts_analysed(([], ["no chart could be placed"]))

    assert window.stack.currentWidget() is window.placement
    assert said and "no chart could be placed" in said[0]


# -- the combined verdict ------------------------------------------------------


def test_the_board_s_own_verdict_is_one_press_away(window, charts):
    """Four cards measure one system, and whether that system passed at every
    position is not any one card's answer. It keeps its screen; what changed is
    that reaching a report no longer goes through it."""
    window._on_charts_analysed((charts, []))
    combined_button(window.result).click()

    assert window.stack.currentWidget() is window.session
    assert len(window.session.session.captures) == 2


def test_it_is_rebuilt_on_every_press(window, charts, monkeypatch):
    """A chart can be judged again between two presses of this, and a session
    held from before that would state a verdict against thresholds that are no
    longer the ones on screen."""
    from deltachrome.pipeline import rescore

    window._on_charts_analysed((charts, []))
    combined_button(window.result).click()
    assert window.session.session.profile.profile_id == FADGI3

    window.stack.setCurrentWidget(window.result)
    window._on_rescored([rescore(run, FADGI4) for run in charts])
    combined_button(window.result).click()
    assert window.session.session.profile.profile_id == FADGI4


def test_a_report_opened_from_the_session_goes_back_to_it(window, charts):
    """And the one the run landed on goes out to the library, because there is
    no session behind it that the reader has seen."""
    window._on_charts_analysed((charts, []))
    window.result.backRequested.emit()
    assert window.stack.currentWidget() is window.library

    window._on_charts_analysed((charts, []))
    combined_button(window.result).click()
    window.session.captureRequested.emit(charts[1].run.run_id)
    assert window.stack.currentWidget() is window.result
    assert window.result.chart_index == 1

    window.result.backRequested.emit()
    assert window.stack.currentWidget() is window.session


def test_a_saved_capture_offers_it_too(window, charts, tmp_path):
    """The charts are in the record beside the one on screen, so combining them
    is a read. Offering it on a fresh run and not on the same capture reopened
    would be the control appearing and vanishing for no reason a reader has."""
    from deltachrome.session import capture_record
    from deltachrome_ui.library import summarise

    record = capture_record(charts, labels=["chart 1", "chart 2"])
    directory = tmp_path / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    path = record.save(directory / f"{record.capture_id}.json")

    window._open_run(summarise(json.loads(path.read_text()), path))
    assert window.result.chart_count == 2
    combined_button(window.result).click()

    assert window.stack.currentWidget() is window.session
    assert len(window.session.session.captures) == 2


# -- a capture is judged as one thing ------------------------------------------


def test_judging_a_board_again_judges_every_chart_of_it(window, charts):
    """A capture record holds one combined verdict against one profile, so a
    board with one card at four star and the rest at three is not a state that
    can be saved or combined. The control stands on one chart's report and is
    asking the question of the capture."""
    from deltachrome_ui.workers import Rescorer

    window._on_charts_analysed((charts, []))
    window._show_chart(1)

    rescored: list = []
    worker = Rescorer(results=list(window._fresh_results.values()),
                      profile_id=FADGI4)
    worker.finished.connect(rescored.append)
    worker.run()
    window._on_rescored(rescored[0])

    assert window.result.chart_index == 1, "the reader stays on the chart they were on"
    assert {run.run.profile.get("profile_id")
            for run in window._fresh_results.values()} == {FADGI4}


def test_saving_a_re_judged_board_saves_the_judgement_on_screen(
    window, charts, tmp_path, monkeypatch
):
    """The bug the fix above exists for. The shell's list of charts is what Save
    writes, and a re-score is a new run with its own id -- so judging chart 2 of
    a board again and pressing Save wrote the judgement it replaced, with the
    header naming a profile the record did not hold."""
    from PySide6.QtWidgets import QMessageBox
    from deltachrome.pipeline import rescore

    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *args, **kwargs: None)
    )
    window._on_charts_analysed((charts, []))
    window._show_chart(1)
    window._on_rescored([rescore(run, FADGI4) for run in charts])
    window._save_result(window.result.result)

    written = list((tmp_path / "runs").glob("*.json"))
    assert len(written) == 1
    record = json.loads(written[0].read_text())
    assert {chart["profile"]["profile_id"] for chart in record["charts"]} == {FADGI4}


def test_a_chart_of_a_saved_capture_opens_when_it_is_asked_for_by_id(
    window, charts, tmp_path, monkeypatch
):
    """The frame map's whole interaction is clicking a chart, and on a saved
    capture that had nowhere to land.

    ``captureRequested`` carries a run id, and a saved board's charts each have
    their own -- but the library lists *captures*, so only the first chart's id
    is ever in it and looking the others up there found nothing. The charts of
    the record already open are searched first, which is where they are.
    """
    from PySide6.QtWidgets import QMessageBox
    from deltachrome.session import capture_record
    from deltachrome_ui.library import summarise

    # So the regression is a failure rather than a modal nobody can dismiss --
    # which is how it presented: the test run stopped dead on "Run not found".
    refusals: list = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: refusals.append(a)))

    record = capture_record(charts, labels=["chart 1", "chart 2"])
    directory = tmp_path / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    path = record.save(directory / f"{record.capture_id}.json")

    window._open_run(summarise(json.loads(path.read_text()), path))
    combined_button(window.result).click()
    assert window.stack.currentWidget() is window.session

    second = charts[1].run.run_id
    assert second != record.capture_id, "the second chart has an id of its own"
    window.session.captureRequested.emit(second)
    assert not refusals, "the chart was refused instead of opened"
    assert window.stack.currentWidget() is window.result
    assert window.result.chart_index == 1
    # And Back still belongs to the session it was reached from.
    window.result.backRequested.emit()
    assert window.stack.currentWidget() is window.session


# -- a capture is saved as a capture, from either of its screens ---------------


def test_the_board_s_own_screen_saves_a_capture_record(
    window, charts, tmp_path, monkeypatch
):
    """Reported from the running application, and it was two bugs in one.

    This screen offered *Save Session*, which writes a session record -- no
    ``run_id``, no ``image``, no ``charts``. The library summarised that as a
    row named after the file (``session-a1b2c3-d4e5f6``) with no capture and no
    target, and opening it built a report out of a record that is not one.

    A one-image board is a **capture**, and saving it means what saving any
    other capture means: one record, named after the image, holding every
    target. The same action and the same words as on a chart's report, because
    a capture is saved once whichever of its screens you are looking at.
    """
    from PySide6.QtWidgets import QMessageBox
    from deltachrome_ui.library import read_record

    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    directory = tmp_path / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("deltachrome_ui.app.ensure_run_dir", lambda: directory)

    window._on_charts_analysed((charts, []))
    assert window.stack.currentWidget() is window.session
    assert window.session.save_mode == "capture"

    save = next(b for b in window.session.findChildren(QPushButton)
                if b.text() == "Save to Library")
    save.click()

    written = list(directory.glob("*.json"))
    assert len(written) == 1
    assert not written[0].name.startswith("session-"), "a session record was written"
    record = read_record(written[0])
    assert len(record["charts"]) == 2, "every target is in the one record"
    assert record["image"]["filename"], "the record is named by its capture"


def test_a_capture_already_in_the_library_offers_no_save(
    window, charts, tmp_path, monkeypatch
):
    """It is already there. A second button writing it again is an invitation to
    make a duplicate under a new id."""
    from PySide6.QtWidgets import QMessageBox
    from deltachrome.session import capture_record
    from deltachrome_ui.library import summarise

    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    record = capture_record(charts, labels=["chart 1", "chart 2"])
    directory = tmp_path / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    path = record.save(directory / f"{record.capture_id}.json")

    window._open_from_library(summarise(json.loads(path.read_text()), path))
    assert window.stack.currentWidget() is window.session
    assert window.session.save_mode == "none"
    assert not [b for b in window.session.findChildren(QPushButton)
                if b.text() in ("Save to Library", "Save Session")]


def test_a_saved_board_opens_on_all_targets_and_a_lone_run_does_not(
    window, charts, tmp_path
):
    """The row's own shape decides, so the library needs no second action."""
    from deltachrome.session import capture_record
    from deltachrome_ui.library import summarise

    directory = tmp_path / "runs"
    directory.mkdir(parents=True, exist_ok=True)

    board = capture_record(charts, labels=["chart 1", "chart 2"])
    board_path = board.save(directory / f"{board.capture_id}.json")
    summary = summarise(json.loads(board_path.read_text()), board_path)
    assert summary.chart_count == 2
    window._open_from_library(summary)
    assert window.stack.currentWidget() is window.session

    lone = capture_record([charts[0]], labels=["chart 1"])
    lone_path = lone.save(directory / f"{lone.capture_id}.json")
    summary = summarise(json.loads(lone_path.read_text()), lone_path)
    assert summary.chart_count == 1
    window._open_from_library(summary)
    assert window.stack.currentWidget() is window.result


def test_the_library_row_says_how_many_targets_the_capture_held(
    window, charts, tmp_path
):
    """Before it is clicked, because such a row opens somewhere different."""
    from deltachrome.session import capture_record
    from deltachrome_ui.library import summarise
    from deltachrome_ui.run_table import _target_search, _target_text

    directory = tmp_path / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    record = capture_record(charts, labels=["chart 1", "chart 2"])
    path = record.save(directory / f"{record.capture_id}.json")
    summary = summarise(json.loads(path.read_text()), path)

    assert _target_text(summary).endswith("×2")
    assert summary.target_name in _target_text(summary)
    # The word a person would type is not in the cell, which shows a sign.
    assert "targets" in _target_search(summary)

    lone = summarise(json.loads(
        capture_record([charts[0]], labels=["chart 1"])
        .save(directory / "lone.json").read_text()), directory / "lone.json")
    assert _target_text(lone) == lone.target_name, "no count where there is one"


def test_opening_a_saved_board_draws_its_outlines_before_any_pixels(
    window, charts, tmp_path
):
    """The screen must not wait on a decode to say where a target is.

    The outlines come from each chart's recorded corners, so the map is complete
    the moment the record is read. The pixels arrive afterwards and only make
    the targets recognisable -- and they are read at the capture's own reduced
    resolution, not decoded whole, which on a JPEG 2000 master is 0.61 s against
    11.2 s.
    """
    from deltachrome.session import capture_record
    from deltachrome_ui.library import summarise

    record = capture_record(charts, labels=["Target 1", "Target 2"])
    directory = tmp_path / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    path = record.save(directory / f"{record.capture_id}.json")

    window._open_from_library(summarise(json.loads(path.read_text()), path))
    block = window.session.frame_block
    assert block is not None, "no frame map was built"
    assert len(block._rows) == 2
    # Complete without pixels: every outline is placed and numbered already.
    assert all(len(c.corners) == 4 for c in block.map._frame.charts)
    assert [c.ordinal for c in block.map._frame.charts] == [1, 2]


def test_a_thumbnail_for_a_capture_no_longer_on_screen_is_dropped(
    window, charts, tmp_path
):
    """A reader can open one capture and another inside six tenths of a second,
    and the first read would otherwise land on the second one's map -- a picture
    of the wrong capture under the right outlines, which is exactly the failure
    the fingerprint match exists to prevent everywhere else."""
    from PySide6.QtGui import QImage
    from deltachrome.session import capture_record
    from deltachrome_ui.library import summarise

    record = capture_record(charts, labels=["Target 1", "Target 2"])
    directory = tmp_path / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    path = record.save(directory / f"{record.capture_id}.json")
    window._open_from_library(summarise(json.loads(path.read_text()), path))

    block = window.session.frame_block
    stray = QImage(4, 3, QImage.Format.Format_RGB888)
    stray.fill(0xFF0000)
    window._on_thumbnail(stray, "a-different-capture")
    assert block.map._frame.image is not stray, "a stale read reached the map"

    window._on_thumbnail(stray, window._thumbnail_key)
    assert block.map._frame.image is stray, "the right one was dropped too"


def test_a_re_run_restores_every_chart_of_a_capture_of_two_models(
    window, charts, tmp_path, monkeypatch
):
    """Reported from the running application on a board of four QA-62 cards and
    an Object Level Target. A capture of two models states no one target at its
    top level, which Re-run read as "target unknown": it identified the frame
    from scratch, found one chart, and set the five saved placements aside. And
    what it did restore went onto the canvas and not into the frame's list, so a
    region search then made its own find the first and only chart."""
    from deltachrome.session import capture_record
    from deltachrome_ui.library import summarise

    directory = tmp_path / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    record = json.loads(
        capture_record(charts, labels=["chart 1", "chart 2"])
        .save(directory / "board.json").read_text()
    )
    record["charts"][1]["target"]["target_id"] = "isa_object_level_1x_v1"
    record["target"]["target_id"] = None
    summary = summarise(record, directory / "board.json")
    assert [chart[0] for chart in summary.charts] == [SG, "isa_object_level_1x_v1"]

    monkeypatch.setattr(window, "load_image", lambda path: None)
    window._reuse_run(summary)
    assert window._pending_setup[0] == SG, "the first chart's target, not None"

    window._restore_charts(window._pending_charts)
    assert [t.target_id for t in window._targets_per_chart()] == [
        SG, "isa_object_level_1x_v1"
    ]
    assert all(d.diagnostics.get("restored") for d in window._detections)


def test_stepping_to_another_chart_keeps_the_metric_on_screen(window, charts):
    """A reader on SFR50 of target 1 steps to target 2 to compare SFR50, and was
    put back on the overview every time."""
    window._fresh_results = {r.run.run_id: r for r in charts}
    window._open_record = None
    window.result.set_result(charts[0], saved=False, chart=(0, 2))
    page = next(k for k in window.result._page_index if k[0] != "page")
    window.result.nav.show_target(page)
    window.result._on_navigate(page)
    assert window.result.current_page == page

    window._show_chart(1)
    assert window.result.chart_index == 1
    assert window.result.current_page == page


def test_stepping_to_another_chart_keeps_the_capture_panel_open(window, charts):
    """Every target of a capture is in the same frame, so the pixels on screen
    are the next target's pixels too. The panel folded on every step, and on a
    saved capture the decode was dropped and offered again."""
    from deltachrome_ui.imageproxy import ImageProxy

    image = charts[0].measurement.image if hasattr(charts[0], "measurement") else None
    if image is None:
        pytest.skip("the fixture's runs carry no decoded image")
    proxy = ImageProxy(image)
    window._fresh_results = {r.run.run_id: r for r in charts}
    window._open_record = None
    window._live_proxy = proxy
    window.result.set_proxy(proxy)
    window.result.set_result(charts[0], saved=False, chart=(0, 2))
    window.result.dock.set_collapsed(False)

    window._show_chart(1)
    assert not window.result.dock.collapsed
    assert window.result.proxy is proxy
