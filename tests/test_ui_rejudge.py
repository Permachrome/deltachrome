"""The report's three controls over its own run.

All three exist because the report is where a mistake becomes visible, and until
now the only thing to do about one was to go back to the library and start again.

* The **capture panel** opens folded away. On most reports it holds one button
  and a sentence saying the button has not been pressed, which is a third of a
  laptop screen spent on the least.
* **Back to placement** is a change of page: the screen the run was set up on is
  still holding the capture, the target and the corners.
* **The profile is a control**, wherever the reading is still in memory. Judging
  the same reading again is a fifth of a second; re-reading the capture to ask
  the same question is a decode.

What is pinned here is mostly what must *not* happen: a re-judgement that quietly
loses the reader's place, a control offered where the engine would refuse it, and
a panel that asks for a capture it has already been given.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from PySide6.QtWidgets import QApplication  # noqa: E402

from deltachrome.pipeline import analyze, rescore  # noqa: E402

SG = "colorchecker_sg_v1"
FADGI3 = "fadgi_3star_documents_unbound_general_2023"
FADGI4 = "fadgi_4star_documents_unbound_general_2023"


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    from fixtures import SyntheticChart

    tmp = tmp_path_factory.mktemp("rejudge")
    chart = SyntheticChart()
    reference = chart.write_reference(tmp / "ref.txt")
    image, corners = chart.render_capture(tmp / "capture.tif", skew=18)
    return image, reference, corners


@pytest.fixture(scope="module")
def result(capture):
    image, reference, corners = capture
    return analyze(
        image, target=SG, reference_path=reference, profile=FADGI3, corners=corners
    )


@pytest.fixture(scope="module")
def proxy(capture):
    from deltachrome.imaging.loader import load_image
    from deltachrome_ui.imageproxy import ImageProxy

    return ImageProxy(load_image(capture[0]))


def report(result, *, saved=False, proxy=None):
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    if proxy is not None:
        view.set_proxy(proxy)
    view.set_result(result, saved=saved)
    return view


def saved_copy(result, tmp_path):
    """The same run as the library would hand it back: numbers, and no pixels."""
    from deltachrome_ui.library import load_result

    return load_result(result.run.save(tmp_path / "run.json"))


# -- the capture panel --------------------------------------------------------


def test_the_capture_panel_opens_folded_away(qt_app, result):
    """It holds one control until a capture is loaded, and it is 384 px wide."""
    view = report(result)
    assert view.dock.collapsed
    assert view.dock.tab.isVisibleTo(view)


def test_the_folded_strip_says_what_opening_it_leads_to(qt_app, result):
    """A chevron says something is folded away and nothing about what -- which
    on the one control that starts a twenty-second decode is the wrong thing to
    be silent about."""
    from deltachrome_ui.capture_dock import LOAD_LABEL

    view = report(result)
    assert view.dock.tab._label == LOAD_LABEL == "Load Capture"
    assert view.dock.prompt.text() == LOAD_LABEL


def test_a_capture_arriving_relabels_the_strip_and_does_not_open_it(
    qt_app, result, proxy
):
    """A capture arriving is not by itself a request to look at it.

    The panel unfolded for one, which meant the next report opened wide because
    the previous one had been looked at -- and a run analyzed in this session is
    handed its master with no gesture at all. Opening stays a gesture. What
    changes is that the strip stops offering to load what is already loaded.
    """
    view = report(result)
    view.set_proxy(proxy)
    assert view.dock.collapsed
    assert view.dock.tab._label == "Capture"


def test_a_fresh_run_keeps_the_capture_it_was_handed(qt_app, result, proxy):
    """The regression this is here for: ``set_result`` clears the panel of the
    previous run, and used to clear this run's capture with it.

    A run analyzed from a master already decoded in memory then showed "Load
    Capture", and pressing it decoded the same file a second time -- 24.6 s on
    this repository's own JPEG 2000 sample.
    """
    view = report(result, proxy=proxy)
    assert view.dock.proxy is proxy
    # Folded away like any other report, and holding the capture -- so opening
    # it shows pixels instead of asking for a decode of a file already decoded.
    assert view.dock.collapsed
    asked: list[int] = []
    view.captureRequested.connect(lambda: asked.append(1))
    view.dock.tab.click()
    assert not view.dock.collapsed
    assert asked == []


def test_a_run_with_no_capture_folds_the_panel_back_up(qt_app, result, proxy):
    view = report(result, proxy=proxy)
    view.set_proxy(None)
    assert view.dock.collapsed
    assert view.dock.tab._label == "Load Capture"


def test_opening_the_strip_is_the_request(qt_app, result):
    """One press, not two. Nobody opens a panel labelled Load Capture in order
    to look at the offer."""
    view = report(result)
    asked: list[int] = []
    view.captureRequested.connect(lambda: asked.append(1))
    view.dock.tab.click()
    assert not view.dock.collapsed
    assert asked == [1]


def test_a_second_press_while_it_is_running_asks_for_nothing(qt_app, result):
    """The guard is not defensive tidiness. Each request is a full decode --
    24.6 s on this repository's JPEG 2000 sample -- and before the bar existed a
    second press during the first was the ordinary thing to do."""
    view = report(result)
    asked: list[int] = []
    view.captureRequested.connect(lambda: asked.append(1))
    view.dock.tab.click()
    view.dock.set_loading(True)
    view.dock.request_load()
    view.dock.prompt.click()
    assert asked == [1]


def test_the_wait_is_shown_on_the_panel_rather_than_over_the_window(qt_app, result):
    """This is work alongside the report, not instead of it: the report stays on
    screen and usable, so the bar belongs where the button was."""
    view = report(result)
    view.dock.set_collapsed(False)
    view.dock.set_loading(True)
    assert view.dock.busy.isVisibleTo(view)
    assert not view.dock.prompt.isVisibleTo(view)


def test_a_failed_load_leaves_the_button_as_the_retry(qt_app, result):
    """And leaves the panel open. Folding it away on a failure would hide the
    only way to try again."""
    view = report(result)
    view.dock.set_collapsed(False)
    view.dock.set_loading(True)
    view.dock.set_loading(False)
    assert not view.dock.collapsed
    assert view.dock.prompt.isVisibleTo(view)
    assert not view.dock.busy.isVisibleTo(view)


def test_an_arriving_capture_takes_the_wait_away(qt_app, result, proxy):
    view = report(result)
    view.dock.set_collapsed(False)
    view.dock.set_loading(True)
    view.set_proxy(proxy)
    assert not view.dock.loading
    assert not view.dock.busy.isVisibleTo(view)
    assert view.dock.overview is not None


# -- back to placement --------------------------------------------------------


def header_icons(view) -> dict[str, str]:
    """The header's icon actions, as ``{kind: tooltip}``.

    Asserting on the tooltip rather than only on the icon's presence, because an
    icon nobody can name is a control nobody can use -- and a shape is not
    self-evidently "open the capture and place it again".
    """
    from deltachrome_ui.widgets import IconButton

    return {b.kind: b.toolTip() for b in view.header.findChildren(IconButton)}


def test_a_fresh_run_offers_the_screen_it_was_set_up_on(qt_app, result):
    view = report(result)
    icons = header_icons(view)
    assert "placement" in icons
    assert "back" in icons
    assert "placement" in icons["placement"].lower()


def test_a_saved_run_does_not(qt_app, result, tmp_path):
    """That screen holds whatever capture this session last placed, which for a
    run opened out of the library is somebody else's chart or none at all.
    Re-running the capture is the offer that actually works there."""
    view = report(saved_copy(result, tmp_path), saved=True)
    icons = header_icons(view)
    assert "placement" not in icons
    assert "rerun" in icons


# -- judging it again ---------------------------------------------------------


def test_the_profile_is_a_control_while_the_reading_is_in_memory(qt_app, result):
    view = report(result)
    assert view.can_rescore
    assert view.picker is not None
    assert view.picker.profile_id == FADGI3


def test_and_a_label_once_it_is_not(qt_app, result, tmp_path):
    """A run rebuilt from its record has every number and no pixels, so judging
    it again is a thing the engine refuses. The control is kept off the screen so
    that nobody has to be refused."""
    from PySide6.QtWidgets import QLabel

    saved = saved_copy(result, tmp_path)
    view = report(saved, saved=True)
    assert not view.can_rescore
    assert view.picker is None
    # It still says what the run was judged against, in the same place. A control
    # that cannot be offered is replaced by the answer, not by a gap.
    text = "\n".join(label.text() for label in view.header.findChildren(QLabel))
    assert saved.run.profile["display_name"] in text


# -- reading a saved run's capture again --


def test_a_fresh_run_has_nothing_to_establish(qt_app, result):
    """The reading is already in memory, so there is nothing for the button to
    read again -- the picker is the control here, not this one."""
    view = report(result)
    assert not view.can_establish
    assert view._establish_button is None


def test_a_saved_run_offers_to_read_its_capture_again(qt_app, result, tmp_path):
    """The record carries its own corners, target and profile, so the button
    that turns the label back into a picker is offered."""
    view = report(saved_copy(result, tmp_path), saved=True)
    assert view.can_establish
    assert view._establish_button is not None
    assert view._establish_button.text() == "Change Guidelines…"


def test_pressing_it_asks_for_nothing_more_than_a_click(qt_app, result, tmp_path):
    view = report(saved_copy(result, tmp_path), saved=True)
    asked = []
    view.establishRequested.connect(lambda: asked.append(True))
    view._establish_button.click()
    assert asked == [True]


def test_it_says_what_is_happening_while_it_runs(qt_app, result, tmp_path):
    """A disabled button that still reads ``Change Guidelines…`` invites a second
    press -- the same failure the capture panel's own busy state exists to
    prevent, so the label changes rather than only the enabled state."""
    view = report(saved_copy(result, tmp_path), saved=True)
    view.set_establishing(True)
    assert not view._establish_button.isEnabled()
    assert view._establish_button.text() == "Reading…"
    view.set_establishing(False)
    assert view._establish_button.isEnabled()
    assert view._establish_button.text() == "Change Guidelines…"


def test_a_record_with_no_placement_offers_nothing_to_establish(
    qt_app, result, tmp_path
):
    """No amount of re-decoding supplies a corner placement nobody recorded, so
    a record this bare gets no button rather than one that always fails."""
    import json

    from deltachrome_ui.library import load_result

    data = result.run.to_dict()
    data["geometry"] = {}
    path = tmp_path / "no_geometry.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    view = report(load_result(path), saved=True)
    assert not view.can_establish
    assert view._establish_button is None


def test_choosing_a_different_level_asks_for_a_re_judgement(qt_app, result):
    view = report(result)
    asked: list[str] = []
    view.rescoreRequested.connect(asked.append)
    assert view.picker.select_profile(FADGI4)
    assert asked == [FADGI4]


def test_choosing_the_one_already_selected_asks_for_nothing(qt_app, result):
    """Re-selecting what is selected happens on the way through the category
    list, and answering it would re-run every metric to arrive at the report
    already on screen."""
    view = report(result)
    asked: list[str] = []
    view.rescoreRequested.connect(asked.append)
    view.picker.select_profile(FADGI3)
    assert asked == []


def test_the_reader_keeps_their_place_across_a_re_judgement(qt_app, result):
    """Changing the level while reading tone response is a question about tone
    response, and the answer is on the page they were already on."""
    from deltachrome_ui.nav import METRIC

    view = report(result)
    where = next(
        target for target in view._page_index
        if target[0] == METRIC and target[-1] == "tone_response_oecf"
    )
    view.show_metric(where[1], where[2])
    assert view.current_page == where
    view.set_result(rescore(result, FADGI4), saved=False, focus=where)
    assert view.current_page == where


def test_a_page_the_new_profile_does_not_have_falls_back_to_the_overview(
    qt_app, result
):
    """A profile declaring different metrics has different pages, so the focus is
    a request rather than an instruction."""
    from deltachrome_ui.nav import METRIC, OVERVIEW, PAGE

    view = report(result)
    view.set_result(
        rescore(result, FADGI4), saved=False,
        focus=(METRIC, "colour", "a_metric_no_profile_declares"),
    )
    assert view.current_page == (PAGE, OVERVIEW)


def test_saving_after_a_re_judgement_saves_what_is_on_screen(qt_app, result):
    """The point of the control. Whichever profile the report is showing is the
    one the record is written against -- there is no separate step, and no way
    for the two to disagree, because the button is rebuilt with the result."""
    from PySide6.QtWidgets import QPushButton

    view = report(result)
    asked: list = []
    view.saveRequested.connect(asked.append)
    view.set_result(rescore(result, FADGI4), saved=False)
    button = next(
        b for b in view.header.findChildren(QPushButton)
        if b.text() == "Save to Library"
    )
    button.click()
    assert asked[0].run.profile["profile_id"] == FADGI4


def test_a_failed_re_judgement_puts_the_control_back(qt_app, result):
    """A combo box naming one profile over a report judged against another is
    the one state this control must never be in."""
    view = report(result)
    view.set_rescoring(True)
    view.picker.blockSignals(True)
    view.picker.select_profile(FADGI4)
    view.picker.blockSignals(False)

    view.set_rescoring(False)
    assert view.picker.isEnabled()
    assert view.picker.profile_id == FADGI3


def test_going_back_to_placement_is_asked_for_rather_than_done_here(qt_app, result):
    """The report does not own the placement screen and cannot know whether it
    still holds this capture. It says what was clicked; the shell decides."""
    from deltachrome_ui.widgets import IconButton

    view = report(result)
    asked: list[int] = []
    view.placementRequested.connect(lambda: asked.append(1))
    next(
        b for b in view.header.findChildren(IconButton) if b.kind == "placement"
    ).click()
    assert asked == [1]


def test_a_run_read_again_offers_re_run_rather_than_another_capture_s_placement(
    qt_app, result
):
    """*Change Guidelines…* makes a saved run live without placing it here, and the
    placement screen still holds whatever was placed last -- reported from the
    running application as "back to the placement" opening a batch's frame
    beside a different capture's report. Such a run asks for a re-run."""
    from deltachrome_ui.result_view import ResultView
    from deltachrome_ui.widgets import IconButton

    view = ResultView()
    view.set_result(result, saved=False, placed_here=False)
    kinds = {b.kind for b in view.header.findChildren(IconButton)}
    assert "rerun" in kinds and "placement" not in kinds
    asked: list = []
    view.rerunRequested.connect(asked.append)
    next(b for b in view.header.findChildren(IconButton) if b.kind == "rerun").click()
    assert asked == [result]


def test_back_says_where_it_goes(qt_app, result):
    """A report reached from All Targets goes back there, and said "the library"."""
    from deltachrome_ui.result_view import ResultView

    view = ResultView()
    view.set_result(result, saved=False)
    assert view.back_button.toolTip() == "Back to the library"
    view.set_back_destination("All Targets")
    assert view.back_button.toolTip() == "Back to All Targets"
    view.set_result(result, saved=False)
    assert view.back_button.toolTip() == "Back to All Targets", "kept across a rebuild"


def test_every_icon_in_the_header_says_what_it_does(qt_app, result, tmp_path):
    """Enforced rather than trusted: :class:`IconButton` refuses to be built
    without a tooltip, and this checks the header actually carries icons for
    that refusal to be about."""
    for view in (report(result),
                 report(saved_copy(result, tmp_path), saved=True)):
        icons = header_icons(view)
        assert icons
        assert all(tip.strip() for tip in icons.values()), icons


def test_a_panel_load_that_fails_does_not_take_the_reader_off_the_report(
    qt_app, result, tmp_path, monkeypatch
):
    """It used to share the *opening* failure handler, which returns to the
    library -- so a master that had moved took the reader off a perfectly good
    report to say so. The report is complete without the pixels; only the panel
    failed, and only the panel should say so."""
    monkeypatch.setenv("DELTACHROME_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.exec", lambda self: None, raising=False
    )
    from deltachrome_ui.app import MainWindow
    from deltachrome_ui.workers import LoadFailure

    window = MainWindow()
    try:
        window.result.set_result(result, saved=False)
        window.stack.setCurrentWidget(window.result)
        window.result.dock.set_collapsed(False)
        window.result.dock.set_loading(True)

        window._on_capture_load_failed(
            LoadFailure("It could not be opened.", "No such file.")
        )
        assert window.stack.currentWidget() is window.result
        assert not window.result.dock.loading
        assert window.result.dock.prompt.isVisibleTo(window)
    finally:
        window.close()


def test_every_report_opens_with_the_panel_folded_away(qt_app, result, proxy):
    """Whether the last one was looked at is not a property of this one."""
    view = report(result, proxy=proxy)
    view.dock.tab.click()
    assert not view.dock.collapsed

    view.set_result(result, saved=False)
    assert view.dock.collapsed


def test_but_a_re_judgement_leaves_it_as_the_reader_left_it(
    qt_app, result, proxy, tmp_path, monkeypatch
):
    """A re-judgement is the run already on screen answering a different
    question, not a different run opening. A reader who opened the panel to look
    at a patch has not asked for it to be shut."""
    monkeypatch.setenv("DELTACHROME_RUN_DIR", str(tmp_path))
    from deltachrome_ui.app import MainWindow

    window = MainWindow()
    try:
        window._proxy = proxy
        window.result.set_proxy(proxy)
        window.result.set_result(result, saved=False)
        window.result.dock.tab.click()
        assert not window.result.dock.collapsed

        window._on_rescored([rescore(result, FADGI4)])
        assert not window.result.dock.collapsed
        assert window.result.dock.proxy is proxy
    finally:
        window.close()


# -- not decoding what is already decoded -------------------------------------


def test_a_capture_already_in_memory_is_not_read_again(
    qt_app, result, proxy, tmp_path, monkeypatch
):
    """Analyze, save, reopen from the library, look at the capture. That is an
    ordinary afternoon, and it used to cost a second full decode of a file the
    process was still holding -- 25.1 s on this repository's JPEG 2000 sample."""
    monkeypatch.setenv("DELTACHROME_RUN_DIR", str(tmp_path))
    from deltachrome_ui.app import MainWindow
    from deltachrome_ui.library import load_result

    window = MainWindow()
    try:
        window._proxy = proxy
        window._image_path = Path(result.run.image["path"])
        window.result.set_result(load_result(result.run.save(tmp_path / "r.json")),
                                 saved=True)
        assert window.result.dock.proxy is None

        window._load_capture_for_report()
        assert window.result.dock.proxy is proxy
        assert not window.result.dock.loading
    finally:
        window.close()


def test_a_file_that_changed_under_us_is_read_again(
    qt_app, result, proxy, tmp_path, monkeypatch
):
    """Matching on the path alone would hand back pixels from before the file
    was replaced, and a report drawn over the wrong capture is the class of
    silent wrongness this project refuses everywhere else."""
    monkeypatch.setenv("DELTACHROME_RUN_DIR", str(tmp_path))
    from deltachrome_ui.app import MainWindow

    window = MainWindow()
    try:
        window._proxy = proxy
        window._image_path = Path(result.run.image["path"])
        record = dict(result.run.image)
        record["fingerprint"] = "not what that file hashes to"
        assert window._already_decoded(record) is None
        assert window._already_decoded(result.run.image) is proxy
    finally:
        window.close()
