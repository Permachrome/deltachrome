"""The batch screens, and the seams between them and the shell.

The engine is tested in ``test_batch.py``. What is here is the things a window
adds: that the count is visible while it runs, that a capture nobody could measure
does not render as a failure, that leaving the setup screen abandons the folder,
and that a batch's records reach the one table that reads records.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deltachrome import batch as B
from deltachrome.metrics.base import MetricResult

pytest.importorskip("PySide6")


@pytest.fixture(scope="module", autouse=True)
def _app():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application


def measured(name: str, overall: str = "PASS", *, claim: bool = False,
             sfr50: float | None = None) -> B.BatchCapture:
    results = []
    if sfr50 is not None:
        results.append(MetricResult(
            metric_id="sfr50", outcome="evaluated", value=sfr50,
            units="% of half-sampling freq", passed=True,
            condition="40 <= value <= 75",
        ))
    return B.BatchCapture(
        path=Path(name), outcome="measured", overall=overall,
        conformance_claim=claim, results=results,
    )


def result_for(captures, tmp_path) -> B.BatchResult:
    from deltachrome.profiles.schema import load_profile

    return B.BatchResult(
        profile=load_profile("fadgi_3star_bound_volumes_general_2023"),
        target_id="isa_object_level_05x_v1", target_name="ISA OLT 0.5X",
        directory=tmp_path, captures=list(captures), created_utc="now",
    )


# -- the progress screen -------------------------------------------------------


def test_the_count_is_visible_from_the_first_capture_to_the_last(tmp_path):
    """A determinate bar over a folder of hundred-megapixel masters is a lie told
    sixteen seconds at a time. The honest answer to "is this working" is which
    capture it is on and how many are left, so that is what the screen says."""
    from deltachrome_ui.batch_view import BatchView

    view = BatchView()
    view.start(tmp_path, 20, "ISA OLT 0.5X", "FADGI 3-star")
    assert view.count.text() == "0 of 20 captures"

    view.advance(1, 20, "a.tif", measured("a.tif"))
    assert view.count.text() == "1 of 20 captures"
    view.advance(7, 20, "g.tif", measured("g.tif"))
    assert view.count.text() == "7 of 20 captures"
    # And nowhere else. "Last: g.tif" used to sit under the count, naming the
    # row that had just been added to the table three lines below it.
    assert view.table.item(view.table.rowCount() - 1, 1).text() == "g.tif"


def test_every_capture_reaches_the_list_as_it_finishes(tmp_path):
    from deltachrome_ui.batch_view import BatchView

    view = BatchView()
    view.start(tmp_path, 3, "T", "P")
    for number, name in enumerate(("a.tif", "b.tif", "c.tif"), start=1):
        view.advance(number, 3, name, measured(name))
        assert view.table.rowCount() == number
    assert view.table.item(2, 1).text() == "c.tif"


def test_a_capture_nobody_could_measure_does_not_read_as_a_failure(tmp_path):
    """Nothing judged it. Drawing it in the failure tone would say the system
    missed a requirement, when what happened is that a frame could not be read."""
    from deltachrome_ui.batch_view import _status_for

    refused = B.BatchCapture(
        path=Path("x.tif"), outcome="refused", reason="no chart in this frame"
    )
    status = _status_for(refused)
    assert status.tone == "unknown"
    assert status.tone != "fail"
    assert "no chart" in status.hint


def test_a_run_that_cannot_be_quoted_is_not_shown_in_the_pass_tone(tmp_path):
    """The rule the whole frontend follows, applied to a batch: every capture
    passed and none of them was measured against its own chart.

    It is said by every row and by the tally, in the qualified tone, and it is
    no longer *also* said as a sentence above them -- a paragraph restating a
    column is one a reader learns to skip.
    """
    from deltachrome_ui.batch_view import DOT, BatchView
    from deltachrome_ui.theme import tone_colors

    nominal = [
        B.BatchCapture(path=Path(name), outcome="measured", overall="PASS",
                       conformance_claim=False,
                       reference_basis="manufacturer_nominal")
        for name in ("a.tif", "b.tif")
    ]
    view = BatchView()
    view.start(tmp_path, 2, "T", "P")
    for number, capture in enumerate(nominal, start=1):
        view.advance(number, 2, capture.name, capture)
    outcome = result_for(nominal, tmp_path)
    assert outcome.conformance_claim is False
    view.show_result(outcome)

    tones = {view.table.item(row, DOT).foreground().color().name()
             for row in range(view.table.rowCount())}
    assert tones == {tone_colors("qualified")[0]}
    assert tone_colors("pass")[0] not in tones
    assert view.count.text() == "", "the run finished and said nothing extra"


def test_a_local_guideline_is_named_in_the_header_because_nothing_else_says_it(
    tmp_path,
):
    """The other half of the sentence this screen used to carry, and the half
    that had to survive it.

    A run loses its conformance claim two ways, and only one of them is a fact
    about the evidence. A local guideline is a decision, recorded, and it
    deliberately does not demote a verdict -- so every row and the tally read a
    plain PASS and no reader would learn what it passed. It belongs beside the
    guideline it is true of, which is the header."""
    from deltachrome_ui.batch_view import BatchView

    view = BatchView()
    view.start(tmp_path, 2, "ISA OLT 0.5X", "House Standard")
    assert view.title.text() == "ISA OLT 0.5X   ·   House Standard"
    assert view.title.toolTip() == ""

    view.start(tmp_path, 2, "ISA OLT 0.5X", "House Standard", local_criteria=True)
    assert "House Standard (local guideline)" in view.title.text()
    assert "published standard" in view.title.toolTip()


def test_the_bar_offers_stopping_while_running_and_leaving_afterwards(tmp_path):
    """One button in two roles, because they are the same place and the same
    question. They are never both available."""
    from deltachrome_ui.batch_view import BatchView

    view = BatchView()
    view.start(tmp_path, 2, "T", "P")
    assert view.primary.text() == "Stop"
    stopped = []
    view.stopRequested.connect(lambda: stopped.append(True))
    view.primary.click()
    assert stopped == [True]

    view.show_result(result_for([measured("a.tif")], tmp_path))
    assert view.primary.text() == "Open in Library"
    assert view.primary.isEnabled()


def test_the_outliers_are_stated_as_a_reading_beside_the_batch_s_own(tmp_path):
    """In the capture's own row, and as two numbers in the metric's own unit.

    It read ``SFR50 4.7σ low``, and σ is the wrong number to print: it is the
    distance from the batch median in median-absolute-deviations, so twenty
    frames of one chart that agree closely make it enormous for a small
    difference -- 27.6 on a real batch, which read as a normal distribution is a
    probability of one in 10^168 and means nothing of the sort. The reading and
    what the rest of the batch read are in the unit the report already uses, and
    the direction falls out of the pair.
    """
    from deltachrome_ui.batch_view import MEDIAN, METRIC, VALUE, BatchView

    steady = [measured(f"c{i}.tif", sfr50=65.0 + (i % 3) * 0.3) for i in range(19)]

    view = BatchView()
    view.start(tmp_path, 20, "T", "P")
    view.show_result(result_for(steady, tmp_path))
    cells = [view.table.item(row, METRIC).text() for row in range(view.table.rowCount())]
    assert cells and not any(cells), "nothing unusual, nothing said"

    view = BatchView()
    view.start(tmp_path, 20, "T", "P")
    view.show_result(result_for(steady + [measured("soft.tif", sfr50=48.0)], tmp_path))
    said = {
        view.table.item(row, 1).text(): tuple(
            view.table.item(row, c).text() for c in (METRIC, VALUE, MEDIAN)
        )
        for row in range(view.table.rowCount())
    }
    assert said["soft.tif"] == ("SFR50", "48.000", "65.300")
    assert all("σ" not in cell for cells in said.values() for cell in cells)
    assert not any(any(cells) for name, cells in said.items() if name != "soft.tif")


# -- the way in ----------------------------------------------------------------


def test_the_library_offers_a_batch_by_name_rather_than_a_glyph():
    """It commits something, which is the test the accent block's own actions
    pass, and a shape cannot say "Batch Analysis"."""
    from deltachrome_ui.library_view import LibraryView

    view = LibraryView()
    assert view.batch_button.text() == "Batch Analysis"
    asked = []
    view.batchRequested.connect(lambda: asked.append(True))
    view.batch_button.click()
    assert asked == [True]


def test_the_placement_button_says_what_pressing_it_starts():
    """A hundred captures behind a button reading "Run Analysis" would be the one
    place in this application where the label understates what it commits."""
    from deltachrome_ui.placement_view import PlacementView

    view = PlacementView()
    assert view.continue_button.text() == "Run Analysis"
    view.set_batch(20)
    assert view.continue_button.text() == "Run Batch of 20"
    view.set_batch(0)
    assert view.continue_button.text() == "Run Analysis"


# -- where the records go ------------------------------------------------------


def test_a_batch_writes_into_its_own_folder_inside_the_run_library(tmp_path):
    from deltachrome_ui.paths import batch_dir, ensure_run_dir

    folder = batch_dir(Path("/captures/Odonata 2026"))
    assert folder.is_dir()
    assert folder.parent == ensure_run_dir()
    assert folder.name.startswith("batch-")
    assert "Odonata-2026" in folder.name, "the capture folder's name is legible in it"


def test_the_library_sees_records_a_batch_wrote_in_a_subfolder(tmp_path):
    """A record the library cannot show is a record nobody can sort, search or
    re-score, which is most of what a record is for."""
    from deltachrome_ui.library import scan_runs

    record = {
        "run_id": "abc", "created_utc": "2026-09-10T00:00:00+00:00",
        "overall": "PASS", "conformance_claim": False,
        "image": {"path": "/x/a.tif"},
        "target": {"target_id": "t", "display_name": "T"},
        "profile": {"profile_id": "p", "display_name": "P"},
        "metrics": [],
    }
    (tmp_path / "loose.json").write_text(json.dumps(record))
    nested = tmp_path / "batch-20260910-Odonata"
    nested.mkdir()
    (nested / "a.json").write_text(json.dumps(record))
    (nested / B.INDEX_NAME).write_text(json.dumps({"captures": []}))

    found = {run.path.name for run in scan_runs(tmp_path)}
    assert found == {"loose.json", "a.json"}, "the index is not a run record"


def test_deleted_records_stay_out_of_the_library_now_that_it_recurses(tmp_path):
    """The trash is a folder inside the library, so a recursive scan walks into it
    unless told not to -- which would make Delete look like it did nothing."""
    from deltachrome_ui.library import TRASH_DIR_NAME, scan_runs

    trash = tmp_path / TRASH_DIR_NAME
    trash.mkdir()
    (trash / "gone.json").write_text(json.dumps({"run_id": "x", "metrics": []}))
    assert scan_runs(tmp_path) == []


def test_a_refusal_takes_the_room_the_finding_columns_would_have_had(tmp_path):
    """It is not a measurement, so it has no metric, value or median -- and a
    separate notes column for it would be a column empty on every measured row,
    sized for its rarest case, starving the three that carry the findings. It
    spans them instead."""
    from deltachrome_ui.batch_view import MEDIAN, METRIC, VALUE, BatchView

    view = BatchView()
    view.start(tmp_path, 2, "T", "P")
    view.advance(1, 2, "a.tif", measured("a.tif"))
    assert [view.table.item(0, c).text() for c in (METRIC, VALUE, MEDIAN)] == \
        ["", "", ""]
    assert view.table.columnSpan(0, METRIC) == 1

    view.advance(2, 2, "b.tif", B.BatchCapture(
        path=Path("b.tif"), outcome="refused", reason="no chart in this frame"
    ))
    assert view.table.item(1, METRIC).text() == "no chart in this frame"
    assert view.table.columnSpan(1, METRIC) == 3


def test_the_capture_list_wears_the_application_s_own_table_style(tmp_path):
    """It never had. The style sheet names ``QTableView#DataTable`` and nothing
    here set the name, so the cells had no padding and sat against the frame,
    the rows did not band, and a selected row came up in Qt's own bright blue
    rather than in the theme's. And a filename elides in the middle, because
    twenty captures of one sitting are identical for their first eighteen
    characters and a cut at the end prints the same text twenty times."""
    from deltachrome_ui.batch_view import CAPTURE, ElideMiddle, BatchView

    view = BatchView()
    assert view.table.objectName() == "DataTable"
    assert view.table.alternatingRowColors()
    assert isinstance(view.table.itemDelegateForColumn(CAPTURE), ElideMiddle)


def test_the_warnings_are_a_count_in_the_table_with_the_text_on_the_pointer(tmp_path):
    """Measured on the twenty Odonata captures: five warnings apiece, four of
    them identical across every capture and one of them 510 characters. A
    column of warning *text* would print the same sentence twenty times, which
    is the redundancy this screen was being cleared of -- so it is a count,
    blank where there is nothing, with the warnings themselves on the pointer.
    The library's own table carries the same column the same way."""
    from deltachrome_ui.batch_view import WARNINGS, BatchView

    clean = measured("a.tif")
    noisy = measured("b.tif")
    noisy.warnings = [
        "No per-unit reference file was supplied, so the chart was measured "
        "against the manufacturer's nominal values for the model.",
        "The placed quadrilateral's aspect ratio differs from the chart's "
        "physical proportions by 302.2%.",
    ]

    view = BatchView()
    view.start(tmp_path, 2, "T", "P")
    view.advance(1, 2, "a.tif", clean)
    view.advance(2, 2, "b.tif", noisy)

    assert view.table.item(0, WARNINGS).text() == "", "blank rather than zero"
    assert view.table.item(1, WARNINGS).text() == "2"
    hint = view.table.item(1, WARNINGS).toolTip()
    assert "nominal values" in hint and "aspect ratio" in hint
    assert max(len(line) for line in hint.splitlines()) < 70, "wrapped, as the library's is"


def test_a_batch_never_identifies_the_chart_it_is_about_to_run(tmp_path, monkeypatch):
    """The rule the engine has always followed, lost at the way in.

    The folder's first capture opened down the ordinary path, the search ran, and
    whatever it named became the target for all hundred. On a frame that is mostly
    *subject* that search returns a confident wrong answer -- a Smithsonian
    specimen sheet carrying an Object Level Target scores as an Image Engineering
    UTT, because nothing better is available to rank. Watched, that is one bad
    suggestion the operator checks; unattended it is a wrong answer per capture.
    """
    from deltachrome_ui.app import MainWindow

    window = MainWindow()
    identified: list[bool] = []
    monkeypatch.setattr(window, "_identify", lambda: identified.append(True))
    monkeypatch.setattr(window, "_open_placement", lambda: None)

    class Proxy:
        image = object()

    # An ordinary run still identifies: the operator sees the placement and checks
    # it before anything is measured, which is what makes a suggestion safe.
    window._pending_batch = None
    window._on_loaded(Proxy(), Path("a.tif"))
    assert identified == [True]

    identified.clear()
    window._pending_batch = (tmp_path, [Path("a.tif"), Path("b.tif")])
    window._on_loaded(Proxy(), Path("a.tif"))
    assert identified == [], "a batch declares its target; it does not guess it"


@pytest.mark.parametrize("mode, path_kw, library_kw", [
    ("file", "ref.txt", None),
    ("library", None, "bank"),
    ("automatic", None, None),
])
def test_a_batch_is_judged_against_the_reference_chosen_for_it(
    tmp_path, monkeypatch, mode, path_kw, library_kw
):
    """The placement screen's reference choice reaches the batch.

    It did not: ``_run_batch`` handed the worker a target, a profile and corners
    and no reference at all, so every batch fell back to manufacturer values --
    and a chart shipping none, the LaserSoft among them, had every frame refused
    with "No reference file was supplied" while the screen said *Pair from
    library*. Found taking the user guide's batch screenshot.
    """
    from deltachrome_ui import app as app_module
    from deltachrome_ui.app import MainWindow

    window = MainWindow()
    choices = window.placement.choices
    chosen = tmp_path / (path_kw or library_kw or "unused")
    monkeypatch.setattr(type(choices), "reference_mode", property(lambda self: mode))
    monkeypatch.setattr(type(choices), "reference_choice", property(lambda self: chosen))
    monkeypatch.setattr(app_module, "batch_dir", lambda folder: tmp_path / "out")
    monkeypatch.setattr(app_module, "run_in_thread", lambda worker, parent: None)
    window._pending_batch = (tmp_path, [tmp_path / "a.tif"])
    window._run_batch([(0, 0), (1, 0), (1, 1), (0, 1)])

    worker = window._batch_worker
    assert worker.reference_path == (chosen if path_kw else None)
    assert worker.reference_library == (chosen if library_kw else None)


def test_the_batch_setup_hands_the_choice_over_rather_than_making_it():
    """A combo box left on whatever happened to be first looks like a decision.
    The chip says it is not one, and is the same chip identification shows when it
    declines -- the choice is being handed over either way."""
    from deltachrome_ui.choices import ChoicesPanel

    choices = ChoicesPanel()
    choices.ask_for_target("because a batch declares its target")
    assert choices.identified is False
    assert choices.target_pill.text() == "Choose the Target"
    assert "declares its target" in choices.target_pill.toolTip()


def test_every_unusual_capture_is_named_in_its_own_row(tmp_path):
    """The cap this used to need is gone with the block it capped.

    Twenty captures of one chart on one rig agree closely enough that a great
    many readings sit several standard deviations from the spread: against a
    4-star profile the same twenty produced nineteen findings, which as a list
    beside the table pushed the run's own captures off the screen. Eight
    findings in eight rows push nothing anywhere, because the rows were already
    there -- so every capture is named and none is elided into "and 3 more".

    Grouping by capture is what still matters: one frame soft on two sharpness
    metrics is one soft frame corroborated twice, not two problems.
    """
    from deltachrome_ui.batch_view import METRIC, BatchView

    steady = [measured(f"c{i}.tif", sfr50=65.0 + (i % 3) * 0.3) for i in range(19)]
    odd = [measured(f"odd{i}.tif", sfr50=48.0 + i) for i in range(8)]
    result = result_for(steady + odd, tmp_path)

    grouped = result.unusual_captures()
    assert len(grouped) == len(odd)
    assert grouped[0][1][0].sigmas >= grouped[-1][1][0].sigmas, "worst first"

    view = BatchView()
    view.start(tmp_path, len(result.captures), "T", "P")
    view.show_result(result)
    said = {
        view.table.item(row, 1).text(): view.table.item(row, METRIC).text()
        for row in range(view.table.rowCount())
    }
    assert all(said[name] for name, _ in grouped), "every one of them, in its row"
    assert not any("more" in text for text in said.values())


def test_the_table_does_not_move_as_the_captures_arrive(tmp_path):
    """The complaint this layout was rebuilt to answer.

    The table used to be re-measured against its own contents after every row
    and re-fixed to that width, so a longer filename widened it and the centred
    layout re-centred under the pointer -- once per capture, for a hundred
    captures. Its width is the body's now, capped at ``TABLE_WIDTH``, and the
    columns that vary are given the slack instead.
    """
    from deltachrome_ui.batch_view import TABLE_WIDTH, BatchView

    view = BatchView()
    view.resize(2000, 900)
    view.show()
    try:
        view.start(tmp_path, 3, "T", "P")
        view.advance(1, 3, "a.tif", measured("a.tif"))
        _app_process_events()
        settled = (view.table.width(), view.table.x(), view.table.y())
        assert 0 < settled[0] <= TABLE_WIDTH

        view.advance(2, 3, "a-very-much-longer-capture-filename.tif",
                     measured("a-very-much-longer-capture-filename.tif"))
        view.advance(3, 3, "c.tif", B.BatchCapture(
            path=Path("c.tif"), outcome="refused",
            reason="no chart could be placed anywhere in this frame at all",
        ))
        _app_process_events()
        assert (view.table.width(), view.table.x(), view.table.y()) == settled

        view.show_result(result_for([measured("a.tif")], tmp_path))
        _app_process_events()
        assert (view.table.width(), view.table.x(), view.table.y()) == settled, \
            "nor when the busy strip goes"
    finally:
        view.close()


def _app_process_events() -> None:
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()


# -- the right of the screen ---------------------------------------------------


def test_the_tally_says_how_the_run_is_coming_out_from_the_first_capture(tmp_path):
    """The one thing the list beside it cannot do.

    A hundred rows read a row at a time; how a hundred rows have come out so far
    is what an operator watching a long batch is actually watching for, and
    reading it off the list means counting dots. Before the first capture there
    is nothing to count, so it says nothing at all.
    """
    from deltachrome_ui.batch_view import BatchView

    view = BatchView()
    view.start(tmp_path, 4, "T", "P")
    assert not view.tally.isVisibleTo(view), "nothing has come out yet"

    for number, name in enumerate(("a.tif", "b.tif", "c.tif"), start=1):
        view.advance(number, 4, name, measured(name))
    view.advance(4, 4, "d.tif", B.BatchCapture(
        path=Path("d.tif"), outcome="refused", reason="no chart in this frame"
    ))

    shown = {
        name.text(): count.text()
        for _dot, name, count in view.tally._rows if name.text()
    }
    assert shown == {"PASS": "3", "not measured": "1"}
    assert view.tally.isVisibleTo(view)


def test_the_tally_counts_worst_first(tmp_path):
    """The order the library's own status column sorts in, for the same reason:
    what needs attention gathers at the top."""
    from deltachrome_ui.batch_view import BatchView

    view = BatchView()
    view.start(tmp_path, 3, "T", "P")
    view.advance(1, 3, "a.tif", measured("a.tif"))
    view.advance(2, 3, "b.tif", measured("b.tif"))
    view.advance(3, 3, "c.tif", measured("c.tif", overall="FAIL"))

    order = [name.text() for _dot, name, _count in view.tally._rows if name.text()]
    assert order == ["FAIL", "PASS"]


def test_the_tally_never_puts_advisory_passes_behind_a_green_number(tmp_path):
    """The rule the whole frontend follows, applied to a count.

    A tally built from the verdict strings would read "PASS 100" in the pass tone
    for a folder every capture of which was judged against the manufacturer's
    values for the model rather than a measurement of the chart. It is built from
    the same ``_status_for`` the rows are drawn with, so the demotion reaches it.
    """
    from deltachrome_ui.batch_view import BatchView
    from deltachrome_ui.theme import tone_colors

    view = BatchView()
    view.start(tmp_path, 2, "T", "P")
    for number, name in enumerate(("a.tif", "b.tif"), start=1):
        view.advance(number, 2, name, B.BatchCapture(
            path=Path(name), outcome="measured", overall="PASS",
            conformance_claim=False, reference_basis="manufacturer_nominal",
        ))

    rows = [(dot, name, count) for dot, name, count in view.tally._rows if name.text()]
    assert len(rows) == 1
    dot, name, count = rows[0]
    assert name.text() == "PASS (ADVISORY)"
    assert count.text() == "2"
    assert tone_colors("pass")[0] not in dot.styleSheet()
    assert tone_colors("qualified")[0] in dot.styleSheet()


def test_the_tally_counts_the_whole_batch_and_not_only_what_was_watched(tmp_path):
    """A resumed run carries captures this screen never saw go past."""
    from deltachrome_ui.batch_view import BatchView

    view = BatchView()
    view.start(tmp_path, 3, "T", "P")
    view.advance(1, 3, "c.tif", measured("c.tif"))
    view.show_result(result_for(
        [measured("a.tif"), measured("b.tif"), measured("c.tif")], tmp_path
    ))

    shown = {
        name.text(): count.text()
        for _dot, name, count in view.tally._rows if name.text()
    }
    assert shown == {"PASS": "3"}


def test_the_findings_name_the_captures_and_do_not_explain_themselves(tmp_path):
    """Standing explanation is not on the screen anywhere else in this
    application, and it was on this one: a thirty-word sentence saying what an
    outlier is, above five lines saying which captures are outliers. The
    sentence never changes, so it is on the column heading; the findings do, so
    they are in the rows."""
    from deltachrome_ui.batch_view import METRIC, BatchView

    view = BatchView()
    view.start(tmp_path, 20, "T", "P")
    steady = [measured(f"c{i}.tif", sfr50=65.0 + (i % 3) * 0.3) for i in range(19)]
    view.show_result(result_for(steady + [measured("soft.tif", sfr50=48.0)], tmp_path))

    said = " ".join(
        view.table.item(row, METRIC).text() for row in range(view.table.rowCount())
    )
    assert "SFR50" in said
    assert "far from its own batch" not in said
    assert "stands out" in view.table.horizontalHeaderItem(METRIC).toolTip()
    for row in range(view.table.rowCount()):
        assert view.table.item(row, METRIC).toolTip() == "", "no sentence on a cell"


def test_the_finished_batch_opens_in_the_library_rather_than_shut(tmp_path, monkeypatch):
    """The library gathers a batch's records under one row that opens, and shut
    is the right state for somebody who came looking for something else.
    Somebody who has just pressed "Open in Library" on the run that wrote them is
    not that person: the answer they asked for would be behind a chevron."""
    monkeypatch.setenv("DELTACHROME_RUN_DIR", str(tmp_path / "library"))
    from deltachrome_ui.app import MainWindow

    folder = tmp_path / "library" / "batch-20260912-143200-Odonata"
    folder.mkdir(parents=True)
    (folder / B.INDEX_NAME).write_text(json.dumps({"captures": []}))
    record = {
        "run_id": "abc", "created_utc": "2026-09-12T10:00:00+00:00",
        "overall": "PASS", "conformance_claim": False,
        "image": {"filename": "c0.tif"},
        "target": {"target_id": "t", "display_name": "T"},
        "profile": {"profile_id": "p", "display_name": "P"},
        "metrics": [],
    }
    (folder / "c0.json").write_text(json.dumps(record))

    window = MainWindow()
    try:
        window._batch_finished(result_for([measured("c0.tif")], folder))
        assert folder in window.library.table.expanded_batches
        index = window.library.proxy.index(0, 0)
        assert window.library.table.isExpanded(index)
    finally:
        window.close()


def test_a_batch_offers_no_per_capture_chart_search():
    """Deferred past the beta. A batch says once, from its first
    capture, when frames hold charts it will not measure; asking every capture
    was a checkbox on this screen and is gone with its option in the engine."""
    import inspect

    import deltachrome.batch as engine
    from deltachrome_ui.placement_view import PlacementView
    from deltachrome_ui.workers import BatchRunner

    view = PlacementView()
    view.set_batch(40)
    assert not hasattr(view, "extra_charts_check")
    assert "extra_charts" not in inspect.signature(engine.run_batch).parameters
    assert "extra_charts" not in inspect.signature(BatchRunner).parameters
