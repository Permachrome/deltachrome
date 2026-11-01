"""The library table: what it shows, how it sorts, and what it can lose.

Three things here are worth a test rather than a look.

**Sorting must not run on the string a person reads.** A date column displaying
``2026-08-22 14:32`` happens to sort correctly as text, which is precisely why
it needs a test: the moment somebody prefers ``22 Aug 2026`` the sort silently
becomes alphabetical and nothing about the screen says so.

**The status dot has five states.** Two would be one, because an advisory pass
would have nowhere to go but green -- and that is the single rule
:mod:`deltachrome_ui.status` exists to state.

**Deleting must not delete.** A run record holds every per-patch value its run
measured, and there is no way to regenerate one except by finding the master and
repeating the placement by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deltachrome_ui.library import read_run, scan_runs, trash_run
from deltachrome_ui.status import TONE_FAIL, TONE_PASS, TONE_QUALIFIED


def write(path: Path, record: dict) -> Path:
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def minimal(**overrides) -> dict:
    record = {
        "run_id": "abc123",
        "created_utc": "2026-08-20T20:15:21+00:00",
        "image": {"filename": "capture.tif", "path": "/masters/capture.tif"},
        "target": {"display_name": "ColorChecker Digital SG"},
        "profile": {"display_name": "FADGI 3-Star"},
        "reference": {"basis": "per_unit_measurement"},
        "overall": "PASS",
        "conformance_claim": True,
        "warnings": [],
    }
    record.update(overrides)
    return record


# -- the basis column --------------------------------------------------------


def test_measured_against_the_standard_is_the_only_conformance_basis(tmp_path):
    assert read_run(write(tmp_path / "r.json", minimal())).basis == "Measured"


def test_nominal_reference_data_is_named_as_such(tmp_path):
    run = read_run(write(tmp_path / "r.json", minimal(
        reference={"basis": "manufacturer_nominal"}, conformance_claim=False,
    )))
    assert run.basis == "Nominal"


def test_a_derived_profile_is_a_separate_departure_from_the_reference_data(tmp_path):
    """Both at once is a real combination and must read as both.

    ``conformance_claim`` is one bit covering two independent things, which is
    why the column is not driven from it: a run judged on nominal values against
    a standard profile and one judged on measured values against moved
    thresholds are equally "not conformance" and are not the same problem.
    """
    run = read_run(write(tmp_path / "r.json", minimal(
        reference={"basis": "manufacturer_nominal"},
        profile={"display_name": "Local", "derived_from": "fadgi_3star"},
        conformance_claim=False,
    )))
    # "Local", not "Derived": what the column reports is what the criteria are,
    # and a saved guideline's parent is provenance the record keeps rather than
    # something a column should name.
    assert run.basis == "Nominal + Local"


def test_a_record_predating_the_basis_field_does_not_claim_to_be_measured(tmp_path):
    """The one wrong answer here is the flattering one."""
    run = read_run(write(tmp_path / "r.json", {
        "overall": "PASS", "conformance_claim": False,
    }))
    assert run.basis == "Unstated"


def test_an_unrecognised_basis_is_passed_through_rather_than_guessed_at(tmp_path):
    run = read_run(write(tmp_path / "r.json", minimal(
        reference={"basis": "spectrophotometer_live"})))
    assert run.basis == "spectrophotometer_live"


# -- warnings ----------------------------------------------------------------


def test_the_warnings_themselves_are_carried_not_only_the_count(tmp_path):
    """A count on a screen that cannot show what was counted is a dead end."""
    run = read_run(write(tmp_path / "r.json", minimal(
        warnings=["patch K8 clipped", "reference is nominal"])))
    assert run.warning_count == 2
    assert run.warnings == ("patch K8 clipped", "reference is nominal")


# -- the date ----------------------------------------------------------------


def test_the_date_reads_as_iso_8601_to_the_minute(tmp_path):
    """Two runs of the same chart on the same day is the ordinary case."""
    run = read_run(write(tmp_path / "r.json", minimal(
        created_utc="2026-08-22T14:32:00+00:00")))
    shown = run.created_display
    assert shown.startswith("2026-08-22") or shown.startswith("2026-08-2")
    assert len(shown) == len("2026-08-22 14:32")
    assert shown[4] == "-" and shown[7] == "-"


def test_the_date_column_sorts_on_the_instant_not_on_what_is_displayed(tmp_path):
    """The trap this guards is that today's format happens to sort correctly.

    Offsets are the case that gives it away: two stamps naming the same minute
    in different zones must not order by the text of the hour.
    """
    from deltachrome_ui.run_table import COLUMN_INDEX, COLUMNS

    column = COLUMNS[COLUMN_INDEX["date"]]
    earlier = read_run(write(tmp_path / "a.json", minimal(
        created_utc="2026-08-22T09:00:00+00:00")))
    later = read_run(write(tmp_path / "b.json", minimal(
        created_utc="2026-08-22T08:00:00-05:00")))  # 13:00 UTC
    assert column.sort(earlier) < column.sort(later)


def test_a_record_with_no_usable_timestamp_still_lists(tmp_path):
    from deltachrome_ui.run_table import COLUMN_INDEX, COLUMNS

    run = read_run(write(tmp_path / "r.json", minimal(created_utc="whenever")))
    assert run.created_display == "whenever"
    assert COLUMNS[COLUMN_INDEX["date"]].sort(run) == float("-inf")


# -- the status dot ----------------------------------------------------------


def test_the_dot_has_a_colour_for_a_qualified_pass_that_is_not_the_pass_colour(tmp_path):
    """The whole reason the dot is not red/green.

    On every ISA target an advisory pass is the ordinary result, so a
    two-colour dot would put the common case in the colour reserved for a claim
    the run did not earn.
    """
    from deltachrome_ui.theme import tone_colors

    passed = read_run(write(tmp_path / "p.json", minimal()))
    # Advisory is about the EVIDENCE: the numbers describe the model rather than
    # this chart. A conformance claim withheld for any other reason is a different
    # thing -- see the test below.
    advisory = read_run(write(tmp_path / "i.json", minimal(
        conformance_claim=False,
        reference={"basis": "manufacturer_nominal"})))
    failed = read_run(write(tmp_path / "f.json", minimal(overall="FAIL")))

    assert passed.status.tone == TONE_PASS
    assert advisory.status.tone == TONE_QUALIFIED
    assert failed.status.tone == TONE_FAIL
    colors = {tone_colors(r.status.tone)[0] for r in (passed, advisory, failed)}
    assert len(colors) == 3


def test_a_run_against_a_local_guideline_is_not_demoted_for_being_one(tmp_path):
    """The correction the guideline builder needed.

    A department that writes its own standard down could never see an unqualified
    verdict, however well the capture was measured -- which taught the
    qualification to mean nothing, since a tone that is always on is a tone nobody
    reads. What the criteria are is a decision, recorded and visible in the
    guideline's own name; what the reference is is a fact about the evidence. Only
    the second demotes.
    """
    local = read_run(write(tmp_path / "d.json", minimal(
        conformance_claim=False,
        profile={"display_name": "Special Collections v2",
                 "derived_from": "FADGI 3-star"},
    )))
    assert local.status.tone == TONE_PASS
    assert "(ADVISORY)" not in local.status.label
    assert "local guideline" in local.status.hint
    assert "FADGI 3-star" not in local.status.hint

    # And a local guideline measured against nominal data is still advisory,
    # because the reference is what was wrong with it.
    both = read_run(write(tmp_path / "b.json", minimal(
        conformance_claim=False,
        reference={"basis": "manufacturer_nominal"},
        profile={"display_name": "Special Collections v2",
                 "derived_from": "FADGI 3-star"},
    )))
    assert both.status.tone == TONE_QUALIFIED


def test_sorting_the_status_column_ascending_gathers_the_failures_first(tmp_path):
    """Worst at the top, because that is the only reason to sort by verdict.

    And the qualified pass has to land between them: it is a real reading, so
    it does not belong with the failures, and it is not a claim, so it must not
    sit among the passes.
    """
    from deltachrome_ui.run_table import COLUMN_INDEX, COLUMNS

    column = COLUMNS[COLUMN_INDEX["status"]]
    cases = [
        ("clean pass", minimal()),
        ("failure", minimal(overall="FAIL")),
        ("advisory pass", minimal(conformance_claim=False,
                                  reference={"basis": "manufacturer_nominal"})),
        ("incomplete", minimal(overall="INCOMPLETE")),
    ]
    runs = [
        (name, read_run(write(tmp_path / f"{i}.json", record)))
        for i, (name, record) in enumerate(cases)
    ]
    order = [name for name, _ in sorted(runs, key=lambda pair: column.sort(pair[1]))]
    assert order == ["failure", "incomplete", "advisory pass", "clean pass"]


def test_and_the_table_itself_sorts_that_way_and_not_only_the_key(qt_app, tmp_path):
    """The test above asserts the declaration and the declaration was right.
    What was broken is what the table did with it.

    The verdict's key is a tuple -- ``(rank, label)``, so that a verdict this
    version does not recognise still lands somewhere deliberate -- and Qt's own
    comparison cannot order a Python tuple carried through a QVariant. Every
    comparison came back false, the rows stayed in the order they were read, and
    clicking the header drew the chevron and moved nothing. That is the shape of
    failure this table is most able to hide: a list that did not move looks like
    a list that was already sorted.
    """
    from PySide6.QtCore import Qt

    from deltachrome_ui.library import scan_runs
    from deltachrome_ui.run_table import (
        COLUMN_INDEX, RunFilterProxy, RunTableModel,
    )

    write(tmp_path / "1.json", minimal(
        image={"filename": "clean.tif"}, created_utc="2026-08-20T09:00:00+00:00"))
    write(tmp_path / "2.json", minimal(
        image={"filename": "failed.tif"}, overall="FAIL",
        created_utc="2026-08-21T09:00:00+00:00"))
    write(tmp_path / "3.json", minimal(
        image={"filename": "advisory.tif"}, conformance_claim=False,
        reference={"basis": "manufacturer_nominal"},
        created_utc="2026-08-22T09:00:00+00:00"))

    model = RunTableModel()
    model.set_runs(scan_runs(tmp_path))
    proxy = RunFilterProxy()
    proxy.setSourceModel(model)

    def order() -> list[str]:
        return [
            proxy.index(row, COLUMN_INDEX["file"]).data()
            for row in range(proxy.rowCount())
        ]

    proxy.sort(COLUMN_INDEX["status"], Qt.SortOrder.AscendingOrder)
    assert order() == ["failed.tif", "advisory.tif", "clean.tif"]
    proxy.sort(COLUMN_INDEX["status"], Qt.SortOrder.DescendingOrder)
    assert order() == ["clean.tif", "advisory.tif", "failed.tif"]


# -- deleting ----------------------------------------------------------------


def test_deleting_moves_the_record_rather_than_destroying_it(tmp_path):
    path = write(tmp_path / "r.json", minimal())
    landed = trash_run(path)
    assert not path.exists()
    assert landed.exists()
    assert json.loads(landed.read_text())["run_id"] == "abc123"
    # And it leaves the library, which is the point of the gesture.
    assert scan_runs(tmp_path) == []


def test_the_trash_does_not_overwrite_its_own_contents(tmp_path):
    """Same capture, same profile, run twice and deleted twice."""
    first = trash_run(write(tmp_path / "r.json", minimal(run_id="one")))
    second = trash_run(write(tmp_path / "r.json", minimal(run_id="two")))
    assert first != second
    assert json.loads(first.read_text())["run_id"] == "one"
    assert json.loads(second.read_text())["run_id"] == "two"


def test_the_trash_is_not_itself_scanned_as_part_of_the_library(tmp_path):
    trash_run(write(tmp_path / "gone.json", minimal()))
    write(tmp_path / "kept.json", minimal())
    assert [r.path.stem for r in scan_runs(tmp_path)] == ["kept"]


# -- the table itself --------------------------------------------------------

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from PySide6.QtWidgets import QApplication  # noqa: E402

from deltachrome_ui.run_table import (  # noqa: E402
    ACTION_COLUMN,
    COLUMN_INDEX,
    COLUMNS,
    RunFilterProxy,
    RunTableModel,
    RunTableView,
    _icon_rects,
)


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def table(qt_app, tmp_path):
    write(tmp_path / "a.json", minimal(
        run_id="a", created_utc="2026-08-22T09:00:00+00:00",
        image={"filename": "golden_thread_0012.jp2"},
        target={"display_name": "ISA Golden Thread Device Level Target"},
        profile={"display_name": "FADGI 4-Star — Prints and Photographs"},
        reference={"basis": "manufacturer_nominal"},
        overall="FAIL", conformance_claim=False, warnings=["patch K8 clipped"],
    ))
    write(tmp_path / "b.json", minimal(
        run_id="b", created_utc="2026-08-21T09:00:00+00:00",
        image={"filename": "sg_morning.tif"},
        target={"display_name": "ColorChecker Digital SG"},
        profile={"display_name": "Metamorfoze Full"},
    ))
    (tmp_path / "broken.json").write_text('{"run_id": "c", "over', encoding="utf-8")

    model = RunTableModel()
    model.set_runs(scan_runs(tmp_path))
    proxy = RunFilterProxy()
    proxy.setSourceModel(model)
    view = RunTableView()
    view.setModel(proxy)
    return view


def shown(view) -> list[str]:
    column = COLUMN_INDEX["file"]
    return [
        view.model().index(row, column).data()
        for row in range(view.model().rowCount())
    ]


def test_every_column_is_searchable_at_once(table):
    proxy = table.model()
    for query, expected in (
        ("golden", ["golden_thread_0012.jp2"]),
        ("metamorfoze", ["sg_morning.tif"]),
        ("sg", ["sg_morning.tif"]),
        ("nominal", ["golden_thread_0012.jp2"]),
    ):
        proxy.set_query(query)
        assert shown(table) == expected, query


def test_a_search_is_words_rather_than_a_substring(table):
    """The query a person actually has spans two fields that are never adjacent."""
    proxy = table.model()
    proxy.set_query("sg metamorfoze")
    assert shown(table) == ["sg_morning.tif"]
    proxy.set_query("sg fadgi")
    assert shown(table) == []


def test_the_verdict_is_searchable_even_though_it_is_painted(table):
    """A dot has no letters, so without this the status column is unqueryable."""
    proxy = table.model()
    proxy.set_query("fail")
    assert shown(table) == ["golden_thread_0012.jp2"]


def test_a_warning_is_searchable_by_its_text_not_only_its_count(table):
    proxy = table.model()
    proxy.set_query("clipped")
    assert shown(table) == ["golden_thread_0012.jp2"]


def test_an_unreadable_record_is_listed_rather_than_dropped(table):
    """Deliberately: a record that vanished because its JSON was truncated is
    worse than a visible bad row."""
    assert "broken.json" in shown(table)


def test_an_unreadable_record_offers_only_the_action_that_can_work(table):
    """There is no report to open and no command to reproduce -- but it can be
    got out of the way, which is the whole reason it is listed at all."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    table.resize(1200, 300)
    row = shown(table).index("broken.json")
    index = table.model().index(row, ACTION_COLUMN)
    fired: list[str] = []
    table.actionTriggered.connect(lambda key, _run: fired.append(key))

    for rect in _icon_rects(table.visualRect(index)):
        QTest.mouseClick(
            table.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, rect.center(),
        )
    assert fired == ["delete"]


def test_every_icon_acts_on_a_readable_run(table):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from deltachrome_ui.run_table import ACTIONS

    table.resize(1200, 300)
    row = shown(table).index("golden_thread_0012.jp2")
    index = table.model().index(row, ACTION_COLUMN)
    fired: list[str] = []
    table.actionTriggered.connect(lambda key, _run: fired.append(key))

    for rect in _icon_rects(table.visualRect(index)):
        QTest.mouseClick(
            table.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier, rect.center(),
        )
    assert fired == [key for key, _label, _hover in ACTIONS]


def test_double_clicking_a_row_opens_it_rather_than_re_running_it(table):
    """Re-running costs a full decode of the master to reach a screen the
    operator was probably not asking for."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    table.resize(1200, 300)
    row = shown(table).index("sg_morning.tif")
    fired: list[str] = []
    table.actionTriggered.connect(lambda key, _run: fired.append(key))
    QTest.mouseDClick(
        table.viewport(), Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        table.visualRect(table.model().index(row, COLUMN_INDEX["file"])).center(),
    )
    assert fired == ["open"]


def test_the_columns_are_laid_out_once_the_model_is_attached(table):
    """A header has no sections until it has a model, so widths set before that
    are applied to columns that do not exist and are silently discarded --
    which showed as the capture name and the warning count sharing a width."""
    from deltachrome_ui.run_table import COLUMNS

    header = table.header()
    assert header.sectionSize(COLUMN_INDEX["date"]) == COLUMNS[
        COLUMN_INDEX["date"]].width
    assert header.sectionSize(COLUMN_INDEX["file"]) != header.sectionSize(
        COLUMN_INDEX["warnings"])


# -- the screen around it ----------------------------------------------------


@pytest.fixture
def library(qt_app, tmp_path, monkeypatch):
    from deltachrome_ui.library_view import LibraryView

    monkeypatch.setenv("DELTACHROME_RUN_DIR", str(tmp_path))
    write(tmp_path / "a.json", minimal(
        run_id="a", image={"filename": "golden_thread_0012.jp2"},
        target={"display_name": "ISA Golden Thread Device Level Target"},
    ))
    write(tmp_path / "b.json", minimal(
        run_id="b", image={"filename": "sg_morning.tif"}))
    return LibraryView()


def accept(monkeypatch, answer) -> None:
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "exec", lambda self: answer)


def test_deleting_through_the_screen_reaches_the_trash(library, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    accept(monkeypatch, QMessageBox.StandardButton.Yes)
    library._delete([library.runs[0]])
    assert len(library.runs) == 1
    assert len(list((Path(library.path_label.text()) / ".trash").glob("*.json"))) == 1


def test_declining_the_confirmation_leaves_the_record_where_it_was(library, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    accept(monkeypatch, QMessageBox.StandardButton.Cancel)
    library._delete([library.runs[0]])
    assert len(library.runs) == 2
    assert not (Path(library.path_label.text()) / ".trash").exists()


def test_the_confirmation_says_the_capture_is_not_touched(library, monkeypatch):
    """The row is named after the image, so "Delete 0034.jp2?" reads as an offer
    to delete the master -- and on a shared drive somebody will act on it."""
    from PySide6.QtWidgets import QMessageBox

    seen: list[str] = []

    def capture(self):
        seen.append(self.text() + "\n" + self.informativeText())
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "exec", capture)
    library._delete([library.runs[0]])
    assert "capture itself is not touched" in seen[0]
    assert ".trash" in seen[0]


def test_an_empty_library_and_an_empty_search_are_not_the_same_message(library):
    """An empty library after a search reads as a library that has been lost."""
    library.search.setText("nothing matches this")
    assert library.body.currentIndex() == 2
    assert "matches" in library.no_match.text()
    assert library.count_label.text() == "0 of 2 runs"

    library.search.setText("")
    assert library.body.currentIndex() == 0
    assert library.count_label.text() == "2 runs"


def test_the_menu_offers_the_single_run_actions_only_for_a_single_run(library):
    """There is no report of forty runs to open, and no one command line that
    reproduces them -- but forty can certainly be cleared out at once."""
    one = library.build_menu(library.runs[:1])
    assert [(a.text(), a.isEnabled()) for a in one.actions() if a.text()] == [
        ("Open", True), ("Re-run", True), ("Copy Command", True),
        ("Export Values…", True), ("Move to Trash", True),
    ]
    both = library.build_menu(library.runs)
    assert [(a.text(), a.isEnabled()) for a in both.actions() if a.text()] == [
        ("Open", False), ("Re-run", False), ("Copy Command", False),
        ("Export Values…", False), ("Move 2 Runs to Trash", True),
    ]


def test_the_library_does_not_offer_to_combine_captures(library):
    """Withdrawn, rather than left in with its record defect unfixed.

    Combining several *captures* is a real question and this was the wrong
    place to ask it: a session assembled by selecting rows in a table is a
    question asked of work already filed, where the pairing an operator
    actually wants is two frames shot in one sitting. It returns beside the
    run. What combining a *board* -- several targets in one frame -- has always
    done is untouched, and is reached from the capture rather than from here.
    """
    for runs in (library.runs[:1], library.runs):
        texts = [a.text() for a in library.build_menu(runs).actions()]
        assert not any("Session" in text or "Combine" in text for text in texts)


def test_a_layout_saved_against_different_columns_is_ignored(library, monkeypatch):
    """Qt's own check is on the section count, which cannot notice that a
    column was replaced by another -- restoring then puts one column's
    remembered width onto a different column."""
    import deltachrome_ui.library_view as view

    saved: dict = {}
    monkeypatch.setattr(view, "remember", lambda key, value: saved.update({key: value}))
    library._save_header()
    stored = saved[view._HEADER_KEY]
    assert stored["columns"] == ",".join(c.key for c in COLUMNS)

    monkeypatch.setattr(view, "remembered", lambda key, default=None: {
        "columns": "status,file,something_else", "state": stored["state"],
    })
    widths = [library.table.header().sectionSize(i)
              for i in range(len(COLUMNS))]
    library._restore_header()
    assert [library.table.header().sectionSize(i)
            for i in range(len(COLUMNS))] == widths
