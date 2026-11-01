"""The session screen: several saved runs read together.

Almost no chart measures everything a standard asks for, so covering one takes two
or three targets in a sitting -- and until this screen the only way to ask the
combined question was a hand-written JSON file and a terminal.

Three things here are worth a test rather than a look.

**It must be assembled from records, not re-measured.** That is the whole design,
and the thing that would quietly stop being true: a later change that reached for
the pixels would still produce a correct report, just one that took a minute
instead of no time and needed masters that may be on a drive nobody has mounted.

**Runs judged against different profiles must be refused.** Pooling readings scored
against different thresholds gives a verdict belonging to no standard, and a
library holding a year of runs against a dozen profiles makes it the easy mistake.

**The two kinds of gap must stay apart.** A metric no chart carries is closed by
shooting another chart; a metric a chart carried and the measurement failed on is
closed by fixing the capture. One heading for both sends an operator shopping when
the placement was wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="frontend tests need the gui extra")

from PySide6.QtWidgets import QApplication  # noqa: E402

from deltachrome.errors import ProfileError  # noqa: E402
from deltachrome.session import session_from_records  # noqa: E402
from deltachrome_ui.session_view import (  # noqa: E402
    SessionView,
    _value_text,
)
from deltachrome_ui.tables import ROW_ROLE  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def record(**overrides) -> dict:
    """A run record with just the keys a session reads.

    Hand-built rather than analyzed, so what the screen needs is stated rather than
    inherited: a session consumes metric entries, a conformance claim and a handful
    of labels, and nothing else in the record.
    """
    base = {
        "run_id": "aaaaaa11",
        "created_utc": "2026-08-28T09:00:00+00:00",
        "image": {"filename": "sg.tif"},
        "target": {"target_id": "colorchecker_sg_v1",
                   "display_name": "ColorChecker Digital SG"},
        "profile": {"profile_id": "fadgi_1star_prints_photographs_2023",
                    "display_name": "FADGI 1-Star"},
        "reference": {"basis": "per_unit_measurement"},
        "overall": "PASS",
        "conformance_claim": True,
        "warnings": [],
        "metrics": [
            {"metric_id": "average_color_accuracy", "outcome": "evaluated",
             "status": "PASS", "value": 1.5, "units": "Delta E2000", "passed": True,
             "condition": "value < 6.5", "per_patch": [], "vectors": []},
            {"metric_id": "sfr50", "outcome": "not_applicable", "status": "N/A",
             "value": None, "passed": None, "per_patch": [], "vectors": []},
            {"metric_id": "bit_depth", "outcome": "evaluated", "status": "PASS",
             "value": 16.0, "units": "bits", "passed": True,
             "condition": "value >= 8", "per_patch": [], "vectors": []},
        ],
    }
    base.update(overrides)
    return base


def edge_record(**overrides) -> dict:
    defaults = {
        "run_id": "bbbbbb22",
        "image": {"filename": "edges.tif"},
        "target": {"target_id": "edge_probe", "display_name": "Edge probe"},
        "metrics": [
            {"metric_id": "average_color_accuracy", "outcome": "not_applicable",
             "status": "N/A", "value": None, "passed": None,
             "per_patch": [], "vectors": []},
            {"metric_id": "sfr50", "outcome": "evaluated", "status": "PASS",
             "value": 55.0, "units": "% of half-sampling", "passed": True,
             "condition": "40 <= value <= 75", "per_patch": [], "vectors": []},
            {"metric_id": "bit_depth", "outcome": "evaluated", "status": "PASS",
             "value": 8.0, "units": "bits", "passed": True,
             "condition": "value >= 8", "per_patch": [], "vectors": []},
        ],
    }
    defaults.update(overrides)
    return record(**defaults)


@pytest.fixture
def session():
    return session_from_records([record(), edge_record()])


# -- what a session is assembled from ---------------------------------------


def test_two_partial_captures_cover_what_neither_does_alone(session):
    by_id = {m.metric_id: m for m in session.metrics}
    assert by_id["average_color_accuracy"].summary()["covered_by"] == ["sg.tif"]
    assert by_id["sfr50"].summary()["covered_by"] == ["edges.tif"]
    assert "average_color_accuracy" not in session.uncovered
    assert "sfr50" not in session.uncovered


def test_nothing_but_the_records_is_needed(session):
    """No image path is even present in these records, and the session is complete.

    The guard on the design: a later change that reached for the pixels would still
    produce a correct report, just one needing masters that may not be mounted.
    """
    assert session.overall in {"PASS", "FAIL", "MEASURED", "INCOMPLETE"}
    assert len(session.captures) == 2
    assert all("path" not in (c or {}) for c in [record()["image"]])


def test_the_screen_renders_a_session(qt_app, session):
    view = SessionView()
    view.set_session(session)
    assert view.session is session
    assert view._scroll.widget() is not None


def test_the_screen_can_be_shown_a_second_session(qt_app, session):
    """A view rebuilt in place is where stale widgets accumulate, and the header is
    the one that would show the previous session's verdict."""
    view = SessionView()
    view.set_session(session)
    view.set_session(session_from_records([record()]))
    assert view._header_slot.count() == 1
    assert view.session.captures[0].name == "sg.tif"


# -- the metric table --------------------------------------------------------


def test_the_table_has_a_row_for_every_metric_the_profile_declares(qt_app, session):
    from deltachrome_ui.tables import DataTable

    view = SessionView()
    view.set_session(session)
    table = view._scroll.widget().findChild(DataTable)
    model = table.model()
    ids = {model.index(r, 0).data(ROW_ROLE)["metric_id"]
           for r in range(model.rowCount())}
    assert {"average_color_accuracy", "sfr50", "bit_depth"} <= ids
    assert len(ids) == len(session.profile.metrics)


def test_agreeing_targets_show_one_value_and_disagreeing_ones_a_range():
    """Listing every reading side by side truncated at two targets. A range says
    the useful half -- whether they agree at all -- in the space there is, and it
    is now the whole of what this screen says about disagreement: the section
    that spelled each one out target by target said the same thing at length."""
    bare = {"texts": []}
    assert _value_text({**bare, "values": {}}) == "—"
    assert _value_text({**bare, "values": {"a": 16.0}}) == "16.000"
    assert _value_text({**bare, "values": {"a": 16.0, "b": 16.0}}) == "16.000"
    assert _value_text({**bare, "values": {"a": 8.0, "b": 16.0}}) == "8.000 – 16.000"


def test_a_metric_whose_reading_is_a_word_shows_the_word():
    """Three of them are not numbers -- the master's container, its colour space,
    its bit depth -- and this table had nothing to format for them, so it printed
    a dash beside a metric that had measured something perfectly well."""
    assert _value_text({"texts": ["JPEG 2000"], "values": {}}) == "JPEG 2000"
    assert _value_text(
        {"texts": ["JPEG 2000", "JPEG 2000"], "values": {}}) == "JPEG 2000"
    assert _value_text({"texts": ["8"], "values": {"a": 8.0}}) == "8", \
        "the word wins: 8.000 claims a precision a count does not have"
    assert _value_text(
        {"texts": ["TIFF", "JPEG 2000"], "values": {}}) == "TIFF, JPEG 2000"


def test_the_table_carries_no_spread_column(qt_app, session):
    """Held back for the first release to see whether the screen is wanted
    without it. What it reported is in the Values column's own range."""
    from deltachrome_ui.tables import DataTable

    view = SessionView()
    view.set_session(session)
    table = view._scroll.widget().findChild(DataTable)
    titles = [c.title for c in table.source.columns]
    assert "Spread" not in titles
    assert "Values" in titles
    assert "Reading(s)" not in titles


# -- refusals ----------------------------------------------------------------


def test_runs_judged_against_different_profiles_cannot_be_combined():
    other = edge_record(profile={"profile_id": "fadgi_4star_prints_photographs_2023",
                                 "display_name": "FADGI 4-Star"})
    with pytest.raises(ProfileError, match="different profiles"):
        session_from_records([record(), other])


def test_the_refusal_says_what_to_do_about_it():
    other = edge_record(profile={"profile_id": "fadgi_4star_prints_photographs_2023"})
    with pytest.raises(ProfileError) as caught:
        session_from_records([record(), other])
    message = str(caught.value)
    assert "one profile" in message
    assert "select only the runs" in message


# -- the two kinds of gap ----------------------------------------------------


def test_a_metric_no_chart_carries_is_not_reported_as_an_incomplete_measurement():
    session = session_from_records([record(), edge_record()])
    assert "color_channel_misregistration" in session.uncovered
    assert "color_channel_misregistration" in session.not_carried
    assert "color_channel_misregistration" not in session.incomplete


def test_a_metric_a_chart_carried_and_failed_on_is_kept_out_of_the_shopping_list():
    incomplete = edge_record(metrics=[
        {"metric_id": "sfr50", "outcome": "incomplete", "status": "INCOMPLETE",
         "value": None, "passed": None, "reason": "no edge yielded a response",
         "per_patch": [], "vectors": []},
    ])
    session = session_from_records([record(), incomplete])
    assert "sfr50" in session.incomplete
    assert "sfr50" in session.uncovered
    assert "sfr50" not in session.not_carried
    # And the explanation must not send the operator shopping for it. The clause
    # that offers "adding a target" names only the metrics no chart here carries.
    shopping = session.explanation.split("measured by no capture")[-1]
    assert "adding a target" in shopping
    assert "sfr50" not in shopping


def test_the_two_kinds_of_gap_are_told_apart_by_the_table(qt_app):
    """They had a prose section each, listing the metric ids under a paragraph.
    Both were the table read aloud: it has a row per metric with a verdict on
    each, and the two gaps are two different verdicts -- ``N/A`` for a metric no
    target here carries, ``INCOMPLETE`` for one a target carried and the
    measurement did not finish on. The distinction is the thing worth keeping
    and the table is where it now lives.

    Nothing was lost from the record: ``not_carried`` and ``incomplete`` are
    still separate on the session, still separate in the explanation, and the
    CLI still prints both.
    """
    from PySide6.QtWidgets import QLabel
    from deltachrome_ui.tables import DataTable

    incomplete = edge_record(metrics=[
        {"metric_id": "sfr50", "outcome": "incomplete", "status": "INCOMPLETE",
         "value": None, "passed": None, "per_patch": [], "vectors": []},
    ])
    session = session_from_records([record(), incomplete])
    view = SessionView()
    view.set_session(session)

    headings = {w.text() for w in view._scroll.widget().findChildren(QLabel)
                if w.objectName() == "SectionHeading"}
    assert "Reached by No Capture" not in headings
    assert "Attempted and Incomplete" not in headings
    assert "Where the Captures Disagree" not in headings

    table = view._scroll.widget().findChild(DataTable)
    model = table.model()
    verdict = next(i for i, c in enumerate(table.source.columns)
                   if c.title == "Verdict")
    by_id = {}
    for r in range(model.rowCount()):
        row = model.index(r, 0).data(ROW_ROLE)
        by_id[row["metric_id"]] = model.index(r, verdict).data()
    assert by_id["sfr50"] == "INCOMPLETE", "carried, and did not finish"
    assert by_id["color_channel_misregistration"] == "N/A", "no target carries it"
    # And the engine still keeps them apart, which is what the CLI prints.
    assert "sfr50" in session.incomplete
    assert "color_channel_misregistration" in session.not_carried


# -- saving ------------------------------------------------------------------


def test_a_saved_session_round_trips_as_json(session, tmp_path):
    """The frontend no longer writes one of these -- combining several captures
    from the library is withdrawn -- but ``deltachrome session
    --save-session`` still does, to a path the caller names. So what is pinned
    here is the shape of the dict, not where it lands."""
    path = Path(tmp_path) / "combined.json"
    path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")
    data = json.loads(path.read_text())
    assert data["overall"] == session.overall
    assert [c["name"] for c in data["captures"]] == ["sg.tif", "edges.tif"]
    assert data["coverage_line"] == session.coverage_line


def test_the_header_text_is_not_boxed_off_from_its_own_bar(qt_app, session):
    """Reported from the running application: a lighter rectangle around the
    heading and the filename, a different colour from the bar they sit in.

    The bar paints itself through ``#HeaderBar``, and a plain ``QWidget`` placed
    inside it to hold the two labels paints the *base* background over that. The
    report's header has always added its text as a bare layout; wrapping it was
    the accident, and it clipped the filename's first letter too.

    Pinned by parentage rather than by pixels: the labels belong to the bar, so
    there is no intermediate widget to paint anything.
    """
    from PySide6.QtWidgets import QLabel

    view = SessionView()
    view.set_session(session)
    bar = view._header_slot.itemAt(0).widget()
    assert bar.objectName() == "HeaderBar"

    heading_label = next(w for w in bar.findChildren(QLabel)
                         if w.text().startswith(("All Targets", "Session of")))
    assert heading_label.parentWidget() is bar, (
        "the header text sits in a wrapper widget, which paints over the bar"
    )


def test_this_screen_offers_no_session_record_in_any_mode(qt_app, session):
    """The record it used to write had no ``run_id``, no ``image`` and no
    ``charts``, so the library listed it as ``session-a1b2c3-d4e5f6`` with an
    empty capture and target and opened it as a report of nothing.

    Combining several *captures* is withdrawn rather than patched: it belongs
    beside the run, not scraped back out of a table of work already filed. What
    this screen still does -- one capture, several targets, reached from the
    capture itself -- is untouched, and ``save_mode="capture"`` still writes the
    ordinary capture record.
    """
    from PySide6.QtWidgets import QPushButton

    assert not hasattr(SessionView, "saveRequested")
    for mode in ("none", "capture", "session"):
        view = SessionView()
        view.set_session(session, save_mode=mode)
        assert not [b for b in view.findChildren(QPushButton)
                    if "Session" in b.text()]


def test_an_unspecified_save_mode_commits_nothing(qt_app, session):
    """The safe answer to "which record should this write" is not to guess one.

    It used to default to ``session``, which is the one mode that wrote a record
    the library could not read back. The shell always says which it means.
    """
    from PySide6.QtWidgets import QPushButton

    view = SessionView()
    view.set_session(session)
    assert view.save_mode == "none"
    assert not [b for b in view.findChildren(QPushButton) if "Save" in b.text()]
