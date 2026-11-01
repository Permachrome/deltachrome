"""The settings screen: what it is for, and the one thing it must not pretend.

It holds what is true *between* runs. Anything that changes an answer belongs
beside the run and travels into the record, which is the rule `options_view.py`
states from the other side -- so the test that matters most here is the boundary
one: nothing on this screen may change a measurement.

The other is where the three folders are. They live together under **General**,
which is what a nav made possible: one place to answer "where does this live", and
the other two pages left to be about their contents rather than about their
location. All three are the operator's to place, and all three defer to their
environment variable, which is how a shared lab drive is pinned regardless of what
a window did.
"""

from __future__ import annotations

import os

from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from deltachrome_ui import paths  # noqa: E402


#: The local reference measurements this test reads. data/ is never committed,
#: so on a machine without it -- a build server -- the test skips rather than
#: fails, as every test needing local captures already does.
needs_local_references = pytest.mark.skipif(
    not Path("data/reference/ColorChecker_SG_M1.txt").is_file()
    or not Path("data/reference/T1910021.cxf").is_file(),
    reason="needs the local reference files in data/reference, which are not committed",
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def view(qt_app, tmp_path, monkeypatch):
    monkeypatch.setenv("DELTACHROME_REFERENCE_DIR", str(tmp_path / "references"))
    monkeypatch.setenv("DELTACHROME_RUN_DIR", str(tmp_path / "runs"))
    from deltachrome_ui.settings_view import SettingsView

    settings = SettingsView()
    settings.refresh()
    return settings


def test_it_shows_where_every_folder_actually_resolves_to(view, tmp_path):
    assert view.run_row.path_label.text() == str(tmp_path / "runs")
    assert view.reference_row.path_label.text() == str(tmp_path / "references")
    from deltachrome.profiles.schema import user_profile_dir

    assert view.guideline_row.path_label.text() == str(user_profile_dir())


def test_every_folder_is_on_one_page(view):
    """One place to answer "where does this live". The reference bank's path used
    to sit beside the bank and the guidelines' beside the guidelines, which meant
    navigating to a thing to find out where it was kept."""
    general = view.pages.widget(0)
    changers = [
        b for b in view.findChildren(type(view.run_row.choose))
        if b.text().startswith("Change") and b is not view.colprof_row.choose
    ]
    assert len(changers) == 3
    assert all(view.isAncestorOf(b) and general.isAncestorOf(b) for b in changers)


def test_a_folder_pinned_by_the_environment_cannot_be_changed_here(view):
    """A control that cannot take effect must not look like one that can, and a
    disabled button with the reason on hover is how this application says that
    everywhere else."""
    assert not view.run_row.choose.isEnabled()
    assert paths.RUN_DIR_ENV in view.run_row.choose.toolTip()


def test_an_unpinned_folder_offers_the_change(qt_app, tmp_path, monkeypatch):
    monkeypatch.delenv("DELTACHROME_RUN_DIR", raising=False)
    monkeypatch.setenv("DELTACHROME_REFERENCE_DIR", str(tmp_path / "references"))
    monkeypatch.setattr(paths, "_SESSION_DIR", tmp_path / "runs")
    from deltachrome_ui.settings_view import SettingsView

    view = SettingsView()
    view.refresh()
    assert view.run_row.choose.isEnabled()
    # And says nothing on hover. A reason on a control is this application's way
    # of saying why it cannot be used, so an enabled one has none to give.
    assert view.run_row.choose.toolTip() == ""


def test_the_bank_folder_is_not_the_run_folder(view, tmp_path):
    """Separate on purpose: a chart's measured values describe a physical object
    and outlive any one project's records, so moving records onto a project drive
    must not take the charts' measurements along."""
    assert paths.user_run_dir() != paths.reference_bank_dir()


def test_no_run_parameter_is_a_free_floating_preference_here(view):
    """The boundary the screen exists to respect, and it is about *anonymity*
    rather than about thresholds.

    A settings screen carrying the percentile method or the ROI inset would
    change what a run measures while leaving two records that look identical and
    were measured differently. The Guideline Builder sets thresholds and is not
    that: a guideline is a named object, every record says which one it was
    judged against, and its departures from the standard are written into it. So
    this checks the three pages that are *not* the builder.
    """
    labels = []
    for index in (0, 1, 2):
        for child in view.pages.widget(index).findChildren(object):
            text = getattr(child, "text", None)
            if callable(text):
                try:
                    labels.append(str(text()))
                except TypeError:
                    pass
    joined = " ".join(labels).lower()
    for parameter in ("percentile", "inset", "delta e", "aggregation", "threshold"):
        assert parameter not in joined, parameter


def test_what_the_builder_changes_is_recorded_by_name(view, tmp_path, monkeypatch):
    """The other half of the same rule, from the side that is allowed to move a
    number: nothing the builder produces is anonymous."""
    monkeypatch.setenv("DELTACHROME_PROFILE_DIR", str(tmp_path / "profiles"))
    from deltachrome.profiles.schema import _index

    _index.cache_clear()
    builder = view.guidelines.builder
    view.guidelines._new()
    assert builder.picker.select_profile("fadgi_4star_prints_photographs_2023")
    builder.name.setText("Recorded By Name")
    next(e for e in builder._edits if e.spec.metric_id == "noise").fields[0].setText("3")
    builder._save()

    saved = builder.editing
    assert saved is not None
    assert saved.profile_id == "recorded_by_name"
    assert saved.local_criteria
    # Recorded by name: the guideline knows noise is a criterion somebody chose
    # and that the rest of the table came across untouched.
    assert "noise" not in saved.inherited
    assert saved.inherited
    _index.cache_clear()


def test_the_library_re_reads_when_the_run_folder_moves(qt_app, tmp_path, monkeypatch):
    """The two screens are separate objects and only the window knows both exist,
    so the signal is the whole of the connection."""
    monkeypatch.delenv("DELTACHROME_RUN_DIR", raising=False)
    monkeypatch.setenv("DELTACHROME_REFERENCE_DIR", str(tmp_path / "references"))
    from deltachrome_ui.settings_view import SettingsView

    view = SettingsView()
    seen = []
    view.runDirChanged.connect(lambda: seen.append(True))
    monkeypatch.setattr(
        "deltachrome_ui.settings_view.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path / "elsewhere"),
    )
    (tmp_path / "elsewhere").mkdir()
    view._choose_run_dir()
    assert seen == [True]
    assert paths.user_run_dir() == tmp_path / "elsewhere"
    paths.set_run_dir(tmp_path / "runs")


@needs_local_references
def test_the_placement_screen_is_told_when_the_bank_changes(qt_app, tmp_path,
                                                            monkeypatch):
    """The bank being empty is the difference between an offer and a dead end, and
    the placement screen is not necessarily on top when a file is added."""
    bank = tmp_path / "references"
    bank.mkdir()
    monkeypatch.setenv("DELTACHROME_REFERENCE_DIR", str(bank))
    monkeypatch.setenv("DELTACHROME_RUN_DIR", str(tmp_path / "runs"))
    from deltachrome_ui.choices import LIBRARY, ChoicesPanel

    choices = ChoicesPanel()
    choices.select_target("colorchecker_sg_v1")
    choices.refresh_bank()
    assert not choices.mode_buttons[LIBRARY].isEnabled()

    from deltachrome.reference.library import ReferenceLibrary

    ReferenceLibrary(bank).add_file("data/reference/ColorChecker_SG_M1.txt")
    choices.refresh_bank()
    assert choices.mode_buttons[LIBRARY].isEnabled()


def test_a_guideline_folder_choice_survives_the_session(qt_app, tmp_path, monkeypatch):
    """Split in two: the engine holds the location for this process and has no
    settings file, and the frontend holds the choice between sessions -- which is
    what every other window preference here does."""
    monkeypatch.delenv("DELTACHROME_PROFILE_DIR", raising=False)
    monkeypatch.setenv(paths.HOME_ENV, str(tmp_path))
    from deltachrome.profiles.schema import set_profile_dir, user_profile_dir

    elsewhere = tmp_path / "shared-guidelines"
    elsewhere.mkdir()
    paths.remember_guideline_dir(elsewhere)
    assert user_profile_dir() == elsewhere

    # A fresh process: the engine's own override is gone and the choice is not.
    set_profile_dir(None)
    assert user_profile_dir() != elsewhere
    paths.restore_guideline_dir()
    assert user_profile_dir() == elsewhere
    set_profile_dir(None)


def test_the_environment_still_outranks_a_remembered_folder(qt_app, tmp_path,
                                                            monkeypatch):
    """Which is how a shared lab drive is pinned regardless of what a window did."""
    monkeypatch.setenv(paths.HOME_ENV, str(tmp_path))
    from deltachrome.profiles.schema import set_profile_dir, user_profile_dir

    paths.remember_guideline_dir(tmp_path / "chosen")
    monkeypatch.setenv("DELTACHROME_PROFILE_DIR", str(tmp_path / "pinned"))
    assert user_profile_dir() == tmp_path / "pinned"
    set_profile_dir(None)
