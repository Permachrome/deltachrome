"""Settings' list of owned targets, and what it does to identification.

It narrows what a capture is searched for and nothing else: the target list on
the placement screen still offers every chart, a search that names nothing says
how many targets it left out, and the stored set is what is *off*, so a target a
later release adds arrives switched on.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from deltachrome.imaging.detect import identifiable_targets  # noqa: E402
from deltachrome_ui import paths  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def nothing_off():
    """The settings file lives for the whole session, so a set left behind here
    would narrow every identification test that runs after this file."""
    paths.set_disabled_targets([])
    yield
    paths.set_disabled_targets([])


def test_only_targets_identification_can_name_are_offered():
    """The white field has nothing on it to identify; turning it off would change
    nothing and say that it did."""
    offered = identifiable_targets()
    assert "flat_field_v1" not in offered
    assert "colorchecker_sg_v1" in offered


def test_nothing_off_searches_exactly_what_it_always_did():
    assert paths.disabled_targets() == set()
    assert paths.identification_candidates() is None


def test_what_is_stored_is_what_is_off():
    paths.set_disabled_targets(["it8_7_1_transmissive_v1"])
    candidates = paths.identification_candidates()
    assert "it8_7_1_transmissive_v1" not in candidates
    # Everything else, including any target this test does not know about, is on.
    assert set(candidates) == set(identifiable_targets()) - {"it8_7_1_transmissive_v1"}


def test_a_target_no_longer_installed_is_ignored():
    paths.set_disabled_targets(["a_chart_that_was_removed_v1"])
    assert paths.identification_candidates() is None


def test_every_target_off_searches_none_of_them(monkeypatch):
    """``None`` is every target and an empty list is none. They used to read alike,
    so turning every target off brought the whole search back."""
    from deltachrome.imaging import detect

    seen = []
    monkeypatch.setattr(detect, "chart_extent", lambda image: None)
    monkeypatch.setattr(
        "deltachrome.targets.registry.load_target",
        lambda target_id: seen.append(target_id) or pytest.fail("searched"),
    )
    result = detect.identify_target(object(), [])
    assert not result.found and not result.candidates
    assert seen == []


def test_a_search_that_names_nothing_says_targets_were_left_out(qt_app):
    from deltachrome.imaging.detect import Identification
    from deltachrome_ui.workers import TargetIdentifier

    owned = identifiable_targets()[:3]
    worker = TargetIdentifier(image=None, target_ids=owned)
    hidden = len(identifiable_targets()) - 3
    declined = Identification(reason="Nothing fitted.")
    named = Identification(target_id=owned[0], found=True, reason="Fits.")
    worker._noted([declined, named])
    assert declined.notes == [
        f"{hidden} targets are turned off in Settings and were not searched for."
    ]
    # A named chart is checked on the canvas; a note on every run is read past.
    assert named.notes == []


def test_nothing_is_said_when_nothing_was_left_out(qt_app):
    from deltachrome.imaging.detect import Identification
    from deltachrome_ui.workers import TargetIdentifier

    declined = Identification(reason="Nothing fitted.")
    TargetIdentifier(image=None)._noted([declined])
    assert declined.notes == []


def test_the_panel_stores_what_is_unticked_and_turns_all_back_on(qt_app):
    from deltachrome_ui.targets_panel import TargetsPanel

    panel = TargetsPanel()
    total = len(identifiable_targets())
    assert set(panel.boxes) == set(identifiable_targets())
    assert panel.all_on.isHidden()

    panel.boxes["it8_7_2_reflective_v1"].setChecked(False)
    assert paths.disabled_targets() == {"it8_7_2_reflective_v1"}
    assert panel.count.text() == f"{total - 1} targets selected for analysis"
    assert not panel.all_on.isHidden()

    panel.all_on.click()
    assert paths.disabled_targets() == set()
    assert all(box.isChecked() for box in panel.boxes.values())


def test_one_target_is_counted_in_the_singular(qt_app):
    from deltachrome_ui.targets_panel import TargetsPanel

    panel = TargetsPanel()
    keep = next(iter(panel.boxes))
    for target_id, box in panel.boxes.items():
        box.setChecked(target_id == keep)
    assert panel.count.text() == "1 target selected for analysis"


def test_a_removed_definition_stays_off_when_the_list_is_edited(qt_app):
    """Uninstalling a definition is not the operator changing their mind."""
    from deltachrome_ui.targets_panel import TargetsPanel

    paths.set_disabled_targets(["a_chart_that_was_removed_v1"])
    panel = TargetsPanel()
    panel.boxes["colorchecker_sg_v1"].setChecked(False)
    assert paths.disabled_targets() == {
        "a_chart_that_was_removed_v1", "colorchecker_sg_v1"
    }


def test_settings_lists_the_targets(qt_app, tmp_path, monkeypatch):
    monkeypatch.setenv("DELTACHROME_REFERENCE_DIR", str(tmp_path / "references"))
    monkeypatch.setenv("DELTACHROME_RUN_DIR", str(tmp_path / "runs"))
    from deltachrome_ui.settings_view import SECTIONS, SettingsView

    view = SettingsView()
    assert "Targets" in SECTIONS
    assert view.pages.count() == len(SECTIONS)
    assert view.pages.widget(SECTIONS.index("Targets")).isAncestorOf(view.targets)


def test_the_command_line_names_its_candidates_explicitly(tmp_path):
    """A script states what it searched rather than reading a window's settings,
    so it reproduces on any machine -- and both refusals come before the decode."""
    from deltachrome.cli import main

    image = tmp_path / "never_read.tif"
    with pytest.raises(SystemExit, match="means nothing without it"):
        main(["analyze", str(image), "--target", "colorchecker_sg_v1",
              "--profile", "fadgi_3star_prints_photographs_2023",
              "--candidates", "colorchecker_sg_v1"])
    with pytest.raises(SystemExit, match="flat_field_v1, which identification"):
        main(["analyze", str(image), "--identify",
              "--profile", "fadgi_3star_prints_photographs_2023",
              "--candidates", "colorchecker_sg_v1", "flat_field_v1"])


def test_targets_are_grouped_by_light_not_by_maker(qt_app):
    """A maker heading files somebody's chart under the wrong name -- the IT8
    layout is printed by several vendors -- where reflected or transmitted light
    is a fact every definition states."""
    from PySide6.QtWidgets import QCheckBox, QLabel

    from deltachrome_ui.targets_panel import TargetsPanel

    panel = TargetsPanel()
    rows = panel._rows
    order, heading = [], None
    for index in range(rows.count()):
        widget = rows.itemAt(index).widget()
        if isinstance(widget, QCheckBox):
            order.append((heading, widget.text()))
        elif isinstance(widget, QLabel):
            heading = widget.text()
    assert [h for h in dict.fromkeys(h for h, _ in order)] == ["Reflective", "Transmissive"]
    assert ("Reflective", "IT8.7/2 Reflective Target") in order
    assert ("Transmissive", "IT8.7/1 Transmissive Target") in order
    for group in ("Reflective", "Transmissive"):
        names = [n for h, n in order if h == group]
        assert names == sorted(names, key=str.casefold)


def test_the_list_does_not_move_when_turn_all_on_appears(qt_app):
    from deltachrome_ui.targets_panel import TargetsPanel

    panel = TargetsPanel()
    panel.resize(700, 900)
    panel.show()
    qt_app.processEvents()
    first = next(iter(panel.boxes.values()))
    before = first.mapTo(panel, first.rect().topLeft())

    panel.boxes["it8_7_2_reflective_v1"].setChecked(False)
    qt_app.processEvents()
    assert panel.all_on.isVisible()
    assert first.mapTo(panel, first.rect().topLeft()) == before
