"""The About section: the license shown is the license shipped."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from deltachrome import __version__  # noqa: E402
from deltachrome_ui import about_panel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_the_license_shown_is_the_repository_license():
    """Held as a constant because a packaged install carries no LICENSE file --
    which is only safe while the copy and the original are the same text."""
    assert about_panel.LICENSE_TEXT == (ROOT / "LICENSE").read_text()


def test_the_copyright_holder_matches_the_license():
    holder = about_panel.COPYRIGHT.removeprefix("© ")
    assert f"Copyright (c) {holder}" in about_panel.LICENSE_TEXT


def test_every_component_is_a_declared_dependency():
    """A component credited and not installed, or renamed upstream, would show
    a dash where its version belongs."""
    pyproject = (ROOT / "pyproject.toml").read_text().lower()
    for _shown, dist, _license, _url in about_panel.COMPONENTS:
        assert f'"{dist.lower()}' in pyproject, dist


def test_it_shows_the_running_version(qt_app):
    panel = about_panel.AboutPanel()
    assert __version__ in panel.version_label.text()
    assert panel.license_text.isReadOnly()


def test_about_is_the_last_settings_section(qt_app):
    from deltachrome_ui.settings_view import SECTIONS, SettingsView

    view = SettingsView()
    assert SECTIONS[-1] == "About"
    assert view.pages.count() == len(SECTIONS)
    assert view.nav.count() == len(SECTIONS)


def test_notices_are_offered_only_where_a_build_put_them(tmp_path, monkeypatch):
    from deltachrome_ui import about_panel

    monkeypatch.delattr(about_panel.sys, "_MEIPASS", raising=False)
    assert about_panel.notices_path() is None
    monkeypatch.setattr(about_panel.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert about_panel.notices_path() is None
    (tmp_path / about_panel.NOTICES_NAME).write_text("notices")
    assert about_panel.notices_path() == tmp_path / about_panel.NOTICES_NAME
