"""Where Deltachrome keeps things by default, and that moving the defaults loses nothing."""

from __future__ import annotations

import json
import sys

import pytest

from deltachrome import home


@pytest.fixture
def user_home(tmp_path, monkeypatch):
    """A person's home directory, with the test suite's pinned home lifted."""
    monkeypatch.setattr(home.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv(home.HOME_ENV, raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return tmp_path


def test_a_new_installation_keeps_a_persons_work_where_they_can_see_it(user_home):
    base = user_home / "Documents" / "Deltachrome"
    assert home.default_folder(home.RUN_LIBRARY) == base / "Run Library"
    assert home.default_folder(home.REFERENCE_FILES) == base / "Reference Files"
    assert home.default_folder(home.GUIDELINES) == base / "Guidelines"


def test_records_already_in_the_old_hidden_folder_go_on_being_used(user_home):
    old = user_home / ".deltachrome" / "runs"
    old.mkdir(parents=True)
    (old / "run.json").write_text("{}")
    assert home.default_folder(home.RUN_LIBRARY) == old
    # Only the folder that holds something stays behind.
    assert home.default_folder(home.GUIDELINES) == (
        user_home / "Documents" / "Deltachrome" / "Guidelines")


def test_an_empty_old_folder_is_not_a_reason_to_stay(user_home):
    (user_home / ".deltachrome" / "references").mkdir(parents=True)
    assert home.default_folder(home.REFERENCE_FILES) == (
        user_home / "Documents" / "Deltachrome" / "Reference Files")


def test_once_the_new_folder_exists_it_wins(user_home):
    old = user_home / ".deltachrome" / "profiles"
    old.mkdir(parents=True)
    (old / "mine.json").write_text("{}")
    new = user_home / "Documents" / "Deltachrome" / "Guidelines"
    new.mkdir(parents=True)
    assert home.default_folder(home.GUIDELINES) == new


def test_a_pinned_home_keeps_its_own_layout(tmp_path, monkeypatch):
    monkeypatch.setenv(home.HOME_ENV, str(tmp_path))
    assert home.default_folder(home.RUN_LIBRARY) == tmp_path / "runs"
    assert home.default_folder(home.REFERENCE_FILES) == tmp_path / "references"
    assert home.default_folder(home.GUIDELINES) == tmp_path / "profiles"
    assert home.settings_home() == tmp_path
    assert home.legacy_settings_file("frontend.json") is None


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS location")
def test_settings_go_where_macos_keeps_them(user_home):
    assert home.settings_home() == (
        user_home / "Library" / "Application Support" / "Deltachrome")


def test_settings_left_in_the_old_home_are_still_read(user_home):
    from deltachrome_ui import paths

    legacy = user_home / ".deltachrome"
    legacy.mkdir()
    chosen = user_home / "projects" / "runs"
    chosen.mkdir(parents=True)
    (legacy / "frontend.json").write_text(json.dumps({"run_dir": str(chosen)}))
    assert paths.user_run_dir() == chosen
