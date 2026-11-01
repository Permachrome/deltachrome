"""Where Deltachrome keeps things when nobody has said where.

Two kinds of thing, kept apart because a person treats them differently:

* **Their work** -- the run library, the reference files, the guidelines -- goes
  in ``~/Documents/Deltachrome``, where they will look for it, back it up and
  find it in a file dialog. It used to be ``~/.deltachrome``, which the Finder
  and Explorer hide, so the records a department keeps as its archive were in a
  folder nobody could see without already knowing the path.
* **The application's own settings** go where each platform keeps them:
  ``~/Library/Application Support/Deltachrome`` on macOS, ``%APPDATA%`` on
  Windows, ``$XDG_CONFIG_HOME`` (or ``~/.config``) elsewhere.

**A folder already in use is not abandoned.** Everything above is a default,
and a default changing must not make anybody's records look as though they have
vanished. So for each folder, if the new location does not exist yet and the
old hidden one holds something, the old one goes on being used -- nothing is
moved behind anybody's back, and Settings shows where each folder is.

``DELTACHROME_HOME`` keeps its meaning: everything, settings included, under the
one folder it names, in the layout it has always had. The test suite depends on
that, and so does a portable install.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HOME_ENV = "DELTACHROME_HOME"

#: The one folder everything used to default under.
LEGACY_HOME_NAME = ".deltachrome"

#: Each folder: its name under the new visible home, and under the old one.
RUN_LIBRARY = ("Run Library", "runs")
REFERENCE_FILES = ("Reference Files", "references")
GUIDELINES = ("Guidelines", "profiles")


def legacy_home() -> Path:
    return Path.home() / LEGACY_HOME_NAME


def documents_home() -> Path:
    """The visible folder a person's work defaults under."""
    return Path.home() / "Documents" / "Deltachrome"


def settings_home() -> Path:
    """Where the application's own settings live."""
    pinned = os.environ.get(HOME_ENV)
    if pinned:
        return Path(pinned)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Deltachrome"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        return (Path(appdata) if appdata else Path.home() / "AppData" / "Roaming") / "Deltachrome"
    config = os.environ.get("XDG_CONFIG_HOME")
    return (Path(config) if config else Path.home() / ".config") / "deltachrome"


def _holds_something(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def default_folder(which: tuple[str, str]) -> Path:
    """The default for one of the three folders, honouring one already in use."""
    visible_name, legacy_name = which
    pinned = os.environ.get(HOME_ENV)
    if pinned:
        return Path(pinned) / legacy_name
    new = documents_home() / visible_name
    old = legacy_home() / legacy_name
    if not new.exists() and _holds_something(old):
        return old
    return new


def legacy_settings_file(name: str) -> Path | None:
    """A settings file left in the old home, if one is there to carry over."""
    if os.environ.get(HOME_ENV):
        return None
    old = legacy_home() / name
    return old if old.is_file() else None
