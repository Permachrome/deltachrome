"""Where the frontend keeps things.

The defaults are the engine's (:mod:`deltachrome.home`), so the application and
the command line agree about where a person's work is: the three folders under
``~/Documents/Deltachrome``, the settings file where the platform keeps
settings. Each is overridable by environment variable, because a shared lab
drive is a reasonable place for a department's records to live.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from deltachrome import home

#: Where saved analysis runs accumulate. The library view reads this directory.
RUN_DIR_ENV = "DELTACHROME_RUN_DIR"

#: Where the bank of per-unit reference files lives. A lab sharing one bank
#: across several workstations is a deployment decision, which is what this is
#: for; Settings can point it elsewhere for an operator who wants their charts'
#: measurements on a project drive.
REFERENCE_DIR_ENV = "DELTACHROME_REFERENCE_DIR"


#: Where the application keeps its own things. Overridable, and that is not a
#: convenience: without it *this* machine's ``~/.deltachrome`` is the only place
#: the settings file can be, so a test suite exercising the settings screen
#: rewrites the operator's own remembered run library -- which is exactly what
#: happened, and is why a person running the tests found their library pointing
#: at a pytest temporary directory that no longer existed. ``conftest.py`` points
#: it somewhere disposable; a portable install can use it for what it says.
HOME_ENV = home.HOME_ENV


def app_home() -> Path:
    """Where the application keeps its own settings.

    Not where a person's work is: that is :func:`default_run_dir` and its two
    siblings, which default somewhere visible.
    """
    return home.settings_home()


def default_run_dir() -> Path:
    return home.default_folder(home.RUN_LIBRARY)


def default_reference_dir() -> Path:
    return home.default_folder(home.REFERENCE_FILES)


def default_guideline_dir() -> Path:
    return home.default_folder(home.GUIDELINES)


#: What the settings file is called. Beside the default run directory rather than
#: inside it, so pointing the library at a project folder does not drop a
#: settings file into the operator's own records.
SETTINGS_NAME = "frontend.json"


def settings_path() -> Path:
    """Where window preferences are remembered. Read every time, never cached.

    A module-level constant here was the bug: it froze this machine's home at
    import, so nothing set afterwards could redirect it.
    """
    return app_home() / SETTINGS_NAME


#: Set for this session by :func:`set_run_dir`. Takes precedence over the stored
#: choice but not over the environment variable, which is how a scripted or
#: shared-drive deployment pins the location regardless of what a window did.
_SESSION_DIR: Path | None = None

#: The same, for the reference bank.
_SESSION_BANK: Path | None = None


def _chosen(env: str, session: Path | None, key: str, default: Path) -> Path:
    """One folder, resolved: environment, this session, the remembered choice, the default.

    **A remembered folder that is no longer there is not used**, and that is the
    one non-obvious clause. A directory is created on demand wherever this points,
    so a path remembered from a scratch folder, an ejected volume or a deleted
    project would be silently *recreated* -- and what the operator sees is an empty
    library at an address they do not recognise, which is exactly the report that
    prompted this. Falling back to the default is the only reading that cannot
    invent a folder somewhere nobody meant. The environment variable is exempt:
    it is a deployment's instruction rather than a remembered gesture, and a share
    that has not mounted yet is not a reason to write records under home.
    """
    from_env = os.environ.get(env)
    if from_env:
        return Path(from_env)
    if session is not None:
        return session
    stored = _settings().get(key)
    if stored and Path(stored).is_dir():
        return Path(stored)
    return default


def user_run_dir() -> Path:
    """The run library directory. Not created here -- see :func:`ensure_run_dir`."""
    return _chosen(RUN_DIR_ENV, _SESSION_DIR, "run_dir", default_run_dir())


def reference_bank_dir() -> Path:
    """The bank of reference files. Not created here -- see :func:`ensure_reference_bank`.

    Separate from the run directory for a reason that is not tidiness --
    reference measurements describe *physical charts* and outlive any one
    project's records, so a person moving their runs onto a project drive should
    not take their charts' measurements with them.
    """
    return _chosen(REFERENCE_DIR_ENV, _SESSION_BANK, "reference_dir",
                   default_reference_dir())


def ensure_reference_bank() -> Path:
    """The bank, created if it is not there.

    It ships empty and is created on first sight rather than at install time, so
    an operator who never pairs a chart never acquires a folder they did not ask
    for.
    """
    path = reference_bank_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _settings() -> dict:
    # The settings file moved when the defaults did. Until this one has been
    # written, the old one is still what the operator chose, so it is read --
    # and the first write puts it in the new place.
    source = settings_path()
    if not source.is_file():
        source = home.legacy_settings_file(SETTINGS_NAME) or source
    try:
        stored = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return stored if isinstance(stored, dict) else {}


def _write_settings(values: dict) -> None:
    path = settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError:
        # A read-only home is not a reason to refuse anything for this session.
        pass


def remembered(key: str, default=None):
    """A stored window preference. Never load-bearing.

    Column widths, which column the library was last sorted on: things whose
    absence costs a person one gesture and whose corruption must cost them
    nothing. Everything here is read with a default and nothing here changes a
    measurement.
    """
    return _settings().get(key, default)


def remember(key: str, value) -> None:
    values = _settings()
    values[key] = value
    _write_settings(values)


def set_run_dir(path: Path) -> Path:
    """Point the library at another folder, and remember it.

    A run record references its capture by path and never sits beside it, so
    which folder holds the records is genuinely the operator's choice: one
    project's records can live with the project while another's stay under home.
    """
    global _SESSION_DIR
    _SESSION_DIR = Path(path)
    remember("run_dir", str(_SESSION_DIR))
    return _SESSION_DIR


def set_reference_bank(path: Path) -> Path:
    """Point the bank at another folder, and remember it.

    Shows a *different bank* rather than moving anything: the files in a bank are
    copies this application made, and their originals are wherever they came
    from. The row on the settings screen says so, which is why the control can
    exist at all.
    """
    global _SESSION_BANK
    _SESSION_BANK = Path(path)
    remember("reference_dir", str(_SESSION_BANK))
    return _SESSION_BANK


def remember_guideline_dir(path: Path) -> None:
    """Point the engine's guideline folder at ``path``, and remember it.

    Split in two on purpose. The engine holds the location for this process and
    has no settings file; this holds the *choice* between sessions, which is what
    every other window preference here does.
    """
    from deltachrome.profiles.schema import set_profile_dir

    set_profile_dir(path)
    remember("guideline_dir", str(path))


def restore_guideline_dir() -> None:
    """Apply a remembered guideline folder. Called once, as the window opens."""
    from deltachrome.profiles.schema import set_profile_dir

    stored = _settings().get("guideline_dir")
    if stored:
        set_profile_dir(Path(stored))


#: The settings key holding the targets an operator has turned off.
DISABLED_TARGETS_KEY = "disabled_targets"


def disabled_targets() -> set[str]:
    """The targets turned off in Settings: ones this operator does not own.

    **Stored as what is off, never as what is on**, so a target a later release
    adds arrives switched on -- an operator who has never opened the list, and one
    who opened it a year ago, both get the new chart searched for. A stored list
    of what is *on* would make every new target invisible to everybody who had
    ever touched the setting.
    """
    stored = _settings().get(DISABLED_TARGETS_KEY)
    if not isinstance(stored, list):
        return set()
    return {str(t) for t in stored}


def set_disabled_targets(target_ids) -> None:
    remember(DISABLED_TARGETS_KEY, sorted(set(target_ids)))


def identification_candidates() -> list[str] | None:
    """The targets identification should choose among, or ``None`` for all of them.

    ``None`` rather than the full list where nothing is off, so the ordinary case
    passes exactly what it always passed and writes nothing into the record. An id
    left over from a target no longer installed is ignored rather than searched.
    """
    from deltachrome.imaging.detect import identifiable_targets

    off = disabled_targets()
    every = identifiable_targets()
    if not off.intersection(every):
        return None
    return [t for t in every if t not in off]


def ensure_run_dir() -> Path:
    path = user_run_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def batch_dir(captures: Path) -> Path:
    """Where one batch's records go: a folder of its own inside the run library.

    **Inside the library, not beside the captures.** A record is the archive and
    belongs with every other record -- sorted, searched and re-scored by the one
    table that reads them. Writing them next to the masters would put a year of
    findings in whatever folder a project happened to use, and take them with it
    when somebody moved the captures.

    **And in a folder of its own**, because a batch is a thing that was done: a
    hundred rows arriving in one library at one instant are hard to tell from a
    hundred that arrived over a month, and the folder is what says which is which
    to anyone looking at the disk rather than at the table. The table sees them
    either way -- the library scans through subfolders.
    """
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = "".join(c if c.isalnum() or c in "-_" else "-" for c in captures.name)[:60]
    folder = ensure_run_dir() / f"batch-{stamp}-{name}".rstrip("-")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# -- camera profiles -----------------------------------------------------------

#: Where colprof is, when somebody has said. Not a folder like the three above:
#: ArgyllCMS is installed by the operator until it ships inside the application,
#: and the engine's own lookup (``DELTACHROME_COLPROF``, then the PATH) still
#: applies when nothing is remembered here.
COLPROF_KEY = "colprof"
#: The copyright line every profile is built with, until changed.
PROFILE_COPYRIGHT_KEY = "profile_copyright"
#: Where the last profile was saved, so the next one opens there.
PROFILE_DIR_KEY = "profile_dir"


def remembered_colprof() -> Path | None:
    """The colprof chosen in Settings, if one was and it is still there."""
    stored = remembered(COLPROF_KEY)
    if not stored:
        return None
    path = Path(stored)
    return path if path.is_file() else None


def resolve_colprof() -> Path | None:
    """The colprof a build will use, or ``None`` where none can be found."""
    from deltachrome.errors import ProfilingError
    from deltachrome.profiling import find_colprof

    try:
        return find_colprof(remembered_colprof())
    except ProfilingError:
        return None
