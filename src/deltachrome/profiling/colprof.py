"""Running ArgyllCMS's ``colprof``, which does the fitting.

**Bundled, unmodified, and found in a fixed order**: named explicitly, named
by ``DELTACHROME_COLPROF``, the copy inside a packaged build, then the
``PATH``. ArgyllCMS is AGPLv3; its author has agreed in writing that an MIT
application with public source may bundle and call an unmodified ``colprof``.
A machine where none is found is offered no profiling at all.

**The order of the white-point flags matters, and it is not documented.**
``-u <scale>`` switches on the automatic white scaling as well as setting the
scale, so ``-ua -u 1.02`` ends relative: the last ``-u`` wins. The scale is
therefore always written before ``-ua``. And a bare ``-u`` takes the next
argument as its scale if that argument is not a flag -- ``-u input`` is a scale
of zero -- so the basename always comes last, after a flag.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from deltachrome.errors import ProfilingError

#: Profile types, by the name Deltachrome gives them, to ``colprof``'s ``-a``
#: letter. The display-only ``X`` and ``Y`` are absent: ``colprof`` ignores them
#: for an input profile.
PROFILE_TYPES = {
    "matrix": "m",
    "shaper": "s",
    "single-shaper": "S",
    "gamma": "g",
    "single-gamma": "G",
    "xyz-table": "x",
    "lab-table": "l",
}
TABLE_TYPES = frozenset({"xyz-table", "lab-table"})

#: What a person is shown for each type, in the order a menu offers them.
PROFILE_TYPE_LABELS = {
    "matrix": "Matrix",
    "single-shaper": "Single Shaper + Matrix",
    "shaper": "Shaper + Matrix",
    "single-gamma": "Single Gamma + Matrix",
    "gamma": "Gamma + Matrix",
    "xyz-table": "XYZ Table",
    "lab-table": "Lab Table",
}


def profile_type_label(kind: str) -> str:
    return PROFILE_TYPE_LABELS.get(kind, kind)


#: What the chart's white is taken to be. ``absolute`` is ``-ua``: the profile's
#: white point is D50, so relative and absolute rendering coincide and the chart
#: reads its *measured* values -- which is what QC compares against, since it reads
#: the media-relative table and never applies ``wtpt``. ``chart`` maps the chart's
#: white to perfect white; ``chart-scaled`` is ``-u``, scaled so that lighter
#: values survive.
WHITE_MODES = ("absolute", "chart", "chart-scaled")

#: ``-ql``, ``-qm``, ``-qh``. ``-qu`` is Argyll's own test mode and is not offered.
QUALITIES = {"low": "l", "medium": "m", "high": "h"}

ENVIRONMENT_VARIABLE = "DELTACHROME_COLPROF"

#: Where a packaged build carries colprof, relative to the bundle's own files.
#: Nothing is shipped there yet; this is so that, once it is, a fresh install
#: finds it with nothing set.
BUNDLED_RELATIVE = Path("argyll") / "bin"


def bundled_colprof() -> Path | None:
    """The colprof inside a packaged build, if this is one and it carries it."""
    root = getattr(sys, "_MEIPASS", None)
    if not getattr(sys, "frozen", False) or root is None:
        return None
    name = "colprof.exe" if sys.platform == "win32" else "colprof"
    path = Path(root) / BUNDLED_RELATIVE / name
    return path if path.is_file() else None


def find_colprof(explicit: str | Path | None = None) -> Path:
    """Where ``colprof`` is, or a refusal saying where it was looked for."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ProfilingError(f"colprof was named at {path}, and is not there.")
        return path
    named = os.environ.get(ENVIRONMENT_VARIABLE)
    if named:
        path = Path(named).expanduser()
        if not path.is_file():
            raise ProfilingError(
                f"{ENVIRONMENT_VARIABLE} names {path}, and colprof is not there."
            )
        return path
    bundled = bundled_colprof()
    if bundled is not None:
        return bundled
    found = shutil.which("colprof")
    if found:
        return Path(found)
    raise ProfilingError(
        "colprof was not found. Install ArgyllCMS and put its bin folder on the "
        f"PATH, or name colprof with {ENVIRONMENT_VARIABLE}."
    )


@dataclass
class ColprofOptions:
    """Every choice that reaches ``colprof``, and nothing that does not."""

    profile_type: str = "shaper"
    white: str = "absolute"
    #: The ``-u`` scale, or ``None`` for none.
    scale: float | None = None
    quality: str = "medium"
    description: str = ""
    copyright: str = ""
    #: ``-Zt``: the header's transparency attribute, for a transmissive target.
    transmissive: bool = False
    #: ``-R``: white Y <= 1, black and primaries positive, for software that
    #: rejects anything else. Costs accuracy; a camera matrix legitimately has a
    #: negative primary.
    restrict_primaries: bool = False
    #: ``-r``: the readings' average deviation, in percent, that a table is
    #: smoothed for; ``None`` for colprof's 0.5. Only the table's fit reads it
    #: -- matrix and shaper builds come out byte-identical at any value.
    smoothing: float | None = None

    def __post_init__(self) -> None:
        if self.profile_type not in PROFILE_TYPES:
            raise ProfilingError(
                f"Unknown profile type {self.profile_type!r}; choose from "
                f"{', '.join(PROFILE_TYPES)}."
            )
        if self.white not in WHITE_MODES:
            raise ProfilingError(
                f"Unknown white {self.white!r}; choose from {', '.join(WHITE_MODES)}."
            )
        if self.quality not in QUALITIES:
            raise ProfilingError(
                f"Unknown quality {self.quality!r}; choose from {', '.join(QUALITIES)}."
            )
        if self.scale is not None:
            if not 0.0 < self.scale <= 200.0:
                raise ProfilingError(f"A white scale of {self.scale} is out of range.")
            # colprof cannot express it: a scale switches on the automatic
            # scaling, which is chart-scaled and not chart.
            if self.white == "chart":
                raise ProfilingError(
                    "A white scale needs the absolute or the scaled chart white."
                )

        if self.smoothing is not None:
            if self.profile_type not in TABLE_TYPES:
                raise ProfilingError("Smoothing applies to a table profile only.")
            if not 0.0 <= self.smoothing <= 100.0:
                raise ProfilingError(f"A smoothing of {self.smoothing}% is out of range.")

    def arguments(self) -> list[str]:
        """The flags, in an order ``colprof`` reads as meant. No file names."""
        args = ["-v", f"-q{QUALITIES[self.quality]}", f"-a{PROFILE_TYPES[self.profile_type]}"]
        if self.scale is not None:
            args += ["-u", f"{self.scale:.6g}"]
        if self.white == "absolute":
            args.append("-ua")
        elif self.white == "chart-scaled" and self.scale is None:
            args.append("-u")
        if self.smoothing is not None:
            args += ["-r", f"{self.smoothing:g}"]
        if self.restrict_primaries:
            args.append("-R")
        if self.transmissive:
            args.append("-Zt")
        # Attached rather than separate, so a value beginning with '-' is still a
        # value and not the next flag.
        if self.description:
            args.append(f"-D{self.description}")
        if self.copyright:
            args.append(f"-C{self.copyright}")
        return args


@dataclass
class ColprofRun:
    """What ``colprof`` produced, and what it said while producing it."""

    icc: bytes
    arguments: list[str]
    log: str = ""
    warnings: list[str] = field(default_factory=list)


def run_colprof(ti3: str, options: ColprofOptions, colprof: Path) -> ColprofRun:
    """Build a profile from ``.ti3`` text, in a folder that is deleted afterwards."""
    with tempfile.TemporaryDirectory(prefix="deltachrome-profile-") as folder:
        base = Path(folder) / "input"
        base.with_suffix(".ti3").write_text(ti3, encoding="ascii", errors="replace")
        output = Path(folder) / "output.icc"
        arguments = options.arguments() + ["-O", str(output), str(base)]
        try:
            done = subprocess.run(
                [str(colprof), *arguments],
                capture_output=True,
                text=True,
                timeout=COLPROF_TIMEOUT_S,
                cwd=folder,
            )
        except subprocess.TimeoutExpired as failure:
            raise ProfilingError(
                f"colprof did not finish within {COLPROF_TIMEOUT_S // 60} minutes."
            ) from failure
        except OSError as failure:
            raise ProfilingError(f"colprof could not be started: {failure}") from failure
        log = (done.stdout or "") + (done.stderr or "")
        if done.returncode != 0 or not output.exists():
            raise ProfilingError(
                "colprof could not build the profile:\n" + _tail(log)
            )
        warnings = [
            line.strip() for line in log.splitlines()
            if line.lstrip().lower().startswith("warning")
        ]
        return ColprofRun(
            icc=output.read_bytes(), arguments=arguments[:-3], log=log, warnings=warnings
        )


#: A table profile at high quality from a large chart takes minutes; anything
#: that takes an hour has stopped.
COLPROF_TIMEOUT_S = 60 * 60


def _tail(log: str, lines: int = 12) -> str:
    return "\n".join(log.strip().splitlines()[-lines:])
