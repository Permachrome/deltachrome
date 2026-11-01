"""The command line that reproduces a run.

Placement used to be the only thing the frontend could hand back, as a bare
``--corners`` string. That was the smaller half of the problem: corners without
the target, the profile, the reference and the run options do not reproduce
anything, and the operator was left assembling the rest from memory.

Everything needed is already to hand in both places this is called from -- the
choices made on the setup screen, and, for a saved run, the record, which stores
what it was judged as and against precisely so it can be re-scored later. So the
whole invocation is assemblable, and the difference between "here are four
numbers" and "here is the command" is that only one of them can be pasted into a
terminal, a build script or a bug report.

The output is deliberately a *shell* command, quoted with :mod:`shlex`, because
that is where it is going. A path with a space in it is the normal case in this
sector -- masters live in directories named after collections.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Iterable, Mapping

#: Run options, in the order they read best on a command line. Anything absent
#: from the mapping is simply not emitted -- an option left alone must not appear
#: as its default, for the same reason the setup screen does not send it.
OPTION_FLAGS: tuple[tuple[str, str], ...] = (
    ("original_size", "--original-size"),
    ("percentile_method", "--percentile-method"),
    ("frequency_units", "--frequency-units"),
    ("roi_inset", "--roi-inset"),
    ("fiducial_extent_mm", "--fiducial-extent"),
)


def _extent(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return f"{float(value[0]):g}x{float(value[1]):g}"
    return str(value)


def corners_argument(corners: Iterable[Iterable[float]]) -> str:
    """The placement as ``--corners`` takes it."""
    return " ".join(f"{float(x):.1f},{float(y):.1f}" for x, y in corners)


def analyze_command(
    *,
    image_path: str | Path | None,
    target_id: str | None,
    profile_id: str | None,
    corners: Iterable[Iterable[float]] | None = None,
    reference_path: str | Path | None = None,
    reference_library: str | Path | None = None,
    options: Mapping[str, Any] | None = None,
    refined: bool = False,
    save_run: bool = False,
) -> str:
    """A complete ``deltachrome analyze`` invocation, ready to paste.

    Missing pieces are omitted rather than guessed at. A command that is short a
    ``--reference`` fails loudly the moment it is run, which is a better outcome
    than one carrying a plausible path to a file describing a different chart.
    """
    parts: list[str] = ["deltachrome", "analyze"]
    if image_path:
        parts.append(shlex.quote(str(image_path)))
    if target_id:
        parts += ["--target", shlex.quote(str(target_id))]
    if profile_id:
        parts += ["--profile", shlex.quote(str(profile_id))]
    if reference_path:
        parts += ["--reference", shlex.quote(str(reference_path))]
    elif reference_library:
        parts += ["--reference-library", shlex.quote(str(reference_library))]
    if corners is not None:
        parts += ["--corners", shlex.quote(corners_argument(corners))]

    values = dict(options or {})
    for key, flag in OPTION_FLAGS:
        if key not in values or values[key] is None:
            continue
        raw = values[key]
        text = _extent(raw) if key == "fiducial_extent_mm" else f"{raw}"
        parts += [flag, shlex.quote(text)]

    # Recorded because it is part of what produced the placement. Re-running
    # without it against the *unrefined* corners is a different measurement; with
    # the refined corners already substituted it would refine twice.
    if refined:
        parts.append("--refine-placement")
    if save_run:
        parts.append("--save-run")
    return " ".join(parts)


def command_for_record(record: Mapping[str, Any]) -> str:
    """The command that reproduces a saved run, from its own record."""
    image = record.get("image") or {}
    target = record.get("target") or {}
    profile = record.get("profile") or {}
    reference = record.get("reference") or {}
    geometry = record.get("geometry") or {}
    sampling = record.get("sampling") or {}

    options: dict[str, Any] = {}
    inset = sampling.get("roi_inset")
    if inset is not None:
        options["roi_inset"] = inset
    for key in ("original_size", "percentile_method", "frequency_units"):
        if record.get(key) is not None:
            options[key] = record[key]

    return analyze_command(
        image_path=image.get("path"),
        target_id=target.get("target_id"),
        profile_id=profile.get("profile_id"),
        corners=geometry.get("corners_px"),
        # A record keeps where its reference came from; a nominal-value run has
        # no file to point at and must not be given one.
        reference_path=reference.get("path"),
        options=options,
        save_run=True,
    )
