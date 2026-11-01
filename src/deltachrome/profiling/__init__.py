"""Building a camera input profile from a capture of a colour chart.

A separate path from QC, and the separation is the point. QC reads a capture
through the profile it carries and judges the colour; this reads the device
values a profile will later stand in front of, and the capture usually carries
no profile at all. The chart is still found, placed and sampled by the same code
QC uses -- that part is not a separate path.

The fitting is ArgyllCMS's ``colprof`` (:mod:`deltachrome.profiling.colprof`).
What this package adds is everything around it: the patches and their reference
values, which patches are left out and why, and the score of the result read
through Deltachrome's own ICC reader rather than ``colprof``'s, whose own check
is CIE76 on the patches it was fitted to.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

from deltachrome.errors import ProfilingError
from deltachrome.imaging.color import delta_e, xyz_to_lab
from deltachrome.imaging.icc import CIE_D50, parse_icc
from deltachrome.profiling.colprof import (
    PROFILE_TYPES,
    QUALITIES,
    TABLE_TYPES,
    WHITE_MODES,
    ColprofOptions,
    ColprofRun,
    find_colprof,
    run_colprof,
)
from deltachrome.profiling.patches import (
    EXCLUDED_CHOSEN,
    EXCLUDED_FLAGGED,
    ProfilePatch,
    ProfilingSet,
    gather_patches,
    merge_repeats,
    profilable,
    repeat_families,
    select_patches,
)
from deltachrome.profiling.automatic import Trial, choose
from deltachrome.profiling.heldout import DEFAULT_FOLDS, held_out_errors
from deltachrome.profiling.scale import SCALE_FOR, compute_scale, ramp_patches, white_patch
from deltachrome.profiling.ti3 import format_ti3

__all__ = [
    "EXCLUDED_CHOSEN",
    "EXCLUDED_FLAGGED",
    "PROFILE_TYPES",
    "QUALITIES",
    "SCALE_FOR",
    "TABLE_TYPES",
    "WHITE_MODES",
    "BuiltProfile",
    "ColprofOptions",
    "ColprofRun",
    "ErrorSummary",
    "ProfilePatch",
    "ProfilingSet",
    "build_automatic",
    "build_profile",
    "check_reference",
    "compute_scale",
    "find_colprof",
    "held_out_errors",
    "format_ti3",
    "gather_patches",
    "merge_repeats",
    "profilable",
    "select_patches",
    "ramp_patches",
    "repeat_families",
    "run_colprof",
    "score_profile",
    "white_patch",
]


@dataclass(frozen=True)
class ErrorSummary:
    """ΔE2000 over a set of patches: what the result table's row holds."""

    count: int
    mean: float
    p90: float
    max: float
    per_patch: dict[str, float] = field(default_factory=dict)

    @classmethod
    def of(cls, errors: Mapping[str, float]) -> "ErrorSummary | None":
        if not errors:
            return None
        values = np.asarray(list(errors.values()), dtype=np.float64)
        return cls(
            count=len(values),
            mean=float(values.mean()),
            p90=float(np.percentile(values, 90)),
            max=float(values.max()),
            per_patch=dict(errors),
        )


@dataclass
class BuiltProfile:
    icc: bytes
    run: ColprofRun
    data: ProfilingSet
    fit: ErrorSummary
    excluded: ErrorSummary | None
    #: Each used patch scored by a build that left it out
    #: (:mod:`deltachrome.profiling.heldout`); ``None`` where it was not asked
    #: for or the chart has too few patches to leave any out.
    held_out: ErrorSummary | None = None
    #: The ``-u`` scale the profile was built with, given or computed; ``None``
    #: for none.
    scale: float | None = None
    #: What a computed scale aimed at -- ``white`` or ``ramp`` -- and the patch
    #: or steps it was read from. ``None`` where the scale was given or absent.
    scale_aim: str | None = None
    scale_basis: str = ""
    #: Every candidate Automatic measured, simplest first; empty where the type
    #: and quality were given.
    trials: list = field(default_factory=list)
    #: What colprof was finally run with, Automatic's choice included.
    options: ColprofOptions | None = None


def check_reference(reference, name: str) -> list[str]:
    """Refuse a reference a profile must not be built against, or say what it is.

    A profile carries its reference's errors into every image made with it, so
    the bar is higher than for QC: a printed aim table is refused outright, and
    only a batch mean shown to sit inside its own population is accepted --
    with a warning, since it is still not this chart's own measurement.
    """
    if reference.is_nominal and not reference.population_is_centred:
        raise ProfilingError(
            f"{name} holds the manufacturer's nominal values, not a measurement of "
            "this chart. A profile needs the chart's own measurement."
        )
    if reference.is_nominal:
        return ["Built against a mean of measured units of this model, not this "
                "chart's own measurement."]
    return []


def score_profile(icc: bytes, patches: list[ProfilePatch]) -> dict[str, float]:
    """ΔE2000 per patch, reading device RGB through ``icc`` as QC would.

    Through :func:`~deltachrome.imaging.icc.parse_icc` and the media-relative
    table it prefers, against CIE D50 -- the same reader and white a later QC run
    of a capture carrying this profile will use. So a figure here is a figure
    there.
    """
    if not patches:
        return {}
    profile = parse_icc(icc)
    rgb = np.stack([patch.rgb for patch in patches])
    lab = xyz_to_lab(profile.to_xyz(rgb), white_point=CIE_D50)
    reference = np.stack([patch.lab for patch in patches])
    errors = delta_e(lab, reference)
    return {patch.patch_id: float(e) for patch, e in zip(patches, errors)}


def build_profile(
    data: ProfilingSet,
    options: ColprofOptions,
    *,
    colprof=None,
    keywords: Mapping[str, str] | None = None,
    scale_for: str | None = None,
    folds: int = DEFAULT_FOLDS,
) -> BuiltProfile:
    """Write the ``.ti3``, run ``colprof`` on it, and score what comes back.

    ``scale_for`` (``white`` or ``ramp``) computes the white scale rather than
    taking it from ``options``: one build with none, the scale read off it
    (:mod:`deltachrome.profiling.scale`), and the build that is kept made at that
    scale. Two runs of ``colprof``, which for a matrix profile is seconds.
    """
    executable = find_colprof(colprof)
    basis = ""
    if scale_for is not None:
        if options.scale is not None:
            raise ProfilingError("Give a scale or compute one, not both.")
        if options.white == "chart":
            raise ProfilingError(
                "A white scale needs the absolute or the scaled chart white."
            )
        first = run_colprof(_ti3(data, options, keywords), options, executable)
        scale, basis = compute_scale(first.icc, data, scale_for)
        options = replace(options, scale=round(scale, 5))
    run = run_colprof(_ti3(data, options, keywords), options, executable)
    fit = ErrorSummary.of(score_profile(run.icc, data.used))
    excluded = ErrorSummary.of(
        score_profile(run.icc, [patch for patch, _ in data.excluded.values()])
    )
    held_out = (
        ErrorSummary.of(held_out_errors(data, options, executable, folds=folds) or {})
        if folds >= 2 else None
    )
    return BuiltProfile(
        icc=run.icc, run=run, data=data, fit=fit, excluded=excluded, held_out=held_out,
        scale=options.scale, scale_aim=scale_for, scale_basis=basis, options=options,
    )


def build_automatic(
    data: ProfilingSet,
    options: ColprofOptions,
    *,
    colprof=None,
    keywords: Mapping[str, str] | None = None,
    scale_for: str | None = None,
    folds: int = DEFAULT_FOLDS,
    types=None,
    qualities=None,
    progress=None,
) -> BuiltProfile:
    """Choose the type and quality by held-out error, then build with them.

    :mod:`deltachrome.profiling.automatic` decides; this builds the winner from
    every used patch and reuses the held-out errors that chose it.
    """
    executable = find_colprof(colprof)
    if scale_for is not None and options.white == "chart":
        raise ProfilingError("A white scale needs the absolute or the scaled chart white.")
    chosen, trials, errors = choose(
        data, options, executable, scale_for=scale_for, folds=folds,
        types=types, qualities=qualities, progress=progress,
    )
    run = run_colprof(_ti3(data, chosen, keywords), chosen, executable)
    # What the winner's scale was read off, said the way build_profile says it.
    basis = ""
    if scale_for == "white":
        basis = white_patch(data).patch_id
    elif scale_for == "ramp":
        basis = f"{len(ramp_patches(data))} ramp steps"
    return BuiltProfile(
        icc=run.icc, run=run, data=data,
        fit=ErrorSummary.of(score_profile(run.icc, data.used)),
        excluded=ErrorSummary.of(
            score_profile(run.icc, [patch for patch, _ in data.excluded.values()])
        ),
        held_out=ErrorSummary.of(errors),
        scale=chosen.scale, scale_aim=scale_for, scale_basis=basis, trials=trials,
        options=chosen,
    )


def _ti3(data: ProfilingSet, options: ColprofOptions, keywords) -> str:
    return format_ti3(data, {**dict(keywords or {}), "settings": " ".join(options.arguments())})
