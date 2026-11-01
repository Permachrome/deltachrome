"""Automatic: the profile type and quality chosen by held-out error.

**Why held-out and not a rule.** The design began with a rule -- a linear capture
gets a matrix, a gamma-encoded one a shaper -- and a real linear SG
capture refuted it: the shaper predicts unseen patches at 1.11 ΔE2000 against the
matrix's 1.29, because the shot carries about 0.3% of stray light in its blacks
and a matrix, passing through zero, cannot take an offset out. The physics was
right about the sensor and wrong about the shot. What predicts colours that are
not on the chart is measured, so it is measured rather than assumed.

**The candidates.** Every matrix-family type at every quality -- seconds each --
and the two table types at low and medium only where the chart has patches
enough to fill a table (:data:`TABLE_MIN_PATCHES`). High-quality tables take a
minute or more a build here and are a choice somebody makes, not one to try
seven times. The gamma types are left out: a shaper does everything a gamma
curve does and more.

**Simpler is the default, and a more elaborate model has to earn its place.**
Held-out error measures the chart's own gamut and nothing beyond it -- not the
highlight brighter than its white, not the pigment more saturated than its
patches -- and that is exactly where a simpler model is safer: a matrix extends
in straight lines, a table extrapolates by guesswork. So the candidates are
walked simplest first, and one replaces the incumbent only by beating it on
**both** the mean and the 90th percentile, each by more than
:data:`TIE_FRACTION` -- the two quantities FADGI judges colour on. Ranking by
the mean alone chose an XYZ table on the LaserSoft whose mean was lowest and
whose 90th percentile was a third worse than a shaper's: it fitted most patches
closely by missing some badly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from deltachrome.errors import ProfilingError
from deltachrome.profiling.colprof import ColprofOptions, TABLE_TYPES, run_colprof
from deltachrome.profiling.heldout import DEFAULT_FOLDS, held_out_errors
from deltachrome.profiling.patches import ProfilingSet
from deltachrome.profiling.scale import compute_scale
from deltachrome.profiling.ti3 import format_ti3

#: Every candidate, simplest first -- which is also the order a tie is broken in.
CANDIDATES: tuple[tuple[str, str], ...] = (
    ("matrix", "medium"),
    ("single-shaper", "low"),
    ("single-shaper", "medium"),
    ("single-shaper", "high"),
    ("shaper", "low"),
    ("shaper", "medium"),
    ("shaper", "high"),
    ("xyz-table", "low"),
    ("xyz-table", "medium"),
    ("lab-table", "low"),
    ("lab-table", "medium"),
)

#: The fewest used patches a table is tried from. A 9-cubed table is 729 nodes;
#: below a couple of hundred patches nearly every one is smoothing, and the SG's
#: 140 built a table whose own patches read 6.4 at the 90th percentile.
TABLE_MIN_PATCHES = 200

#: How much better a more elaborate candidate must be, on the mean and on the
#: 90th percentile alike, to replace a simpler one: 2%. Five-fold held-out error
#: is itself an estimate, and a difference inside this is not one to buy
#: extrapolation risk with.
TIE_FRACTION = 0.02


@dataclass(frozen=True)
class Trial:
    """One candidate, as Automatic measured it."""

    profile_type: str
    quality: str
    scale: float | None
    held_out_mean: float
    held_out_p90: float


def candidates(data: ProfilingSet, *, types: Sequence[str] | None = None,
               qualities: Sequence[str] | None = None) -> list[tuple[str, str]]:
    """The candidates worth trying here, simplest first.

    ``types`` and ``qualities`` narrow it, which is how a type chosen by hand
    with quality left automatic is served.
    """
    chosen = [
        (kind, quality) for kind, quality in CANDIDATES
        if (types is None or kind in types)
        and (qualities is None or quality in qualities)
        and (kind not in TABLE_TYPES or len(data.used) >= TABLE_MIN_PATCHES)
    ]
    if types is not None and qualities is not None and not chosen:
        # A combination Automatic would not try of its own accord -- a table at
        # high quality, a gamma type -- is still the operator's to ask for.
        chosen = [(kind, quality) for kind in types for quality in qualities]
    if not chosen:
        raise ProfilingError(
            f"No profile type can be tried from {len(data.used)} patches"
            + (f"; a table needs at least {TABLE_MIN_PATCHES}." if types else ".")
        )
    return chosen


def pick(trials: Sequence[Trial]) -> Trial:
    """The winner among ``trials``, which are given simplest first."""
    margin = 1.0 - TIE_FRACTION
    winner = trials[0]
    for trial in trials[1:]:
        if (trial.held_out_mean < winner.held_out_mean * margin
                and trial.held_out_p90 < winner.held_out_p90 * margin):
            winner = trial
    return winner


def choose(
    data: ProfilingSet,
    options: ColprofOptions,
    colprof: Path,
    *,
    scale_for: str | None,
    folds: int = DEFAULT_FOLDS,
    types: Sequence[str] | None = None,
    qualities: Sequence[str] | None = None,
    progress=None,
) -> tuple[ColprofOptions, list[Trial], dict[str, float]]:
    """The options to build with, every trial, and the winner's held-out errors.

    Each candidate is built once in full to read its scale off, where a scale is
    computed, and then held out at that scale -- so candidates are compared as
    they would be saved. ``progress``, where given, is called before each
    candidate with its type, its quality, its position from 1 and the count.
    """
    trials: list[Trial] = []
    errors_of: dict[tuple[str, str], dict[str, float]] = {}
    tried = candidates(data, types=types, qualities=qualities)
    for number, (kind, quality) in enumerate(tried, start=1):
        if progress is not None:
            progress(kind, quality, number, len(tried))
        trial_options = replace(options, profile_type=kind, quality=quality)
        if scale_for is not None:
            first = run_colprof(format_ti3(data), trial_options, colprof)
            scale, _ = compute_scale(first.icc, data, scale_for)
            trial_options = replace(trial_options, scale=round(scale, 5))
        errors = held_out_errors(data, trial_options, colprof, folds=max(folds, 2))
        if errors is None:
            raise ProfilingError(
                "Too few patches to hold any out, so nothing can be chosen "
                "automatically. Give --type and --quality."
            )
        values = np.asarray(list(errors.values()))
        trials.append(Trial(kind, quality, trial_options.scale,
                            float(values.mean()), float(np.percentile(values, 90))))
        errors_of[(kind, quality)] = errors

    winner = pick(trials)
    chosen = replace(options, profile_type=winner.profile_type, quality=winner.quality,
                     scale=winner.scale)
    return chosen, trials, errors_of[(winner.profile_type, winner.quality)]
