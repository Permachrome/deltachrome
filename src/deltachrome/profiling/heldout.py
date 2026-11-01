"""Held-out error: each patch scored by a profile that never saw it.

The fit -- the used patches scored through the profile built from them -- says
the profile *can* match the chart, and nothing more: a profile with freedom to
spare passes through every patch and does something strange between them. What
the material will meet is colours that are not on the chart, and the honest
prediction of that is a patch the profile was not allowed to learn from.

**K-fold.** The used patches are split into ``folds`` groups; the profile is
built once per group with that group left out, and each group is scored by the
build that did not see it. Every patch is scored exactly once. The profile that
is saved is not one of these: it is built from every used patch, and these
builds exist only to measure it.

**Repeats go into one fold together.** The SG prints the same near-white and
greys again and again round its border, and a held-out white whose twin is in
the training set is not held out at all -- it is the fit, reported under the
wrong name. Patches whose references agree to within :data:`REPEAT_DE` are one
group and are always left out together.

**The scale is held at the saved profile's.** A computed scale is read off the
build it corrects, and recomputing it per fold would measure a slightly
different procedure for twice the cost. ``colprof`` applies it after the fit, so
it moves every fold's readings uniformly and the comparison stays fair.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from deltachrome.imaging.color import delta_e
from deltachrome.profiling.colprof import ColprofOptions, run_colprof
from deltachrome.profiling.patches import MIN_PATCHES, ProfilePatch, ProfilingSet
from deltachrome.profiling.ti3 import format_ti3

#: How many groups the used patches are split into. Five is the usual choice and
#: costs five further builds -- seconds for a matrix profile.
DEFAULT_FOLDS = 5

#: References this close (ΔE2000) are the same colour printed twice. Well above
#: a spectrophotometer's repeatability and well below any two colours a chart
#: prints on purpose.
REPEAT_DE = 1.0

#: Fixed, so the same capture gives the same folds and the same figure.
FOLD_SEED = 20261026


def repeat_groups(patches: list[ProfilePatch]) -> list[list[int]]:
    """Indices of ``patches`` grouped so that repeats of one colour share a group."""
    groups: list[list[int]] = []
    anchors: list[np.ndarray] = []
    for index, patch in enumerate(patches):
        if anchors:
            distances = delta_e(np.stack(anchors), np.broadcast_to(patch.lab, (len(anchors), 3)))
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) < REPEAT_DE:
                groups[nearest].append(index)
                continue
        groups.append([index])
        anchors.append(patch.lab)
    return groups


def assign_folds(patches: list[ProfilePatch], folds: int) -> list[list[int]]:
    """Patch indices per fold: whole repeat groups, shuffled, dealt round-robin.

    Dealt largest group first, each to the fold that is currently smallest, so
    the SG's dozen repeated whites cannot leave one fold with nothing else in it.
    """
    groups = repeat_groups(patches)
    order = np.random.default_rng(FOLD_SEED).permutation(len(groups))
    ranked = sorted((groups[i] for i in order), key=len, reverse=True)
    dealt: list[list[int]] = [[] for _ in range(folds)]
    for group in ranked:
        min(dealt, key=len).extend(group)
    return [sorted(fold) for fold in dealt if fold]


def held_out_errors(
    data: ProfilingSet,
    options: ColprofOptions,
    colprof: Path,
    *,
    folds: int = DEFAULT_FOLDS,
) -> dict[str, float] | None:
    """ΔE2000 per used patch, each from a build that left it out.

    ``None`` where the chart has too few patches to leave any out and still build
    a profile from the rest.
    """
    from deltachrome.profiling import score_profile

    used = data.used
    split = assign_folds(used, folds)
    if len(split) < 2 or any(len(used) - len(fold) < MIN_PATCHES for fold in split):
        return None

    def one(fold: list[int]) -> dict[str, float]:
        left_out = set(fold)
        training = replace(
            data,
            used=[patch for i, patch in enumerate(used) if i not in left_out],
            excluded={},
        )
        run = run_colprof(format_ti3(training), options, colprof)
        return score_profile(run.icc, [used[i] for i in fold])

    errors: dict[str, float] = {}
    # Each build is its own colprof process, so they run side by side.
    with ThreadPoolExecutor(max_workers=min(len(split), os.cpu_count() or 1)) as pool:
        for result in pool.map(one, split):
            errors.update(result)
    return errors
