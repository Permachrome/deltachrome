"""The white scale (``colprof -u <scale>``), computed rather than tuned by hand.

**Why it can be computed exactly.** ``colprof`` applies the scale *after* the
fit: it multiplies the profile's white point by it and divides the relative
output by it, leaving the absolute response unchanged (``xicc/xmatrix.c``,
``xicc/xfit.c``). Deltachrome reads the relative output -- the media-relative
table, and never ``wtpt`` -- so a scale ``s`` divides every XYZ Deltachrome reads
by ``s``, uniformly, moving lightness and leaving hue and chroma where they were.
Build once at no scale, read the patches through the result, and the scale
that puts any chosen reading anywhere is arithmetic. A second build at that
scale lands there by construction, in the reader QC will use.

**Two things to aim for**, the operator's choice:

- ``white``: the brightest neutral patch reads its reference L* exactly. What
  an operator tuning ``-u`` by hand is usually doing -- a hand-tuned LaserSoft
  profile was tuned to exactly this patch. **Brightest by the measurement, never
  by the capture**: the patch recording the highest value in the image may only
  be the one the lights fell on hardest, and aiming at it would build the
  illumination's fall-off into the profile.
- ``ramp``: the worst ΔL*2000 along the whole ramp is as small as it can be,
  which is the quantity FADGI's tone response judges. It can pass a capture that
  matching white alone narrowly fails.
"""

from __future__ import annotations

import numpy as np

from deltachrome.errors import ProfilingError
from deltachrome.imaging.color import delta_L_2000, xyz_to_lab
from deltachrome.imaging.icc import CIE_D50, parse_icc
from deltachrome.profiling.patches import ProfilePatch, ProfilingSet
from deltachrome.targets.roles import DEFAULT_ACHROMATIC_CHROMA_MAX

SCALE_FOR = ("white", "ramp")

#: The range searched for a ramp scale. colprof accepts up to 200, but a scale
#: outside a factor of two is an exposure problem, not something to absorb.
SCALE_BOUNDS = (0.5, 2.0)


def white_patch(data: ProfilingSet) -> ProfilePatch | None:
    """The used patch whose *reference* is the lightest neutral on the chart.

    Neutral by the threshold role resolution uses, so this and the ramp agree on
    what neutral means. Not limited to the declared ramp: a chart's lightest
    neutral is often elsewhere -- the LaserSoft's sits at the top of a colour
    ramp, the SG's in its border. Used patches only, so a clipped white is never
    the aim.
    """
    neutral = [
        patch for patch in data.used
        if float(np.hypot(patch.lab[1], patch.lab[2])) < DEFAULT_ACHROMATIC_CHROMA_MAX
    ]
    return max(neutral, key=lambda patch: float(patch.lab[0]), default=None)


def ramp_patches(data: ProfilingSet) -> list[ProfilePatch]:
    """The used patches of the neutral ramp, lightest first.

    Only used ones: a ramp step left out because it clipped has device values
    that are not its colour, and a scale aimed at it would aim at the clipping.
    """
    # By every id a reading stands for, so a step merged into a family whose
    # first member is elsewhere on the chart is still the ramp's.
    used = {pid: patch for patch in data.used for pid in patch.ids}
    steps = {id(used[pid]): used[pid] for pid in data.ramp if pid in used}
    return sorted(steps.values(), key=lambda patch: -float(patch.lab[0]))


def compute_scale(icc: bytes, data: ProfilingSet, aim: str) -> tuple[float, str]:
    """The scale for ``aim``, given a profile built with none, and what it aimed at."""
    if aim not in SCALE_FOR:
        raise ProfilingError(f"Unknown scale aim {aim!r}; choose from {', '.join(SCALE_FOR)}.")
    profile = parse_icc(icc)

    if aim == "white":
        white = white_patch(data)
        if white is None:
            raise ProfilingError(
                "No usable neutral patch to compute a white scale from. Give --scale "
                "instead."
            )
        built_y = float(profile.to_xyz(white.rgb[None])[0, 1])
        return built_y / float(white.xyz[1]), white.patch_id

    steps = ramp_patches(data)
    if not steps:
        raise ProfilingError(
            "The chart's neutral ramp has no usable step, so there is nothing to "
            "compute a ramp scale from. Give --scale instead."
        )
    built = profile.to_xyz(np.stack([patch.rgb for patch in steps]))
    reference = np.stack([patch.lab for patch in steps])

    def worst(log_scale: float) -> float:
        lab = xyz_to_lab(built / np.exp(log_scale), white_point=CIE_D50)
        return float(delta_L_2000(lab, reference).max())

    from scipy.optimize import minimize_scalar

    found = minimize_scalar(
        worst, bounds=tuple(np.log(SCALE_BOUNDS)), method="bounded",
        options={"xatol": 1e-7},
    )
    return float(np.exp(found.x)), f"{len(steps)} ramp steps"
