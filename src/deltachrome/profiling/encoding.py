"""Whether a capture's device values are linear in light: information, not a rule.

It was going to decide the profile type -- a linear capture gets a matrix -- and
a real linear SG capture refuted that, the shaper predicting unseen patches
better because of stray light in the blacks
(:mod:`deltachrome.profiling.automatic`). What is left is worth saying anyway: an
export that was meant to be linear and is not is a mistake caught before a profile
is built on it.

**The exponent of a power law through the neutral ramp**, device value against
the reference's Y, fitted in log-log. A linear encoding reads 1.0 and a gamma 2.2
encoding 0.45. Steps darker than :data:`MIN_Y` are left out, because the blacks
are where flare lifts a linear capture off zero and a log-log fit is most
sensitive to it: the SG reads 0.99 above it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deltachrome.profiling.patches import ProfilingSet
from deltachrome.profiling.scale import ramp_patches
from deltachrome.targets.roles import DEFAULT_ACHROMATIC_CHROMA_MAX

#: The darkest reference Y the fit reads, as a fraction of white.
MIN_Y = 0.02

#: The fewest steps an exponent is fitted through.
MIN_STEPS = 3

#: Below this the encoding is not linear. The geometric mean of linear (1.0) and
#: gamma 2.2 (0.45), so a capture is called whichever it is nearer in ratio.
LINEAR_EXPONENT_MIN = 0.67


@dataclass(frozen=True)
class Encoding:
    #: Device value ~ Y ** exponent through the neutrals.
    exponent: float
    steps: int

    @property
    def linear(self) -> bool:
        return self.exponent >= LINEAR_EXPONENT_MIN

    @property
    def label(self) -> str:
        return "Linear" if self.linear else f"Gamma {1.0 / self.exponent:.1f}"


def detect_encoding(data: ProfilingSet) -> Encoding | None:
    """The neutral ramp's exponent, or ``None`` where too few neutrals are used.

    The ramp if the chart declares one, otherwise every used neutral.
    """
    steps = ramp_patches(data) or [
        patch for patch in data.used
        if float(np.hypot(patch.lab[1], patch.lab[2])) < DEFAULT_ACHROMATIC_CHROMA_MAX
    ]
    y = np.array([float(patch.xyz[1]) for patch in steps])
    device = np.array([float(np.mean(patch.rgb)) for patch in steps])
    keep = (y >= MIN_Y) & (device > 0.0)
    if int(keep.sum()) < MIN_STEPS:
        return None
    exponent = float(np.polyfit(np.log(y[keep]), np.log(device[keep]), 1)[0])
    return Encoding(exponent=exponent, steps=int(keep.sum()))
